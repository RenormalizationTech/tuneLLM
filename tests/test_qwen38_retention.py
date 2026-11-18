import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from scripts.qwen38_retention import (
    compare_scores, label_likelihood, ledger_event, matched, run_ifeval,
    score_mc, score_predictions, validate_fixture,
    adapter_identity, prepare, sha, score_run, selected_checkpoint,
)


def fixture():
    return {
        "general_mc": [{"id": "mc", "subject": "ethics", "prompt": "Question\nAnswer:",
                        "choices": [" A", " B", " C", " D"], "answer_index": 1}],
        "gsm8k": [{"id": "gsm", "user": "How many?", "assistant": "Work.\n#### 2"}],
        "ifeval": [{"key": 3, "prompt": "Uppercase", "instruction_id_list": ["change_case:english_capital"], "kwargs": [{}]}],
    }


SMALL = {"general_mc": 1, "gsm8k": 1, "ifeval": 1}


def predictions():
    return [
        {"id": "mc", "task": "general_mc", "choice_loglikelihood": [-3., -.2, -1., -2.], "predicted_index": 1},
        {"id": "gsm", "task": "gsm8k", "output": "The answer is 2", "hit_token_limit": False},
        {"id": "3", "task": "ifeval", "output": "HELLO", "hit_token_limit": False},
    ]


def fake_ifeval(rows, predicted):
    return {"total": 1, "prompt_strict_correct": 1, "records": [{"id": "3", "prompt_strict": True, "instruction_strict": [True]}]}


def test_likelihood_argmax_is_consistent_with_full_vocab_log_softmax_and_ties():
    logits = [1000., 1002., 1001., 999.]
    normalizer = 1002 + math.log(sum(math.exp(x - 1002) for x in logits + [998., 1003.]))
    log_probs = [x - normalizer for x in logits]
    assert label_likelihood(logits) == label_likelihood(log_probs) == 1
    assert label_likelihood([1., 1., 0., 0.]) == 0
    scored = score_mc(fixture()["general_mc"], [predictions()[0]])
    assert scored["correct"] == 1 and scored["total"] == 1


@pytest.mark.parametrize("values", [[0, 1, 2], [0, 1, 2, float("nan")], [0, 1, 2, float("inf")], [0, 1, 2, True]])
def test_invalid_likelihood_scores_rejected(values):
    with pytest.raises(ValueError):
        label_likelihood(values)


def test_single_label_boundary_check_reused_without_ml_imports():
    from scripts.benchmark_screen import choice_token_ids
    class Tokens:
        def __call__(self, text, add_special_tokens=False):
            if text in [" A", " B", " C", " D"]:
                return {"input_ids": [[" A", " B", " C", " D"].index(text) + 10]}
            if text.startswith("Question\nAnswer:"):
                suffix = text[len("Question\nAnswer:"):]
                return {"input_ids": [1, 2] + ([10 + [" A", " B", " C", " D"].index(suffix)] if suffix else [])}
            raise AssertionError(text)
    assert choice_token_ids(Tokens(), fixture()["general_mc"]) == [10, 11, 12, 13]
    class Merges(Tokens):
        def __call__(self, text, **kwargs):
            if text.endswith("Answer: A"):
                return {"input_ids": [1, 999]}
            return super().__call__(text, **kwargs)
    with pytest.raises(ValueError, match="boundary"):
        choice_token_ids(Merges(), fixture()["general_mc"])


def test_valid_fixture_and_malformed_item_rejection():
    assert validate_fixture(fixture(), expected_counts=SMALL) == ["general_mc", "gsm8k", "ifeval"]
    invalid = []
    for field, value in [("answer_index", True), ("answer_index", 4), ("choices", ["A", "B", "C", "D"]), ("prompt", "")]:
        f = fixture(); f["general_mc"][0][field] = value; invalid.append(f)
    f = fixture(); f["gsm8k"][0]["assistant"] = "no numerical marker"; invalid.append(f)
    f = fixture(); f["ifeval"][0]["kwargs"] = []; invalid.append(f)
    f = fixture(); f["ifeval"][0]["key"] = "mc"; invalid.append(f)
    f = fixture(); f["general_mc"] = []; invalid.append(f)
    for f in invalid:
        with pytest.raises(ValueError):
            validate_fixture(f, expected_counts=SMALL)


def test_prediction_ids_counts_and_likelihood_disagreement_fail_closed():
    with pytest.raises(ValueError, match="IDs/counts"):
        matched(fixture()["general_mc"], [])
    with pytest.raises(ValueError):
        matched(fixture()["general_mc"], [{"id": "mc"}, {"id": "mc"}])
    p = predictions()[0]; p["predicted_index"] = 3
    with pytest.raises(ValueError, match="disagrees"):
        score_mc(fixture()["general_mc"], [p])


def test_numeric_and_strict_gsm8k_scores_distinct_and_all_denominators_retained():
    report = score_predictions(fixture(), predictions(), fake_ifeval)
    assert report["gsm8k"]["correct"] == 1 and report["gsm8k"]["strict_correct"] == 0
    assert report["ifeval"]["correct"] == 1 and report["ifeval"]["total"] == 1
    assert report["general_mc"]["details"][0]["id"] == "mc"
    for rows in [predictions()[:-1], predictions() + [predictions()[0]], predictions() + [{"id": "unknown", "task": "unknown"}]]:
        with pytest.raises(ValueError):
            score_predictions(fixture(), rows, fake_ifeval)
    malformed = predictions(); malformed[1].pop("hit_token_limit")
    with pytest.raises(ValueError, match="truncation"):
        score_predictions(fixture(), malformed, fake_ifeval)


def arm_scores(arm, correct):
    tasks = {}
    for task, count in [("general_mc", 8), ("gsm8k", 8), ("ifeval", 6), ("physics_mc", 4)]:
        bits = correct.get(task, [True] * count)
        tasks[task] = {"total": count, "correct": sum(bits),
                       "details": [{"id": task + str(i), "correct": value} for i, value in enumerate(bits)]}
    return {"arm": arm, "comparison_sha256": "same", "tasks": tasks}


def test_matched_gate_thresholds_and_physics_diagnostic_only():
    counts = {"general_mc": 8, "gsm8k": 8, "ifeval": 6, "physics_mc": 4}
    base = arm_scores("base", {})
    candidate = arm_scores("qasper", {"general_mc": [False] * 5 + [True] * 3,
        "gsm8k": [False] * 5 + [True] * 3, "ifeval": [False] * 3 + [True] * 3, "physics_mc": [False] * 4})
    comparison = compare_scores(base, candidate, counts)
    assert comparison["retention_pass"]
    assert len(comparison["tasks"]["general_mc"]["broken_ids"]) == 5
    assert comparison["tasks"]["physics_mc"]["gate_pass"] is None
    candidate["tasks"]["ifeval"]["details"][-1]["correct"] = False
    candidate["tasks"]["ifeval"]["correct"] -= 1
    assert not compare_scores(base, candidate, counts)["retention_pass"]


def test_paired_fixed_broken_ids_and_mismatched_denominators_rejected():
    counts = {"general_mc": 8, "gsm8k": 8, "ifeval": 6, "physics_mc": 4}
    base = arm_scores("base", {"general_mc": [False] + [True] * 7})
    candidate = arm_scores("prl", {})
    comparison = compare_scores(base, candidate, counts)
    assert comparison["tasks"]["general_mc"]["fixed_ids"] == ["general_mc0"]
    altered = copy.deepcopy(candidate); altered["comparison_sha256"] = "different"
    with pytest.raises(ValueError):
        compare_scores(base, altered, counts)
    altered = copy.deepcopy(candidate); altered["tasks"]["gsm8k"]["details"].pop()
    with pytest.raises(ValueError):
        compare_scores(base, altered, counts)
    altered = copy.deepcopy(candidate); altered["tasks"].pop("ifeval")
    with pytest.raises(ValueError):
        compare_scores(base, altered, counts)
    with pytest.raises(ValueError, match="registered task count"):
        compare_scores(base, candidate)  # real CLI cannot use toy denominators


def test_total_resource_ledger_reserves_unfinished_runs_and_debits_actual_time(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger_event(ledger, "arm1", 14400, 8)
    ledger_event(ledger, "arm2", 14400, 8)
    with pytest.raises(ValueError, match="exhausted"):
        ledger_event(ledger, "arm3", 14400, 8)
    ledger_event(ledger, "arm1", 14400, 8, {"status": "incomplete", "elapsed_seconds": 60})
    ledger_event(ledger, "arm3", 3600, 8)
    with pytest.raises(ValueError, match="already attempted"):
        ledger_event(ledger, "arm1", 14400, 8)
    with pytest.raises(ValueError, match="cap mismatch"):
        ledger_event(ledger, "different", 3600, 10)
    assert json.loads(ledger.read_text())["arms"]["arm2"]["status"] == "reserved"


def test_parent_adapter_hash_helper_and_config_task_checks_on_synthetic_files(tmp_path):
    training = tmp_path / "training"
    adapter = training / "checkpoints/step-00001"
    adapter.mkdir(parents=True)
    base = tmp_path / "base"
    base.mkdir()
    manifest = tmp_path / "nf4-manifest.json"
    manifest.write_text("{}")
    saved = {"r": 16, "lora_alpha": 32, "lora_dropout": .05, "task_type": "CAUSAL_LM", "bias": "none",
             "base_model_name_or_path": str(base), "target_modules": ["model.layers.0.mlp.up_proj"]}
    (adapter / "adapter_config.json").write_text(json.dumps(saved))
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic metadata fixture, never load tensors")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in adapter.iterdir()}
    (training / "result.json").write_text(json.dumps({"completed": True, "selected": {"path": str(adapter), "adapter_files": files}}))
    cfg = {"task": "qasper", "nf4_artifact_manifest_sha256": sha(manifest)}
    (training / "config.json").write_text(json.dumps(cfg))
    assert adapter_identity(adapter, "qasper", base, manifest)["adapter_files"] == files
    with pytest.raises(ValueError, match="task"):
        adapter_identity(adapter, "prl", base, manifest)
    (adapter / "adapter_model.safetensors").write_bytes(b"changed after training save")
    with pytest.raises(ValueError, match="changed after save"):
        adapter_identity(adapter, "qasper", base, manifest)


def test_training_run_resolves_only_completed_selected_checkpoint(tmp_path):
    (tmp_path / "result.json").write_text('{"completed":false}')
    with pytest.raises(ValueError, match="training must complete"):
        selected_checkpoint(tmp_path)
    path = tmp_path / "checkpoints/step-00003"
    (tmp_path / "result.json").write_text(json.dumps({"completed": True, "selected": {"path": str(path)}}))
    assert selected_checkpoint(tmp_path) == str(path.resolve())


def test_cpu_preparation_freezes_exact_counts_and_never_loads_runtime(tmp_path, monkeypatch):
    import scripts.qwen38_retention as retention
    import scripts.qwen38_science_runtime as runtime
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU preparation must never call model runtime")
    monkeypatch.setattr(runtime, "load_runtime", forbidden)
    monkeypatch.setattr(runtime, "versions", lambda: {"synthetic": "CPU-only"})
    monkeypatch.setattr(retention, "ROOT", tmp_path)
    source = fixture()
    fixed = {}
    for task, count in [("general_mc", 128), ("gsm8k", 128), ("ifeval", 32)]:
        fixed[task] = []
        for i in range(count):
            row = copy.deepcopy(source[task][0])
            row["key" if task == "ifeval" else "id"] = (1000 + i) if task == "ifeval" else f"{task}-{i}"
            fixed[task].append(row)
    benchmark = tmp_path / "data/benchmark-screen-01"
    benchmark.mkdir(parents=True)
    fixture_path = benchmark / "test.json"
    fixture_path.write_text(json.dumps(fixed))
    (benchmark / "provenance.json").write_text(json.dumps({"fixture_sha256": {"test": sha(fixture_path)}}))
    manifest = tmp_path / "results/qwen38_27b_nf4_artifact_01.json"
    manifest.parent.mkdir()
    manifest.write_text("{}")
    for name in retention.SOURCES:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# synthetic source snapshot\n")
    base = tmp_path / "base"
    base.mkdir()
    (base / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(retention, "verify_ifeval_assets", lambda: {"fixture": True})
    monkeypatch.setattr(retention, "run_ifeval", lambda *args: {"packages": {"nltk": "3.9.2"}, "python": "synthetic"})
    class Tokens:
        pad_token_id, eos_token_id = 0, 1
        def __call__(self, text, **kwargs):
            if text in [" A", " B", " C", " D"]:
                return {"input_ids": [10 + [" A", " B", " C", " D"].index(text)]}
            for choice in [" A", " B", " C", " D"]:
                if text.endswith("Answer:" + choice):
                    return {"input_ids": [ord(c) + 100 for c in text[:-len(choice)]] + self(choice)["input_ids"]}
            return {"input_ids": [ord(c) + 100 for c in text]}
        def apply_chat_template(self, rows, **kwargs):
            assert kwargs["return_dict"] is False and kwargs["enable_thinking"] is False
            return "user:" + rows[0]["content"] + "\nassistant:"
    args = SimpleNamespace(output=tmp_path / "out", fixture=fixture_path, gpu="0", max_seconds=14400,
        ledger_total_hours=12, arm="base", adapter=None, physics=False, base=base,
        scorer_python=sys.executable, ledger=tmp_path / "ledger.json")
    report = prepare(args, Tokens())
    assert report["counts"] == {"general_mc": 128, "gsm8k": 128, "ifeval": 32}
    assert not report["gpu_launched"] and not args.ledger.exists()
    assert not (args.output / "result.json").exists()
    frozen = json.loads((args.output / "prepared.json").read_text())
    assert frozen[0]["prompt"] == fixed["general_mc"][0]["prompt"]  # no MMLU chat wrapping
    assert {r["max_new_tokens"] for r in frozen if r["task"] == "ifeval"} == {1024}
    assert {r["max_new_tokens"] for r in frozen if r["task"] == "gsm8k"} == {512}
    # New arm registrations share exactly the same batching/runtime comparison
    # binding; adapter identity is separately checked by the synthetic helper test.
    batch_bindings = []
    monkeypatch.setattr(retention, "adapter_identity", lambda *args: {"synthetic": True})
    for arm in ("base", "qasper", "prl"):
        batched_args = copy.copy(args)
        batched_args.output = tmp_path / ("batch4-" + arm)
        batched_args.batch_size = 4
        batched_args.arm = arm
        batched_args.adapter = None if arm == "base" else tmp_path / ("adapter-" + arm)
        prepare(batched_args, Tokens())
        registration = json.loads((batched_args.output / "registration.json").read_text())
        binding = registration["comparison_binding"]
        assert binding["protocol"]["batch_size"] == 4 and binding["protocol"]["mc_batch_size"] == 1
        assert [len(p["ids"]) for p in binding["batches"] if p["task"] == "general_mc"] == [1] * 128
        assert len([p for p in binding["batches"] if p["task"] == "gsm8k"]) == 32
        assert len([p for p in binding["batches"] if p["task"] == "ifeval"]) == 8
        batch_bindings.append(binding)
    assert batch_bindings[0] == batch_bindings[1] == batch_bindings[2]
    with pytest.raises(FileExistsError):
        prepare(args, Tokens())
    # Exercise the complete CPU scoring path with synthetic execution records;
    # the fake model loader remains forbidden throughout.
    records = []
    for task, rows in fixed.items():
        for row in rows:
            if task == "general_mc":
                records.append({"task": task, "id": row["id"], "choice_loglikelihood": [-3., -.2, -1., -2.]})
            else:
                records.append({"task": task, "id": str(row["key"]) if task == "ifeval" else row["id"],
                                "output": "#### 2", "hit_token_limit": False})
    predictions_path = args.output / "predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    (args.output / "result.json").write_text(json.dumps({"completed": True,
        "registration_sha256": sha(args.output / "registration.json"), "predictions_sha256": sha(predictions_path)}))
    def official_mock(rows, predictions, python):
        return {"packages": {"nltk": "3.9.2"}, "scores": {"total": len(rows), "prompt_strict_correct": 0,
            "records": [{"id": str(row["key"]), "prompt_strict": False, "instruction_strict": [False]} for row in rows]}}
    monkeypatch.setattr(retention, "run_ifeval", official_mock)
    scored = score_run(args.output)
    assert scored["tasks"]["general_mc"]["correct"] == 128
    assert scored["tasks"]["gsm8k"]["strict_correct"] == 128
    assert scored["tasks"]["ifeval"]["correct"] == 0
    predictions_path.write_text(predictions_path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after execution"):
        score_run(args.output)


def test_pinned_official_ifeval_cpu_subprocess_on_synthetic_outputs():
    python = Path(__file__).resolve().parents[1] / ".venv/bin/python"
    if not python.exists():
        pytest.skip("local legacy CPU scorer environment unavailable")
    rows = fixture()["ifeval"]
    english = ("This is an English response about a scientific experiment. The researchers measure the temperature "
               "and record their results. They explain the method clearly so that other people can repeat the experiment.")
    yes = run_ifeval(rows, [{"id": "3", "output": english.upper()}], python)
    no = run_ifeval(rows, [{"id": "3", "output": english}], python)
    assert yes["scores"]["prompt_strict_correct"] == 1
    assert no["scores"]["prompt_strict_correct"] == 0
    assert yes["packages"]["nltk"] == "3.9.2"


def batch_rows():
    return ([{'id': 'mc0', 'task': 'general_mc', 'input_ids': [7, 8], 'prompt': 'raw', 'max_new_tokens': 0}]
        + [{'id': f'g{i}', 'task': 'gsm8k', 'input_ids': [20 + i] * n, 'prompt': f'Invented {i}', 'max_new_tokens': 512}
           for i, n in enumerate([2, 5, 1, 3, 4])]
        + [{'id': 'instruction', 'task': 'ifeval', 'input_ids': [9, 8, 7], 'prompt': 'Synthetic instruction', 'max_new_tokens': 1024}])


def batch_binding(rows, size=4):
    from scripts.qwen38_retention import batch_plan, digest
    return {'prepared_sha256': digest(rows), 'protocol': {'batch_size': size, 'mc_batch_size': 1, 'pad_token_id': 0,
            'eos_token_id': 1, 'batch_policy': 'source-order contiguous per task; MC singleton; manual left padding'},
            'batches': batch_plan(rows, size, 0, 1)}


def test_source_order_batch_groups_partial_batch_and_three_arm_parity():
    from scripts.qwen38_retention import batch_plan, digest, validate_batch_binding
    rows = batch_rows()
    base, qasper, prl = [batch_plan(copy.deepcopy(rows), 4, 0, 1) for _ in range(3)]
    assert base == qasper == prl
    assert [p['ids'] for p in base] == [['mc0'], ['g0', 'g1', 'g2', 'g3'], ['g4'], ['instruction']]
    assert base[1]['prompt_lengths'] == [2, 5, 1, 3]
    assert base[1]['padding_lengths'] == [3, 0, 4, 2]
    assert base[2]['padding_lengths'] == [0]
    validate_batch_binding(batch_binding(rows), rows)
    old = {'prepared_sha256': digest(rows), 'protocol': {'batch_size': 1}}
    validate_batch_binding(old, rows)
    old['protocol']['batch_size'] = 4
    with pytest.raises(ValueError, match='historical'): validate_batch_binding(old, rows)


@pytest.mark.parametrize('change', ['size', 'groups', 'padding', 'tokens', 'eos', 'mcsize'])
def test_batch_registration_mismatch_rejected(change):
    from scripts.qwen38_retention import validate_batch_binding
    rows = batch_rows(); binding = batch_binding(rows)
    if change == 'size': binding['protocol']['batch_size'] = 1
    elif change == 'groups': binding['batches'][1]['ids'].reverse()
    elif change == 'padding': binding['batches'][1]['padding_lengths'][0] += 1
    elif change == 'tokens': rows[1]['input_ids'][0] += 1
    elif change == 'eos': binding['protocol']['eos_token_id'] = 0
    else: binding['protocol']['mc_batch_size'] = 4
    with pytest.raises(ValueError): validate_batch_binding(binding, rows)


def test_left_padding_first_eos_and_pad_tokens_before_eos_are_preserved():
    from scripts.qwen38_retention import batch_plan, generation_predictions
    from scripts.batched_generation import left_pad
    members = batch_rows()[1:5]
    plan = batch_plan(members, 4, 0, 1)[0]
    prefixes = left_pad([r['input_ids'] for r in members], 0)
    assert prefixes['attention_mask'][0] == [0, 0, 0, 1, 1]
    suffixes = [[11, 1, 0, 0], [0, 12, 1, 0], [1, 0, 0, 0], [4, 0, 3, 2]]
    # Tiny cap here exercises extraction independently of the production512cap.
    plan['max_new_tokens'] = 4
    output = [prefix + suffix for prefix, suffix in zip(prefixes['input_ids'], suffixes)]
    predictions = generation_predictions(members, output, plan, 1, 0)
    assert [p['generated_ids'] for p in predictions] == [[11, 1], [0, 12, 1], [1], [4, 0, 3, 2]]
    assert [p['hit_token_limit'] for p in predictions] == [False, False, False, True]
    assert [p['stop_reason'] for p in predictions] == ['eos', 'eos', 'eos', 'token_limit']
    assert predictions[0]['padding_length'] == 3
    output[0][0] = 99
    with pytest.raises(ValueError, match='prefix'): generation_predictions(members, output, plan, 1, 0)


def test_generate_group_uses_explicit_cached_bf16_settings_and_shared_timing(monkeypatch):
    from contextlib import nullcontext
    from scripts.qwen38_retention import batch_plan, generate_group
    class Tensor:
        def __init__(self, value): self.value = value
        def tolist(self): return self.value
    casts = []
    fake = SimpleNamespace(long='long', bfloat16='bf16', tensor=lambda value, **kwargs: Tensor(value),
        inference_mode=nullcontext, autocast=lambda device, dtype: casts.append((device, dtype)) or nullcontext(),
        cuda=SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
             max_memory_allocated=lambda: 100, max_memory_reserved=lambda: 200))
    monkeypatch.setitem(sys.modules, 'torch', fake)
    class Model:
        def generate(self, **kwargs):
            assert kwargs['eos_token_id'] == 1 and kwargs['pad_token_id'] == 0
            assert kwargs['use_cache'] and kwargs['logits_to_keep'] == 1
            assert not kwargs['do_sample'] and kwargs['num_beams'] == 1 and not kwargs['return_dict_in_generate']
            assert kwargs['attention_mask'].value[0] == [0, 0, 0, 1, 1]
            return Tensor([r + [30, 1, 0] for r in kwargs['input_ids'].value])
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1, decode=lambda ids, **kw: str(ids))
    members = batch_rows()[1:5]
    plan = batch_plan(members, 4, 0, 1)[0]
    predicted, batch = generate_group(Model(), tokenizer, members, plan)
    assert casts == [('cuda', 'bf16')]
    assert all('seconds' not in p and p['batch_seconds'] == batch['batch_seconds'] for p in predicted)
    assert batch['generated_tokens'] == 8 and all(p['generated_ids'] == [30, 1] for p in predicted)
    assert 'not exclusive per-item' in batch['timing_scope']


def test_saved_batch_metadata_rejects_timing_and_group_changes():
    from scripts.qwen38_retention import batch_plan, generation_predictions, validate_prediction_groups
    from scripts.batched_generation import left_pad
    members = batch_rows()[1:5]; binding = batch_binding(members)
    plan = binding['batches'][0]
    prefixes = left_pad([r['input_ids'] for r in members], 0)['input_ids']
    predictions = generation_predictions(members, [p + [9, 1] for p in prefixes], plan, 1, 0)
    scope = 'shared synchronized batch generation; not exclusive per-item time'
    for p in predictions: p.update(batch_seconds=2.0, timing_scope=scope)
    batches = [{**plan, 'batch_seconds': 2.0}]
    validate_prediction_groups(binding, predictions, batches)
    predictions[0]['seconds'] = 2.0
    with pytest.raises(ValueError, match='metadata'): validate_prediction_groups(binding, predictions, batches)
