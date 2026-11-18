import copy
import json
from pathlib import Path

import pytest

from scripts.qwen38_qasper_analysis import (
    EXPECTED_BOOTSTRAP, EXPECTED_GENERATION, EXPECTED_GATES, analyze, canonical,
    evaluate_metrics, load, load_protocol, sha, snapshot_module, snapshot_prompts,
    text_sha, verify_training_binding,
)

ROOT = Path(__file__).resolve().parents[1]
# Synthetic tests construct their own hash-bound snapshots from committed code.
# Historical study verification below still requires the actual run artifacts.
SOURCES = ROOT / "scripts"
SYNTHETIC_SOURCES = (
    "qwen38_eval_science.py", "qwen38_qasper_protocol.py", "qwen38_science_data.py",
    "qwen38_science_runtime.py", "qwen38_science_score.py",
)


class Tokenizer:
    eos_token_id = 1
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) + 10 for c in text]}
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_dict"] is False and kwargs["enable_thinking"] is False
        return "user:" + messages[0]["content"] + "\nassistant:"
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i - 10) for i in ids if i != 1)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value) + "\n")


def write_rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(r) + "\n" for r in records))


def sample_rows():
    records = []
    for i in range(403):
        unanswerable = 267 <= i < 309
        records.append({"id": f"q{i:03d}", "paper_id": f"paper{i % 4}", "user": f"Evidence and question {i}",
            "primary": i < 309, "evidence_strict_primary": i < 267,
            "full_text_in_context": True, "annotation_disagreement": False,
            "nontext_or_unmatched_evidence": False, "nontext_evidence": False, "unmatched_text_evidence": False,
            "paragraphs": [{"id": "p000000", "text": "Evidence"}], "context_ids": ["p000000"],
            "references": [{"answer": "" if unanswerable else "correct", "unanswerable": unanswerable,
                "evidence": [] if unanswerable else ["p000000"], "missing_context_evidence": []}]})
    return records


def response(correct=True, abstain=False, evidence=True):
    return canonical({"answer": "" if abstain else ("correct" if correct else "wrong"),
                      "unanswerable": abstain, "evidence": [] if abstain or not evidence else ["p000000"]})


@pytest.fixture
def bundle(tmp_path):
    prepared = tmp_path / "prepared"
    evaluation = tmp_path / "outputs/qwen38-qasper-evaluation-01"
    source = evaluation / "source"
    source.mkdir(parents=True)
    for name in SYNTHETIC_SOURCES:
        (source / name).write_bytes((SOURCES / name).read_bytes())
    source_hashes = {p.name: sha(p) for p in source.glob("*.py")}
    final = sample_rows()
    dev = [{**final[0], "id": "dev0", "paper_id": "development-only"}]
    for split, data in (("test", final), ("validation", dev), ("train", [{**dev[0], "id": "train0", "paper_id": "training-only"}])):
        write_rows(prepared / (split + ".jsonl"), data)
    base = tmp_path / "base"
    base.mkdir()
    (base / "tokenizer.json").write_text("synthetic tokenizer file")
    manifest = {"task": "qasper", "output_sha256": {s + ".jsonl": sha(prepared / (s + ".jsonl")) for s in ("train", "validation", "test")},
        "tokenizer_sha256": {"tokenizer.json": sha(base / "tokenizer.json")}}
    write(prepared / "manifest.json", manifest)
    write_rows(evaluation / "prompt_development.jsonl", dev)
    nf4 = tmp_path / "nf4-manifest.json"
    write(nf4, {"files": {}})
    cfg = {"task": "qasper", "base": str(base), "versions": {"torch": "synthetic-test-only"},
        "nf4_artifact_manifest": str(nf4), "nf4_artifact_manifest_sha256": sha(nf4),
        "rank": 16, "alpha": 32, "dropout": .05, "seed": 42, "epochs": 1, "max_length": 8192,
        "selection": "minimum development NLL; never final outputs"}
    protocols = {}
    for run_name in ("qwen38-qasper-01", "qwen38-qasper-02"):
        run = tmp_path / "outputs" / run_name
        run.mkdir()
        write_rows(run / "inputs/train.jsonl", [{"id": "t", "input_ids": [1, 2]}])
        write_rows(run / "inputs/validation.jsonl", [{"id": "v", "input_ids": [1, 2]}])
        p = run / "source/scripts/qwen38_nf4_smoke.py"
        p.parent.mkdir(parents=True); p.write_text("# synthetic frozen training source\n")
        trainer = run / "source/scripts/qwen38_train_science.py"
        trainer.write_text("# " + run_name + "\n")
        protocol = {"config": cfg, "input_sha256": {s: sha(run / "inputs" / (s + ".jsonl")) for s in ("train", "validation")},
            "source_sha256": {"scripts/qwen38_nf4_smoke.py": sha(p), "scripts/qwen38_train_science.py": sha(trainer)},
            "selected_ids": {"train": ["t"], "validation": ["v"]}, "target_tokens": {"train": 1, "validation": 1}}
        write(run / "config.json", cfg); write(run / "protocol.json", protocol)
        write(tmp_path / "results/protocols" / (run_name + ".json"), protocol)
        protocols[run_name] = protocol
    old_hash = sha(tmp_path / "outputs/qwen38-qasper-01/protocol.json")
    protocol = {"generation": copy.deepcopy(EXPECTED_GENERATION), "success_gates": copy.deepcopy(EXPECTED_GATES),
        "bootstrap": copy.deepcopy(EXPECTED_BOOTSTRAP), "source_sha256": source_hashes,
        "prompt_candidates": ["base", "verify", "schema"], "prepared_manifest_sha256": sha(prepared / "manifest.json"),
        "final_input_sha256": sha(prepared / "test.jsonl"), "development_input_sha256": sha(prepared / "validation.jsonl"),
        "prompt_selection_input_sha256": sha(evaluation / "prompt_development.jsonl"), "prompt_selection_ids": ["dev0"],
        "prompt_selection": "largest strict primary Answer F1 among base/verify/schema; ties retain base then verify then schema",
        "training_protocol_sha256": old_hash, "counts": {"test": {"records": 403, "primary": 309, "evidence_strict_primary": 267}},
        "limits": ["Synthetic fixture, no model outputs"]}
    write(evaluation / "registration.json", protocol)
    write(tmp_path / "results/protocols/qwen38-qasper-evaluation-01.json", protocol)
    amendment = tmp_path / "results/protocols/qwen38-qasper-integrity-amendment-01.json"
    write(amendment, {"supersedes_prepared_only_training": "qwen38-qasper-01", "training_run": "qwen38-qasper-02",
        "old_training_protocol_sha256": old_hash,
        "new_training_protocol_sha256": sha(tmp_path / "outputs/qwen38-qasper-02/protocol.json"),
        "evaluation_registration_sha256": sha(evaluation / "registration.json"),
        "old_run_optimization_started": False, "new_optimization_started_at_amendment": False})
    training = tmp_path / "outputs/qwen38-qasper-02"
    adapter = training / "checkpoints/step-00001"
    adapter.mkdir(parents=True)
    write(adapter / "adapter_config.json", {"r": 16, "lora_alpha": 32, "lora_dropout": .05, "task_type": "CAUSAL_LM",
        "bias": "none", "base_model_name_or_path": str(base)})
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic; never load tensors")
    files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in adapter.iterdir()}
    write(training / "result.json", {"completed": True, "steps": 1, "protocol_sha256": sha(training / "protocol.json"),
        "selected": {"path": str(adapter), "adapter_files": files}})
    prompts = snapshot_prompts(source / "qwen38_eval_science.py")
    implementation = source / "selection_implementation.py"
    implementation.write_text("# frozen synthetic development-only selector\n")
    write(evaluation / "prompt_selection.json", {"chosen": "verify", "strict_primary_answer_f1": {"base": .1, "verify": .2, "schema": .1},
        "selection_input_sha256": protocol["prompt_selection_input_sha256"], "selection_implementation_sha256": sha(implementation),
        "rule": protocol["prompt_selection"], "final_test_used": False})
    tokenizer = Tokenizer()
    mapping = {"prompt_selection_sha256": sha(evaluation / "prompt_selection.json"), "training_amendment_sha256": sha(amendment)}
    for arm, template in [("base", "base"), ("prompt", "verify"), ("adapter", "base")]:
        run = tmp_path / "outputs" / ("final-" + arm)
        run.mkdir()
        rcfg = {"task": "qasper", "purpose": "final", "template": template, "template_text": prompts[template],
            "input_sha256": protocol["final_input_sha256"], "input_path": str(prepared / "test.jsonl"),
            "ids": [r["id"] for r in final], "max_new_tokens": 512, "max_context": 8192, "seed": 42, "batch_size": 1,
            "cached_generation": True, "enable_thinking": False, "do_sample": False, "gpu": "0", "max_seconds": 14400,
            "base_manifest_sha256": cfg["nf4_artifact_manifest_sha256"], "versions": cfg["versions"],
            "base_preparation": "same PEFT floating casts and rank16 wrapper; disabled adapter for unadapted arm",
            "adapter": str(adapter) if arm == "adapter" else None,
            "adapter_files": {k: v["sha256"] for k, v in files.items()} if arm == "adapter" else {},
            "source_sha256": {"scripts/qwen38_eval_science.py": source_hashes["qwen38_eval_science.py"],
                "scripts/qwen38_science_runtime.py": source_hashes["qwen38_science_runtime.py"],
                "scripts/qwen38_nf4_smoke.py": protocols["qwen38-qasper-02"]["source_sha256"]["scripts/qwen38_nf4_smoke.py"]}}
        write(run / "registration.json", rcfg)
        status = {"completed": True, "generations": 403, "registration_sha256": sha(run / "registration.json"),
                  "runtime": {"casts": "identical"}, "elapsed_seconds": 1.0}
        if arm == "adapter":
            from scripts.qwen38_eval_adapter import validate
            write(run.with_name(run.name + "-adapter-validation.json"), validate(adapter, "qasper", "base"))
            status["adapter_tensor_equality"] = True
        write(run / "result.json", status)
        generated = []
        for i, row in enumerate(final):
            unanswerable = 267 <= i < 309
            output = response(correct=arm == "adapter" or i < (20 if arm == "base" else 40), abstain=unanswerable)
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompts[template] + row["user"]}], return_dict=False, enable_thinking=False)
            generated.append({"id": row["id"], "paper_id": row["paper_id"], "rendered_prompt": prompt,
                "input_ids": tokenizer(prompt)["input_ids"], "generated_ids": tokenizer(output)["input_ids"] + [1],
                "output": output, "ended_with_eos": True, "hit_token_limit": False, "seconds": .01})
        write_rows(run / "predictions.jsonl", generated)
        mapping[arm] = {"run": str(run), "registration_sha256": sha(run / "registration.json"),
                        "result_sha256": sha(run / "result.json"), "predictions_sha256": sha(run / "predictions.jsonl")}
    mapping_file = tmp_path / "runs.json"
    write(mapping_file, mapping)
    return {"root": tmp_path, "prepared": prepared, "evaluation": evaluation, "amendment": amendment,
        "mapping": mapping_file, "output": tmp_path / "analysis.json", "tokenizer": tokenizer,
        "final": final, "protocol": protocol, "training": training, "adapter": adapter}


def run_analysis(b):
    return analyze(b["evaluation"], b["prepared"], b["amendment"], b["mapping"], b["output"], b["tokenizer"], b["root"])


def rebind_run(b, arm):
    mapping = load(b["mapping"])
    run = Path(mapping[arm]["run"])
    status = load(run / "result.json")
    status["registration_sha256"] = sha(run / "registration.json")
    write(run / "result.json", status)
    for name, key in [("registration.json", "registration_sha256"), ("result.json", "result_sha256"), ("predictions.jsonl", "predictions_sha256")]:
        mapping[arm][key] = sha(run / name)
    write(b["mapping"], mapping)


def test_complete_analysis_frozen_scorer_paper_ci_gates_and_pending_retention(bundle):
    before = {p: sha(p) for p in bundle["root"].rglob("*") if p.is_file()}
    result = run_analysis(bundle)
    analysis = result["analysis"]
    assert analysis["better_answer_baseline"] == "prompt"
    assert analysis["science_gates_pass"] and analysis["retention"]["status"] == "pending"
    assert analysis["recommended_adapter"] is None and not result["checkpoint_selection_from_final"]
    assert result["prompt_selection"]["selected_prompt_text_sha256"] == text_sha(snapshot_prompts(bundle["evaluation"] / "source/qwen38_eval_science.py")["verify"])
    assert analysis["format_gate"]["denominator"] == 403 and analysis["evidence_gate"]["denominator"] == 267
    assert analysis["false_unanswerable"]["arms"]["adapter"]["denominator"] == 267
    assert all(sha(p) == expected for p, expected in before.items())
    for mode in ("strict", "lenient"):
        assert result["arms"]["adapter"]["scores"][mode]["groups"]["primary"]["denominator"] == 309
        interval = analysis["bootstrap"][mode]["base"]
        assert interval["draws"] == 10000 and interval["seed"] == 20261117 and interval["papers"] == 4
    with pytest.raises(FileExistsError):
        run_analysis(bundle)


def test_base_selected_prompt_alias_reuses_identical_baseline(bundle):
    selection_file = bundle["evaluation"] / "prompt_selection.json"
    s = load(selection_file); s["chosen"] = "base"; s["strict_primary_answer_f1"] = {"base": .2, "verify": .2, "schema": .1}; write(selection_file, s)
    mapping = load(bundle["mapping"]); mapping["prompt_selection_sha256"] = sha(selection_file); mapping["prompt"] = {"alias": "base"}; write(bundle["mapping"], mapping)
    result = run_analysis(bundle)
    assert result["aliases"] == {"prompt": "base"}
    assert result["analysis"]["better_answer_baseline"] == "base"
    assert result["analysis"]["bootstrap"]["strict"]["prompt"]["alias_of"] == "base"
    assert result["arms"]["base"]["scores"] == result["arms"]["prompt"]["scores"]


@pytest.mark.parametrize("case", ["missing", "duplicate", "wrong_input", "wrong_path", "incomplete", "template", "purpose", "settings", "software", "source", "prompt_tokens", "output_text", "truncation", "paper", "reload", "adapter_weights"])
def test_adversarial_generation_mappings_fail_closed(bundle, case):
    mapping = load(bundle["mapping"])
    run = Path(mapping["adapter"]["run"])
    if case in {"wrong_input", "wrong_path", "template", "purpose", "settings", "software", "source"}:
        cfg = load(run / "registration.json")
        if case == "wrong_input": cfg["input_sha256"] = "0" * 64
        elif case == "wrong_path": cfg["input_path"] = str(bundle["prepared"] / "validation.jsonl")
        elif case == "template": cfg["template"] = "verify"
        elif case == "purpose": cfg["purpose"] = "development"
        elif case == "settings": cfg["batch_size"] = 2
        elif case == "software": cfg["versions"] = {"torch": "different"}
        elif case == "source": cfg["source_sha256"]["scripts/qwen38_eval_science.py"] = "0" * 64
        write(run / "registration.json", cfg)
    elif case in {"incomplete", "reload"}:
        status = load(run / "result.json"); status["completed" if case == "incomplete" else "adapter_tensor_equality"] = False; write(run / "result.json", status)
    elif case == "adapter_weights":
        (bundle["adapter"] / "adapter_model.safetensors").write_bytes(b"changed after save")
    else:
        predicted = [json.loads(s) for s in (run / "predictions.jsonl").read_text().splitlines()]
        if case == "missing": predicted.pop()
        elif case == "duplicate": predicted[-1] = copy.deepcopy(predicted[0])
        elif case == "prompt_tokens": predicted[0]["input_ids"][0] += 1
        elif case == "output_text": predicted[0]["output"] = "tampered"
        elif case == "truncation": predicted[0]["hit_token_limit"] = True
        elif case == "paper": predicted[0]["paper_id"] = "different"
        write_rows(run / "predictions.jsonl", predicted)
    rebind_run(bundle, "adapter")  # semantic rejection even with internally refreshed hashes
    with pytest.raises(ValueError):
        run_analysis(bundle)
    assert not bundle["output"].exists()


def test_unpinned_prediction_change_rejected_by_explicit_mapping(bundle):
    run = Path(load(bundle["mapping"])["base"]["run"])
    with (run / "predictions.jsonl").open("a") as stream: stream.write("\n")
    with pytest.raises(ValueError, match="hash differs"):
        run_analysis(bundle)


def test_registration_gates_scorer_and_amendment_tamper_rejected(bundle):
    source = bundle["evaluation"] / "source/qwen38_science_score.py"
    source.write_text(source.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="hash differs"):
        run_analysis(bundle)


def test_wrong_prompt_choice_cannot_be_selected_from_final_results(bundle):
    path = bundle["evaluation"] / "prompt_selection.json"
    receipt = load(path); receipt["chosen"] = "schema"; write(path, receipt)
    mapping = load(bundle["mapping"]); mapping["prompt_selection_sha256"] = sha(path); write(bundle["mapping"], mapping)
    with pytest.raises(ValueError, match="development/tie rule"):
        run_analysis(bundle)


def metric_fixture():
    final = sample_rows()
    scorer = snapshot_module(SOURCES / "qwen38_science_score.py", sha(SOURCES / "qwen38_science_score.py"))
    baseline = {r["id"]: response(False, 267 <= i < 309) for i, r in enumerate(final)}
    candidate = {r["id"]: response(True, 267 <= i < 309) for i, r in enumerate(final)}
    protocol = {"success_gates": copy.deepcopy(EXPECTED_GATES), "bootstrap": copy.deepcopy(EXPECTED_BOOTSTRAP)}
    return final, scorer, baseline, candidate, protocol


@pytest.mark.parametrize("invalid,passes", [(8, True), (9, False)])
def test_invalid_json_gate_uses_all403_and_lenient_cannot_rescue_it(invalid, passes):
    final, scorer, base, candidate, protocol = metric_fixture()
    for row in final[-invalid:]:
        candidate[row["id"]] = "correct"  # lenient answer accepted; strict JSON invalid
    reports = {"base": scorer.score_qasper(final, base), "prompt": scorer.score_qasper(final, base),
               "adapter": scorer.score_qasper(final, candidate)}
    result = evaluate_metrics(final, reports, scorer, protocol, True)
    assert result["gate_results"]["invalid_json"] is passes
    assert result["format_gate"]["invalid_outputs"] == invalid
    assert result["format_gate"]["fraction"] == invalid / 403
    assert reports["adapter"]["lenient"]["groups"]["all_sampled"]["invalid_outputs"] == 0
    assert result["science_gates_pass"] is passes
    assert protocol["success_gates"] == EXPECTED_GATES  # no threshold mutation


@pytest.mark.parametrize("lost,passes", [(5, True), (6, False)])
def test_evidence_gate_uses267_strict_subset_not_unanswerable_empty_sets(lost, passes):
    final, scorer, base, candidate, protocol = metric_fixture()
    for row in final[:lost]: candidate[row["id"]] = response(True, evidence=False)
    reports = {"base": scorer.score_qasper(final, base), "prompt": scorer.score_qasper(final, base),
               "adapter": scorer.score_qasper(final, candidate)}
    result = evaluate_metrics(final, reports, scorer, protocol, True)
    assert result["gate_results"]["evidence_loss"] is passes
    assert result["evidence_gate"]["denominator"] == 267
    assert result["evidence_gate"]["delta_vs_better_baseline"] == pytest.approx(-lost / 267)


@pytest.mark.parametrize("false_count,passes", [(13, True), (14, False)])
def test_false_unanswerable_gate_excludes_nonprimary_and_true_unanswerables(false_count, passes):
    final, scorer, base, candidate, protocol = metric_fixture()
    for row in final[:false_count] + final[309:]: candidate[row["id"]] = response(abstain=True)
    reports = {"base": scorer.score_qasper(final, base), "prompt": scorer.score_qasper(final, base),
               "adapter": scorer.score_qasper(final, candidate)}
    result = evaluate_metrics(final, reports, scorer, protocol, True)
    assert result["false_unanswerable"]["arms"]["adapter"] == {"count": false_count, "denominator": 267, "rate": false_count / 267}
    assert result["gate_results"]["false_unanswerable"] is passes
    assert result["false_unanswerable"]["delta_vs_better_baseline"] == false_count / 267


def test_positive_gain_alone_cannot_pass_a_ci_crossing_zero():
    final, scorer, base, candidate, protocol = metric_fixture()
    reports = {"base": scorer.score_qasper(final, base), "prompt": scorer.score_qasper(final, base),
               "adapter": scorer.score_qasper(final, candidate)}
    actual_point = reports["adapter"]["strict"]["groups"]["primary"]["answer_f1"] - reports["base"]["strict"]["groups"]["primary"]["answer_f1"]
    class CrossingCI:
        @staticmethod
        def paired_paper_bootstrap(a, b, metric, draws, seed):
            assert draws == 10000 and seed == 20261117 and metric == "answer_f1"
            return {"difference_a_minus_b": actual_point, "ci95": [-.01, 1.]}
    result = evaluate_metrics(final, reports, CrossingCI(), protocol, True)
    assert result["gate_results"]["answer_gain_each_baseline"]
    assert not result["gate_results"]["positive_bootstrap_ci_each_baseline"]
    assert not result["science_gates_pass"]


def test_scientific_training_change_is_not_an_integrity_only_amendment(bundle):
    run = bundle["training"]
    protocol = load(run / "protocol.json")
    protocol["config"]["rank"] = 32
    write(run / "protocol.json", protocol); write(run / "config.json", protocol["config"])
    write(bundle["root"] / "results/protocols/qwen38-qasper-02.json", protocol)
    amendment = load(bundle["amendment"])
    amendment["new_training_protocol_sha256"] = sha(run / "protocol.json")
    write(bundle["amendment"], amendment)
    with pytest.raises(ValueError, match="scientific training settings"):
        verify_training_binding(bundle["amendment"], sha(bundle["amendment"]), bundle["protocol"], bundle["evaluation"], bundle["root"])


def test_real_frozen_registration_and_amendment01_bind_without_final_outputs():
    evaluation = ROOT / "outputs/qwen38-qasper-evaluation-01"
    prepared = ROOT / "outputs/qwen38-science-01/prepared-qasper-tokenizer-01"
    amendment = ROOT / "results/protocols/qwen38-qasper-integrity-amendment-01.json"
    if not all(p.exists() for p in [amendment, evaluation / "registration.json", prepared / "manifest.json",
                                   ROOT / "outputs/qwen38-qasper-02/protocol.json"]):
        pytest.skip("local registered study unavailable")
    protocol, _, final, _, _ = load_protocol(evaluation, prepared)
    binding = verify_training_binding(amendment, sha(amendment), protocol, evaluation)
    assert len(final) == 403 and binding["training"].name == "qwen38-qasper-02"
    assert binding["protocol_chain"]["0-new-protocol"] == load(amendment)["new_training_protocol_sha256"]


def test_mutation_during_analysis_refuses_mixed_input_report(bundle, monkeypatch):
    import scripts.qwen38_qasper_analysis as module
    original = module.evaluate_metrics
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        run = Path(load(bundle["mapping"])["base"]["run"])
        with (run / "predictions.jsonl").open("a") as stream: stream.write("\n")
        return result
    monkeypatch.setattr(module, "evaluate_metrics", mutate)
    with pytest.raises(ValueError, match="input changed before report write"):
        run_analysis(bundle)
    assert not bundle["output"].exists()
