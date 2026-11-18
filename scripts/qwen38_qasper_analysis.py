"""CPU-only final QASPER analysis using the registered frozen scorer.

No generation/checkpoint selection is performed. Run mappings pin all completed
generation artifacts, including prediction hashes absent from the older runner's
receipt. Prompt selection must already be final. Retention remains pending.

--runs JSON schema:
  {"prompt_selection_sha256": "...", "training_amendment_sha256": "...",
   "base": {"run": "outputs/...", "registration_sha256": "...",
            "result_sha256": "...", "predictions_sha256": "..."},
   "prompt": {same four fields for the selected-prompt final arm},
   "adapter": {same four fields for the selected-adapter final arm}}
Only when development selected base, use "prompt": {"alias": "base"}.
All hashes are SHA256 of exact file bytes, captured after generation completes.
No generation directory is discovered or selected automatically.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSION = "qwen38-qasper-analysis-v1"
EXPECTED_COUNTS = {"records": 403, "primary": 309, "evidence_strict_primary": 267}
EXPECTED_GENERATION = {"max_new_tokens": 512, "sequence_cap": 8192, "seed": 42,
                       "batch_size": 1, "thinking": False, "cached": True, "do_sample": False}
EXPECTED_GATES = {"answer_f1_gain_each_baseline": .05, "paired_paper_bootstrap_ci_excludes_zero": True,
    "max_evidence_f1_loss_vs_better_answer_baseline": .02, "max_invalid_json_fraction": .02,
    "max_false_unanswerable_increase": .05,
    "retention": {"gsm8k_total": 128, "gsm8k_max_net_loss": 5, "general_mmlu_total": 128,
                  "general_mmlu_max_net_loss": 5, "ifeval_total": 32, "ifeval_max_strict_net_loss": 3}}
EXPECTED_BOOTSTRAP = {"draws": 10000, "seed": 20261117, "unit": "paper", "method": "paired percentile95"}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def load(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def rows(path):
    return [json.loads(s, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
            for s in Path(path).read_text().split("\n") if s.strip()]


def unique_records(records, name):
    if not isinstance(records, list) or any(not isinstance(r, dict) or not isinstance(r.get("id"), str) or not r["id"] for r in records):
        raise ValueError(f"{name}: nonempty string IDs required")
    keyed = {r["id"]: r for r in records}
    if len(keyed) != len(records):
        raise ValueError(f"{name}: duplicate IDs")
    return keyed


def rooted(root, path):
    result = Path(path)
    return (result if result.is_absolute() else root / result).resolve()


def checked_hash(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64 or sha(path) != expected:
        raise ValueError(f"{label}: hash differs")


def snapshot_module(path, expected):
    checked_hash(path, expected, "frozen scorer")
    spec = importlib.util.spec_from_file_location("qwen38_frozen_qasper_" + expected[:12], path)
    module = importlib.util.module_from_spec(spec)
    # Compile directly instead of writing __pycache__ into the frozen snapshot.
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError("frozen scorer changed while loading")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def snapshot_prompts(path):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PROMPTS" for t in node.targets):
            result = ast.literal_eval(node.value)
            if set(result) != {"base", "verify", "schema"} or any(not isinstance(v, str) for v in result.values()):
                break
            return result
    raise ValueError("frozen generation prompt definitions missing/malformed")


def load_protocol(evaluation, prepared, repo_root=ROOT):
    evaluation, prepared, repo_root = map(Path, (evaluation, prepared, repo_root))
    protocol_file = evaluation / "registration.json"
    registry = repo_root / "results/protocols" / (evaluation.name + ".json")
    if protocol_file.read_bytes() != registry.read_bytes():
        raise ValueError("evaluation registration differs from reviewed registry")
    protocol = load(protocol_file)
    if protocol["generation"] != EXPECTED_GENERATION or protocol["success_gates"] != EXPECTED_GATES or protocol["bootstrap"] != EXPECTED_BOOTSTRAP:
        raise ValueError("registered generation/gates/bootstrap changed")
    if protocol.get("prompt_candidates") != ["base", "verify", "schema"]:
        raise ValueError("registered prompt candidates/tie order differ")
    checked_hash(prepared / "manifest.json", protocol["prepared_manifest_sha256"], "prepared manifest")
    manifest = load(prepared / "manifest.json")
    for name, expected in manifest["output_sha256"].items():
        path = rooted(prepared, name)
        if not path.is_relative_to(prepared.resolve()):
            raise ValueError("prepared payload path escapes directory")
        checked_hash(path, expected, "prepared payload")
    checked_hash(prepared / "test.jsonl", protocol["final_input_sha256"], "final inputs")
    checked_hash(prepared / "validation.jsonl", protocol["development_input_sha256"], "development inputs")
    checked_hash(evaluation / "prompt_development.jsonl", protocol["prompt_selection_input_sha256"], "prompt-selection inputs")
    for name, expected in protocol["source_sha256"].items():
        checked_hash(evaluation / "source" / name, expected, "evaluation source snapshot")
    final = rows(prepared / "test.jsonl")
    unique_records(final, "final inputs")
    actual = {"records": len(final), "primary": sum(r["primary"] for r in final),
              "evidence_strict_primary": sum(r["evidence_strict_primary"] for r in final)}
    if actual != EXPECTED_COUNTS or any(protocol["counts"]["test"][k] != n for k, n in actual.items()):
        raise ValueError("frozen403/309/267 denominators changed")
    if any(type(r["primary"]) is not bool or type(r["evidence_strict_primary"]) is not bool
           or (r["evidence_strict_primary"] and not r["primary"]) for r in final):
        raise ValueError("invalid frozen subset membership")
    for row in final:
        if not row["references"] or any(type(ref["unanswerable"]) is not bool for ref in row["references"]):
            raise ValueError("typed nonempty answer references required")
        if row["primary"] and len({ref["unanswerable"] for ref in row["references"]}) != 1:
            raise ValueError("mixed-answerability primary references forbidden")
        if row["evidence_strict_primary"] and any(ref["unanswerable"] or not (ref["evidence"] or ref.get("unmatched_text_evidence")) for ref in row["references"]):
            raise ValueError("empty-reference evidence gate membership forbidden")
    subset = rows(evaluation / "prompt_development.jsonl")
    unique_records(subset, "prompt development")
    if [r["id"] for r in subset] != protocol["prompt_selection_ids"] or set(r["paper_id"] for r in subset) & {r["paper_id"] for r in final}:
        raise ValueError("prompt-selection IDs differ or overlap final papers")
    scorer = snapshot_module(evaluation / "source/qwen38_science_score.py", protocol["source_sha256"]["qwen38_science_score.py"])
    prompts = snapshot_prompts(evaluation / "source/qwen38_eval_science.py")
    return protocol, manifest, final, scorer, prompts


def load_prompt_selection(evaluation, protocol, expected_hash, prompts):
    path = evaluation / "prompt_selection.json"
    checked_hash(path, expected_hash, "completed prompt selection")
    selection = load(path)
    if selection.get("final_test_used") is not False or selection.get("selection_input_sha256") != protocol["prompt_selection_input_sha256"]:
        raise ValueError("prompt selection lacks registered development-only binding")
    scores = selection.get("strict_primary_answer_f1")
    if not isinstance(scores, dict) or set(scores) != set(protocol["prompt_candidates"]) or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
        raise ValueError("all three finite development selection scores required")
    chosen = max(protocol["prompt_candidates"], key=lambda key: scores[key])
    if selection.get("chosen") != chosen or selection.get("rule") != protocol["prompt_selection"]:
        raise ValueError("selected prompt contradicts registered development/tie rule")
    implementation = evaluation / "source/selection_implementation.py"
    checked_hash(implementation, selection["selection_implementation_sha256"], "prompt-selection implementation")
    prompt_hash = text_sha(prompts[chosen])
    for key in ("chosen_prompt_sha256", "selected_prompt_sha256"):
        if key in selection and selection[key] != prompt_hash:
            raise ValueError("selected prompt text hash differs")
    return {"chosen": chosen, "selection_sha256": sha(path), "selected_prompt_text_sha256": prompt_hash,
            "selection_implementation_sha256": sha(implementation), "development_scores": scores,
            "final_test_used": False}


def verify_training_binding(amendment, expected_hash, protocol, evaluation, repo_root=ROOT):
    """Bind registered amendment01 run01->run02; never select from final answers."""
    repo_root, amendment = Path(repo_root), Path(amendment)
    checked_hash(amendment, expected_hash, "training integrity amendment01")
    current = load(amendment)
    required = {"training_run", "old_training_protocol_sha256", "new_training_protocol_sha256", "evaluation_registration_sha256",
                "supersedes_prepared_only_training", "old_run_optimization_started", "new_optimization_started_at_amendment"}
    if not required <= set(current):
        raise ValueError("unsupported training amendment schema")
    if current["old_training_protocol_sha256"] != protocol["training_protocol_sha256"]:
        raise ValueError("training amendment does not reach original evaluation binding")
    chain = [current]
    old_protocol = None
    hashes = {}
    verified_files = {str(amendment): expected_hash}
    for index, change in enumerate(chain):
        if change["evaluation_registration_sha256"] != sha(evaluation / "registration.json") or change["old_run_optimization_started"] is not False or change["new_optimization_started_at_amendment"] is not False:
            raise ValueError("amendment lacks pre-optimization evaluation binding")
        for role, run_name, expected in [("old", change["supersedes_prepared_only_training"], change["old_training_protocol_sha256"]),
                                         ("new", change["training_run"], change["new_training_protocol_sha256"])]:
            if not isinstance(run_name, str) or Path(run_name).name != run_name:
                raise ValueError("training amendment must name local run directories")
            run = repo_root / "outputs" / run_name
            registry = repo_root / "results/protocols" / (run_name + ".json")
            checked_hash(registry, expected, "training protocol registry")
            checked_hash(run / "protocol.json", expected, "frozen training protocol")
            verified_files[str(registry)] = expected
            verified_files[str(run / "protocol.json")] = expected
            value = load(run / "protocol.json")
            if load(run / "config.json") != value["config"]:
                raise ValueError("training config differs from frozen protocol")
            verified_files[str(run / "config.json")] = sha(run / "config.json")
            for split, file_hash in value["input_sha256"].items():
                checked_hash(run / "inputs" / (split + ".jsonl"), file_hash, "encoded training inputs")
                verified_files[str(run / "inputs" / (split + ".jsonl"))] = file_hash
            for name, file_hash in value["source_sha256"].items():
                checked_hash(run / "source" / name, file_hash, "frozen training source")
                verified_files[str(run / "source" / name)] = file_hash
            if value["config"]["task"] != "qasper":
                raise ValueError("wrong training task")
            if role == "old":
                if (run / "training.jsonl").exists() and (run / "training.jsonl").stat().st_size:
                    raise ValueError("superseded run performed optimization")
                if (run / "result.json").exists() and load(run / "result.json").get("steps", 0) != 0:
                    raise ValueError("superseded run performed optimization")
                old_protocol = value
            else:
                for field in ("selected_ids", "input_sha256", "target_tokens"):
                    if value[field] != old_protocol[field]:
                        raise ValueError("integrity amendment changed training data/exposure")
                old_cfg, new_cfg = old_protocol["config"], value["config"]
                metadata = {"gpu", "repo_root", "python_executable"}
                if {k: v for k, v in old_cfg.items() if k not in metadata} != {k: v for k, v in new_cfg.items() if k not in metadata}:
                    raise ValueError("integrity amendment changed scientific training settings")
            hashes[f"{index}-{role}-protocol"] = expected
    training = repo_root / "outputs" / current["training_run"]
    cfg = load(training / "config.json")
    checked_hash(cfg["nf4_artifact_manifest"], cfg["nf4_artifact_manifest_sha256"], "training NF4 manifest")
    verified_files[cfg["nf4_artifact_manifest"]] = cfg["nf4_artifact_manifest_sha256"]
    return {"training": training, "config": cfg, "protocol": value, "amendment_sha256": sha(amendment),
            "protocol_chain": hashes, "source_amendment": current, "verified_files": verified_files}


def validate_arm(run_spec, expected_template, adapter_arm, final, input_path, protocol, prompts, training, tokenizer, repo_root=ROOT):
    if not isinstance(run_spec, dict) or set(run_spec) != {"run", "registration_sha256", "result_sha256", "predictions_sha256"}:
        raise ValueError("explicit run/registration/result/prediction hash mapping required")
    run = rooted(Path(repo_root), run_spec["run"])
    for name, key in [("registration.json", "registration_sha256"), ("result.json", "result_sha256"), ("predictions.jsonl", "predictions_sha256")]:
        checked_hash(run / name, run_spec[key], "generation " + name)
    cfg, status = load(run / "registration.json"), load(run / "result.json")
    ids = [r["id"] for r in final]
    if status.get("completed") is not True or type(status.get("generations")) is not int or status["generations"] != len(ids) or status.get("error"):
        raise ValueError("complete generation arm with all frozen IDs required")
    if status.get("registration_sha256") != sha(run / "registration.json"):
        raise ValueError("generation result registration binding differs")
    if "predictions_sha256" in status and status["predictions_sha256"] != run_spec["predictions_sha256"]:
        raise ValueError("generation receipt prediction hash differs")
    expected = {"task": "qasper", "purpose": "final", "template": expected_template, "template_text": prompts[expected_template],
        "input_sha256": protocol["final_input_sha256"], "ids": ids,
        "max_new_tokens": 512, "max_context": 8192, "seed": 42, "batch_size": 1,
        "cached_generation": True, "enable_thinking": False, "do_sample": False,
        "base_manifest_sha256": training["config"]["nf4_artifact_manifest_sha256"], "versions": training["config"]["versions"],
        "base_preparation": "same PEFT floating casts and rank16 wrapper; disabled adapter for unadapted arm"}
    for key, value in expected.items():
        if cfg.get(key) != value or (type(value) in (bool, int) and type(cfg.get(key)) is not type(value)):
            raise ValueError(f"wrong generation template/settings/input: {key}")
    if Path(cfg["input_path"]).resolve() != Path(input_path).resolve():
        raise ValueError("generation points to wrong final input path")
    source_hashes = {"scripts/qwen38_eval_science.py": protocol["source_sha256"]["qwen38_eval_science.py"],
                     "scripts/qwen38_science_runtime.py": protocol["source_sha256"]["qwen38_science_runtime.py"],
                     "scripts/qwen38_nf4_smoke.py": training["protocol"]["source_sha256"]["scripts/qwen38_nf4_smoke.py"]}
    if cfg.get("source_sha256") != source_hashes:
        raise ValueError("generation source/version hashes differ from frozen runtime")
    if not isinstance(cfg.get("gpu"), str) or not cfg["gpu"].isdecimal() or type(cfg.get("max_seconds")) is not int or not 0 < cfg["max_seconds"] <= 43200:
        raise ValueError("bounded single-GPU generation registration required")
    adapter_audit, extra_files = None, {}
    if adapter_arm:
        from scripts.qwen38_eval_adapter import validate
        adapter = Path(cfg["adapter"]).resolve() if cfg.get("adapter") else None
        if adapter is None or adapter.parents[1] != training["training"].resolve():
            raise ValueError("adapter does not belong to amended training run")
        adapter_audit = validate(adapter, "qasper", "base")
        if load(run.with_name(run.name + "-adapter-validation.json")) != adapter_audit:
            raise ValueError("adapter validation sidecar differs from selected training-time hashes")
        sidecar = run.with_name(run.name + "-adapter-validation.json")
        extra_files[str(sidecar)] = sha(sidecar)
        extra_files[str(training["training"] / "result.json")] = adapter_audit["training_result_sha256"]
        for name, item in adapter_audit["adapter_files"].items():
            extra_files[str(adapter / name)] = item["sha256"]
        if cfg["adapter_files"] != {k: v["sha256"] for k, v in adapter_audit["adapter_files"].items()} or status.get("adapter_tensor_equality") is not True:
            raise ValueError("generation lacks exact selected-adapter reload identity")
        saved = load(adapter / "adapter_config.json")
        for key, value in {"r": 16, "lora_alpha": 32, "lora_dropout": .05, "task_type": "CAUSAL_LM", "bias": "none", "base_model_name_or_path": training["config"]["base"]}.items():
            if saved.get(key) != value:
                raise ValueError("selected adapter configuration/base identity differs")
        result = load(training["training"] / "result.json")
        if result.get("protocol_sha256") != sha(training["training"] / "protocol.json"):
            raise ValueError("selected adapter training result protocol binding differs")
    elif cfg.get("adapter") is not None or cfg.get("adapter_files") != {}:
        raise ValueError("baseline must be unadapted")
    predictions = rows(run / "predictions.jsonl")
    by_id = unique_records(predictions, "generation predictions")
    if set(by_id) != set(ids):
        raise ValueError("missing/wrong generation IDs")
    for row in final:
        p = by_id[row["id"]]
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompts[expected_template] + row["user"]}],
            tokenize=False, return_dict=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if p.get("paper_id") != row["paper_id"] or p.get("rendered_prompt") != prompt or p.get("input_ids") != prompt_ids or len(prompt_ids) + 512 > 8192:
            raise ValueError("prediction paper/prompt/token IDs differ from registered final inputs")
        tokens = p.get("generated_ids")
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= 512 or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("invalid generated token IDs/count")
        ended = tokens[-1] == tokenizer.eos_token_id
        if (type(p.get("ended_with_eos")) is not bool or p["ended_with_eos"] != ended
                or type(p.get("hit_token_limit")) is not bool or p["hit_token_limit"] != (len(tokens) == 512 and not ended)
                or tokenizer.eos_token_id in tokens[:-1]
                or not isinstance(p.get("output"), str) or tokenizer.decode(tokens, skip_special_tokens=True) != p["output"]):
            raise ValueError("prediction text/EOS/truncation differs from generated tokens")
        if type(p.get("seconds")) not in (int, float) or not math.isfinite(p["seconds"]) or p["seconds"] < 0:
            raise ValueError("finite per-item generation timing required")
    return {"predictions": {key: p["output"] for key, p in by_id.items()}, "runtime": status["runtime"],
        "provenance": {"path": str(run), **run_spec, "adapter_audit": adapter_audit, "versions": cfg["versions"],
            "source_sha256": source_hashes, "template": expected_template, "template_sha256": text_sha(prompts[expected_template]),
            "verified_identity_files": extra_files},
        "generation": {"items": len(predictions), "truncated": sum(p["hit_token_limit"] for p in predictions),
            "truncated_ids": [p["id"] for p in predictions if p["hit_token_limit"]], "generated_tokens": sum(len(p["generated_ids"]) for p in predictions),
            "item_seconds": sum(p["seconds"] for p in predictions), "elapsed_seconds": status.get("elapsed_seconds")}}


def _roundoff_ge(value, threshold):
    return value >= threshold or math.isclose(value, threshold, rel_tol=0, abs_tol=1e-12)


def evaluate_metrics(final, reports, scorer, protocol, prompt_alias=False):
    """Frozen strict gates; lenient results remain diagnostic, never gate rescue."""
    primary_ids = {r["id"] for r in final if r["primary"]}
    answerable_ids = {r["id"] for r in final if r["primary"] and all(not ref["unanswerable"] for ref in r["references"])}
    best = "base" if reports["base"]["strict"]["groups"]["primary"]["answer_f1"] >= reports["prompt"]["strict"]["groups"]["primary"]["answer_f1"] else "prompt"
    differences, comparisons = {}, {}
    for mode in ("strict", "lenient"):
        differences[mode] = {}
        for name in ("base", "prompt"):
            groups = {}
            for group, candidate in reports["adapter"][mode]["groups"].items():
                baseline = reports[name][mode]["groups"][group]
                if candidate["denominator"] != baseline["denominator"]:
                    raise ValueError("scored denominator differs between arms")
                keys = set(candidate) - {"denominator"}
                groups[group] = {"denominator": candidate["denominator"], "delta_adapter_minus_baseline": {
                    key: candidate[key] - baseline[key] if candidate[key] is not None and baseline[key] is not None else None for key in keys}}
                groups[group]["invalid_json_fraction_delta"] = ((candidate["invalid_outputs"] - baseline["invalid_outputs"]) / candidate["denominator"]
                    if candidate["denominator"] else None)
            differences[mode][name] = groups
        mode_comparisons = {}
        for name in ("base", "prompt"):
            if name == "prompt" and prompt_alias:
                mode_comparisons[name] = {**mode_comparisons["base"], "alias_of": "base"}
                continue
            candidate = [r for r in reports["adapter"][mode]["records"] if r["id"] in primary_ids]
            baseline = [r for r in reports[name][mode]["records"] if r["id"] in primary_ids]
            mode_comparisons[name] = scorer.paired_paper_bootstrap(candidate, baseline, metric="answer_f1",
                draws=protocol["bootstrap"]["draws"], seed=protocol["bootstrap"]["seed"])
            point = reports["adapter"][mode]["groups"]["primary"]["answer_f1"] - reports[name][mode]["groups"]["primary"]["answer_f1"]
            if not math.isclose(mode_comparisons[name]["difference_a_minus_b"], point, rel_tol=0, abs_tol=1e-12):
                raise ValueError("bootstrap point differs from frozen primary score difference")
        comparisons[mode] = mode_comparisons
    gates = protocol["success_gates"]
    gains = {name: {"gain": comparisons["strict"][name]["difference_a_minus_b"],
        "ci95": comparisons["strict"][name]["ci95"],
        "gain_pass": _roundoff_ge(comparisons["strict"][name]["difference_a_minus_b"], gates["answer_f1_gain_each_baseline"]),
        "ci_pass": comparisons["strict"][name]["ci95"][0] > 0} for name in ("base", "prompt")}
    def false_unanswerable(name):
        scored = [r for r in reports[name]["strict"]["records"] if r["id"] in answerable_ids]
        return {"count": sum(r["abstained"] for r in scored), "denominator": len(scored),
                "rate": sum(r["abstained"] for r in scored) / len(scored) if scored else None}
    false = {name: false_unanswerable(name) for name in reports}
    invalid = reports["adapter"]["strict"]["groups"]["all_sampled"]["invalid_outputs"]
    invalid_fraction = invalid / len(final)
    evidence_delta = reports["adapter"]["strict"]["groups"]["evidence_strict_primary"]["evidence_f1"] - reports[best]["strict"]["groups"]["evidence_strict_primary"]["evidence_f1"]
    false_delta = false["adapter"]["rate"] - false[best]["rate"] if answerable_ids else None
    passed = {"answer_gain_each_baseline": all(g["gain_pass"] for g in gains.values()),
        "positive_bootstrap_ci_each_baseline": all(g["ci_pass"] for g in gains.values()),
        "evidence_loss": _roundoff_ge(evidence_delta, -gates["max_evidence_f1_loss_vs_better_answer_baseline"]),
        "invalid_json": _roundoff_ge(gates["max_invalid_json_fraction"], invalid_fraction),
        "false_unanswerable": false_delta is not None and _roundoff_ge(gates["max_false_unanswerable_increase"], false_delta)}
    return {"better_answer_baseline": best, "better_baseline_rule": "strict primary AnswerF1; base wins ties",
        "answer_comparisons": gains, "bootstrap": comparisons, "group_differences": differences,
        "evidence_gate": {"group": "evidence_strict_primary", "denominator": reports["adapter"]["strict"]["groups"]["evidence_strict_primary"]["denominator"], "delta_vs_better_baseline": evidence_delta},
        "format_gate": {"group": "all_sampled", "denominator": len(final), "invalid_outputs": invalid, "fraction": invalid_fraction,
                        "maximum_fraction": gates["max_invalid_json_fraction"]},
        "false_unanswerable": {"definition": "strict valid unanswerable flag on primary rows where every original reference is answerable; strict invalid outputs count against format, not abstention",
            "arms": false, "delta_vs_better_baseline": false_delta,
            "delta_vs_each_baseline": {name: false["adapter"]["rate"] - false[name]["rate"] if answerable_ids else None for name in ("base", "prompt")}},
        "gate_parser": "strict; lenient metrics/intervals are diagnostic only", "gate_results": passed,
        "science_gates_pass": all(passed.values()), "retention": {"status": "pending", "registered_gates": gates["retention"]},
        "recommended_adapter": None, "overall_status": "pending_retention" if all(passed.values()) else "science_gates_failed_retention_pending",
        "numeric_comparison": "absolute1e-12 tolerance solely for float roundoff at inclusive point-estimate boundaries; CI lower bound strictly >0"}


def analyze(evaluation, prepared, amendment, mapping_file, output, tokenizer=None, repo_root=ROOT):
    evaluation, prepared, amendment, mapping_file, output, repo_root = map(lambda p: Path(p).resolve(), (evaluation, prepared, amendment, mapping_file, output, repo_root))
    if output.exists():
        raise FileExistsError(output)
    mapping_hash = sha(mapping_file)
    mapping = load(mapping_file)
    if set(mapping) != {"base", "prompt", "adapter", "prompt_selection_sha256", "training_amendment_sha256"}:
        raise ValueError("explicit base/prompt/adapter mappings and selection/amendment hashes required")
    protocol, manifest, final, scorer, prompts = load_protocol(evaluation, prepared, repo_root)
    selection = load_prompt_selection(evaluation, protocol, mapping["prompt_selection_sha256"], prompts)
    training = verify_training_binding(amendment, mapping["training_amendment_sha256"], protocol, evaluation, repo_root)
    for name, expected in manifest["tokenizer_sha256"].items():
        checked_hash(Path(training["config"]["base"]) / name, expected, "frozen tokenizer")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(training["config"]["base"], local_files_only=True, trust_remote_code=False)
    final_path = prepared / "test.jsonl"
    arms = {"base": validate_arm(mapping["base"], "base", False, final, final_path, protocol, prompts, training, tokenizer, repo_root)}
    alias = selection["chosen"] == "base"
    if alias:
        if mapping["prompt"] != {"alias": "base"}:
            raise ValueError("base-selected prompt baseline must alias the identical base run")
        arms["prompt"] = arms["base"]
    else:
        arms["prompt"] = validate_arm(mapping["prompt"], selection["chosen"], False, final, final_path, protocol, prompts, training, tokenizer, repo_root)
    arms["adapter"] = validate_arm(mapping["adapter"], "base", True, final, final_path, protocol, prompts, training, tokenizer, repo_root)
    if any(arm["runtime"] != arms["base"]["runtime"] for arm in arms.values()):
        raise ValueError("runtime preparation/casts differ between arms")
    reports = {name: scorer.score_qasper(final, arm["predictions"]) for name, arm in arms.items()}
    report = {"version": VERSION, "protocol": protocol, "evaluation_registration_sha256": sha(evaluation / "registration.json"),
        "run_mapping_sha256": mapping_hash, "prepared_manifest_sha256": sha(prepared / "manifest.json"),
        "final_input_sha256": sha(final_path), "final_ids_sha256": text_sha(canonical([r["id"] for r in final])),
        "frozen_source_sha256": protocol["source_sha256"], "analysis_source_sha256": sha(__file__), "python": platform.python_version(),
        "adapter_validation_helper_sha256": arms["adapter"]["provenance"]["adapter_audit"]["validation_script_sha256"],
        "training_binding": {k: v for k, v in training.items() if k not in {"training", "config", "protocol"}},
        "training_run": str(training["training"]), "prompt_selection": selection, "aliases": {"prompt": "base"} if alias else {},
        "arms": {name: {"provenance": arm["provenance"], "generation": arm["generation"], "scores": reports[name]} for name, arm in arms.items()},
        "analysis": evaluate_metrics(final, reports, scorer, protocol, alias), "checkpoint_selection_from_final": False,
        "limitations": protocol["limits"] + ["Generation outputs lack upstream committed prediction hashes; explicit caller-supplied run mapping pins them before analysis", "Lexical/synthetic scorer parity does not establish full official benchmark equivalence"]}
    pinned = {str(mapping_file): mapping_hash, str(evaluation / "registration.json"): report["evaluation_registration_sha256"],
        str(repo_root / "results/protocols" / (evaluation.name + ".json")): report["evaluation_registration_sha256"],
        str(prepared / "manifest.json"): protocol["prepared_manifest_sha256"],
        str(evaluation / "prompt_selection.json"): selection["selection_sha256"],
        str(evaluation / "source/selection_implementation.py"): selection["selection_implementation_sha256"],
        str(evaluation / "prompt_development.jsonl"): protocol["prompt_selection_input_sha256"], **training["verified_files"]}
    pinned.update({str(evaluation / "source" / name): value for name, value in protocol["source_sha256"].items()})
    pinned.update({str(prepared / name): value for name, value in manifest["output_sha256"].items()})
    pinned.update({str(Path(training["config"]["base"]) / name): value for name, value in manifest["tokenizer_sha256"].items()})
    for arm in arms.values():
        provenance = arm["provenance"]
        path = Path(provenance["path"])
        pinned.update({str(path / filename): provenance[key] for filename, key in [
            ("registration.json", "registration_sha256"), ("result.json", "result_sha256"), ("predictions.jsonl", "predictions_sha256")]})
        pinned.update(provenance["verified_identity_files"])
    for filename, expected in pinned.items():
        checked_hash(filename, expected, "analysis input changed before report write")
    report["verified_input_sha256"] = pinned
    payload = canonical(report) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", default="outputs/qwen38-qasper-evaluation-01")
    p.add_argument("--prepared", default="outputs/qwen38-science-01/prepared-qasper-tokenizer-01")
    p.add_argument("--training-amendment", default="results/protocols/qwen38-qasper-integrity-amendment-01.json")
    p.add_argument("--runs", required=True, help="JSON with base/prompt/adapter run mappings and expected hashes")
    p.add_argument("--output", required=True, help="new analysis JSON file; never overwritten")
    a = p.parse_args()
    report = analyze(ROOT / a.evaluation, ROOT / a.prepared, ROOT / a.training_amendment, ROOT / a.runs, ROOT / a.output)
    print(canonical({"output": a.output, "science_gates_pass": report["analysis"]["science_gates_pass"],
                     "retention": "pending", "recommended_adapter": None}))


if __name__ == "__main__":
    main()
