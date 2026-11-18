"""CPU-only, no-upload staging for separately evaluated 27B science adapters.

Contracts: QASPER analysis JSON from qwen38_qasper_analysis; PRL final/scores.json
plus its registered review binding; complete candidate/base retention scores.json;
public NF4 publish_release receipt with exact computational hashes and revision.
Incomplete PRL judging requires --experimental; incomplete generations/retention,
unverified bases, or identity failures are never bypassed. No merged exports.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import random
import shutil
import types
import zipfile
import io

from scripts.stage_physics_release import json_bytes, private_preflight, redact_metadata
from scripts.qwen38_eval_adapter import validate as validate_adapter

ROOT = Path(__file__).resolve().parents[1]
BASE = "RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4"
TARGETS = {"qasper": "Qwen3.8-27B-tuneLLM-QASPER-QLoRA", "prl": "Qwen3.8-27B-tuneLLM-PRL-Abstract-QLoRA"}
TRAINING_RUNS = {"qasper": "qwen38-qasper-02", "prl": "qwen38-prl-02"}  # CLI defaults only
TRAINING_COUNTS = {"qasper": {"train": 482, "validation": 101}, "prl": {"train": 57, "validation": 9}}
PREPARED = {"qasper": "prepared-qasper-tokenizer-01", "prl": "prepared-prl-audited-01"}
COMPUTE = {"chat_template.jinja", "config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
VERSION = "qwen38-stage-adapter-v1"
BATCH4_CAVEAT = ("Batch4 is not token-equivalent to batch1: 2/4 invented preflight probes differed, "
    "while disabled and zero-active adapters agreed within each batch size. Retention comparisons require "
    "the exact same batch4 groups and padding across all retention arms. Final science generation remains "
    "a distinct batch1 measurement; these results must not be pooled. This is descriptive evidence, not "
    "a demonstrated explanation of the batching difference.")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().split("\n") if s.strip()]


def checked(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or sha(path) != expected:
        raise ValueError(f"bound file hash differs: {Path(path).name}")


def safe_json(value, root=ROOT):
    """Reuse legacy credential rejection/home redaction; sanitize extra local paths.

    Applied to metadata only. Adapter weights and scientific/source text are
    never rewritten; portable adapter config is separately checked for two-field
    changes. Credentials are rejected rather than redacted into publishable data.
    """
    value = redact_metadata(value, Path(root))
    def clean(item):
        if isinstance(item, dict): return {clean(k): clean(v) for k, v in item.items()}
        if isinstance(item, list): return [clean(v) for v in item]
        if isinstance(item, str):
            item = re.sub(r"(?<![:\w])/(?:tmp|mnt|opt|workspace|srv)/[^\s\"'<>]+", "<LOCAL_PATH>", item)
            return re.sub(r"[A-Za-z]:\\[^\s\"'<>]+", "<LOCAL_PATH>", item)
        return item
    return json_bytes(clean(value), Path(root))


def frozen_module(path, expected):
    checked(path, expected)
    module = types.ModuleType("staged_frozen_" + expected[:12])
    module.__file__ = str(path)
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def training_identity(task, training, root):
    result, config, protocol = [load(training / name) for name in ("result.json", "config.json", "protocol.json")]
    registry = root / "results/protocols" / (training.name + ".json")
    if registry.read_bytes() != (training / "protocol.json").read_bytes() or config != protocol["config"]:
        raise ValueError("training protocol/config registry differs")
    if result.get("completed") is not True or result.get("protocol_sha256") != sha(training / "protocol.json"):
        raise ValueError("completed registered training required")
    if config["task"] != task or config["training_counts"] != TRAINING_COUNTS[task]:
        raise ValueError("training task/counts differ from registered study")
    prepared_dir = root / "outputs/qwen38-science-01" / PREPARED[task]
    checked(prepared_dir / "manifest.json", config["prepared_manifest_sha256"])
    prepared_manifest = load(prepared_dir / "manifest.json")
    if prepared_manifest["task"] != task:
        raise ValueError("training preparation task differs")
    if set(protocol["input_sha256"]) != {"train", "validation"}:
        raise ValueError("both training input splits required")
    for split, expected in protocol["input_sha256"].items():
        checked(training / "inputs" / (split + ".jsonl"), expected)
        encoded = rows(training / "inputs" / (split + ".jsonl"))
        if ([r["id"] for r in encoded] != protocol["selected_ids"][split]
                or len(set(r["id"] for r in encoded)) != len(encoded)
                or len(encoded) != config["training_counts"][split]):
            raise ValueError("training selected-ID counts differ")
        checked(prepared_dir / (split + ".jsonl"), config["prepared_input_sha256"][split])
        checked(prepared_dir / (split + ".jsonl"), prepared_manifest["output_sha256"][split + ".jsonl"])
        eligible = {r["id"] for r in rows(prepared_dir / (split + ".jsonl")) if r.get("training_eligible") is True}
        if not set(protocol["selected_ids"][split]) <= eligible:
            raise ValueError("selected training IDs lack prepared eligibility")
    if task == "prl":
        application = prepared_manifest["audit_application"]
        audit_dir = root / "outputs/qwen38-prl-claim-audit-01"
        if application["eligible"] != TRAINING_COUNTS[task] or application.get("AI_audit_only") is not True:
            raise ValueError("PRL both-pass audit eligibility differs")
        # Application snapshots include the isolation/binding witnesses as well
        # as claim-audit files; these live together in the prepared evidence copy.
        for name, expected in application["audit_evidence_sha256"].items():
            checked(prepared_dir / "claim_audit" / name, expected)
        checked(audit_dir / "result.json", application["audit_result_sha256"])
        audit_result = load(audit_dir / "result.json")
        if audit_result.get("completed") is not True or audit_result.get("eligible") != TRAINING_COUNTS[task]:
            raise ValueError("completed PRL training claim audit required")
        records = audit_result["records"]
        if len({r["id"] for r in records}) != len(records):
            raise ValueError("duplicate PRL audit IDs")
        for split in ("train", "validation"):
            audited = {r["id"] for r in records if r["split"] == split and r["eligible"] is True and r["pass_eligible"] == [True, True]}
            if audited != set(protocol["selected_ids"][split]):
                raise ValueError("selected PRL IDs differ from both-pass claim audit")
    for name, expected in protocol["source_sha256"].items():
        checked(training / "source" / name, expected)
    checked(config["nf4_artifact_manifest"], config["nf4_artifact_manifest_sha256"])
    adapter = Path(result["selected"]["path"]).resolve()
    if adapter.parents[1] != training.resolve():
        raise ValueError("selected checkpoint belongs to another training run")
    audit = validate_adapter(adapter, task, "base")
    saved = load(adapter / "adapter_config.json")
    for key, expected in {"r": 16, "lora_alpha": 32, "lora_dropout": .05, "bias": "none", "task_type": "CAUSAL_LM",
                          "base_model_name_or_path": config["base"]}.items():
        if saved.get(key) != expected:
            raise ValueError("saved adapter configuration differs from registered runtime")
    return {"adapter": adapter, "audit": audit, "config": config, "protocol": protocol, "result": result, "saved": saved,
            "prepared_manifest": prepared_manifest}


def public_base(receipt_file, config, revision=None):
    receipt = load(receipt_file)
    if (receipt.get("status") not in {"published_remote_hashes_verified", "published_and_verified"}
            or receipt.get("private") is not False or receipt.get("repo_id") != BASE):
        raise ValueError("verified public 27B NF4 publication receipt required")
    published = receipt.get("revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", published) or (revision is not None and revision != published):
        raise ValueError("exact immutable NF4 revision required")
    artifact = load(config["nf4_artifact_manifest"])
    for name in COMPUTE:
        expected, actual = artifact["files"][name], receipt.get("files", {}).get(name, {})
        if actual.get("sha256") != expected["sha256"] or actual.get("bytes") != expected["bytes"]:
            raise ValueError("published NF4 compute files differ from training base")
    return receipt, published


def retention_report(directory, task, identity, root):
    report, reg, status = [load(directory / name) for name in ("scores.json", "registration.json", "result.json")]
    if (status.get("completed") is not True or status.get("registration_sha256") != sha(directory / "registration.json")
            or report["registration_sha256"] != status["registration_sha256"]
            or report["predictions_sha256"] != status["predictions_sha256"]):
        raise ValueError("completed hash-bound retention arm required")
    checked(directory / "predictions.jsonl", report["predictions_sha256"])
    binding = reg["comparison_binding"]
    if hashlib.sha256(canonical(binding).encode()).hexdigest() != report["comparison_sha256"] or reg["comparison_sha256"] != report["comparison_sha256"]:
        raise ValueError("retention comparison binding differs")
    fixture_path = root / "data/benchmark-screen-01/test.json"
    expected_fixture = load(root / "data/benchmark-screen-01/provenance.json")["fixture_sha256"]["test"]
    checked(fixture_path, expected_fixture)
    if binding["fixture_sha256"] != expected_fixture:
        raise ValueError("retention used another fixed fixture")
    checked(directory / "fixture.json", reg["fixture_copy_sha256"])
    checked(directory / "prepared.json", reg["prepared_copy_sha256"])
    fixture = load(fixture_path)
    tasks = binding["tasks"]
    if (not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks)
            or set(tasks) != set(report["tasks"]) or any(name not in fixture for name in tasks)
            or load(directory / "fixture.json") != {name: fixture[name] for name in tasks}):
        raise ValueError("retention copied fixture differs from registered source projection")
    if hashlib.sha256(canonical(load(directory / "prepared.json")).encode()).hexdigest() != binding["prepared_sha256"]:
        raise ValueError("retention prepared prompt binding differs")
    for name in ("scripts/qwen38_science_runtime.py", "scripts/qwen38_nf4_smoke.py"):
        if binding["source_sha256"].get(name) != identity["protocol"]["source_sha256"].get(name) or name not in binding["source_sha256"]:
            raise ValueError("retention preparation source differs from training runtime")
    for name, score in report["tasks"].items():
        key = "key" if name == "ifeval" else "id"
        if {str(r["id"]) for r in score["details"]} != {str(r[key]) for r in fixture[name]}:
            raise ValueError("retention item IDs differ from fixed source")
    predictions = rows(directory / "predictions.jsonl")
    for name, score in report["tasks"].items():
        predicted = [r["id"] for r in predictions if r["task"] == name]
        if len(predicted) != score["total"] or len(set(predicted)) != len(predicted) or {str(i) for i in predicted} != {str(r["id"]) for r in score["details"]}:
            raise ValueError("retention prediction IDs/counts differ from score")
    if len(predictions) != sum(t["total"] for t in report["tasks"].values()):
        raise ValueError("unknown/extra retention predictions")
    if report["arm"] != task or reg["arm"] != task or binding["base_manifest_sha256"] != identity["config"]["nf4_artifact_manifest_sha256"]:
        raise ValueError("retention arm/base identity differs")
    if task == "base":
        if reg["adapter"] is not None:
            raise ValueError("unadapted retention baseline required")
    else:
        saved_audit = reg["adapter_audit"]
        # Retention enriches the parent's selected-file audit with config data;
        # verify those extra bindings without confusing them with another adapter.
        extras = set(saved_audit) - set(identity["audit"])
        if (Path(reg["adapter"]).resolve() != identity["adapter"]
                or any(saved_audit.get(k) != v for k, v in identity["audit"].items())
                or extras and (extras != {"training_config_sha256", "saved_config"}
                    or saved_audit["training_config_sha256"] != sha(identity["adapter"].parents[1] / "config.json")
                    or saved_audit["saved_config"] != identity["saved"])):
            raise ValueError("retention evaluated a different selected adapter/config")
    for name, expected in binding["source_sha256"].items():
        checked(directory / "source" / name, expected)
    return report, reg


def paired_retention(candidate_dir, base_dir, task, identity, root):
    candidate, reg = retention_report(candidate_dir, task, identity, root)
    baseline, _ = retention_report(base_dir, "base", identity, root)
    checked(base_dir / "scores.json", candidate["baseline_scores_sha256"])
    name = "scripts/qwen38_retention.py"
    helper = frozen_module(candidate_dir / "source" / name, reg["comparison_binding"]["source_sha256"][name])
    computed = helper.compare_scores(baseline, candidate)
    if computed != candidate.get("comparison"):
        raise ValueError("paired retention gates differ from matched item scores")
    return {"candidate": candidate, "baseline": baseline, "comparison": computed,
        "batching": {"generation_batch_size": reg["comparison_binding"].get("protocol", {}).get("batch_size", 1), "mc_batch_size": 1}}


def qasper_science(path, identity, root, tokenizer=None):
    report = load(path)
    if report.get("version") != "qwen38-qasper-analysis-v1" or report.get("checkpoint_selection_from_final") is not False:
        raise ValueError("completed frozen QASPER analysis required")
    if Path(report["training_run"]).resolve() != identity["adapter"].parents[1] or report["arms"]["adapter"]["provenance"]["adapter_audit"] != identity["audit"]:
        raise ValueError("science report evaluated another selected adapter")
    if not report.get("verified_input_sha256"):
        raise ValueError("QASPER analysis lacks frozen input/prediction hashes")
    for filename, expected in report["verified_input_sha256"].items(): checked(filename, expected)
    checked(root / "scripts/qwen38_qasper_analysis.py", report["analysis_source_sha256"])
    evaluation = root / "outputs/qwen38-qasper-evaluation-01"
    checked(evaluation / "registration.json", report["evaluation_registration_sha256"])
    if load(evaluation / "registration.json") != report["protocol"]:
        raise ValueError("QASPER report protocol differs from frozen registration")
    counts = report["protocol"]["counts"]["test"]
    if any(counts.get(k) != n for k, n in {"records": 403, "primary": 309, "evidence_strict_primary": 267}.items()):
        raise ValueError("fixed QASPER final denominators differ")
    analysis = report["analysis"]
    if not analysis["gate_results"] or any(type(v) is not bool for v in analysis["gate_results"].values()) or analysis["science_gates_pass"] != all(analysis["gate_results"].values()):
        raise ValueError("QASPER science gate summary inconsistent")
    frozen = frozen_module(root / "scripts/qwen38_qasper_analysis.py", report["analysis_source_sha256"])
    prepared = root / "outputs/qwen38-science-01/prepared-qasper-tokenizer-01"
    protocol, _, final, scorer, prompts = frozen.load_protocol(evaluation, prepared, root)
    amendment = root / "results/protocols/qwen38-qasper-integrity-amendment-01.json"
    training = frozen.verify_training_binding(amendment, report["training_binding"]["amendment_sha256"], protocol, evaluation, root)
    if training["training"].resolve() != identity["adapter"].parents[1]:
        raise ValueError("QASPER amended training binding differs")
    selection = frozen.load_prompt_selection(evaluation, protocol, report["prompt_selection"]["selection_sha256"], prompts)
    if selection != report["prompt_selection"] or report["aliases"] != ({"prompt": "base"} if selection["chosen"] == "base" else {}):
        raise ValueError("selected prompt identity differs")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(training["config"]["base"], local_files_only=True, trust_remote_code=False)
    data_path = prepared / "test.jsonl"
    checked(data_path, report["final_input_sha256"])
    final = rows(data_path)
    scored, runtimes = {}, []
    for arm, arm_report in report["arms"].items():
        provenance = arm_report["provenance"]
        spec = {k: provenance[k] for k in ("run", "registration_sha256", "result_sha256", "predictions_sha256")}
        validated = frozen.validate_arm(spec, selection["chosen"] if arm == "prompt" else "base", arm == "adapter",
            final, data_path, protocol, prompts, training, tokenizer, root)
        runtimes.append(validated["runtime"])
        if validated["provenance"] != provenance or validated["generation"] != arm_report["generation"]:
            raise ValueError("QASPER generation provenance differs from replay")
        generation = Path(provenance["path"])
        for filename, field in [("registration.json", "registration_sha256"), ("result.json", "result_sha256"), ("predictions.jsonl", "predictions_sha256")]:
            checked(generation / filename, provenance[field])
        status = load(generation / "result.json")
        if status.get("completed") is not True or status.get("generations") != 403:
            raise ValueError("complete QASPER final generation arm required")
        predicted = rows(generation / "predictions.jsonl")
        if len(predicted) != 403 or len({r["id"] for r in predicted}) != 403 or {r["id"] for r in predicted} != {r["id"] for r in final}:
            raise ValueError("QASPER generation IDs differ")
        scored[arm] = scorer.score_qasper(final, {r["id"]: r["output"] for r in predicted})
        if scored[arm] != arm_report["scores"]:
            raise ValueError("QASPER metrics differ from frozen-scorer replay")
    if set(scored) != {"base", "prompt", "adapter"} or any(r != runtimes[0] for r in runtimes):
        raise ValueError("complete matched-runtime QASPER arms required")
    recomputed = frozen.evaluate_metrics(final, scored, scorer, report["protocol"], report["aliases"].get("prompt") == "base")
    if recomputed != analysis:
        raise ValueError("QASPER gates/intervals differ from frozen analysis replay")
    return report, {"completed": True, "passed": analysis["science_gates_pass"], "incomplete": False}


def prl_science(path, identity, root):
    report, final = load(path), path.parent
    evaluation = final.parent
    protocol = load(evaluation / "registration.json")
    if (protocol.get("gates") != {"preference_each_baseline": .6, "paper_bootstrap_lower_bound_above": .5,
            "unsupported_mean_increase_each_baseline_at_most": 0, "complete_papers": 40}
            or protocol.get("bootstrap") != {"draws": 10000, "seed": 20261117, "unit": "paper", "method": "paired percentile95"}):
        raise ValueError("PRL frozen gates/bootstrap differ")
    if (root / "results/protocols" / (evaluation.name + ".json")).read_bytes() != (evaluation / "registration.json").read_bytes():
        raise ValueError("PRL review registration differs")
    binding = load(final / "binding.json")
    checked(evaluation / "registration.json", binding["registration_sha256"])
    checked(final / "plans.json", binding["plans_sha256"])
    # Hardware amendments bind a replacement run without changing study data or settings.
    amendment_path = root / "results/protocols/qwen38-prl-hardware-amendment-01.json"
    if amendment_path.exists():
        amendment = load(amendment_path)
        checked(identity["adapter"].parents[1] / "protocol.json", amendment["new_training_protocol_sha256"])
        if amendment["training_run"] != identity["adapter"].parents[1].name or amendment["evaluation_registration_sha256"] != sha(evaluation / "registration.json"):
            raise ValueError("PRL hardware amendment training/evaluation binding differs")
        old_run = root / "outputs" / amendment["superseded_prepared_only_run"]
        checked(old_run / "protocol.json", amendment["old_training_protocol_sha256"])
        old = load(old_run / "protocol.json")
        new = identity["protocol"]
        changes = {k: [old["config"].get(k), new["config"].get(k)] for k in old["config"] | new["config"]
                   if old["config"].get(k) != new["config"].get(k)}
        if (any(old[k] != new[k] for k in ("selected_ids", "input_sha256", "target_tokens"))
                or {k: v for k, v in old["config"].items() if k != "gpu"} != {k: v for k, v in new["config"].items() if k != "gpu"}
                or amendment.get("optimization_started_at_amendment") is not False
                or changes != amendment["config_changes"] or set(changes) - {"gpu"}
                or (old_run / "training.jsonl").exists() and (old_run / "training.jsonl").stat().st_size
                or (old_run / "result.json").exists() and load(old_run / "result.json").get("steps", 0)):
            raise ValueError("PRL hardware amendment changed science or superseded optimized run")
    checked(evaluation / "final.jsonl", protocol["inputs"]["final"])
    source = rows(evaluation / "final.jsonl")
    ids = {r["id"] for r in source}
    if len(source) != 40 or len(ids) != 40:
        raise ValueError("all40 fresh PRL final papers required")
    for name, expected in protocol["source_sha256"].items(): checked(evaluation / "source" / name, expected)
    selection = load(evaluation / "prompt_selection.json")
    if selection.get("completed") is not True or selection.get("final_used") is not False or selection.get("chosen") not in {"base", "verify"} or selection.get("identical_baselines") != (selection["chosen"] == "base"):
        raise ValueError("PRL frozen development prompt selection required")
    generations = {}
    for arm, record in binding["generations"].items():
        run = root / record["run"]
        checked(run / "registration.json", record["registration_sha256"])
        checked(run / "predictions.jsonl", record["predictions_sha256"])
        cfg, status = load(run / "registration.json"), load(run / "result.json")
        if status.get("completed") is not True or status["registration_sha256"] != record["registration_sha256"] or cfg["input_sha256"] != protocol["inputs"]["final"]:
            raise ValueError("PRL final generations incomplete or wrong input")
        expected = {"task": "prl", "purpose": "final", "template": selection["chosen"] if arm == "prompt" else "base",
            "max_new_tokens": 512, "max_context": 8192, "seed": 42, "batch_size": 1,
            "cached_generation": True, "enable_thinking": False, "do_sample": False,
            "base_manifest_sha256": identity["config"]["nf4_artifact_manifest_sha256"], "versions": identity["config"]["versions"]}
        if any(cfg.get(k) != v for k, v in expected.items()) or status.get("generations") != 40:
            raise ValueError("PRL final generation settings/template/base differ")
        if arm != "adapter" and cfg.get("adapter") is not None:
            raise ValueError("PRL baseline must be unadapted")
        predicted = rows(run / "predictions.jsonl")
        if len(predicted) != 40 or {r["id"] for r in predicted} != ids:
            raise ValueError("PRL final generation IDs differ")
        generations[arm] = {r["id"]: r for r in predicted}
        if arm == "adapter" and (Path(cfg["adapter"]).resolve() != identity["adapter"] or cfg["adapter_files"] != {k: v["sha256"] for k, v in identity["audit"]["adapter_files"].items()}):
            raise ValueError("PRL science used another adapter")
    if "adapter" not in binding["generations"] or "base" not in binding["generations"]:
        raise ValueError("PRL base/selected adapter generations required")
    plans = load(final / "plans.json")
    if len(plans) != 80 or {(p["paper_id"], p["pass"]) for p in plans} != {(key, p) for key in ids for p in (0, 1)}:
        raise ValueError("registered80 PRL review packets required")
    helper = frozen_module(evaluation / "source/qwen38_prl_review.py", protocol["source_sha256"]["qwen38_prl_review.py"])
    from scripts import tone_judge
    checked(root / "scripts/tone_judge.py", protocol["source_sha256"]["tone_judge.py"])
    checked(tone_judge.__file__, protocol["source_sha256"]["tone_judge.py"])
    from scripts import qwen38_prl_claim_audit
    checked(qwen38_prl_claim_audit.__file__, protocol["source_sha256"]["qwen38_prl_claim_audit.py"])
    grouped = {}
    source_by_id = {r["id"]: r for r in source}
    packet_indices = {}
    for pass_index in (0, 1):
        order = list(source)
        random.Random(20261117 + pass_index).shuffle(order)
        packet_indices.update({(row["id"], pass_index): i for i, row in enumerate(order)})
    for plan in plans:
        directory = final / plan["directory"]
        if not directory.resolve().is_relative_to(final.resolve()): raise ValueError("review plan escapes final directory")
        checked(directory / "payload.json", plan["payload_sha256"])
        payload = load(directory / "payload.json")
        if payload["evidence_excerpt"] != source_by_id[plan["paper_id"]]["selected_body"] or set(plan["mapping"].values()) != set(generations):
            raise ValueError("review evidence/arm mapping differs from final generations")
        for candidate in payload["candidates"]:
            arm = plan["mapping"][candidate["label"]]
            if candidate["abstract"] != helper.clean(generations[arm][plan["paper_id"]]["output"]):
                raise ValueError("review candidate differs from generated output")
        expected_payload, expected_mapping = helper.packet(source_by_id[plan["paper_id"]],
            {arm: generations[arm][plan["paper_id"]]["output"] for arm in generations}, plan["pass"],
            packet_indices[plan["paper_id"], plan["pass"]])
        if payload != expected_payload or plan["mapping"] != expected_mapping:
            raise ValueError("PRL frozen rubric/sentences/blinding differ")
        review = load(directory / "checked.json") if (directory / "checked.json").exists() else {"valid": False}
        if review.get("valid") is True:
            attempt = load(directory / "attempt.json")
            if attempt["canonical_stdin_sha256"] != tone_judge.canonical_hash(payload): raise ValueError("PRL review attempt payload differs")
            outer, models = tone_judge.outer_result(attempt)
            actual = helper.check(tone_judge.result_json(outer["result"]), payload)
            if any(review.get(k) != v for k, v in actual.items()) or review.get("model_ids") != models:
                raise ValueError("PRL checked judgment differs from saved response replay")
        grouped.setdefault(plan["paper_id"], []).append((plan, payload, review))
    complete_ids = [key for key, pair in grouped.items() if len(pair) == 2 and {p[0]["pass"] for p in pair} == {0, 1} and all(r[2].get("valid") is True for r in pair)]
    if report.get("completed") is not True:
        if (type(report.get("complete_papers")) is not int or report.get("expected_papers") != 40
                or report["complete_papers"] != len(complete_ids) or not 0 <= len(complete_ids) < 40 or report.get("science_gates_pass") is True):
            raise ValueError("malformed incomplete PRL judging report")
        return report, {"completed": False, "passed": False, "incomplete": True}
    if len(complete_ids) != 40: raise ValueError("complete PRL report lacks80 valid judgments")
    checked(evaluation / "prompt_selection.json", report["prompt_selection_sha256"])
    baselines = {"base"} if selection["identical_baselines"] else {"base", "prompt"}
    if set(binding["generations"]) != baselines | {"adapter"}:
        raise ValueError("PRL baseline alias/generation binding differs")
    records = report["records"]
    if report.get("papers") != 40 or len(records) != 40 or {r["id"] for r in records} != ids or set(report["preference"]) != baselines:
        raise ValueError("complete unique PRL review records/baselines required")
    expected_records = []
    for key, pair in sorted(grouped.items()):
        preferences = {name: [] for name in baselines}
        labels, empty = {}, {}
        for plan, payload, review in pair:
            reverse = {arm: label for label, arm in plan["mapping"].items()}
            for baseline in baselines:
                a, b = reverse["adapter"], reverse[baseline]
                winner = review["winners"][":".join(sorted([a, b]))]
                preferences[baseline].append(.5 if winner == "tie" else float(winner == a))
            for arm, label in reverse.items():
                candidate = next(c for c in payload["candidates"] if c["label"] == label)
                empty[arm] = not candidate["abstract"]
                labels.setdefault(arm, []).append([review["labels"][s["sid"]] for s in candidate["sentences"]])
        unsupported, fractions, partial = {}, {}, {}
        for arm, values in labels.items():
            if len(values[0]) != len(values[1]): raise ValueError("review sentence counts differ across passes")
            worst = [max(v, key=helper.LABELS.index) for v in zip(*values)]
            unsupported[arm] = 1 if empty[arm] else sum(v in {"UNSUPPORTED", "UNCERTAIN"} for v in worst)
            denominator = sum(v != "FRAMING" for v in worst)
            fractions[arm] = 1 if empty[arm] else unsupported[arm] / denominator if denominator else 0
            partial[arm] = worst.count("PARTIAL")
        expected_records.append({"id": key, "preferences": {a: sum(v) / 2 for a, v in preferences.items()},
            "unsupported": unsupported, "unsupported_fraction": fractions, "partial": partial})
    if records != expected_records: raise ValueError("PRL report records differ from checked review replay")
    for field, record_field in (("unsupported_mean", "unsupported"), ("unsupported_nonframing_fraction_macro", "unsupported_fraction")):
        expected = {arm: sum(r[record_field][arm] for r in records) / 40 for arm in generations}
        if report.get(field) != expected:
            raise ValueError("PRL claim aggregates differ from checked judgments")
    if (report.get("AI_review_only") is not True or report.get("identical_baselines") != selection["identical_baselines"]
            or report.get("baseline_comparisons") != len(baselines) or report.get("retention_gate_pending") is not True):
        raise ValueError("PRL review scope/alias metadata differs")
    passed = True
    import numpy as np
    for baseline in baselines:
        mean = sum(r["preferences"][baseline] for r in records) / 40
        delta = sum(r["unsupported"]["adapter"] - r["unsupported"][baseline] for r in records) / 40
        p, d = report["preference"][baseline], report["unsupported_difference"][baseline]
        if not math.isclose(mean, p["mean"], abs_tol=1e-12) or not math.isclose(delta, d["mean"], abs_tol=1e-12):
            raise ValueError("PRL review aggregates differ from per-paper judgments")
        if len(p["ci95"]) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in p["ci95"]) or p["ci95"][0] > p["ci95"][1]:
            raise ValueError("invalid PRL bootstrap interval")
        values = np.array([r["preferences"][baseline] for r in records])
        delta_values = np.array([r["unsupported"]["adapter"] - r["unsupported"][baseline] for r in records])
        draw = np.random.default_rng(20261117).integers(0, 40, (10000, 40))
        for actual, expected in [(p["ci95"], np.quantile(values[draw].mean(axis=1), [.025, .975])),
                                  (d["ci95"], np.quantile(delta_values[draw].mean(axis=1), [.025, .975]))]:
            if not np.allclose(actual, expected, rtol=0, atol=1e-12): raise ValueError("PRL interval differs from registered paper bootstrap")
        passed = passed and mean >= .6 and p["ci95"][0] > .5 and delta <= 0
    if report.get("science_gates_pass") is not passed:
        raise ValueError("PRL science gate summary inconsistent")
    return report, {"completed": True, "passed": passed, "incomplete": False}


def prl_attributions(root, selected_ids):
    records = []
    paths = [(root / "data/qwen38-prl-abstract-01/historical-reextracted-02" / (s + ".jsonl"), s) for s in ("train", "validation", "test")]
    paths.append((root / "data/qwen38-prl-abstract-01/fresh-acquisition-01/fresh.jsonl", "fresh_test"))
    hashes = {}
    for path, split in paths:
        hashes[str(path)] = sha(path)
        papers = rows(path)
        if len(papers) != {"train": 70, "validation": 10, "test": 20, "fresh_test": 40}[split]:
            raise ValueError("PRL attribution split counts differ")
        for paper in papers:
            proof = paper.get("authoritative_license", {})
            if proof.get("verified") is not True or not re.fullmatch(r"https?://creativecommons.org/licenses/by/4\.0/?", proof.get("license", "")):
                raise ValueError("per-paper verified CC-BY4 license required")
            for key in ("title", "authors", "doi", "source_url", "source_sha256", "arxiv_version", "id", "paper_id"):
                if not paper.get(key): raise ValueError("per-paper attribution field missing: " + key)
            if not re.fullmatch(r"[0-9a-f]{64}", paper["source_sha256"]) or not re.fullmatch(r"[0-9a-f]{64}", proof.get("response_sha256", "")):
                raise ValueError("per-paper source/license proof hashes required")
            records.append({k: paper[k] for k in ("id", "paper_id", "title", "authors", "doi", "source_url", "source_sha256", "arxiv_version", "authoritative_license")} | {
                "split": split, "journal_reference": paper.get("journal-ref"), "license": "CC-BY-4.0",
                "used_for_optimization": paper["id"] in selected_ids["train"], "used_for_checkpoint_selection": paper["id"] in selected_ids["validation"],
                "changes": "Versioned arXiv source cleaned; title/abstract removed from input; fixed whole-block head/tail excerpt; AI claim audit filters training/development only."})
    if len({r["paper_id"] for r in records}) != 140:
        raise ValueError("PRL attribution papers overlap")
    for split in ("train", "validation"):
        if not set(selected_ids[split]) <= {r["id"] for r in records if r["split"] == split}:
            raise ValueError("selected PRL IDs absent from attributed source split")
    return {"papers": records, "input_sha256": hashes, "fresh_test_scope": "fresh to project; possible pretraining exposure; arXiv sources associated with PRL publications"}


def prl_diagnostics(path, science_path, root):
    """Secondary replay after prl_science has validated all primary judgments.

    The registered diagnostics source is pinned prospectively; its imported token
    F1 helper is additionally recorded/archived here, without an upstream parity
    claim. No summary or diagnostic changes the primary/retention gates.
    """
    path, science_path, root = Path(path).resolve(), Path(science_path).resolve(), Path(root).resolve()
    science = load(science_path)
    if science.get("completed") is not True or science.get("papers") != 40:
        raise ValueError("secondary final diagnostics require complete PRL science")
    final, evaluation = science_path.parent, science_path.parent.parent
    protocol_path = root / "results/protocols/qwen38-prl-diagnostics-01.json"
    protocol = load(protocol_path)
    checked(evaluation / "registration.json", protocol["primary_evaluation_registration_sha256"])
    if (root / "results/protocols" / (evaluation.name + ".json")).read_bytes() != (evaluation / "registration.json").read_bytes():
        raise ValueError("diagnostics primary registration differs")
    registration = load(evaluation / "registration.json")
    checked(evaluation / "final.jsonl", registration["inputs"]["final"])
    source = root / "scripts/qwen38_prl_diagnostics.py"
    helper = frozen_module(source, protocol["source_sha256"])
    from scripts import qwen38_science_score
    dependency = root / "scripts/qwen38_science_score.py"
    checked(dependency, sha(qwen38_science_score.__file__))
    report = load(path)
    if report["source_sha256"] != protocol["source_sha256"] or report["input_sha256"] != registration["inputs"]["final"]:
        raise ValueError("secondary diagnostic source/input binding differs")
    binding = load(final / "binding.json")
    checked(evaluation / "registration.json", binding["registration_sha256"])
    checked(final / "plans.json", binding["plans_sha256"])
    records = rows(evaluation / "final.jsonl")
    ids = {r["id"] for r in records}
    if len(records) != 40 or len(ids) != 40 or set(report["arms"]) != {"base", "adapter"}:
        raise ValueError("all40 unique papers and base/adapter secondary arms required")
    bound = {str(p): sha(p) for p in (path, science_path, protocol_path, source, dependency,
        evaluation / "registration.json", evaluation / "final.jsonl", final / "binding.json", final / "plans.json")}
    expected = {"scope": "secondary custom lexical diagnostics; no semantic faithfulness claim",
        "source_sha256": protocol["source_sha256"], "input_sha256": registration["inputs"]["final"], "arms": {}}
    for arm in ("base", "adapter"):
        run = root / binding["generations"][arm]["run"]
        checked(run / "predictions.jsonl", binding["generations"][arm]["predictions_sha256"])
        checked(run / "registration.json", binding["generations"][arm]["registration_sha256"])
        cfg, status = load(run / "registration.json"), load(run / "result.json")
        if (status.get("completed") is not True or status["registration_sha256"] != sha(run / "registration.json")
                or cfg["input_sha256"] != registration["inputs"]["final"]
                or report["arms"][arm]["predictions_sha256"] != binding["generations"][arm]["predictions_sha256"]):
            raise ValueError("secondary predictions differ from completed final science binding")
        predicted = rows(run / "predictions.jsonl")
        if len(predicted) != 40 or {r["id"] for r in predicted} != ids:
            raise ValueError("secondary prediction IDs/count differ")
        lookup = {r["id"]: r for r in predicted}
        expected["arms"][arm] = {"records": [{"id": r["id"], **helper.describe(r, lookup[r["id"]]["output"])} for r in records],
            "predictions_sha256": binding["generations"][arm]["predictions_sha256"]}
        bound.update({str(run / name): sha(run / name) for name in ("predictions.jsonl", "registration.json", "result.json")})
    for plan in load(final / "plans.json"):
        directory = (final / plan["directory"]).resolve()
        if not directory.is_relative_to(final): raise ValueError("secondary review path escapes final stage")
        for name in ("payload.json", "checked.json", "attempt.json"):
            bound[str(directory / name)] = sha(directory / name)
    expected["fixed_copy_baseline"] = [{"id": r["id"], **helper.describe(r, r["copy_baseline"]["text"])} for r in records]
    expected["reviewer_diagnostics"] = helper.agreement(final)
    if canonical(report) != canonical(expected):
        raise ValueError("secondary diagnostics differ from describe/agreement replay")
    summary = {}
    for arm, values in [(arm, value["records"]) for arm, value in expected["arms"].items()] + [("fixed_copy", expected["fixed_copy_baseline"])]:
        means = {}
        for metric in ("words", "novel_unique_8gram_fraction", "custom_whitespace_lcs_f1", "normalized_token_f1"):
            valid = [r[metric] for r in values if r[metric] is not None]
            means[metric] = {"mean": sum(valid) / len(valid) if valid else None, "denominator": len(valid)}
        summary[arm] = {"records": len(values), "means": means}
    for filename, expected_hash in bound.items(): checked(filename, expected_hash)
    return {"report": report, "summary": summary, "provenance": {"verified_input_sha256": bound,
        "registered_source_sha256": protocol["source_sha256"], "registered_protocol_sha256": sha(protocol_path),
        "token_f1_dependency_sha256": sha(dependency), "dependency_scope": "imported helper recorded at packaging; not separately pinned by the original diagnostics registration",
        "gates_affected": False}, "source": source, "protocol": protocol_path, "dependency": dependency}


def qasper_baseline_completion(science, root, tokenizer=None):
    """Optional full CPU assembly proof, after the frozen science validation."""
    root = Path(root).resolve()
    provenance = science["arms"]["base"]["provenance"]
    directory = Path(provenance["path"]).resolve()
    if not directory.is_relative_to(root / "outputs"): raise ValueError("baseline path escapes outputs")
    for name, key in (("registration.json", "registration_sha256"), ("result.json", "result_sha256"), ("predictions.jsonl", "predictions_sha256")):
        checked(directory / name, provenance[key])
    registration, result = load(directory / "registration.json"), load(directory / "result.json")
    if "completion" not in registration:
        if "completion" in result or "assembler_source_sha256" in registration:
            raise ValueError("inconsistent baseline completion declaration")
        return None
    from scripts import qwen38_complete_science as helper
    source = root / "scripts/qwen38_complete_science.py"
    for value in (registration.get("assembler_source_sha256"), result.get("assembler_source_sha256"),
            registration["completion"].get("assembler_source_sha256")):
        checked(source, value); checked(helper.__file__, value)
    proof = helper.validate_assembly(directory, root=root, tokenizer=tokenizer)
    if proof != registration["completion"] or proof != result["completion"]:
        raise ValueError("baseline completion differs from full CPU helper proof")
    amendment = root / helper.AMENDMENT
    checked(amendment, proof["amendment_sha256"])
    original, suffix = [Path(s["run"]) for s in proof["segments"]]
    checked(original / "result.json", proof["original_failed_result_sha256"])
    bound = {**proof["bound_files_sha256"], str(source): sha(source), str(amendment): sha(amendment),
        **{str(directory / name): provenance[key] for name, key in (("registration.json", "registration_sha256"),
            ("result.json", "result_sha256"), ("predictions.jsonl", "predictions_sha256"))}}
    metadata = {"proof": proof, "original_failed_receipt": load(original / "result.json"),
        "suffix": {"ids": proof["suffix_ids"], "registration": load(suffix / "registration.json"), "result": load(suffix / "result.json")},
        "assembled": {"path": str(directory), "registration_sha256": provenance["registration_sha256"],
            "result_sha256": provenance["result_sha256"], "predictions_sha256": provenance["predictions_sha256"],
            "status": {k: v for k, v in result.items() if k != "completion"}},
        "verified_input_sha256": bound, "raw_prediction_texts_exported": False}
    for name, expected in bound.items(): checked(name, expected)
    return {"metadata": metadata, "source": source, "amendment": amendment, "bound": bound}


def synthetic_prompt(task):
    """Tiny invented content in the registered base instruction format."""
    if task == "qasper":
        return ('Answer using only the numbered paper paragraphs. Return only JSON with keys '
            '"answer" (string), "unanswerable" (boolean), "evidence" (array of paragraph IDs). '
            'For insufficient evidence return an empty answer and evidence array with unanswerable=true.\n\n'
            '[p000000] Results\nIn a synthetic experiment, the blue sample cooled to 2 K.\n\n'
            'Question: What temperature did the blue sample reach?')
    if task == "prl":
        return ('Write a grounded physics abstract in at most 200 words using only the supplied paper body excerpt, '
            'which may omit its middle and nonprose material. Preserve quantities and limitations; do not invent claims. '
            'Return only the abstract.\n\n'
            'We study a synthetic two-site lattice model. Exact diagonalization gives a gap of two energy units. '
            'This calculation concerns only the two-site system; no thermodynamic-limit conclusion is established.')
    raise ValueError("unknown adapter task")


def card(task, status, science, retention, revision, config, experimental, prompt_selection=None, diagnostics=None, completion=None):
    if diagnostics is not None and task != "prl": raise ValueError("secondary diagnostics are PRL-only")
    if completion is not None and task != "qasper": raise ValueError("baseline completion is QASPER-only")
    target = TARGETS[task]
    retention_batch_size = retention.get("batching", retention["candidate"].get("batching", {}) if "candidate" in retention else {}).get("generation_batch_size", 1)
    state = ("INCOMPLETE FINAL JUDGING — EXPERIMENTAL; NOT RECOMMENDED" if status["incomplete"] else
             "EXPERIMENTAL; NOT RECOMMENDED" if experimental else "Passed registered task and retention gates on fixed subsets")
    lines = ["---", "license: apache-2.0", "library_name: peft", "pipeline_tag: text-generation", f"base_model: {BASE}",
        "base_model_relation: adapter", "tags:", "- qlora", "- text-only", "- research", "---", f"# {target}", "", f"**{state}**", "",
        "Experimental labels do not establish improved capability. Gate-passing labels apply only to the recorded task, prompts and small subsets.", "",
        f"This package contains a separate adapter over [{BASE}](https://huggingface.co/{BASE}/tree/{revision}), pinned to `{revision}`.",
        "Modification notice (Apache-2.0): tuneLLM converted the upstream Qwen checkpoint to a text-only NF4 base by removing vision/MTP components and quantizing its language weights, then trained this separate task-specific QLoRA adapter. The adapter targets QASPER answer/evidence JSON or PRL-associated body-to-abstract generation, as identified by this package. These modifications are not an upstream Qwen release.",
        "The original selected adapter tensor file is copied byte-for-byte. The copied configuration changes only its portable base repository and revision; config_diff.json records the hashes and field differences.", ""]
    if task == "qasper":
        if completion is not None:
            lines += ["**Baseline execution disclosure: the original 10000-second phase failed after 395/403 questions. The eight never-previously-evaluated suffix questions were completed once in a separate phase capped at 900 seconds.**", "",
                "All 403 questions have first-time coverage through two sequential GPU0 executions. The original failed receipt is preserved. The total 12-hour QASPER evaluation pool and all quality gates remain unchanged. This is not an uninterrupted original-cap run. See baseline_completion.json and baseline_completion_amendment.json for the segment receipts, exact suffix IDs and source hashes.", ""]
        lines += ["Trained on 482 QASPER questions; checkpoint selection used 101 development questions. Final: 403 questions, 309 primary and 267 evidence-strict primary. Prompt selection used a separate frozen development prefix.",
            "Public NLP papers may have appeared in pretraining. Paragraph-ID evidence metrics and the text-only primary subset are custom; this is not the full official benchmark or a physics-specialization result.",
            "One configuration and seed were evaluated for one epoch. The prompt baseline was the best of three fixed templates on 35 development questions. Answer F1 measures reference-answer overlap; interpret the gains under this protocol rather than as a general scientific-comprehension result.", "",
            "| Arm | Strict primary Answer F1 | Lenient primary Answer F1 | Invalid strict JSON / all403 | Invalid lenient / all403 |", "|---|---:|---:|---:|---:|"]
        for name in ("base", "prompt", "adapter"):
            score = science["arms"][name]["scores"]
            lines.append(f"| {name} | {score['strict']['groups']['primary']['answer_f1']:.6f} | {score['lenient']['groups']['primary']['answer_f1']:.6f} | {score['strict']['groups']['all_sampled']['invalid_outputs']} / 403 | {score['lenient']['groups']['all_sampled']['invalid_outputs']} / 403 |")
        lines += ["", "| Adapter comparison | Strict gain (95% paper CI) | Lenient gain (95% paper CI) |", "|---|---|---|"]
        threshold = science["protocol"]["success_gates"]["answer_f1_gain_each_baseline"]
        mode_pass = {}
        for mode in ("strict", "lenient"):
            mode_pass[mode] = all(v["difference_a_minus_b"] >= threshold - 1e-12 and v["ci95"][0] > 0
                for v in science["analysis"]["bootstrap"][mode].values())
        for name in ("base", "prompt"):
            values = [science["analysis"]["bootstrap"][mode][name] for mode in ("strict", "lenient")]
            cells = [f"{v['difference_a_minus_b']:.6f} ({v['ci95'][0]:.6f}, {v['ci95'][1]:.6f})" for v in values]
            lines.append(f"| vs {name} | {cells[0]} | {cells[1]} |")
        if mode_pass["strict"] and not mode_pass["lenient"]:
            lines += ["", "**FORMAT-DRIVEN RESULT: strict gain/CI conditions pass while lenient gain/CI conditions do not. This does not establish improved answer capability beyond response formatting.**"]
        if science.get("aliases", {}).get("prompt") == "base":
            lines += ["", "The development-selected prompt is base; the two baselines alias the same outputs. No stronger-prompt claim is made."]
        selection_hash = science.get("prompt_selection", {}).get("selected_prompt_text_sha256")
        if selection_hash:
            lines += ["", f"Selected prompt prefix SHA256: `{selection_hash}`."]
        lines += ["", "Full strict/lenient groups, evidence scores, invalid JSON, false abstentions, paired paper-bootstrap intervals and gate outcomes are in science_report.json."]
        primary = science["arms"]["adapter"]["scores"]["strict"]["groups"]["primary"]
        base_primary = science["arms"]["base"]["scores"]["strict"]["groups"]["primary"]
        n = primary.get("unanswerable_denominator", 0)
        if n and primary.get("unanswerable_accuracy") is not None and base_primary.get("unanswerable_accuracy") is not None:
            lines += ["", f"Gold-unanswerable detection is a small descriptive cohort: adapter {round(primary['unanswerable_accuracy'] * n)}/{n}, base {round(base_primary['unanswerable_accuracy'] * base_primary['unanswerable_denominator'])}/{base_primary['unanswerable_denominator']}. These counts do not establish improved unanswerable detection."]
        lines += ["", "Figure/table cases are outside the text-only primary gate. Missing visual inputs and empty-text-evidence scoring complicate their evidence scores; subgroup results remain descriptive."]
    else:
        lines += ["57 training and 9 development papers passed both AI claim-audit passes. Final evaluation uses 40 CC-BY arXiv papers associated with PRL, fresh to this project; they are not proven absent from pretraining. The old 20 test papers remain historical diagnostics.",
            "All 57 selected training papers and all 40 fresh evaluation papers have cond-mat categories. Author overlap was not excluded. Paper/version and near-duplicate exclusions do not establish author-held-out or pretraining-clean generalization.",
            "The task produces a grounded abstract of at most 200 words from a fixed body excerpt that may omit the middle. Judging uses two fresh contexts from the same AI family as the advisor; this is not independent human scientific validation.", ""]
        lines += ["Development judging may be incomplete: the registered fallback selects base and aliases identical baselines when the frozen selection record so indicates. This establishes no stronger-prompt claim. See the prompt selection provenance.", ""]
        if prompt_selection:
            lines += [f"Actual development choice: **{prompt_selection['chosen']}**; identical-baseline alias: **{prompt_selection['identical_baselines']}**."]
            if prompt_selection.get("fallback_reason"):
                lines += [f"Registered fallback reason: {prompt_selection['fallback_reason']}."]
        if status["completed"]:
            if not status["passed"]:
                lines += [f"**For this abstract task, recommend the unadapted [{BASE}](https://huggingface.co/{BASE}/tree/{revision}). This adapter failed the completed registered science gates.**", ""]
            lines += ["Preference gives half credit to ties, averages two passes per paper, then averages papers; it is not a raw win rate.", "",
                "| Comparison | Adapter preference score | 95% paper-bootstrap CI | Unsupported-claim mean difference | 95% difference CI |", "|---|---:|---|---:|---|"]
            for name, metric in science["preference"].items():
                difference = science["unsupported_difference"][name]
                lines.append(f"| adapter vs {name} | {metric['mean']:.6f} | {metric['ci95']} | {difference['mean']:.6f} | {difference['ci95']} |")
            if any(d["mean"] > 0 and d["ci95"][0] <= 0 <= d["ci95"][1] for d in science["unsupported_difference"].values()):
                lines += ["", "The zero-increase unsupported-claim point gate failed near zero and its interval includes zero; this does not demonstrate a large factual regression."]
        else:
            lines += [f"**Final judging is incomplete: {science['complete_papers']}/40 papers have both valid passes. No complete final preference/fidelity estimate or task success is established.**"]
        if diagnostics is not None:
            lines += ["", "Secondary custom diagnostics (not metrics from a ROUGE package; no gate or selection effect). LCS/token F1 compare with the reference abstract; novel unique 8-grams compare with the supplied excerpt.", "",
                "| Arm | Mean words | Novel unique 8-gram fraction | Custom whitespace LCS F1 | Normalized token F1 |",
                "|---|---:|---:|---:|---:|"]
            for name in ("base", "adapter", "fixed_copy"):
                values = diagnostics["summary"][name]["means"]
                def cell(key):
                    value = values[key]
                    return (f"{value['mean']:.6f}" if value["mean"] is not None else "n/a") + f" (n={value['denominator']})"
                lines.append(f"| {name} | {cell('words')} | {cell('novel_unique_8gram_fraction')} | {cell('custom_whitespace_lcs_f1')} | {cell('normalized_token_f1')} |")
            lines += ["", "Means retain all 40 papers except undefined novel-8-gram fractions for outputs shorter than eight words; each cell reports its denominator. Fixed-copy empty cases are retained. Full per-paper and reviewer-agreement diagnostics are in secondary_diagnostics.json."]
            novelty = lambda arm: diagnostics["summary"][arm]["means"]["novel_unique_8gram_fraction"]["mean"]
            if (status["completed"] and not status["passed"] and any(m["mean"] < .5 for m in science["preference"].values())
                    and novelty("base") is not None and novelty("adapter") is not None and novelty("adapter") < novelty("base")):
                lines += ["", "Source overlap increased while judged quality fell. The causal mechanism is unproven; lexical overlap does not establish support for a claim."]
    lines += ["", "| Retention task | Base correct | Adapter correct | Net change |", "|---|---:|---:|---:|"]
    for name, metric in retention["comparison"]["tasks"].items():
        lines.append(f"| {name} | {metric['baseline_correct']} | {metric['candidate_correct']} | {metric['delta_correct']} |")
    lines += ["", "Retention gates allow at most five fewer numeric GSM8K and general-MMLU correct answers, and three fewer strict IFEval prompt passes. Numeric GSM8K uses last-number fallback; strict-format results are separately retained. Physics MMLU, if present, is diagnostic. Point limits do not establish non-inferiority.",
        f"The NF4 smoke's cached/full-prefix numerical discrepancy remains unresolved. Matching two smoke token sequences does not establish cache equivalence. Compared arms use identical PEFT preparation and cached generation. Science generation uses batch1; retention generation uses batch{retention_batch_size} with identical fixed groups across base/QASPER/PRL, and retention MC likelihood scoring remains batch1.", "",
        *([BATCH4_CAVEAT, ""] if retention_batch_size == 4 else []),
        "## Load the portable adapter", "", "Use the exact environment in environment.lock.txt. This recipe restores stored NF4 directly; it supplies no fresh quantization configuration. PEFT k-bit preparation casts nonquantized floating base parameters to FP32, as in both measured base and adapter arms. The tiny task example below is invented and contains no test-paper text. Resolve the adapter revision once after publication, then retain the printed immutable SHA for reproduction.", "", "```python",
        "import torch", "from huggingface_hub import HfApi", "from transformers import AutoTokenizer, Qwen3_5ForCausalLM", "from peft import PeftModel, prepare_model_for_kbit_training",
        f'base_id = "{BASE}"', f'base_revision = "{revision}"', f'adapter_id = "RenormalizationTech/{target}"',
        'adapter_revision = HfApi().model_info(adapter_id).sha  # Resolve once; retain this immutable SHA.',
        'print("Resolved adapter release:", adapter_revision)',
        "tokenizer = AutoTokenizer.from_pretrained(base_id, revision=base_revision, trust_remote_code=False)",
        'base = Qwen3_5ForCausalLM.from_pretrained(base_id, revision=base_revision, dtype=torch.bfloat16,',
        '    device_map={"": 0}, attn_implementation="sdpa", use_safetensors=True, trust_remote_code=False)',
        'base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,',
        '    gradient_checkpointing_kwargs={"use_reentrant": False})', "base.config.use_cache = False",
        "model = PeftModel.from_pretrained(base, adapter_id, revision=adapter_revision,",
        "    is_trainable=False, autocast_adapter_dtype=True)", "model.eval()",
        'user = ' + repr(synthetic_prompt(task)),
        'prompt = tokenizer.apply_chat_template([{ "role": "user", "content": user}],',
        "    tokenize=False, return_dict=False, add_generation_prompt=True, enable_thinking=False)",
        'inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda:0")',
        'with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):',
        "    tokens = model.generate(**inputs, use_cache=True, do_sample=False, num_beams=1, max_new_tokens=512,",
        "        temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,",
        "        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, logits_to_keep=1)",
        'print(tokenizer.decode(tokens[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))',
        "# For matched base comparisons, use this same prepared model:", "# with model.disable_adapter():", "#     base_tokens = model.generate(**inputs, ...same settings...)", "```", "",
        "This staging step performed no model load or generation. Portable-config runtime replay remains a separate pinned-download check before claiming a verified adapter publication.", "",
        "## Attribution and reproduction", "", "Model/code: Apache-2.0. QASPER and paper texts retain CC-BY-4.0 terms; these are separate from model licensing. Retention data: MMLU/GSM8K MIT; IFEval and its Google Research scorer Apache-2.0. See data_licenses.json, provenance.json and paper_attributions.json where applicable.",
        "The source/environment snapshots and selected checkpoint metadata preserve the original experiment. Sanitized metadata paths use placeholders; input hashes always describe the original local bytes. No base weights or merged adapter export is included."]
    return "\n".join(lines) + "\n"


def stage(task, training, science_path, retention_dir, retention_base, publication, output=None, manifest_path=None,
          experimental=False, base_revision=None, root=ROOT, diagnostics=None):
    if diagnostics is not None and task != "prl": raise ValueError("secondary diagnostics are PRL-only")
    root = Path(root).resolve()
    training, science_path, retention_dir, retention_base, publication = [Path(p).resolve() for p in (training, science_path, retention_dir, retention_base, publication)]
    output = Path(output).resolve() if output else root / "artifacts/releases" / TARGETS[task]
    manifest_path = Path(manifest_path).resolve() if manifest_path else root / "outputs" / f"qwen38-{task}-adapter-release-manifest-01.json"
    if output != root / "artifacts/releases" / TARGETS[task] or manifest_path.is_relative_to(output):
        raise ValueError("task-specific release folder and external explicit-file manifest required")
    if output.exists() or manifest_path.exists(): raise FileExistsError("preserve prior staging/manifest")
    identity = training_identity(task, training, root)
    receipt, revision = public_base(publication, identity["config"], base_revision)
    retention = paired_retention(retention_dir, retention_base, task, identity, root)
    science, status = qasper_science(science_path, identity, root) if task == "qasper" else prl_science(science_path, identity, root)
    completion = qasper_baseline_completion(science, root) if task == "qasper" else None
    secondary = prl_diagnostics(diagnostics, science_path, root) if diagnostics is not None else None
    all_pass = status["completed"] and status["passed"] and retention["comparison"]["retention_pass"]
    if not all_pass and not experimental:
        raise ValueError("failed/incomplete science or retention gates require explicit --experimental")
    saved = identity["saved"]
    portable = {**copy.deepcopy(saved), "base_model_name_or_path": BASE, "revision": revision}
    config_bytes = json_bytes(portable, root)
    if json.loads(config_bytes) != portable:
        raise ValueError("portable config has machine paths outside the two permitted changes")
    changed = {k: {"old": saved.get(k), "new": portable.get(k)} for k in set(saved) | set(portable) if saved.get(k) != portable.get(k)}
    if set(changed) - {"base_model_name_or_path", "revision"}: raise ValueError("unexpected adapter config modification")
    weights = identity["adapter"] / "adapter_model.safetensors"
    weights_hash = identity["audit"]["adapter_files"][weights.name]["sha256"]
    payloads = {"adapter_config.json": config_bytes, "science_report.json": safe_json(science, root),
        "retention_adapter.json": safe_json(retention["candidate"], root), "retention_base.json": safe_json(retention["baseline"], root),
        "retention_comparison.json": safe_json(retention["comparison"], root), "base_publication.json": safe_json(receipt, root),
        "selected_checkpoint.json": safe_json({"selected": identity["result"]["selected"], "audit": identity["audit"],
            "training_config": identity["config"], "training_protocol_sha256": sha(training / "protocol.json")}, root),
        "config_diff.json": safe_json({"allowed_fields": ["base_model_name_or_path", "revision"], "changes": changed,
            "original_config_sha256": sha(identity["adapter"] / "adapter_config.json"), "portable_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "weights_sha256": weights_hash, "weights_byte_identical": True}, root),
        "LICENSE": (root / "LICENSE").read_bytes()}
    payloads["prepared_manifest.json"] = safe_json(identity["prepared_manifest"], root)
    if completion is not None:
        payloads["baseline_completion.json"] = safe_json(completion["metadata"], root)
        payloads["baseline_completion_amendment.json"] = safe_json(load(completion["amendment"]), root)
    for label, directory in (("base", retention_base), ("adapter", retention_dir)):
        payloads[f"retention_registration_{label}.json"] = safe_json(load(directory / "registration.json"), root)
    if secondary is not None:
        payloads["secondary_diagnostics.json"] = safe_json({**secondary["report"],
            "verified_summary": secondary["summary"], "verification": secondary["provenance"]}, root)
        payloads["secondary_diagnostics_protocol.json"] = safe_json(load(secondary["protocol"]), root)
    environment = root / "outputs/qwen38-science-01/environment.freeze.txt"
    payloads["environment.lock.txt"] = "\n".join(json.loads(safe_json(environment.read_text().splitlines(), root))).encode() + b"\n"
    provenance = {"version": VERSION, "task": task, "repo_id": "RenormalizationTech/" + TARGETS[task],
        "experimental": bool(experimental), "recommended": bool(all_pass and not experimental), "science_status": status,
        "retention_batching": retention["batching"],
        "batching_caveat": BATCH4_CAVEAT if retention["batching"]["generation_batch_size"] == 4 else None,
        "recommended_for_prl_abstract_task": BASE if task == "prl" and status["completed"] and not status["passed"] else None,
        "secondary_diagnostics": secondary["provenance"] if secondary is not None else None,
        "baseline_completion": completion["metadata"] if completion is not None else None,
        "all_science_and_retention_gates_pass": bool(all_pass), "public_base": {"repo_id": BASE, "revision": revision,
            "receipt_sha256": sha(publication), "training_base_manifest_sha256": identity["config"]["nf4_artifact_manifest_sha256"]},
        "input_sha256": {str(science_path): sha(science_path), str(retention_dir / "scores.json"): sha(retention_dir / "scores.json"),
            str(retention_base / "scores.json"): sha(retention_base / "scores.json"), str(training / "result.json"): sha(training / "result.json"),
            str(retention_dir / "registration.json"): sha(retention_dir / "registration.json"),
            str(retention_base / "registration.json"): sha(retention_base / "registration.json")},
        "source_sha256": identity["protocol"]["source_sha256"], "environment_sha256": sha(environment),
        "stager_sha256": sha(__file__), "sanitizer_sha256": sha(root / "scripts/stage_physics_release.py"),
        "portable_runtime_check": "pending separate pinned-download model replay; CPU staging only"}
    if task == "prl": payloads["paper_attributions.json"] = safe_json(prl_attributions(root, identity["protocol"]["selected_ids"]), root)
    data = {"model_and_code": {"license": "Apache-2.0", "upstream": "Qwen/Qwen3.8-27B", "revision": "37561816e66b5b9153b1b64c10968c18c1e6f4a0"},
        "qasper": {"license": "CC-BY-4.0", "source": "https://huggingface.co/datasets/allenai/qasper",
            "acquisition": load(root / "data/qasper-v03-01/manifest.json")},
        "prl_arxiv_sources": {"license": "CC-BY-4.0", "terms": "Per-version authoritative license/author attribution in paper_attributions.json; PRL-associated arXiv sources, not publisher PDFs"},
        "retention": {"mmlu": "MIT", "gsm8k": "MIT", "ifeval": "Apache-2.0", "provenance": load(root / "data/benchmark-screen-01/provenance.json")}}
    payloads["data_licenses.json"] = safe_json(data, root)
    evaluation = root / "outputs" / ("qwen38-qasper-evaluation-01" if task == "qasper" else "qwen38-prl-evaluation-01")
    for name in ("registration.json", "prompt_selection.json"):
        payloads["evaluation_" + name] = safe_json(load(evaluation / name), root)
    provenance["evaluation_sha256"] = {name: sha(evaluation / name) for name in ("registration.json", "prompt_selection.json")}
    provenance["retention_source_sha256"] = load(retention_dir / "registration.json")["comparison_binding"]["source_sha256"]
    provenance["evaluation_source_sha256"] = load(evaluation / "registration.json")["source_sha256"]
    if task == "prl":
        provenance["review_evidence_sha256"] = {str(p): sha(p) for p in sorted(science_path.parent.rglob("*.json"))}
        for record in load(science_path.parent / "binding.json")["generations"].values():
            directory = root / record["run"]
            provenance["review_evidence_sha256"].update({str(directory / name): sha(directory / name)
                for name in ("registration.json", "result.json", "predictions.jsonl")})
    else:
        provenance["review_evidence_sha256"] = science["verified_input_sha256"]
    provenance["amendments_sha256"] = {}
    for name in ("qwen38-qasper-integrity-amendment-01.json", "qwen38-qasper-retention-hardware-amendment-01.json") if task == "qasper" else ("qwen38-prl-hardware-amendment-01.json", "qwen38-prl-development-fallback-01.json"):
        path = root / "results/protocols" / name
        if path.exists():
            provenance["amendments_sha256"][name] = sha(path)
            payloads[name] = safe_json(load(path), root)
    path = root / "results/protocols/qwen38-retention-batch-amendment-01.json"
    if path.exists():
        provenance["amendments_sha256"][path.name] = sha(path)
        payloads[path.name] = safe_json(load(path), root)
    payloads["provenance.json"] = safe_json(provenance, root)
    payloads["README.md"] = card(task, status, science, retention, revision, identity["config"], experimental,
        load(evaluation / "prompt_selection.json"), secondary, completion).encode()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as source:
        inventory = {}
        def add(name, path, expected):
            checked(path, expected)
            raw = Path(path).read_bytes()
            private_preflight(raw, root)
            source.writestr(name, raw)
            inventory[name] = expected
        for name, expected in identity["protocol"]["source_sha256"].items():
            add(name, training / "source" / name, expected)
        for name, expected in provenance["evaluation_source_sha256"].items():
            add("evaluation/" + name, evaluation / "source" / name, expected)
        for name, expected in provenance["retention_source_sha256"].items():
            add("retention/" + name, retention_dir / "source" / name, expected)
        if secondary is not None:
            for name, path in (("qwen38_prl_diagnostics.py", secondary["source"]),
                    ("qwen38-prl-diagnostics-01.json", secondary["protocol"]), ("qwen38_science_score.py", secondary["dependency"])):
                add("diagnostics/" + name, path, secondary["provenance"]["verified_input_sha256"][str(path)])
        if completion is not None:
            add("completion/qwen38_complete_science.py", completion["source"], completion["bound"][str(completion["source"])])
            add("completion/qwen38-qasper-baseline-completion-amendment-01.json", completion["amendment"],
                completion["bound"][str(completion["amendment"])])
        add("staging/qwen38_stage_adapter.py", __file__, sha(__file__))
        add("staging/stage_physics_release.py", root / "scripts/stage_physics_release.py", provenance["sanitizer_sha256"])
        source.writestr("scripts/__init__.py", "")
    payloads["source_inventory.json"] = safe_json(inventory, root)
    payloads["source.zip"] = archive.getvalue()
    for name, raw in payloads.items():
        if name != "source.zip": private_preflight(raw, root)
    # Detect any checkpoint/report changes during validation before writing.
    if validate_adapter(identity["adapter"], task, "base") != identity["audit"]: raise ValueError("selected adapter changed during staging")
    for path, expected in provenance["input_sha256"].items(): checked(path, expected)
    if secondary is not None:
        for path, expected in secondary["provenance"]["verified_input_sha256"].items(): checked(path, expected)
    if completion is not None:
        for path, expected in completion["bound"].items(): checked(path, expected)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(weights, output / weights.name)
    if sha(output / weights.name) != weights_hash: raise ValueError("copied adapter weights changed")
    for name, raw in payloads.items():
        with (output / name).open("xb") as stream: stream.write(raw)
    manifest = {"repo_id": "RenormalizationTech/" + TARGETS[task], "private": False, "files": {
        p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(output.iterdir())}}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as stream: stream.write(canonical(manifest) + "\n")
    return {"staged": str(output), "manifest": str(manifest_path), "repo_id": manifest["repo_id"],
        "recommended": provenance["recommended"], "experimental": experimental, "incomplete_science": status["incomplete"],
        "upload_performed": False, "GPU_started": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=list(TARGETS), required=True)
    p.add_argument("--training-run")
    p.add_argument("--science-report")
    p.add_argument("--retention")
    p.add_argument("--retention-base", default="outputs/qwen38-retention-base-04")
    p.add_argument("--nf4-publication", default="results/qwen38_nf4_huggingface_publication.json")
    p.add_argument("--base-revision")
    p.add_argument("--output")
    p.add_argument("--manifest")
    p.add_argument("--experimental", action="store_true")
    p.add_argument("--diagnostics", help="optional registered PRL final secondary diagnostics JSON; PRL-only")
    a = p.parse_args()
    if a.diagnostics and a.task != "prl": p.error("--diagnostics is PRL-only")
    training = ROOT / (a.training_run or "outputs/" + TRAINING_RUNS[a.task])
    science = ROOT / (a.science_report or ("outputs/qwen38-qasper-final-analysis-01.json" if a.task == "qasper" else "outputs/qwen38-prl-evaluation-01/final/scores.json"))
    result = stage(a.task, training, science, ROOT / (a.retention or f"outputs/qwen38-retention-{a.task}-01"),
        ROOT / a.retention_base, ROOT / a.nf4_publication, ROOT / a.output if a.output else None,
        ROOT / a.manifest if a.manifest else None, a.experimental, a.base_revision,
        **({"diagnostics": ROOT / a.diagnostics} if a.diagnostics else {}))
    print(canonical(result))


if __name__ == "__main__": main()
