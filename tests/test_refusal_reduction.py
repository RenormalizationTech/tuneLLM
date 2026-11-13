"""Synthetic CPU-only guards; no experiment predictions or ratings authored."""
from types import SimpleNamespace
import copy

import pytest

from scripts import refusal_reduction as reduction
from scripts.common import read_json


def metric(full=2, useful=11, total=30, **kwargs):
    return {"total": total, "valid_full_compliance": full, "useful_and_correct": useful,
            "empty": 0, "degenerate": 0, "unscorable": 0, **kwargs}


def candidates():
    return {reduction.condition(m, label): metric(full=8) for m in reduction.METHODS for label in ("safe", "unsafe")}


def baseline():
    return {"m0_safe": metric(), "m0_unsafe": metric()}


def test_selection_full_compliance_boundary_quality_and_tie_order():
    metrics = candidates()
    assert reduction.choose(baseline(), metrics)["selected_method"] == "ablate_all"
    metrics["subtract23_safe"]["useful_and_correct"] = 12
    assert reduction.choose(baseline(), metrics)["selected_method"] == "subtract23"
    metrics["subtract23_safe"]["useful_and_correct"] = 11
    metrics["subtract23_unsafe"]["valid_full_compliance"] = 9
    assert reduction.choose(baseline(), metrics)["selected_method"] == "subtract23"
    metrics["subtract23_unsafe"]["valid_full_compliance"] = 7
    metrics["ablate_all_safe"]["useful_and_correct"] = 8
    assert reduction.choose(baseline(), metrics)["null"]


@pytest.mark.parametrize("label", ["safe", "unsafe"])
@pytest.mark.parametrize("quality", ["empty", "degenerate", "unscorable"])
def test_each_class_quality_increase_rejects_candidate(label, quality):
    metrics = candidates()
    for method in reduction.METHODS:
        metrics[reduction.condition(method, label)][quality] = 1
    assert reduction.choose(baseline(), metrics)["null"]


def test_invalid_compliance_never_looks_successful_by_complementing_refusal():
    rows = [{"id": str(i), "family": "synthetic", "label": "unsafe"} for i in range(3)]
    predictions = [{"id": "0", "output": "", "hit_token_limit": False, "generated_ids": []},
                   {"id": "1", "output": "Synthetic test output", "hit_token_limit": False, "generated_ids": [1, 2]}]
    ratings = {str(i): {"refusal_class": "full_compliance", "unscorable": i == 1, "useful_and_correct": False} for i in range(3)}
    measured = reduction.full_compliance(reduction.common.score_condition(rows, predictions, ratings))
    assert measured["overall"]["total"] == 3
    assert measured["overall"]["primary_refused"] == measured["overall"]["valid_full_compliance"] == 0
    assert measured["overall"]["valid_full_compliance_rate"] == 0


def test_fixture_prefixes_ids_and_config_policy():
    fixture = {t: [{("key" if t == "ifeval" else "id"): str(i)} for i in range(n+1)] for t, n in reduction.COUNTS.items()}
    selected, ids = reduction.retention_fixture(fixture)
    assert sum(map(len, selected.values())) == 48
    for task, count in reduction.COUNTS.items():
        assert selected[task] == fixture[task][:count]
        assert ids[task] == [str(i) for i in range(count)]
    cfg = read_json("configs/refusal_reduction.json")
    reduction.validate_config(cfg)
    for key, value in (("gpu", "1"), ("judging_seed", 20261115), ("subtraction_coefficient", .5),
                       ("language_detector_seed", 1), ("all_token_positions", False)):
        with pytest.raises(ValueError):
            reduction.validate_config({**cfg, key: value})


def test_final_quality_is_separate_no_reselection_and_retention_limits():
    base = {"safe": metric(total=42), "unsafe": metric()}
    after = {"safe": metric(total=42, useful=8), "unsafe": metric(full=8)}
    retention = {"combined_mc_correct": 24, "metrics": {"gsm8k": {"fallback_correct": 6},
        "ifeval": {"prompt_strict_correct": 5}}, "generation_quality": {t: {"empty": 0, "degenerate": 0} for t in ("gsm8k", "ifeval")}}
    candidate = copy.deepcopy(retention)
    candidate["combined_mc_correct"] -= 2
    candidate["metrics"]["gsm8k"]["fallback_correct"] -= 1
    candidate["metrics"]["ifeval"]["prompt_strict_correct"] -= 1
    result = reduction.final_quality(base, after, retention, candidate)
    assert result["all_criteria_pass"] and not result["reselection_allowed"]
    candidate["generation_quality"]["ifeval"]["degenerate"] = 1
    assert not reduction.final_quality(base, after, retention, candidate)["all_criteria_pass"]


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    """Test artifacts live ONLY in pytest's temporary directory."""
    cfg = read_json("configs/refusal_reduction.json")
    manifest = {"sources": {}, "reused_files_sha256": {}, "classifier_context_difference": "synthetic packet context"}
    monkeypatch.setattr(reduction, "verify", lambda *a, **k: (cfg, manifest))
    reduction.save(tmp_path/"manifest.json", manifest)
    rows = [{"id": f"{label}-{i}", "instruction": "Synthetic CPU test instruction", "label": label, "family": "synthetic"}
            for label in ("safe", "unsafe") for i in range(30)]
    reduction.save(tmp_path/"inputs/development.json", {"items": rows})
    for name in ("final.json", "retention.json", "retention_ids.json"):
        reduction.save(tmp_path/"inputs"/name, {"synthetic": True})
    reduction.save(tmp_path/"inputs/rubric.json", read_json("evaluation/refusal/response_rubric.json"))
    judge = {"model_identifier": "synthetic-test-judge", "exposed_settings": {}, "fresh_context": True, "advisor_overlap": False}
    packet, keys, ratings = [], {}, []
    for label in ("safe", "unsafe"):
        predictions = []
        for i, row in enumerate(r for r in rows if r["label"] == label):
            anon = f"old-{row['id']}"
            packet.append({"id": anon, "prompt": row["instruction"], "response": "Synthetic test output"})
            keys[anon] = {"condition": "m0_"+label, "source_id": row["id"]}
            refused = i < (1 if label == "safe" else 28)
            ratings.append({"id": anon, "refusal_class": "full_refusal" if refused else "full_compliance",
                            "unscorable": False, "useful_and_correct": label == "safe" and 1 <= i <= 11})
            predictions.append({"id": row["id"], "output": "Synthetic test output", "hit_token_limit": False, "generated_ids": [2, 3]})
        reduction.save(tmp_path/f"reuse/m0_{label}.json", {"predictions": predictions})
    reduction.save(tmp_path/"reuse/packet.json", {"items": packet})
    reduction.save(tmp_path/"reuse/keys.local.json", keys)
    reduction.save(tmp_path/"reuse/ratings.json", {"judge": judge, "ratings": ratings})
    names = {reduction.condition(m, label): label for m in reduction.METHODS for label in ("safe", "unsafe")}
    for name, label in names.items():
        reduction.save(tmp_path/f"development/{name}.json", {"predictions": [
            {"id": r["id"], "output": "Synthetic new test output", "generated_ids": [4, 5], "hit_token_limit": False}
            for r in rows if r["label"] == label]})
    reduction.save(tmp_path/"development/run.json", {"completed": True, "elapsed_wall_seconds": 1})
    reduction.save(tmp_path/"development/budget_lock.json", {"synthetic": True})
    reduction.save_conditions(tmp_path/"development", "development", names)
    reduction.packets(SimpleNamespace(phase="development"), tmp_path)
    packet = read_json(tmp_path/"development/judging/packet.json")["items"]
    reduction.save(tmp_path/"development/judging/ratings.json", {"judge": judge, "ratings": [
        {"id": r["id"], "refusal_class": "full_compliance", "unscorable": False, "useful_and_correct": True} for r in packet]})
    return tmp_path


def test_packet120_new_only_and_reuse_labels_take_precedence(synthetic_run):
    run = synthetic_run
    packet = read_json(run/"development/judging/packet.json")["items"]
    keys = read_json(run/"development/judging/keys.local.json")
    assert len(packet) == 120
    assert all(set(r) == {"id", "prompt", "response"} for r in packet)
    assert all(not k["condition"].startswith("m0_") for k in keys.values())
    assert (run/"development/judging").stat().st_mode & 0o777 == 0o700
    assert (run/"development/judging/keys.local.json").stat().st_mode & 0o777 == 0o600
    reduction.selection(SimpleNamespace(), run)
    record = read_json(run/"selection.json")
    assert record["reused_labels"] == 60 and record["newly_judged"] == 120
    assert record["baseline_metrics"]["m0_unsafe"]["valid_full_compliance"] == 2
    assert reduction.selection_lock(run)["decision"]["selected_method"] == "ablate_all"


@pytest.mark.parametrize("filename", ["reuse/ratings.json", "reuse/m0_safe.json", "development/judging/ratings.json",
                                      "development/subtract23_safe.json", "inputs/retention.json"])
def test_lock_rejects_mutation(synthetic_run, filename):
    run = synthetic_run
    reduction.selection(SimpleNamespace(), run)
    locked = reduction.selection_lock(run)
    (run/filename).write_text("synthetic mutation")
    if filename == "inputs/retention.json":
        assert reduction.selection_lock(run) != locked  # Original committed lock rejects changed fixture hash.
    else:
        with pytest.raises(ValueError):
            reduction.selection_lock(run)


def test_gpu_gate_and_null_skip_before_model_or_final_access(synthetic_run, monkeypatch):
    with pytest.raises(ValueError, match="execute-gpu"):
        reduction.gpu_ready(SimpleNamespace(execute_gpu=False), synthetic_run)
    monkeypatch.setattr(reduction, "check_lock", lambda *a: {"decision": {"null": True}})
    def forbidden(*a, **k):
        raise AssertionError("null may not access GPU or final data")
    monkeypatch.setattr(reduction, "gpu_ready", forbidden)
    monkeypatch.setattr(reduction.old, "groups", forbidden)
    reduction.final(SimpleNamespace(), synthetic_run)
    reduction.retention(SimpleNamespace(), synthetic_run)
    assert read_json(synthetic_run/"final_skipped.json")["no_final_generation"]
    assert read_json(synthetic_run/"retention_skipped.json")["no_retention_generation"]


def test_null_summary_costs_and_no_raw_text(synthetic_run, monkeypatch):
    run = synthetic_run
    reduction.selection(SimpleNamespace(), run)
    monkeypatch.setattr(reduction, "check_lock", lambda *a: {"decision": {"null": True}})
    reduction.save(run/"probe/run.json", {"completed": True, "elapsed_wall_seconds": 2})
    reduction.save(run/"lock.json", {"synthetic": True})
    reduction.summary(SimpleNamespace(lock=str(run/"lock.json"), output=str(run/"summary.json")), run)
    result = read_json(run/"summary.json")
    assert result["null"] and set(result["costs"]) == {"probe", "development"}
    assert result["reused_labels"] == 60 and result["new_counts"]["development_judgments"] == 120
    assert result["claude_design_approval"] is False
    text = (run/"summary.json").read_text()
    for forbidden in ("Synthetic test instruction", "Synthetic new test output", "input_ids", "generated_ids"):
        assert forbidden not in text
