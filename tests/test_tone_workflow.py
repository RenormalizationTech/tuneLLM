"""Immutable CPU workflow guards; synthetic files only, no actual model outputs."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import tone_steering as runner
from scripts.tone_fixture import authored


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def mock_run(tmp_path, monkeypatch):
    cfg = {"judging_seed": 42}
    monkeypatch.setattr(runner, "verify_run", lambda run: (cfg, {}))
    rows = authored()["extraction"][:2]
    write(tmp_path / "inputs/extraction.json", rows)
    write(tmp_path / "inputs/rubric.json", {"instruction": "Unit test only"})
    # Deliberately not real model responses: tests exercise file integrity only.
    predictions = [{"id": r["id"], "output": "unit-test text", "generated_ids": [4], "hit_token_limit": False} for r in rows]
    for condition in ("neutral", "warm"):
        write(tmp_path / f"extraction/{condition}.json", {"predictions": predictions})
    write(tmp_path / "extraction/run.json", {"completed": True, "unit_test": True})
    runner.packets(SimpleNamespace(stage="extraction"), tmp_path)
    return rows


def test_packet_permissions_and_prediction_binding(tmp_path, monkeypatch):
    mock_run(tmp_path, monkeypatch)
    directory = tmp_path / "extraction/judging"
    assert directory.stat().st_mode & 0o777 == 0o700
    assert (directory / "keys.local.json").stat().st_mode & 0o777 == 0o600
    runner.verify_packets(directory.parent)
    write(tmp_path / "extraction/warm.json", {"changed": True})
    with pytest.raises(ValueError, match="source changed"):
        runner.verify_packets(directory.parent)


def lock_fixture(tmp_path):
    write(tmp_path / "manifest.json", {"unit_test": True})
    write(tmp_path / "development/budget_lock.json", {"budget": {"cap": 256}})
    write(tmp_path / "extraction/vectors.json", {"unit_test": True})
    (tmp_path / "extraction/vectors.pt").write_bytes(b"unit test placeholder, not tensor data")
    write(tmp_path / "inputs/final.json", [{"unit_test": "never read as prompts"}])
    directory = tmp_path / "development"
    write(directory / "default.json", {"unit_test": True})
    write(directory / "conditions.json", {"conditions": ["default"],
          "prediction_sha256": {"default": runner.stream_hash(directory / "default.json")}})
    judges = directory / "judging"
    for name in ("content_ratings.json", "warmth_ratings.json", "keys.local.json"):
        write(judges / name, {"unit_test": True})
    write(judges / "packet_manifest.json", {"packet_files": {"keys.local.json": runner.stream_hash(judges / "keys.local.json")},
          "source_files": {"conditions.json": runner.stream_hash(directory / "conditions.json"),
                           "default.json": runner.stream_hash(directory / "default.json")}})
    selection = {"split": "development", "final_used_for_selection": False,
                 "decision": {"condition": "default", "layer": None, "multiplier": 0},
                 "manifest_sha256": runner.stream_hash(tmp_path / "manifest.json"),
                 "budget_lock_sha256": runner.stream_hash(directory / "budget_lock.json"),
                 "development_conditions_sha256": runner.stream_hash(directory / "conditions.json"),
                 "vectors_sha256": runner.stream_hash(tmp_path / "extraction/vectors.pt"),
                 "extraction_vectors_report_sha256": runner.stream_hash(tmp_path / "extraction/vectors.json"),
                 "rating_sha256": {k: runner.stream_hash(judges / f"{k}_ratings.json") for k in ("content", "warmth")}}
    write(tmp_path / "selection.json", selection)
    return selection


@pytest.mark.parametrize("changed", ["manifest.json", "development/budget_lock.json", "development/conditions.json",
    "development/default.json", "development/judging/content_ratings.json", "development/judging/warmth_ratings.json",
    "extraction/vectors.pt", "extraction/vectors.json"])
def test_mutating_any_selection_input_before_lock_is_rejected(tmp_path, changed):
    lock_fixture(tmp_path)
    assert runner.final_lock(tmp_path)["selection_fixed"]
    (tmp_path / changed).write_text("mutated after selection")
    with pytest.raises(ValueError):
        runner.final_lock(tmp_path)


def test_final_manifest_freezes_final_fixture_and_selection_without_reselection(tmp_path, monkeypatch):
    lock_fixture(tmp_path)
    lockfile = tmp_path / "registered.json"
    write(lockfile, runner.final_lock(tmp_path))
    monkeypatch.setattr(runner, "committed", lambda filename: None)
    runner.check_lock(tmp_path, lockfile)
    write(tmp_path / "inputs/final.json", [{"mutated": True}])
    with pytest.raises(ValueError, match="manifest"):
        runner.check_lock(tmp_path, lockfile)


def test_no_final_file_read_before_lock_or_gpu_authorization(tmp_path, monkeypatch):
    def no_read(*args):
        raise AssertionError("unauthorized filesystem/model access")
    monkeypatch.setattr(runner, "verify_run", no_read)
    with pytest.raises(ValueError, match="execute-gpu"):
        runner.gpu_ready(SimpleNamespace(execute_gpu=False), tmp_path)
    monkeypatch.setattr(runner, "gpu_ready", lambda args, run: ({}, {}))
    with pytest.raises(ValueError, match="lock required"):
        runner.final_generate(SimpleNamespace(lock=None), tmp_path)


def test_extraction_fixed_first50_and_task_completion_required():
    rows = authored()["extraction"]
    predictions = [{"id": r["id"], "output": "Maya sorting two towels on Monday.",
                    "generated_ids": [7]*10, "ended_eos": False, "hit_token_limit": False} for r in rows]
    # Fixed-check function mocked only here to isolate eligibility ordering/AI rules.
    original = runner.literal_checks
    runner.literal_checks = lambda row, output: {"complete": True, "unsupported_literal": False}
    try:
        content = {c: {r["id"]: {"coverage": {f["key"]: True for f in r["required_facts"]},
            "task_completed": True, "unsupported_claims": 0, "contradicted_claims": 0} for r in rows} for c in ("neutral", "warm")}
        preferences = {"neutral__warm": {r["id"]: 1. for r in rows}}
        cfg = {"pair_length_ratio": [.85, 1.25], "retain_pairs": 50}
        content["warm"][rows[0]["id"]]["task_completed"] = False
        retained, audit = runner.eligible_pairs(rows, predictions, predictions, content, preferences, cfg)
        assert retained == [r["id"] for r in rows[1:51]]
        assert not audit[0]["checks"]["ai_content"]
        assert len({r["family"] for r in rows if r["id"] in retained}) == 16
    finally:
        runner.literal_checks = original


def test_exclusive_write_never_overwrites_phase_result(tmp_path):
    filename = tmp_path / "phase.json"
    runner.save(filename, {"first": True})
    with pytest.raises(FileExistsError):
        runner.save(filename, {"second": True})
    assert json.loads(filename.read_text()) == {"first": True}
