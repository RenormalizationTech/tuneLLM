import json
from types import SimpleNamespace

import pytest

from scripts import run_physics_final as final


def test_final_preparation_rejects_unfixed_stage_before_any_test_read(tmp_path, monkeypatch):
    filename = tmp_path / "selection.json"
    filename.write_text(json.dumps({"stage": "lr_pilot", "split": "development"}))
    args = SimpleNamespace(plan=str(tmp_path / "plan.json"), selection=str(filename))
    with pytest.raises(ValueError, match="fixed development rank"):
        final.prepare(args)
    assert not (tmp_path / "plan.json").exists()


def test_changed_fixed_input_blocks_execution_before_gpu_or_protocol(tmp_path, monkeypatch):
    scientific_input = tmp_path / "fixture.json"
    scientific_input.write_text("original")
    plan = {"frozen_sha256": {str(scientific_input): final.digest(scientific_input)}}
    filename = tmp_path / "plan.json"
    filename.write_text(json.dumps(plan))
    scientific_input.write_text("changed")
    with pytest.raises(ValueError, match="fixed final input changed"):
        final.execute(SimpleNamespace(plan=str(filename)))


def test_existing_result_without_invocation_cannot_be_reused(tmp_path):
    output = tmp_path / "existing-screen"
    output.mkdir()
    (output / "run.json").write_text('{"completed":true}')
    plan = {"frozen_sha256": {}, "preparations": {},
            "jobs": [{"output": str(output), "kind": "screen", "command": ["must-not-run"]}]}
    filename = tmp_path / "plan.json"
    filename.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="no fixed-plan invocation"):
        final.execute(SimpleNamespace(plan=str(filename)))


def test_all_fail_descriptive_choice_never_claims_passing_candidate():
    rows = [{"id": "r8", "rank": 8, "learning_rate": 5e-5,
             "best_epoch": {"token_weighted_nll": 2.0}},
            {"id": "r32", "rank": 32, "learning_rate": 5e-5,
             "best_epoch": {"token_weighted_nll": 1.9}}]
    assert final.choose({"decision": {"fallback_base": True}, "candidates": rows})["id"] == "r32"
