"""Training/evaluation lineage checks without weights, GPU or network."""
import json

import pytest

from scripts.qwen38_eval_adapter import validate
from scripts.qwen38_science_runtime import sha


def selected_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoints/step-00001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"synthetic bytes")
    (checkpoint / "adapter_config.json").write_text("{}")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in checkpoint.iterdir()}
    (tmp_path / "config.json").write_text(json.dumps({"task": "qasper"}))
    (tmp_path / "result.json").write_text(json.dumps({"completed": True,
        "selected": {"path": str(checkpoint), "adapter_files": files}}))
    return checkpoint


def test_selected_adapter_files_are_verified_before_loading(tmp_path):
    checkpoint = selected_checkpoint(tmp_path)
    assert len(validate(checkpoint, "qasper", "base")["adapter_files"]) == 2
    (checkpoint / "adapter_model.safetensors").write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="changed after save"):
        validate(checkpoint, "qasper", "base")


def test_wrong_task_or_prompt_is_rejected(tmp_path):
    checkpoint = selected_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="identical training task prompt"):
        validate(checkpoint, "qasper", "verify")
    with pytest.raises(ValueError, match="selected checkpoint"):
        validate(checkpoint, "prl", "base")
