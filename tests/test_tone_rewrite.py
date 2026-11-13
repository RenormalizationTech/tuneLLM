"""CPU-only B construction tests; no actual responses, generation or GPU calls."""
import copy
import json
from types import SimpleNamespace

import pytest

from scripts import tone_rewrite as rewrite
from scripts.common import read_json
from scripts.tone_fixture import authored


def test_config_changes_only_b_protocol_and_template_exact():
    a, b = read_json("configs/tone_steering_02.json"), read_json("configs/tone_steering_02b.json")
    assert [k for k in a if a[k] != b[k]] == ["protocol"]
    rows = authored()["extraction"]
    predictions = [{"id": r["id"], "output": "Unit test placeholder."} for r in rows]
    prompts, budgets = rewrite.rewrite_rows(rows, predictions)
    assert prompts[0]["user"] == ("Original scenario:\n" + rows[0]["user"] + "\n\nNeutral message:\nUnit test placeholder.\n\n"
        "Rewrite this message in a warm, encouraging, considerate tone. Keep every stated fact and the same requested message; "
        "add no facts, flattery or extra claims. Use at most 6 words.\nReturn only the rewritten message.")
    assert len(prompts) == 80 and [p["id"] for p in prompts] == [r["id"] for r in rows]
    assert all(p["system"] == rewrite.NEUTRAL_SYSTEM for p in prompts)
    assert all(b["instructed_max_words"] == 6 for b in budgets)


def test_word_budget_is_diagnostic_not_a_new_gate():
    budgets = [{"id": "test", "instructed_max_words": 3}]
    result = rewrite.compliance(budgets, [{"id": "test", "output": "one two three four"}])
    assert result[0]["instruction_complied"] is False
    assert result[0]["word_budget_is_hard_eligibility_gate"] is False
    with pytest.raises(ValueError, match="order"):
        rewrite.compliance(budgets, [{"id": "changed", "output": "test"}])


def write(filename, value):
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(value))


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    from scripts.batched_generation import render_prompt
    from transformers import AutoTokenizer
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)
        def __call__(self, text, **kwargs):
            return {"input_ids": list(map(ord, text))}
    tokenizer = Tokenizer()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    source, run = tmp_path / "a", tmp_path / "b"
    cfg = read_json("configs/tone_steering_02b.json")
    source_cfg = {**cfg, "protocol": "docs/tone_steering_protocol_02.md"}
    manifest = {"sources": {"unit-test-core": "unchanged"}, "snapshot": "unit-test-tokenizer"}
    write(source / "config.json", source_cfg)
    write(source / "manifest.json", manifest)
    rows = authored()["extraction"]
    write(source / "inputs/extraction.json", rows)
    write(run / "inputs/extraction.json", rows)
    predictions = [{"id": r["id"], "output": "Synthetic test metadata, not an experiment response.",
        "rendered_prompt": (prompt := render_prompt(tokenizer, {**r, "system": rewrite.NEUTRAL_SYSTEM})),
        "input_ids": tokenizer(prompt)["input_ids"]} for r in rows]
    write(source / "extraction/neutral.json", {"predictions": predictions})
    write(source / "extraction/run.json", {"completed": True, "representation": "bf16", "physical_gpu": "0", "gpu_uuid": "unit-test-uuid"})
    neutral_sha = rewrite.stream_hash(source / "extraction/neutral.json")
    registration = {"a_neutral_sha256": neutral_sha, "rewrite_source_sha256": rewrite.stream_hash(rewrite.__file__),
        "a_manifest_sha256": rewrite.stream_hash(source / "manifest.json"),
        "a_input_extraction_sha256": rewrite.stream_hash(source / "inputs/extraction.json"),
        "a_gpu_uuid": "unit-test-uuid", "rewrite_template": rewrite.TEMPLATE}
    monkeypatch.setattr(rewrite, "verify_run", lambda directory: (cfg, manifest))
    return run, source, registration, neutral_sha


def test_input_validation_binds_neutral_bytes_original_ids_context_and_gpu(source_fixture):
    run, source, registration, sha = source_fixture
    before = (source / "extraction/neutral.json").read_bytes()
    _, _, prompts, budgets, filename = rewrite.validate_inputs(run, source, registration, required_neutral_sha=sha)
    assert len(prompts) == len(budgets) == 80 and filename.read_bytes() == before
    for field in ("a_neutral_sha256", "rewrite_source_sha256", "a_manifest_sha256", "a_input_extraction_sha256", "a_gpu_uuid", "rewrite_template"):
        with pytest.raises(ValueError):
            rewrite.validate_inputs(run, source, {**registration, field: "modified"}, required_neutral_sha=sha)


def test_source_prompt_tampering_rejected_even_if_other_metadata_is_resealed(source_fixture):
    run, source, registration, _ = source_fixture
    file = source / "extraction/neutral.json"
    value = json.loads(file.read_text())
    value["predictions"][0]["rendered_prompt"] = "different context"
    write(file, value)
    sha = rewrite.stream_hash(file)
    with pytest.raises(ValueError, match="context"):
        rewrite.validate_inputs(run, source, {**registration, "a_neutral_sha256": sha}, required_neutral_sha=sha)


def test_default_execution_is_cpu_validation_only(tmp_path, monkeypatch):
    registration = tmp_path / "reg.json"
    write(registration, {})
    monkeypatch.setattr(rewrite, "validate_inputs", lambda *a: ({}, {}, [], [], tmp_path / "unused"))
    def forbidden(*args, **kwargs):
        raise AssertionError("no GPU or model operations during default validation")
    monkeypatch.setattr(rewrite, "gpu_ready", forbidden)
    monkeypatch.setattr(rewrite, "load_model", forbidden)
    rewrite.execute(SimpleNamespace(run=str(tmp_path / "b"), source_run=str(tmp_path / "a"),
                                    registration=str(registration), execute_gpu=False))
