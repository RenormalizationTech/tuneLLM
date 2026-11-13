"""Offline publication guards; no Hub access or model allocation."""
import copy
import hashlib
import json

import pytest

from scripts.verify_physics_publication import (
    BASE, BASE_FILES, BASE_REVISION, aligned_probes, check_pin,
    compare_probes, publication_files, verify_files,
)


def record(content):
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def probes():
    tasks = [{"id": str(i), "task": task} for i in range(10) for task in ("title",)]
    ids = [str(i) for i in range(8)]
    predictions = [{"id": identifier, "batch_index": 0, "batch_position": i,
                    "batch_ids": ids, "input_ids": [1, i], "generated_ids": [i, 2]}
                   for i, identifier in enumerate(ids)]
    return tasks, predictions


def test_base_pin_requires_published_revision_not_upstream():
    config = {"base_model_name_or_path": BASE, "revision": BASE_REVISION}
    check_pin(config)
    for update in ({"base_model_name_or_path": "Qwen/Qwen3-4B"},
                   {"revision": "1cfa9a7208912126459214e8b04321603b3df60c"}, {"revision": None}):
        with pytest.raises(ValueError, match="pin"):
            check_pin({**config, **update})
    assert len(BASE_FILES) == 12


def test_byte_hash_and_size_both_required_without_mutation(tmp_path):
    filename = tmp_path / "weights.safetensors"
    original = b"packed state"
    filename.write_bytes(original)
    files = {filename.name: record(original)}
    assert verify_files(tmp_path, files) == files
    filename.write_bytes(b"Packed state")  # Same length, different SHA.
    with pytest.raises(ValueError, match="bytes differ"):
        verify_files(tmp_path, files)
    filename.write_bytes(original + b"!")
    with pytest.raises(ValueError, match="bytes differ"):
        verify_files(tmp_path, files)
    assert filename.read_bytes() == original + b"!"


@pytest.mark.parametrize("name", ["../outside", "/outside", "..\\outside"])
def test_reject_manifest_path_escape(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe"):
        verify_files(tmp_path, {name: record(b"")})


def test_first_eight_titles_preserve_order_and_batch():
    tasks, predictions = probes()
    tasks.insert(1, {"id": "abstract", "task": "abstract"})
    selected, references = aligned_probes(tasks, predictions)
    assert [r["id"] for r in selected] == [str(i) for i in range(8)]
    checks = compare_probes(references, copy.deepcopy(references))
    assert checks["all_input_ids_equal"] and checks["all_generated_ids_equal"]
    assert checks["logits_compared"] is False and checks["max_abs_logit_difference"] is None
    changed = copy.deepcopy(references)
    changed[0]["generated_ids"] = [99]
    changed[1]["input_ids"] = [99]
    checks = compare_probes(references, changed)
    assert checks["exact_generated_ids_count"] == checks["exact_input_ids_count"] == 7
    assert not checks["all_generated_ids_equal"] and not checks["all_input_ids_equal"]
    assert "id" not in checks and "predictions" not in checks


@pytest.mark.parametrize("fault", ["duplicate_task", "duplicate_prediction", "missing", "reorder", "batch"])
def test_reject_unaligned_or_ambiguous_reference(fault):
    tasks, predictions = probes()
    if fault == "duplicate_task":
        tasks.append(tasks[0])
    elif fault == "duplicate_prediction":
        predictions.append(predictions[0])
    elif fault == "missing":
        predictions.pop()
    elif fault == "reorder":
        predictions.reverse()
    else:
        predictions[0]["batch_ids"] = ["other"]
    with pytest.raises(ValueError):
        aligned_probes(tasks, predictions)


def test_publication_requires_exact_reviewed_private_revision(tmp_path):
    files = {n: record(n.encode()) for n in ("adapter_config.json", "adapter_model.safetensors")}
    manifest = {"repo_id": "owner/private-adapter", "private": True, "files": files}
    filename = tmp_path / "manifest.json"
    filename.write_text(json.dumps(manifest))
    publication = {"repo_id": manifest["repo_id"], "private": True,
                   "status": "published_remote_hashes_verified", "revision": "a" * 40,
                   "manifest_sha256": record(filename.read_bytes())["sha256"], "files": files}
    assert publication_files(publication, manifest, filename) == files
    for update in ({"private": False}, {"revision": "main"}, {"manifest_sha256": "bad"},
                   {"status": "uploaded_pending_hash_verification"}, {"files": {}}):
        with pytest.raises(ValueError):
            publication_files({**publication, **update}, manifest, filename)


def test_generated_alignment_cannot_hide_reordering():
    _, predictions = probes()
    with pytest.raises(ValueError, match="alignment"):
        compare_probes(predictions, list(reversed(predictions)))
