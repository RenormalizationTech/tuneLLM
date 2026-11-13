"""Offline report integrity and mocked documentation-only Hub transactions."""
import copy
import json
import shutil

import pytest

from scripts.common import ROOT
from scripts import stage_benchmark_reports
from scripts.stage_benchmark_reports import build, compact_conditions, compact_pairs, publication_inventory
from scripts.update_hf_documentation import (
    ALLOW, BASE_PIN, DEPENDENCIES, PREFIX, PostVerificationError, execute, fingerprint, validate_stage,
)


def staged(tmp_path, suffix="NF4"):
    before = {"README.md": fingerprint(b"old card"), "model.safetensors": fingerprint(b"weights"),
              "config.json": fingerprint(b"config"), "tokenizer.json": fingerprint(b"tokens"),
              "evaluation.json": fingerprint(b"historical scores"), ".gitattributes": fingerprint(b"LFS")}
    report = {"conditions": {"nf4_base": {"correct": 1}}}
    if suffix == "GGUF":
        report = {"conditions": {k: {} for k in ("bf16_gguf", "q8", "q4")}}
    updates = {"README.md": b"Reviewed card\n", "benchmark_summary.json": json.dumps(report).encode()}
    for name, data in updates.items():
        (tmp_path / name).write_bytes(data)
    manifest = {"repo_id": PREFIX + suffix, "private": suffix != "GGUF", "parent_commit": "a" * 40,
                "allowed_changes": sorted(ALLOW), "updates": {n: fingerprint(b) for n, b in updates.items()},
                "expected_inventory": before}
    if manifest["repo_id"] in DEPENDENCIES:
        manifest["pinned_dependency"] = DEPENDENCIES[manifest["repo_id"]]
    return manifest, before, updates


class Remote:
    def __init__(self, manifest, before):
        self.head, self.private = manifest["parent_commit"], manifest["private"]
        self.files = copy.deepcopy(before)
        self.calls = []
        self.dependency = BASE_PIN.copy()
        self.tamper_after = False
        self.race = False

    def inventory(self):
        return self.head, self.private, copy.deepcopy(self.files)

    def json_file(self, name, revision):
        assert name == "adapter_config.json" and revision == self.head
        return self.dependency

    def commit(self, updates, parent_commit):
        self.calls.append((updates, parent_commit))
        if self.race or parent_commit != self.head:
            raise RuntimeError("parent_commit conflict")
        assert set(updates) == ALLOW
        self.files.update({n: fingerprint(data) for n, data in updates.items()})
        if self.tamper_after:
            self.files["model.safetensors"] = fingerprint(b"changed model")
        self.head = "b" * 40
        return self.head


@pytest.fixture
def report_root(tmp_path):
    # Tests reconstruct only metadata, never depend on ignored real run outputs.
    for name in ("results/physics_prl_final.json", "results/huggingface_publication.json",
                 "results/nf4_huggingface_publication.json", "results/physics_huggingface_publication.json",
                 "releases/Qwen3-4B-tuneLLM-GGUF/provenance.json", "results/physics_prl_protocol.json",
                 "configs/benchmark_sources.json", "results/qlora_real_01.json"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    filename = tmp_path / "results/physics_prl_final.json"
    final = json.loads(filename.read_text())
    for condition in final["conditions"].values():
        directory = tmp_path / condition["evaluation_run"]
        directory.mkdir(parents=True)
        data = json.dumps(condition["summary"]).encode()
        (directory / "summary.json").write_bytes(data)
        condition["raw_result_sha256"]["summary.json"] = fingerprint(data)["sha256"]
        data = json.dumps({"completed": True, "git_head": {"stdout": "a" * 40},
                           "source_sha256": {"scripts/benchmark_screen.py": "b" * 64}}).encode()
        (directory / "run.json").write_bytes(data)
        condition["run_sha256"] = fingerprint(data)["sha256"]
    filename.write_text(json.dumps(final))
    return tmp_path


def test_dry_run_is_read_only_and_apply_preserves_all_non_doc_bytes(tmp_path):
    manifest, before, updates = staged(tmp_path)
    remote = Remote(manifest, before)
    assert execute(manifest, tmp_path, remote)["applied"] is False
    assert remote.calls == [] and remote.files == before
    result = execute(manifest, tmp_path, remote, apply=True)
    assert result["post_verified"] and result["revision"] == "b" * 40
    assert remote.calls == [(updates, "a" * 40)]
    for name in before:
        if name != "README.md":
            assert before[name] == result["post_inventory"][name]


@pytest.mark.parametrize("fault", ["head", "visibility", "weight", "addition", "historical"])
def test_remote_drift_blocks_commit(tmp_path, fault):
    manifest, before, _ = staged(tmp_path)
    remote = Remote(manifest, before)
    if fault == "head":
        remote.head = "c" * 40
    elif fault == "visibility":
        remote.private = False
    elif fault == "addition":
        remote.files["extra.json"] = fingerprint(b"extra")
    else:
        remote.files["evaluation.json" if fault == "historical" else "model.safetensors"] = fingerprint(b"changed")
    with pytest.raises(ValueError):
        execute(manifest, tmp_path, remote, apply=True)
    assert remote.calls == []


@pytest.mark.parametrize("fault", ["forbidden", "changed", "visibility", "existing_report", "parent", "symlink"])
def test_local_allowlist_and_reviewed_bytes_fail_closed(tmp_path, fault):
    manifest, _, _ = staged(tmp_path)
    if fault == "forbidden":
        manifest["updates"]["adapter_config.json"] = fingerprint(b"bad")
    elif fault == "changed":
        (tmp_path / "README.md").write_bytes(b"unreviewed")
    elif fault == "visibility":
        manifest["private"] = False
    elif fault == "existing_report":
        manifest["expected_inventory"]["benchmark_summary.json"] = fingerprint(b"old report")
    elif fault == "parent":
        manifest["parent_commit"] = "main"
    else:
        (tmp_path / "README.md").unlink()
        (tmp_path / "README.md").symlink_to(tmp_path / "benchmark_summary.json")
    with pytest.raises(ValueError):
        validate_stage(manifest, tmp_path)


def test_concurrent_commit_uses_parent_cas_and_never_retries(tmp_path):
    manifest, before, _ = staged(tmp_path)
    remote = Remote(manifest, before)
    remote.race = True
    with pytest.raises(RuntimeError, match="conflict"):
        execute(manifest, tmp_path, remote, apply=True)
    assert len(remote.calls) == 1 and remote.head == manifest["parent_commit"]


def test_post_verification_failure_retains_remote_revision(tmp_path):
    manifest, before, _ = staged(tmp_path)
    remote = Remote(manifest, before)
    remote.tamper_after = True
    with pytest.raises(PostVerificationError) as exc:
        execute(manifest, tmp_path, remote, apply=True)
    assert exc.value.record["revision"] == "b" * 40
    assert exc.value.record["applied"] and not exc.value.record["post_verified"]


@pytest.mark.parametrize("suffix", ["PRL-Writing-QLoRA", "GSM8K-QLoRA"])
def test_adapter_dependency_cannot_be_changed(tmp_path, suffix):
    manifest, before, _ = staged(tmp_path, suffix)
    remote = Remote(manifest, before)
    remote.dependency["revision"] = "c" * 40
    with pytest.raises(ValueError, match="pin"):
        execute(manifest, tmp_path, remote, apply=True)
    assert remote.calls == []


def test_public_report_excludes_private_conditions_and_paths(tmp_path):
    manifest, _, _ = staged(tmp_path, "GGUF")
    validate_stage(manifest, tmp_path)
    report = {"conditions": {n: {} for n in ("bf16_gguf", "q8", "q4", "nf4_base")}}
    data = json.dumps(report).encode()
    (tmp_path / "benchmark_summary.json").write_bytes(data)
    manifest["updates"]["benchmark_summary.json"] = fingerprint(data)
    with pytest.raises(ValueError, match="only GGUF"):
        validate_stage(manifest, tmp_path)


def test_summary_uses_saved_counts_and_only_public_metadata(report_root):
    full, views, _ = build(report_root)
    public = views[PREFIX + "GGUF"]
    assert public["items_per_condition"] == 384
    assert set(public["conditions"]) == {"bf16_gguf", "q8", "q4"}
    text = json.dumps(public)
    assert all(x not in text for x in ("/home/", "outputs/", "nf4", "writing", "paper_text", "input_ids"))
    assert full["earlier_gsm8k_experiment"]["base_fallback_correct"] == 121
    assert full["earlier_gsm8k_experiment"]["adapter_fallback_correct"] == 101
    assert full["conditions"]["nf4_base"]["scores"]["gsm8k"]["correct"] == 117


def test_changed_counts_and_invalid_pairing_are_rejected():
    final = json.loads((ROOT / "results/physics_prl_final.json").read_text())
    conditions = compact_conditions(final)
    final["paired_comparisons"]["bf16_gguf__q8"]["gsm8k"]["broken"] += 1
    with pytest.raises(ValueError, match="reconcile"):
        compact_pairs(final, conditions)
    final["conditions"]["q8"]["summary"]["gsm8k"]["fallback_correct"] += 1
    with pytest.raises(ValueError, match="reviewed"):
        compact_conditions(final)


def test_legacy_publication_requires_every_verified_file():
    with pytest.raises(ValueError, match="inventory"):
        publication_inventory({"files": ["README.md", "model.bin"],
                               "verified_files": [{"file": "README.md", "sha256": "a" * 64}]})


def test_evaluated_weight_must_match_published_provenance(monkeypatch, report_root):
    original = stage_benchmark_reports.read
    def changed(filename):
        result = original(filename)
        if filename.name == "physics_prl_final.json":
            result["conditions"]["q4"]["gguf_sha256"] = "0" * 64
        return result
    monkeypatch.setattr(stage_benchmark_reports, "read", changed)
    with pytest.raises(ValueError, match="checksum"):
        build(report_root)
