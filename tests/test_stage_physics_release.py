import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import pytest

from scripts.stage_physics_release import (BASE, MANIFEST, SCORING_MODULES, TRAINING_MODULES,
                                           bytes_sha, corpus_attributions, portable_nf4_report, private_preflight, redact_metadata, sha,
                                           stage, verify_base)


def write(filename, value):
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


@pytest.fixture
def project(tmp_path):
    root, run = tmp_path, tmp_path / "outputs/physics-candidate"
    inputs = run / "inputs"
    inputs.mkdir(parents=True)
    cfg = {"model_id": "Qwen/Qwen3-4B", "revision": "a" * 40, "representation": "nf4",
           "rank": 32, "learning_rate": 5e-5, "epochs": 3, "seed": 42, "steps_per_epoch": 69,
           "max_sequence_length": 1024, "per_device_train_batch_size": 1,
           "repo_root": str(root), "nf4_checkpoint": str(root / "artifacts/nf4/base")}
    write(run / "config.json", cfg)
    write(run / "learning_curve.json", [{"step": step, "epoch": epoch, "eval_loss": nll}
                                        for epoch, step, nll in ((1, 69, 2), (2, 138, 1.9), (3, 207, 1.95))])
    checkpoint = run / "checkpoints/checkpoint-138"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"synthetic immutable adapter bytes")
    original_config = {"base_model_name_or_path": cfg["nf4_checkpoint"], "revision": cfg["revision"],
                       "r": 32, "lora_alpha": 64, "lora_dropout": 0.05, "target_modules": ["q_proj"], "bias": "none"}
    write(checkpoint / "adapter_config.json", original_config)
    source_names = sorted(set(TRAINING_MODULES) | set(SCORING_MODULES))
    source_hashes = {}
    for name in source_names:
        original, frozen = root / name, run / "source" / name
        original.parent.mkdir(parents=True, exist_ok=True)
        frozen.parent.mkdir(parents=True, exist_ok=True)
        original.write_text("# synthetic tracked source\nVALUE = 1\n")
        frozen.write_bytes(original.read_bytes())
        source_hashes[name] = sha(original)
    (root / "docs").mkdir()
    (root / "docs/reproduction.md").write_text(f"Local workspace: {root}/outputs/run\n")
    (root / ".env").write_text("must never enter source archive")
    tracked = source_names + ["docs/reproduction.md", ".env", "data/raw.txt", ".git/config"]
    manifest, raw_hashes = [], {}
    number = 0
    for split, count in (("train", 70), ("validation", 10), ("test", 20)):
        rows = []
        for _ in range(count):
            number += 1
            row = {"id": f"paper{number}v1", "paper_id": f"paper{number}", "text": f"Scientific text {number}.\nEquation x=1."}
            rows.append(row)
            manifest.append({**{key: row[key] for key in ("id", "paper_id")}, "split": split,
                             "arxiv_version": "v1", "authors": "Paper Author", "title": f"Paper {number}",
                             "doi": f"10.1103/paper{number}", "source_url": f"https://arxiv.org/src/paper{number}v1",
                             "source_sha256": "b" * 64, "clean_text_sha256": bytes_sha(row["text"].encode()),
                             "license": "http://creativecommons.org/licenses/by/4.0/",
                             "authoritative_license": {"verified": True, "license": "http://creativecommons.org/licenses/by/4.0/"}})
        filename = inputs / f"{split}.jsonl"
        filename.write_text("".join(json.dumps(row) + "\n" for row in rows))
        raw_hashes[split] = sha(filename)
    write(inputs / "corpus_provenance.json", {"papers": manifest, "local_path": str(root / "data/physics")})
    snapshot = root / ".hf/downloaded-private-base"
    snapshot.mkdir(parents=True)
    for name, content in {"LICENSE": b"Apache License, Version 2.0\n", "tokenizer.json": b"{}\n",
                          "tokenizer_config.json": b"{}\n", "config.json": b"{}\n",
                          "added_tokens.json": b"{}\n", "chat_template.jinja": b"template\n",
                          "generation_config.json": b"{}\n", "merges.txt": b"merges\n", "vocab.json": b"{}\n",
                          "special_tokens_map.json": b"{}\n", "model.safetensors.index.json": b"{}\n",
                          "model-00001-of-00002.safetensors": b"synthetic packed weights1",
                          "model-00002-of-00002.safetensors": b"synthetic packed weights2"}.items():
        (snapshot / name).write_bytes(content)
    base_files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in snapshot.iterdir()}
    publication = {"status": "published_and_verified", "private": True, "repo_id": BASE, "revision": "f" * 40,
                   "files": base_files,
                   "remote_load_check": {"repo_id": BASE, "revision": "f" * 40, "exact_last_token_logits": True,
                                         "exact_input_and_generated_ids": True, "requantization_forbidden": True,
                                         "max_abs_logit_difference": 0.0},
                   "download_check": {"repo_id": BASE, "revision": "f" * 40, "all_sha256_equal": True,
                                      "snapshot": str(snapshot)}}
    write(root / "results/nf4_huggingface_publication.json", publication)
    nltk_file = root / "artifacts/tools/nltk/tokenizers/punkt_tab/english/abbrev_types.txt"
    nltk_file.parent.mkdir(parents=True)
    nltk_file.write_text("synthetic sentence-tokenizer resource\n")
    write(root / "results/physics_nltk_resource.json",
          {"files": {"tokenizers/punkt_tab/english/abbrev_types.txt": sha(nltk_file)}})
    parent_files = {name: entry for name, entry in base_files.items() if name != "LICENSE"}
    parent_files["provenance.json"] = {"bytes": 22, "sha256": bytes_sha(b"unpublished provenance")}
    write(inputs / "nf4_report.json", {"completed": True, "model_id": cfg["model_id"], "revision": cfg["revision"],
                                       "checks": {"fixed_generated_ids_equal": True}, "files": parent_files,
                                       "total_bytes": sum(value["bytes"] for value in parent_files.values())})
    write(root / "results/physics_prl_corpus_quality.json", {"unicode_replacement_characters": 5,
                                                           "papers_with_replacement_characters": ["paper1v1", "paper2v1"],
                                                           "split_sha256": raw_hashes, "current_data_not_mutated": True,
                                                           "latex_limit": "No guarantee of compilable standalone LaTeX."})
    protocol = {"raw_corpus_sha256": raw_hashes, "source_sha256": source_hashes,
                "counts": {"train": {"loss_tokens_per_epoch": 522380, "chunks": 550}}}
    write(run / "protocol.json", protocol)
    frozen_files = [run / "config.json", run / "protocol.json"] + [p for p in inputs.rglob("*") if p.is_file()] + [p for p in (run / "source").rglob("*") if p.is_file()]
    write(run / "frozen.json", {"sha256": {str(p.relative_to(run)): sha(p) for p in frozen_files}})
    training = {"completed": True, "status": "complete", "protocol_sha256": sha(run / "protocol.json"),
                "frozen_manifest_sha256": sha(run / "frozen.json"), "global_step": 207,
                "exposure": {"actual_loss_tokens": 1567140},
                "replay": {"exact_adapter_tensors": True, "exact_logits": True, "exact_generated_ids": True},
                "execution_provenance": {"git_head": {"stdout": "e" * 40},
                                         "pip_freeze": {"returncode": 0, "stdout": "torch==2.6.0\npeft==0.18.1"}}}
    write(run / "run.json", training)
    selection_file = root / "results/rank_selection.json"
    selected = {"id": "candidate", "rank": 32, "learning_rate": 5e-5, "gate": {"passed": True, "checks": {}},
                "best_epoch": {"epoch": 2, "step": 138, "checkpoint": str(checkpoint)},
                "checkpoint_adapter_sha256": sha(checkpoint / "adapter_model.safetensors")}
    selection = {"stage": "rank", "split": "development", "candidates": [selected],
                 "decision": {"selected_id": "candidate", "selected_checkpoint": str(checkpoint), "selected_epoch": 2, "fallback_base": False}}
    write(selection_file, selection)
    lr_file = root / "results/physics_lr_selection.json"
    write(lr_file, {**selection, "stage": "lr_pilot"})
    final_file = root / "results/final.json"
    final = {"status": "computational_evaluation_complete_human_writing_review_pending", "writing_quality_improvement_established": False,
             "lr_selection_report_sha256": sha(lr_file),
             "nltk_resource_report_sha256": sha(root / "results/physics_nltk_resource.json"),
             "selection_report": str(selection_file), "selection_report_sha256": sha(selection_file),
             "limits": ["Human ratings pending; small subsets."], "conditions": {}}
    for label, adapter in (("nf4_base_fewshot", None), ("nf4_adapter", str(checkpoint))):
        directory = root / "outputs" / label
        directory.mkdir()
        write(directory / "run.json", {"completed": True, "adapter": adapter, "source_sha256": source_hashes})
        write(directory / "writing_predictions.json", [{"id": "writing1", "output": "A scientific title"}])
        final["conditions"][label] = {"representation": "nf4", "checkpoint": adapter, "evaluation_run": str(directory),
                                     "run_sha256": sha(directory / "run.json"), "prompt_condition": "few-shot" if adapter is None else "zero-shot",
                                     "summary": {"gsm8k": {"total": 128, "fallback_correct": 100 if adapter else 110},
                                                 "writing": {"total": 1, "nonempty": 1, "automatic_fidelity_flags": 0}},
                                     "raw_result_sha256": {"writing_predictions.json": sha(directory / "writing_predictions.json")}}
    benchmarks = root / "data/benchmark-screen-01"
    benchmarks.mkdir(parents=True)
    for split in ("development", "test"):
        write(benchmarks / f"{split}.json", {"fixture": split})
    write(benchmarks / "writing_test.json", {"source_split": "test", "tasks": [{"id": "writing1", "task": "title", "source": "Paper evidence."}]})
    write(benchmarks / "writing_provenance.json", {"files": {"writing_test.json": {"sha256": sha(benchmarks / "writing_test.json")}}})
    official = root / "artifacts/tools/ifeval/evaluation_lib.py"
    official.parent.mkdir(parents=True)
    official.write_text("# pinned official scorer\n")
    write(benchmarks / "provenance.json", {"fixture_sha256": {split: sha(benchmarks / f"{split}.json") for split in ("development", "test")},
                                          "ifeval_source_sha256": {"evaluation_lib.py": sha(official)}})
    for condition in final["conditions"].values():
        condition["fixture_sha256"] = sha(benchmarks / "test.json")
        condition["writing_sha256"] = sha(benchmarks / "writing_test.json")
    write(final_file, final)
    write(root / "results/physics_prl_protocol.json", {"design": "fixed unchanged",
                                                     "benchmark_provenance": json.loads((benchmarks / "provenance.json").read_text()),
                                                     "writing_provenance_sha256": sha(benchmarks / "writing_provenance.json")})
    write(root / "results/physics_prl_protocol_addendum_01.json", {"original_global_protocol_sha256": sha(root / "results/physics_prl_protocol.json"), "all_fail": "base recommended"})
    write(root / "outputs/benchmark_dataset_licenses.json", [{"repository": repo, "revision": "d" * 40, "reported_license": license}
                                                            for repo, license in (("cais/mmlu", ["mit"]), ("openai/gsm8k", ["mit"]), ("google/IFEval", "apache-2.0"))])
    args = SimpleNamespace(checkpoint=str(checkpoint), training_run=str(run), selection_report=str(selection_file),
                           final_report=str(final_file), adapter_condition="nf4_adapter", output="artifacts/releases/physics", descriptive_only=False)
    return SimpleNamespace(root=root, run=run, checkpoint=checkpoint, selection_file=selection_file, final_file=final_file,
                           args=args, publication=publication, tracked=tracked)


def stage_fake(project):
    git = SimpleNamespace(stdout=("\0".join(project.tracked) + "\0").encode())
    with patch("scripts.stage_physics_release.subprocess.run", return_value=git):
        return stage(project.args, root=project.root)


def test_private_payload_preserves_weights_data_and_only_changes_dependency_fields(project):
    before = {p: sha(p) for p in project.checkpoint.iterdir()}
    original = json.loads((project.checkpoint / "adapter_config.json").read_text())
    manifest = stage_fake(project)
    out = project.root / project.args.output
    exported = json.loads((out / "adapter_config.json").read_text())
    assert exported == {**original, "base_model_name_or_path": BASE, "revision": "f" * 40}
    assert exported["revision"] != "a" * 40
    assert all(sha(p) == value for p, value in before.items())
    assert sha(out / "adapter_model.safetensors") == before[project.checkpoint / "adapter_model.safetensors"]
    for split in ("train", "validation", "test"):
        assert (out / f"reproduction_data/{split}.jsonl").read_bytes() == (project.run / f"inputs/{split}.jsonl").read_bytes()
    attribution = json.loads((out / "paper_attributions.json").read_text())
    assert len(attribution["papers"]) == 100
    assert "config.json" not in manifest["files"] and "model.safetensors" not in manifest["files"]
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["budget"]["checkpoint_loss_tokens"] == 1044760
    assert provenance["budget"]["full_run_loss_tokens"] == 1567140
    assert provenance["budget"]["checkpoint_step"] == 138
    assert provenance["budget"]["full_run_steps"] == 207
    assert manifest["private"] is True
    assert json.loads((project.root / MANIFEST).read_text()) == manifest
    for name, expected in manifest["files"].items():
        assert sha(out / name) == expected["sha256"]
    packet = json.loads((out / "human_review/packet.json").read_text())
    assert packet["ratings"].startswith("pending") and packet["pairs"][0]["style_preference"] is None
    assert "nf4_adapter" not in (out / "human_review/packet.json").read_text()
    assert not (out / "human_review/answer_key.json").exists()
    key_file = project.root / "outputs/physics_release_review_key.json"
    assert provenance["human_review_key_sha256"] == sha(key_file)
    assert "lr_selection.json" in manifest["files"]
    with zipfile.ZipFile(out / "reproduction.zip") as archive:
        assert not any(name.startswith((".git/", "data/", ".env")) for name in archive.namelist())
        assert archive.read("scripts/physics_train.py") == (project.root / "scripts/physics_train.py").read_bytes()
        assert str(project.root).encode() not in archive.read("docs/reproduction.md")
        assert b"<PROJECT_ROOT>" in archive.read("docs/reproduction.md")
    assert "protocol_addendum_01.json" in manifest["files"]
    assert "benchmark_dataset_licenses.json" in manifest["files"]
    assert "corpus_quality.json" in manifest["files"]
    portable = json.loads((out / "reproduction_data/nf4_checkpoint_report.json").read_text())
    assert len(portable["files"]) == 12 and "provenance.json" not in portable["files"]
    assert portable["revision"] == "a" * 40
    assert portable["published_nf4_dependency"]["revision"] == "f" * 40
    assert portable["parent_report_sha256"] == sha(project.run / "inputs/nf4_report.json")
    card = (out / "README.md").read_text()
    assert "prepared 70/10/20" in card and "All 100" in card and "dropout 0.05" in card
    assert "5 replacement characters" in card and "not guaranteed" in card


def test_existing_output_refused_without_touching_checkpoint(project):
    output = project.root / project.args.output
    output.mkdir(parents=True)
    (output / "keep").write_text("existing")
    with pytest.raises(FileExistsError):
        stage_fake(project)
    assert (output / "keep").read_text() == "existing"


def test_no_final_report_or_changed_fixed_selection_cannot_stage(project):
    final = json.loads(project.final_file.read_text())
    final["selection_report_sha256"] = "0" * 64
    write(project.final_file, final)
    with pytest.raises(ValueError, match="exact fixed selection"):
        stage_fake(project)
    assert not (project.root / project.args.output).exists()
    project.final_file.rename(project.final_file.with_suffix(".not-ready"))
    with pytest.raises(FileNotFoundError):
        stage_fake(project)


def test_all_fail_requires_explicit_descriptive_flag_and_honest_card(project):
    selection = json.loads(project.selection_file.read_text())
    selection["decision"] = {"selected_id": None, "fallback_base": True}
    selection["candidates"][0]["gate"] = {"passed": False, "checks": {"math_drop": {"passed": False}}}
    write(project.selection_file, selection)
    final = json.loads(project.final_file.read_text())
    final["selection_report_sha256"] = sha(project.selection_file)
    write(project.final_file, final)
    with pytest.raises(ValueError, match="descriptive-only"):
        stage_fake(project)
    project.args.descriptive_only = True
    result = stage_fake(project)
    card = (project.root / project.args.output / "README.md").read_text()
    assert result["descriptive_only"] and "failed development selection" in card
    assert "base remains recommended" in card and "math_drop" in card
    assert "Human writing-quality improvement is not established" in card


def test_frozen_training_and_measured_scoring_source_drift_fail(project):
    source = project.root / "scripts/physics_train.py"
    original = source.read_bytes()
    source.write_bytes(original + b"# drift\n")
    with pytest.raises(ValueError, match="training source differs"):
        stage_fake(project)
    source.write_bytes(original)
    scoring = project.root / "scripts/benchmark_screen.py"
    scoring.write_text("# changed scoring module\n")
    with pytest.raises(ValueError, match="scoring source differs"):
        stage_fake(project)
    assert not (project.root / MANIFEST).exists()


def test_private_preflight_and_redaction_never_rewrite_scientific_text():
    root = Path("/workspace/example")
    assert redact_metadata({"path": str(root / "run")}, root) == {"path": "<PROJECT_ROOT>/run"}
    with pytest.raises(ValueError, match="credential field"):
        redact_metadata({"hf_token": "should not export"}, root)
    with pytest.raises(ValueError, match="credential-like"):
        private_preflight(b"hf_" + b"a" * 32, root)
    with pytest.raises(ValueError, match="scientific text/code must not be rewritten"):
        private_preflight(b"paper text includes /workspace/example/local", root)


def test_verified_private_published_base_must_pass_gpu_download_checks(project):
    verify_base(project.publication)
    value = copy.deepcopy(project.publication)
    value["private"] = False
    with pytest.raises(ValueError, match="private NF4 publication"):
        verify_base(value)
    value = copy.deepcopy(project.publication)
    value["remote_load_check"]["requantization_forbidden"] = False
    with pytest.raises(ValueError, match="GPU/hash verification"):
        verify_base(value)


def test_all100_attributions_require_verified_ccby_and_clean_text_hash(project):
    filename = project.run / "inputs/corpus_provenance.json"
    provenance = json.loads(filename.read_text())
    provenance["papers"][0]["authoritative_license"]["verified"] = False
    write(filename, provenance)
    protocol = json.loads((project.run / "protocol.json").read_text())
    with pytest.raises(ValueError, match="attribution verification failed"):
        corpus_attributions(project.run, protocol, project.root)
    provenance["papers"][0]["authoritative_license"]["verified"] = True
    provenance["papers"][0]["clean_text_sha256"] = "0" * 64
    write(filename, provenance)
    with pytest.raises(ValueError, match="attribution verification failed"):
        corpus_attributions(project.run, protocol, project.root)


def test_prepared_data_with_local_paths_is_rejected_without_rewriting(project):
    filename = project.run / "inputs/train.jsonl"
    rows = [json.loads(line) for line in filename.read_text().split("\n") if line]
    rows[0]["text"] = f"Scientific text with local path {project.root}/private"
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    filename.write_bytes(raw)
    protocol = json.loads((project.run / "protocol.json").read_text())
    protocol["raw_corpus_sha256"]["train"] = sha(filename)
    with pytest.raises(ValueError, match="scientific text/code must not be rewritten"):
        corpus_attributions(project.run, protocol, project.root)
    assert filename.read_bytes() == raw


def test_global_protocol_freezes_fixture_and_scorer_provenance(project):
    filename = project.root / "data/benchmark-screen-01/provenance.json"
    value = json.loads(filename.read_text())
    value["fixture_sha256"]["test"] = "0" * 64
    write(filename, value)
    with pytest.raises(ValueError, match="frozen global protocol"):
        stage_fake(project)
    assert not (project.root / project.args.output).exists()


def test_portable_report_passes_actual_verify_nf4_without_raw_provenance(project):
    from scripts.physics_train import verify_nf4
    snapshot = Path(project.publication["download_check"]["snapshot"])
    original_file = project.run / "inputs/nf4_report.json"
    original_hash = sha(original_file)
    assert not (snapshot / "provenance.json").exists()
    portable = portable_nf4_report(original_file, project.publication, snapshot)
    filename = project.root / "portable_nf4_report.json"
    write(filename, portable)
    actual = verify_nf4({"nf4_report": str(filename), "nf4_checkpoint": str(snapshot), "revision": "a" * 40})
    assert actual["completed"] is True
    assert len(actual["files"]) == 12
    assert actual["total_bytes"] == sum(entry["bytes"] for entry in actual["files"].values())
    assert actual["portability"]["omitted_original_files"] == ["provenance.json"]
    assert actual["checks"] == json.loads(original_file.read_text())["checks"]
    assert sha(original_file) == original_hash
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"tampered packed weights")
    with pytest.raises(ValueError, match="standalone NF4 checkpoint changed"):
        verify_nf4({"nf4_report": str(filename), "nf4_checkpoint": str(snapshot), "revision": "a" * 40})
    with pytest.raises(ValueError, match="published computational bytes changed"):
        portable_nf4_report(original_file, project.publication, snapshot)
