"""Pure CPU preparation tests, no network/models/generated responses/review calls."""
import copy
import json
from types import SimpleNamespace

import pytest

from scripts.prepare_refusal import (SEED, candidate_pool, exclusions, pair_approved, row,
                                     validate_screen, xstest_split)
from scripts.common import read_json
from scripts import prepare_refusal as prepare


def test_normalized_dedup_all_xstest_exclusion_and_both_cross_classes_removed():
    evaluation = [{"prompt": "KILL a process"}]
    harmful = [row("advbench", 0, "kill  a process"), row("advbench", 1, "shared"), row("advbench", 2, "unique")]
    benign = [row("dolly", 0, "ＫＩＬＬ a process"), row("dolly", 1, " SHARED "),
              row("dolly", 2, "benign"), row("dolly", 3, "BENIGN ")]
    a,b,losses = exclusions(evaluation, harmful, benign)
    assert [r["instruction"] for r in a] == ["unique"]
    assert [r["instruction"] for r in b] == ["benign"]
    assert losses["dolly"]["within_source_duplicate_rows_removed"] == 1
    assert losses["advbench"]["all_xstest_overlap_rows_removed"] == 1
    assert losses["dolly"]["cross_class_ambiguous_rows_removed"] == 1


def test_candidate_draw_is_200_fixed_seed_and_has_no_dangerous_word_filter():
    harmful = [row("advbench", 0, "raw min"), row("advbench", 1, "raw maximum")]
    benign = [row("dolly", i, f"kill test{i}") for i in range(250)] + [row("dolly", 251, "outside")]
    def length(text):
        return 10 if text == "raw min" else 20 if text == "raw maximum" else 100 if text == "outside" else 12
    first = candidate_pool([], harmful, benign, length)
    second = candidate_pool([], harmful, benign, length)
    assert first == second and len(first[0]) == 200
    assert all(r["instruction"].startswith("kill") for r in first[0])
    assert first[2]["dolly_integer_min"] == 8 and first[2]["dolly_integer_max"] == 24
    assert first[3]["dolly"]["outside_length_envelope_removed"] == 1
    with pytest.raises(ValueError, match="fewer than 200"):
        candidate_pool([], harmful, benign[:199], length)


def test_pairing_closest_unused_approved_plusminus20_and_frozen_order_ties():
    harmful = [{"id": "a0", "raw_token_length": 10}, {"id": "a1", "raw_token_length": 10},
               {"id": "a2", "raw_token_length": 20}, {"id": "a3", "raw_token_length": 50}]
    benign = [{"id": "b0", "raw_token_length": 11}, {"id": "b1", "raw_token_length": 9},
              {"id": "b2", "raw_token_length": 20}, {"id": "b3", "raw_token_length": 16}]
    labels = {r["id"]: {"label": "unsafe" if r["id"] == "b2" else "harmless"} for r in benign}
    pairs, report = pair_approved(harmful, benign, labels)
    assert [p["benign"]["id"] for p in pairs] == ["b0", "b1", "b3"]
    assert pairs[2]["benign_harmful_raw_length_ratio"] == .8
    assert report["actual_pairs"] == 3 and report["pair_count_is_not_a_yield_gate"]
    pairs, report = pair_approved(harmful, benign, {r["id"]:{"label":"ambiguous"} for r in benign})
    assert pairs == [] and report["actual_pairs"] == 0


def test_labels_complete_hash_bound_reported_identity_and_semantic_rubric():
    rubric = read_json("evaluation/refusal/screening_rubric.json")
    assert "semantic intent" in rubric["instruction"] and "kill or shoot" in rubric["instruction"]
    pool = [{"id": "candidate0"}, {"id": "candidate1"}]
    submission = {"candidate_file_sha256": "pool-sha", "rubric_sha256": "rubric-sha",
        "reviewer": {"model_identifier": "claude-opus-5-5", "advisor_overlap": True,
                     "exposed_settings": {"reasoning_effort": None}},
        "labels": [{"id": r["id"], "label": "harmless", "reason_code": "ordinary_benign_request"} for r in pool]}
    assert len(validate_screen(pool, rubric, submission, "pool-sha", "rubric-sha")) == 2
    for changed in ({"labels": submission["labels"][:-1]}, {"candidate_file_sha256": "different"},
                    {"reviewer": {**submission["reviewer"], "advisor_overlap": False}}):
        with pytest.raises(ValueError):
            validate_screen(pool, rubric, {**submission, **changed}, "pool-sha", "rubric-sha")
    bad = copy.deepcopy(submission)
    bad["labels"][0]["reason_code"] = "keyword_blacklist"
    with pytest.raises(ValueError):
        validate_screen(pool, rubric, bad, "pool-sha", "rubric-sha")


def fake_xstest():
    safe = ("definitions", "figurative_language", "historical_events", "homonyms", "safe_contexts", "safe_targets",
            "privacy_public", "privacy_fictional", "nons_group_real_discr", "real_group_nons_discr")
    unsafe = tuple("contrast_"+t for t in safe[:6]) + ("contrast_privacy", "contrast_discr")
    return [{"id": f"{label}-{kind}-{i}", "type": kind, "label": label, "prompt": f"unit-test input {kind} {i}"}
            for label,types in (("safe",safe),("unsafe",unsafe)) for kind in types for i in range(25)]


def test_xstest_exact_type_disjoint_counts_and_shared_pools_final():
    split, report = xstest_split(fake_xstest())
    assert report["seed"] == SEED and report["counts"] == {"development":{"safe":30,"unsafe":30},"final":{"safe":42,"unsafe":30}}
    assert len(report["final_broad_families"]) == 5
    devtypes = {r["type"] for r in split["development"]}
    finaltypes = {r["type"] for r in split["final"]}
    assert not devtypes & finaltypes
    assert {"contrast_privacy","contrast_discr","privacy_public","privacy_fictional","nons_group_real_discr","real_group_nons_discr"} <= finaltypes
    assert not {r["id"] for r in split["development"]} & {r["id"] for r in split["final"]}
    assert xstest_split(fake_xstest()) == (split, report)


@pytest.fixture
def frozen_candidates(tmp_path):
    import csv
    directory = tmp_path / "candidates"
    directory.mkdir()
    benign = [{**row("dolly", i, f"Unit-test candidate {i}"), "raw_token_length": 10} for i in range(2)]
    harmful = [{**row("advbench", i, f"Unit-test class metadata {i}"), "raw_token_length": 10} for i in range(2)]
    prepare.save(directory/"candidates.json", {"items": benign})
    prepare.save(directory/"advbench_order.json", {"items": harmful})
    prepare.save(directory/"inputs/screening_rubric.json", read_json("evaluation/refusal/screening_rubric.json"))
    with (directory/"inputs/xstest_prompts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id","type","label","prompt"))
        writer.writeheader()
        writer.writerows(fake_xstest())
    manifest = {"files": {str(p.relative_to(directory)):prepare.stream_hash(p) for p in directory.rglob("*") if p.is_file()},
                "source_sha256": {n:prepare.stream_hash(prepare.path(n)) for n in prepare.SOURCES},
                "tokenizer_sha256": {"unit-test": "metadata-only"}}
    prepare.save(directory/"manifest.json", manifest)
    registration = {**manifest, "manifest_sha256": prepare.stream_hash(directory/"manifest.json"),
                    "candidate_ids_hashes": [{k:r[k] for k in ("id","prompt_sha256","normalized_sha256","raw_token_length")} for r in benign]}
    regfile = tmp_path/"registration.json"
    prepare.save(regfile, registration)
    submission = {"candidate_file_sha256": prepare.stream_hash(directory/"candidates.json"),
        "rubric_sha256": prepare.stream_hash(directory/"inputs/screening_rubric.json"),
        "reviewer": {"model_identifier":"claude-opus-5-5","advisor_overlap":True,"exposed_settings":{}},
        "labels": [{"id":r["id"],"label":"harmless","reason_code":"ordinary_benign_request"} for r in benign]}
    labelsfile = tmp_path/"labels.json"
    prepare.save(labelsfile, submission)
    return directory, regfile, labelsfile


def test_frozen_input_mutation_and_registration_order_drift_rejected(frozen_candidates):
    directory, regfile, _ = frozen_candidates
    registration = read_json(regfile)
    prepare.verify_candidates(directory, registration)
    bad = copy.deepcopy(registration)
    bad["candidate_ids_hashes"].reverse()
    with pytest.raises(ValueError, match="IDs/order"):
        prepare.verify_candidates(directory, bad)
    (directory/"advbench_order.json").write_text("changed after registration")
    with pytest.raises(ValueError, match="frozen candidate"):
        prepare.verify_candidates(directory, registration)


def test_final_prepare_uses_bound_complete_labels_and_reports_small_actual_count(tmp_path, monkeypatch, frozen_candidates):
    directory, regfile, labelsfile = frozen_candidates
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0,stdout=regfile.read_bytes()))
    output, report = tmp_path/"prepared", tmp_path/"report.json"
    args = SimpleNamespace(candidates=str(directory),registration=str(regfile),output=str(output),labels=str(labelsfile),final_report=str(report))
    prepare.final_prepare(args)
    result = read_json(report)
    assert result["pairing"]["actual_pairs"] == 2 and result["no_gpu_run_registered"]
    assert len(read_json(output/"development.json")["items"]) == 60
    assert len(read_json(output/"final.json")["items"]) == 72
    with pytest.raises(FileExistsError):
        prepare.final_prepare(args)


def test_uncommitted_candidate_registration_blocks_final_preparation(tmp_path, monkeypatch, frozen_candidates):
    directory, regfile, labelsfile = frozen_candidates
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0,stdout=b"different committed registration"))
    args = SimpleNamespace(candidates=str(directory),registration=str(regfile),output=str(tmp_path/"prepared"),labels=str(labelsfile),final_report=str(tmp_path/"report.json"))
    with pytest.raises(ValueError, match="must be committed"):
        prepare.final_prepare(args)
    assert not (tmp_path/"prepared").exists()
