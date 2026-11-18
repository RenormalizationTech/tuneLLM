import copy
import json
from pathlib import Path
import random

import pytest

from scripts import qwen38_prl_claim_audit as checker, tone_judge
from scripts.qwen38_apply_prl_audit import AuditIncomplete, apply_audit, canonical, read_json, sha_bytes, sha_text

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(value) + "\n")


@pytest.fixture
def audit_fixture(tmp_path):
    prepared, audit = tmp_path / "prepared", tmp_path / "audit"
    prepared.mkdir(); audit.mkdir()
    splits, hashes = {}, {}
    for split, count in (("train", 70), ("validation", 10), ("test", 20)):
        rows = [{"id": f"{split}-{i:02d}", "paper_id": f"paper-{split}-{i:02d}", "split": split,
                 "selected_body": "Evidence supports this result.", "assistant": "This result is supported.",
                 "user": "Write an abstract from evidence.", "training_eligible": False,
                 "training_eligibility_reason": "audit pending", "historical_test_diagnostic": split == "test",
                 "prompt_tokens": 10, "target_tokens_with_eos": 6, "sequence_tokens": 16,
                 "encoded_row_sha256": sha_text(f"{split}-{i}")} for i in range(count)]
        splits[split] = rows
        path = prepared / (split + ".jsonl")
        path.write_text("".join(canonical(r) + "\n" for r in rows))
        hashes[path.name] = sha_bytes(path.read_bytes())
    (prepared / "source_manifest.json").write_text('{"original_sources":"synthetic"}\n')
    hashes["source_manifest.json"] = sha_bytes((prepared / "source_manifest.json").read_bytes())
    manifest = {"task": "prl", "version": "synthetic-preparation", "output_sha256": hashes,
        "counts": {s: {"records": len(rows), "training_eligible": 0} for s, rows in splits.items()},
        "original_split_paper_ids": {s: [r["paper_id"] for r in rows] for s, rows in splits.items()},
        "eligible_training_order": {"seed": 42}, "eligible_development_order": {"seed": 42}}
    write(prepared / "manifest.json", manifest)
    papers = []
    for split in ("train", "validation"):
        for row in splits[split]:
            papers.append({"id": row["id"], "paper_id": row["paper_id"], "split": split,
                "body_excerpt": row["selected_body"], "abstract": row["assistant"],
                "abstract_sentences": [{"index": 0, "text": row["assistant"]}],
                "excerpt_sha256": sha_text(row["selected_body"]), "abstract_sha256": sha_text(row["assistant"]),
                "target_words": len(row["assistant"].split()), "body_nonempty": True})
    write(audit / "papers.json", papers)
    write(audit / "rubric.json", checker.RUBRIC)
    plans = []
    for seed in (20261117, 20261118):
        order = [p["id"] for p in papers]
        random.Random(seed).shuffle(order)
        plans.append({"seed": seed, "batches": [order[i:i + 10] for i in range(0, 80, 10)]})
    write(audit / "plans.json", plans)
    source_hashes = {}
    for name in ("qwen38_prl_claim_audit.py", "tone_judge.py", "qwen38_judge_runtime.py"):
        data = (ROOT / "scripts" / name).read_bytes()
        for destination in (audit / "source" / name, tmp_path / "scripts" / name):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        source_hashes[name] = sha_bytes(data)
    runtime = tmp_path / "outputs/qwen38-judge-runtime-01"
    write(runtime / "audit.json", {"passed": True})
    write(runtime / "binding.json", {"requested_model": tone_judge.MODEL})
    protocol = {"prepared": "prepared", "prepared_manifest_sha256": sha_bytes((prepared / "manifest.json").read_bytes()),
        "papers_sha256": sha_bytes((audit / "papers.json").read_bytes()), "rubric_sha256": sha_bytes((audit / "rubric.json").read_bytes()),
        "plans_sha256": sha_bytes((audit / "plans.json").read_bytes()), "source_sha256": source_hashes,
        "runtime_binding": {"requested_model": tone_judge.MODEL}, "isolation_audit_sha256": sha_bytes((runtime / "audit.json").read_bytes()),
        "maximum_calls": 16, "attempts_per_batch": 1, "eligibility": "both registered passes"}
    write(audit / "registration.json", protocol)
    write(tmp_path / "results/protocols/audit.json", protocol)
    (audit / "execution.lock").write_bytes(b"")
    lookup = {r["id"]: r for r in papers}
    for pass_index, plan in enumerate(plans):
        for batch_index, batch in enumerate(plan["batches"]):
            directory = audit / f"pass-{pass_index+1}/batch-{batch_index:02d}"
            payload = {"rubric": checker.RUBRIC, "expected_ids": batch,
                "papers": [{k: lookup[i][k] for k in ("id", "body_excerpt", "abstract_sentences")} for i in batch]}
            submitted = []
            for key in batch:
                rejected = pass_index == 1 and key in {"train-00", "validation-00"}
                submitted.append({"id": key, "sentences": [{"index": 0, "label": "UNSUPPORTED" if rejected else "SUPPORTED",
                    "quotes": [] if rejected else [lookup[key]["body_excerpt"]], "reason": "Synthetic audit fixture."}]})
            value = {"papers": submitted}
            accepted = checker.checked_submission(value, batch, lookup)
            outer = {"subtype": "success", "is_error": False, "modelUsage": {tone_judge.MODEL: {}}, "result": json.dumps(value)}
            attempt = {"returncode": 0, "stdout": json.dumps(outer), "stderr": "", "call_started": True,
                "canonical_stdin_sha256": tone_judge.canonical_hash(payload),
                "isolation": {"git_not_repository": True, "git_environment_clean": True, "returncode": 128}}
            write(directory / "payload.json", payload)
            write(directory / "call_started.json", {"started": True, "payload_sha256": tone_judge.canonical_hash(payload)})
            write(directory / "attempt.json", attempt)
            write(directory / "checked.json", {"ratings": accepted, "model_ids": [tone_judge.MODEL]})
    records = [{"id": p["id"], "split": p["split"], "eligible": p["id"] not in {"train-00", "validation-00"},
                "pass_eligible": [True, p["id"] not in {"train-00", "validation-00"}],
                "excerpt_sha256": p["excerpt_sha256"], "abstract_sha256": p["abstract_sha256"]} for p in papers]
    write(audit / "result.json", {"completed": True, "AI_audit_only": True, "eligible": {"train": 69, "validation": 9},
        "records": records, "source_registration_sha256": sha_bytes((audit / "registration.json").read_bytes()),
        "data_limited": False, "agreement": .975, "cohens_kappa": 0.0})
    return tmp_path, prepared, audit


def apply_fixture(fixture, output="applied"):
    root, prepared, audit = fixture
    return apply_audit(prepared, audit, root / output, root)


def test_application_retains_all_rows_exact_prompts_targets_and_old_test(audit_fixture):
    root, prepared, audit = audit_fixture
    before = {p: sha_bytes(p.read_bytes()) for directory in (prepared, audit) for p in directory.rglob("*") if p.is_file()}
    report = apply_fixture(audit_fixture)
    assert report["retained"] == {"train": 70, "validation": 10, "test": 20}
    assert report["eligible"] == {"train": 69, "validation": 9} and not report["GPU_started"]
    assert all(sha_bytes(p.read_bytes()) == h for p, h in before.items())
    out = root / "applied"
    assert (out / "test.jsonl").read_bytes() == (prepared / "test.jsonl").read_bytes()
    for split, expected in (("train", 69), ("validation", 9)):
        original = [json.loads(s) for s in (prepared / (split + ".jsonl")).read_text().splitlines()]
        updated = [json.loads(s) for s in (out / (split + ".jsonl")).read_text().splitlines()]
        assert len(original) == len(updated) and sum(r["training_eligible"] for r in updated) == expected
        for old, new in zip(original, updated):
            changed = {k for k in set(old) | set(new) if old.get(k) != new.get(k)}
            assert changed <= {"training_eligible", "training_eligibility_reason", "claim_audit"}
            assert new["claim_audit"]["eligible"] == all(new["claim_audit"]["pass_eligible"])
    manifest = read_json(out / "manifest.json")
    assert manifest["eligible_training_order"]["prefix_preview_count"] == 69
    assert manifest["eligible_development_order"]["prefix_preview_count"] == 9
    assert manifest["counts"]["test"]["training_eligible"] == 0
    for name, expected in manifest["output_sha256"].items():
        assert sha_bytes((out / name).read_bytes()) == expected
    with pytest.raises(FileExistsError):
        apply_fixture(audit_fixture)


@pytest.mark.parametrize("case", ["missing_result", "incomplete"])
def test_incomplete_audit_waits_without_creating_output(audit_fixture, case):
    root, _, audit = audit_fixture
    if case == "missing_result":
        (audit / "result.json").unlink()
    else:
        result = read_json(audit / "result.json"); result["completed"] = False; write(audit / "result.json", result)
    with pytest.raises(AuditIncomplete):
        apply_fixture(audit_fixture)
    assert not (root / "applied").exists()


@pytest.mark.parametrize("case", ["duplicate", "missing", "split", "excerpt", "abstract", "one_pass", "counts", "registration_hash", "typed_flag"])
def test_result_tamper_missing_and_duplicate_rejected(audit_fixture, case):
    root, _, audit = audit_fixture
    result = read_json(audit / "result.json")
    if case == "duplicate": result["records"][-1] = copy.deepcopy(result["records"][0])
    elif case == "missing": result["records"].pop()
    elif case == "split": result["records"][0]["split"] = "test"
    elif case == "excerpt": result["records"][0]["excerpt_sha256"] = "0" * 64
    elif case == "abstract": result["records"][0]["abstract_sha256"] = "0" * 64
    elif case == "one_pass": result["records"][0]["eligible"] = True
    elif case == "counts": result["eligible"]["train"] = 70
    elif case == "registration_hash": result["source_registration_sha256"] = "0" * 64
    elif case == "typed_flag": result["records"][0]["pass_eligible"][0] = 1
    write(audit / "result.json", result)
    with pytest.raises(ValueError):
        apply_fixture(audit_fixture)
    assert not (root / "applied").exists()


@pytest.mark.parametrize("case", ["prepared_file", "prepared_manifest", "registry", "payload", "checked", "missing_checked", "raw_attempt", "extra_batch", "isolation"])
def test_all_registered_inputs_and_saved_pass_outputs_checked(audit_fixture, case):
    root, prepared, audit = audit_fixture
    directory = audit / "pass-1/batch-00"
    if case == "prepared_file": (prepared / "source_manifest.json").write_text("tampered")
    elif case == "prepared_manifest":
        m = read_json(prepared / "manifest.json"); m["task"] = "qasper"; write(prepared / "manifest.json", m)
    elif case == "registry": (root / "results/protocols/audit.json").write_text("{}")
    elif case == "payload":
        p = read_json(directory / "payload.json"); p["papers"][0]["body_excerpt"] += " tampered"; write(directory / "payload.json", p)
    elif case == "checked":
        c = read_json(directory / "checked.json"); next(iter(c["ratings"].values()))["eligible"] = False; write(directory / "checked.json", c)
    elif case == "missing_checked": (directory / "checked.json").unlink()
    elif case == "raw_attempt":
        a = read_json(directory / "attempt.json"); a["canonical_stdin_sha256"] = "0" * 64; write(directory / "attempt.json", a)
    elif case == "extra_batch": (audit / "pass-3/batch-00").mkdir(parents=True)
    elif case == "isolation": write(root / "outputs/qwen38-judge-runtime-01/audit.json", {"passed": False})
    with pytest.raises((ValueError, FileNotFoundError)):
        apply_fixture(audit_fixture)
    assert not (root / "applied").exists()


def test_completed_local_application_matches_audit_and_trainer_pool():
    """Optional read-only integration against the completed local audit/artifact."""
    prepared = ROOT / "outputs/qwen38-science-01/prepared-prl-tokenizer-04"
    output = ROOT / "outputs/qwen38-science-01/prepared-prl-audited-01"
    audit = ROOT / "outputs/qwen38-prl-claim-audit-01"
    if not (output / "manifest.json").exists() or not (audit / "result.json").exists():
        pytest.skip("completed local audit application is not available")
    from scripts.qwen38_train_science import paper_subset
    manifest, result = read_json(output / "manifest.json"), read_json(audit / "result.json")
    eligible = {r["id"]: r for r in result["records"]}
    assert result["completed"] and manifest["task"] == "prl"
    assert manifest["audit_application"]["audit_result_sha256"] == sha_bytes((audit / "result.json").read_bytes())
    for name, expected in manifest["output_sha256"].items():
        assert sha_bytes((output / name).read_bytes()) == expected
    for split, count in (("train", 70), ("validation", 10), ("test", 20)):
        original = [json.loads(s) for s in (prepared / (split + ".jsonl")).read_text().splitlines()]
        rows = [json.loads(s) for s in (output / (split + ".jsonl")).read_text().splitlines()]
        assert len(rows) == count and [r["id"] for r in rows] == [r["id"] for r in original]
        assert all((a["user"], a["assistant"], a["selected_body"], a["encoded_row_sha256"]) ==
                   (b["user"], b["assistant"], b["selected_body"], b["encoded_row_sha256"]) for a, b in zip(original, rows))
        if split != "test":
            accepted = [r for r in rows if r["training_eligible"]]
            assert len(accepted) == result["eligible"][split] == manifest["counts"][split]["training_eligible"]
            assert all(r["training_eligible"] == eligible[r["id"]]["eligible"] for r in rows)
            assert [r["id"] for r in paper_subset(accepted, len(accepted), 42)] == manifest[
                "eligible_training_order" if split == "train" else "eligible_development_order"]["prefix_preview_ids"]
        else:
            assert (output / "test.jsonl").read_bytes() == (prepared / "test.jsonl").read_bytes()
            assert all(not r["training_eligible"] for r in rows)
