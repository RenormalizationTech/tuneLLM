"""Apply a completed, registered two-pass AI claim audit to immutable PRL inputs.

CPU/stdlib only; no tokenization, model loading, judging, downloads or training.
Original train70/dev10/test20 rows and all prompts/targets remain intact. Only
train/dev eligibility and its provenance change. Test JSONL is byte-identical.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import fcntl
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"train": 70, "validation": 10, "test": 20}
VERSION = "qwen38-apply-prl-audit-v1"


class AuditIncomplete(RuntimeError):
    """A completed result is required; no output is created while waiting."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_text(text):
    return sha_bytes(text.encode("utf-8"))


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def safe_file(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("manifest path must stay inside its directory")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()) or not result.is_file():
        raise ValueError("manifest file missing or escapes its directory")
    return result


def unique_rows(rows, label):
    if not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get("id"), str) or not r["id"] for r in rows):
        raise ValueError(f"{label}: nonempty string IDs required")
    result = {r["id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"{label}: duplicate IDs")
    return result


def verified_prepared(prepared, expected_manifest):
    manifest_bytes = (prepared / "manifest.json").read_bytes()
    if sha_bytes(manifest_bytes) != expected_manifest:
        raise ValueError("registered prepared input manifest hash differs")
    manifest = read_json(prepared / "manifest.json")
    if manifest.get("task") != "prl":
        raise ValueError("fixed PRL task required")
    output_hashes = manifest.get("output_sha256")
    if not isinstance(output_hashes, dict) or not {s + ".jsonl" for s in COUNTS} <= set(output_hashes):
        raise ValueError("all original prepared split hashes required")
    payloads = {}
    for name, expected in output_hashes.items():
        data = safe_file(prepared, name).read_bytes()
        if sha_bytes(data) != expected:
            raise ValueError(f"prepared output hash differs: {name}")
        payloads[name] = data
    splits, all_rows = {}, []
    for split, count in COUNTS.items():
        rows = [json.loads(line) for line in payloads[split + ".jsonl"].decode().split("\n") if line.strip()]
        unique_rows(rows, split)
        if len(rows) != count or manifest["counts"][split]["records"] != count:
            raise ValueError("original70/10/20 split counts required")
        if any(r.get("split") != split or r.get("training_eligible") is not False for r in rows):
            raise ValueError("original split tags and unaudited false eligibility required")
        if any(not isinstance(r.get("paper_id"), str) or not r["paper_id"] for r in rows):
            raise ValueError("paper IDs required")
        if [r["paper_id"] for r in rows] != manifest["original_split_paper_ids"][split]:
            raise ValueError("original paper split/order differs")
        splits[split] = rows
        all_rows.extend(rows)
    unique_rows(all_rows, "all prepared splits")
    base_ids = [re.sub(r"v\d+$", "", r["paper_id"]) for r in all_rows]
    if len(set(base_ids)) != len(base_ids):
        raise ValueError("paper/version overlap within or across splits")
    return manifest, manifest_bytes, payloads, splits


def verified_audit(audit, prepared, repo_root):
    """Replay saved responses with the pinned pure audit checker, never invoke AI."""
    result_file = audit / "result.json"
    if not result_file.exists():
        raise AuditIncomplete("claim audit has no completed result yet")
    result = read_json(result_file)
    if result.get("completed") is not True:
        raise AuditIncomplete("claim audit result is not complete")
    protocol_file = audit / "registration.json"
    protocol = read_json(protocol_file)
    registry = repo_root / "results/protocols" / (audit.name + ".json")
    if registry.read_bytes() != protocol_file.read_bytes():
        raise ValueError("audit registration differs from reviewed registry")
    protocol_hash = sha_bytes(protocol_file.read_bytes())
    if result.get("source_registration_sha256") != protocol_hash or result.get("AI_audit_only") is not True:
        raise ValueError("result registration binding/AI-audit disclosure differs")
    if (repo_root / protocol["prepared"]).resolve() != prepared:
        raise ValueError("audit is bound to different prepared inputs")
    manifest, original_manifest, payloads, splits = verified_prepared(prepared, protocol["prepared_manifest_sha256"])
    audited = splits["train"] + splits["validation"]
    lookup = unique_rows(audited, "audited source")
    evidence = {"registration.json": protocol_file.read_bytes(), "result.json": result_file.read_bytes(),
                "reviewed_registration.json": registry.read_bytes(), "prepared_manifest.json": original_manifest}
    for name, field in [("papers.json", "papers_sha256"), ("rubric.json", "rubric_sha256"), ("plans.json", "plans_sha256")]:
        data = (audit / name).read_bytes()
        if sha_bytes(data) != protocol[field]:
            raise ValueError(f"audit registered input hash differs: {name}")
        evidence[name] = data
    for name, expected in protocol["source_sha256"].items():
        data = safe_file(audit / "source", name).read_bytes()
        if sha_bytes(data) != expected or sha_bytes(safe_file(repo_root / "scripts", name).read_bytes()) != expected:
            raise ValueError("pinned audit checker/runtime source differs")
        evidence["source/" + name] = data
    # Importing these modules does not invoke the CLI; only pure saved-response
    # parsers/checkers below are called. Their exact sources are verified above.
    helper = importlib.import_module("scripts.qwen38_prl_claim_audit")
    tone = importlib.import_module("scripts.tone_judge")
    for module in (helper, tone):
        if sha_bytes(Path(module.__file__).read_bytes()) != protocol["source_sha256"][Path(module.__file__).name]:
            raise ValueError("imported audit checker differs from pinned source")
    runtime = repo_root / "outputs/qwen38-judge-runtime-01"
    isolation = (runtime / "audit.json").read_bytes()
    if sha_bytes(isolation) != protocol["isolation_audit_sha256"] or read_json(runtime / "audit.json").get("passed") is not True:
        raise ValueError("registered judge isolation audit differs or failed")
    if read_json(runtime / "binding.json") != protocol["runtime_binding"]:
        raise ValueError("stored judge runtime binding differs")
    evidence["judge_isolation_audit.json"] = isolation
    evidence["judge_runtime_binding.json"] = (runtime / "binding.json").read_bytes()
    papers = read_json(audit / "papers.json")
    paper_lookup = unique_rows(papers, "audit papers")
    records = unique_rows(result.get("records"), "audit result")
    if set(paper_lookup) != set(lookup) or set(records) != set(lookup):
        raise ValueError("complete exact train/dev audit IDs required; no historical test records")
    for key, original in lookup.items():
        paper, record = paper_lookup[key], records[key]
        if paper.get("split") != original["split"] or record.get("split") != original["split"] or paper.get("paper_id") != original["paper_id"]:
            raise ValueError("audit paper/result split or paper ID differs")
        excerpt, abstract = original["selected_body"], original["assistant"]
        if paper.get("body_excerpt") != excerpt or paper.get("abstract") != abstract:
            raise ValueError("audited excerpt/abstract differs from frozen row")
        for field, expected in [("excerpt_sha256", sha_text(excerpt)), ("abstract_sha256", sha_text(abstract))]:
            if paper.get(field) != expected or record.get(field) != expected:
                raise ValueError("per-row excerpt/abstract hash differs")
        units = [{"index": i, "text": text} for i, text in enumerate(helper.sentences(abstract))]
        if (not units or paper.get("abstract_sentences") != units or type(paper.get("target_words")) is not int
                or paper["target_words"] != len(abstract.split()) or paper.get("body_nonempty") is not bool(excerpt.strip())):
            raise ValueError("audit sentence/word/excerpt metadata differs")
        if (type(record.get("eligible")) is not bool or not isinstance(record.get("pass_eligible"), list)
                or len(record["pass_eligible"]) != 2 or any(type(v) is not bool for v in record["pass_eligible"])
                or record["eligible"] != all(record["pass_eligible"])):
            raise ValueError("strict boolean both-pass eligibility required")
    plans = read_json(audit / "plans.json")
    if (not isinstance(plans, list) or len(plans) != 2 or protocol.get("attempts_per_batch") != 1
            or protocol.get("maximum_calls") != 16 or read_json(audit / "rubric.json") != helper.RUBRIC):
        raise ValueError("registered two-pass/16-call audit rubric required")
    expected_dirs, passes = set(), []
    for pass_index, plan in enumerate(plans):
        batches = plan["batches"]
        ids = [i for batch in batches for i in batch]
        if len(batches) != 8 or any(len(b) != 10 for b in batches) or len(ids) != len(set(ids)) or set(ids) != set(lookup):
            raise ValueError("each pass must cover all80 unique papers in registered batches")
        ratings = {}
        for batch_index, batch in enumerate(batches):
            relative = f"pass-{pass_index + 1}/batch-{batch_index:02d}"
            expected_dirs.add(relative)
            directory = audit / relative
            payload = {"rubric": helper.RUBRIC, "expected_ids": batch,
                "papers": [{k: paper_lookup[key][k] for k in ("id", "body_excerpt", "abstract_sentences")} for key in batch]}
            if read_json(directory / "payload.json") != payload:
                raise ValueError("audit payload differs from registered paper batch")
            start = read_json(directory / "call_started.json")
            attempt = read_json(directory / "attempt.json")
            expected_stdin = tone.canonical_hash(payload)
            if start.get("started") is not True or start.get("payload_sha256") != expected_stdin or attempt.get("canonical_stdin_sha256") != expected_stdin or attempt.get("call_started") is not True:
                raise ValueError("audit call/attempt is not bound to its exact payload")
            outer, models = tone.outer_result(attempt)
            checked = helper.checked_submission(tone.result_json(outer.get("result")), batch, paper_lookup)
            saved = read_json(directory / "checked.json")
            if saved != {"ratings": checked, "model_ids": models}:
                raise ValueError("saved checked output differs from raw-attempt replay")
            ratings.update(checked)
            for filename in ("payload.json", "call_started.json", "attempt.json", "checked.json"):
                evidence[relative + "/" + filename] = (directory / filename).read_bytes()
        passes.append(ratings)
    actual_dirs = {str(p.relative_to(audit)) for p in audit.glob("pass-*/batch-*") if p.is_dir()}
    if actual_dirs != expected_dirs:
        raise ValueError("unexpected/missing audit batch directories")
    for key, row in records.items():
        if row["pass_eligible"] != [p[key]["eligible"] for p in passes]:
            raise ValueError("result eligibility differs from replayed audit passes")
    counts = {s: sum(r["eligible"] for r in records.values() if r["split"] == s) for s in ("train", "validation")}
    if (not isinstance(result.get("eligible"), dict) or any(type(v) is not int for v in result["eligible"].values())
            or result["eligible"] != counts or type(result.get("data_limited")) is not bool
            or result["data_limited"] != (counts["train"] < 40 or counts["validation"] < 5)):
        raise ValueError("aggregate audit eligibility/floor differs")
    agreement = sum(r["pass_eligible"][0] == r["pass_eligible"][1] for r in records.values()) / len(records)
    pa, pb = [sum(r["pass_eligible"][i] for r in records.values()) / len(records) for i in range(2)]
    expected = pa * pb + (1 - pa) * (1 - pb)
    kappa = (agreement - expected) / (1 - expected) if expected < 1 else None
    for key, value in [("agreement", agreement), ("cohens_kappa", kappa)]:
        observed = result.get(key)
        if ((value is None and observed is not None) or (value is not None and
                (type(observed) not in (int, float) or not math.isfinite(observed) or not math.isclose(observed, value, abs_tol=1e-12)))):
            raise ValueError("audit agreement summary differs from record flags")
    return manifest, payloads, splits, result, records, evidence


def apply_audit(prepared, audit, output, repo_root=ROOT):
    prepared, audit, output, repo_root = [Path(p).resolve() for p in (prepared, audit, output, repo_root)]
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(prepared) or output.is_relative_to(audit):
        raise ValueError("new output must not be inside either immutable input directory")
    lock_path = audit / "execution.lock"
    with (lock_path.open("rb") if lock_path.exists() else nullcontext()) as lock:
        if lock is not None:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise AuditIncomplete("audit executor still owns the input directory") from None
        manifest, payloads, splits, result, records, evidence = verified_audit(audit, prepared, repo_root)
        # Existing preparation sources stay untouched. Bind the additional pure
        # metadata helper and this application's exact source separately.
        from scripts.qwen38_prepare_inputs import eligible_order
        metadata_helper = importlib.import_module("scripts.qwen38_prepare_inputs")
        application_source = Path(__file__).read_bytes()
        helper_source = Path(metadata_helper.__file__).read_bytes()
        provenance = {"version": VERSION, "prepared_manifest_sha256": sha_bytes((prepared / "manifest.json").read_bytes()),
            "audit_registration_sha256": sha_bytes(evidence["registration.json"]), "audit_result_sha256": sha_bytes(evidence["result.json"]),
            "audit_evidence_sha256": {name: sha_bytes(data) for name, data in evidence.items()},
            "application_source_sha256": sha_bytes(application_source), "metadata_helper_sha256": sha_bytes(helper_source),
            "eligible": result["eligible"], "data_limited": result["data_limited"], "AI_audit_only": True,
            "rule": "both registered passes eligible; no rubric relaxation; all original rows retained",
            "scope": "train/development eligibility only; old test unchanged and diagnostic",
            "output_integrity": "registered inputs checked; saved audit outputs replayed and hashes captured at application time"}
        for split in ("train", "validation"):
            updated = []
            for row in splits[split]:
                record = records[row["id"]]
                updated.append({**row, "training_eligible": record["eligible"],
                    "training_eligibility_reason": "both-pass AI support audit accepted" if record["eligible"] else "rejected by registered two-pass AI support audit",
                    "claim_audit": {"eligible": record["eligible"], "pass_eligible": record["pass_eligible"],
                        "excerpt_sha256": record["excerpt_sha256"], "abstract_sha256": record["abstract_sha256"],
                        "result_sha256": provenance["audit_result_sha256"], "AI_audit_only": True}})
            splits[split] = updated
            payloads[split + ".jsonl"] = "".join(canonical(r) + "\n" for r in updated).encode()
        for name, data in evidence.items():
            payloads["claim_audit/" + name] = data
        payloads["claim_audit/application_source.py"] = application_source
        payloads["claim_audit/metadata_helper_source.py"] = helper_source
        payloads["claim_audit/application.json"] = (canonical(provenance) + "\n").encode()
        new_manifest = {**manifest, "audit_application": provenance,
            "prl_grounding": "Registered both-pass AI screening applied; not human factual validation. Evidence, targets and original diagnostics unchanged."}
        new_manifest["counts"] = {split: {**manifest["counts"][split], "training_eligible": sum(r["training_eligible"] for r in rows)} for split, rows in splits.items()}
        for split, key in [("train", "eligible_training_order"), ("validation", "eligible_development_order")]:
            count = sum(r["training_eligible"] for r in splits[split])
            new_manifest[key] = eligible_order(splits[split], max(1, count), manifest.get(key, {}).get("seed", 42))
        new_manifest["output_sha256"] = {name: sha_bytes(data) for name, data in payloads.items()}
        # Recheck every read-only source after replay and before writing output.
        verified_prepared(prepared, provenance["prepared_manifest_sha256"])
        for name, data in evidence.items():
            if name in {"reviewed_registration.json", "prepared_manifest.json", "judge_isolation_audit.json", "judge_runtime_binding.json"}:
                continue
            if (audit / name).read_bytes() != data:
                raise ValueError("audit evidence changed during application")
        output.mkdir(parents=True, exist_ok=False)
        for name, data in payloads.items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
            if sha_bytes(path.read_bytes()) != new_manifest["output_sha256"][name]:
                raise ValueError("written output hash differs")
        with (output / "manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(canonical(new_manifest) + "\n")
    return {"output": str(output), "retained": dict(COUNTS), "eligible": result["eligible"],
            "data_limited": result["data_limited"], "manifest_sha256": sha_bytes((output / "manifest.json").read_bytes()),
            "GPU_started": False, "AI_audit_only": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", default="outputs/qwen38-science-01/prepared-prl-tokenizer-04")
    p.add_argument("--audit", default="outputs/qwen38-prl-claim-audit-01")
    p.add_argument("--output", required=True)
    p.add_argument("--wait", action="store_true", help="wait read-only for the parent's audit; never invoke the judge")
    a = p.parse_args()
    while True:
        try:
            result = apply_audit(ROOT / a.prepared, ROOT / a.audit, ROOT / a.output)
            print(canonical(result), flush=True)
            return
        except AuditIncomplete as exc:
            if not a.wait:
                p.error(str(exc))
            print(canonical({"waiting": str(exc)}), flush=True)
            time.sleep(30)


if __name__ == "__main__":
    main()
