"""CPU-only refusal data preparation: freeze candidates, then use adviser labels.

No network, model loads, GPU access, generation or judging. Raw instruction
lengths use only the cached pinned Qwen tokenizer. Targets/responses are unused.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import shutil
import subprocess

from scripts.common import ROOT, path, read_json
from scripts.save_nf4 import stream_hash
from scripts.refusal_data_audit import normalized, PINS, POOLED_CONTRASTS

SEED = 20261114
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
CATEGORIES = {"open_qa", "brainstorming", "classification"}
RUBRIC = "evaluation/refusal/screening_rubric.json"
SOURCES = ("scripts/prepare_refusal.py", "scripts/refusal_data_audit.py", RUBRIC)


def digest_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save(filename, value):
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def row(source, index, instruction, category=None):
    return {"id": f"{source}-{index:05d}", "source_row": index, "instruction": instruction,
            "prompt_sha256": digest_text(instruction), "normalized_sha256": digest_text(normalized(instruction)),
            **({"category": category} if category is not None else {})}


def load_sources(raw, audit):
    if audit["pins"] != PINS:
        raise ValueError("audited immutable source pins differ")
    data = {}
    for name in ("xstest_prompts.csv", "advbench_harmful_behaviors.csv", "dolly.jsonl"):
        filename = raw / name
        expected = audit["files"][name]
        if stream_hash(filename) != expected["sha256"] or filename.stat().st_size != expected["bytes"]:
            raise ValueError("audited raw source changed")
        data[name] = filename.read_text(encoding="utf-8-sig")
    xstest = list(csv.DictReader(io.StringIO(data["xstest_prompts.csv"], newline="")))
    # The target column and Dolly responses are never accessed or propagated.
    harmful = [row("advbench", i, r["goal"]) for i, r in enumerate(csv.DictReader(io.StringIO(data["advbench_harmful_behaviors.csv"], newline="")))]
    dolly = [row("dolly", i, r["instruction"], r["category"]) for i, line in enumerate(data["dolly.jsonl"].split("\n"))
             if line.strip() and (r := json.loads(line))["category"] in CATEGORIES and not r["context"].strip()
             and r["instruction"].strip()]
    return xstest, harmful, dolly


def exclusions(xstest, harmful, dolly):
    evaluation = {normalized(r["prompt"]) for r in xstest}
    losses = {}
    cleaned = {}
    for name, rows in (("advbench", harmful), ("dolly", dolly)):
        seen, unique = set(), []
        for r in rows:
            key = normalized(r["instruction"])
            if key not in seen:
                unique.append(r)
                seen.add(key)
        without_xstest = [r for r in unique if normalized(r["instruction"]) not in evaluation]
        losses[name] = {"input_rows": len(rows), "within_source_duplicate_rows_removed": len(rows)-len(unique),
                        "all_xstest_overlap_rows_removed": len(unique)-len(without_xstest)}
        cleaned[name] = without_xstest
    cross = {normalized(r["instruction"]) for r in cleaned["advbench"]} & {normalized(r["instruction"]) for r in cleaned["dolly"]}
    for name in cleaned:
        before = len(cleaned[name])
        cleaned[name] = [r for r in cleaned[name] if normalized(r["instruction"]) not in cross]
        losses[name]["cross_class_ambiguous_rows_removed"] = before-len(cleaned[name])
        losses[name]["remaining_rows"] = len(cleaned[name])
    return cleaned["advbench"], cleaned["dolly"], losses


def candidate_pool(xstest, harmful, dolly, length):
    # Envelope depends on ALL raw audited AdvBench goals, before any Dolly draw.
    raw_lengths = [length(r["instruction"]) for r in harmful]
    if not raw_lengths or min(raw_lengths) <= 0:
        raise ValueError("nonempty positive raw AdvBench lengths required")
    lo, hi = max(1, (4*min(raw_lengths))//5), (6*max(raw_lengths)+4)//5
    envelope = {"advbench_raw_min": min(raw_lengths), "advbench_raw_max": max(raw_lengths),
                "dolly_integer_min": lo, "dolly_integer_max": hi,
                "rounding": "outward: floor(0.8*raw_min), ceil(1.2*raw_max); minimum1", "computed_before_draw": True}
    harmful, dolly, losses = exclusions(xstest, harmful, dolly)
    harmful = [{**r, "raw_token_length": length(r["instruction"])} for r in harmful]
    dolly = [{**r, "raw_token_length": length(r["instruction"])} for r in dolly]
    filtered = [r for r in dolly if lo <= r["raw_token_length"] <= hi]
    losses["dolly"].update({"outside_length_envelope_removed": len(dolly)-len(filtered), "envelope_eligible": len(filtered)})
    if len(filtered) < 200:
        raise ValueError("fewer than 200 eligible candidates; stop, no replacement pool")
    random.Random(SEED).shuffle(filtered)
    random.Random(SEED).shuffle(harmful)
    return filtered[:200], harmful, envelope, losses


def tokenizer():
    from transformers import AutoTokenizer
    snapshot = path(".hf/hub/models--Qwen--Qwen3-4B/snapshots") / MODEL_REVISION
    if not (snapshot / "tokenizer.json").is_file():
        raise ValueError("pinned tokenizer must already be cached; downloads prohibited")
    instance = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
             "added_tokens.json", "vocab.json", "merges.txt", "config.json")
    hashes = {n: stream_hash(snapshot/n) for n in names if (snapshot/n).is_file()}
    return lambda text: len(instance(text, add_special_tokens=False)["input_ids"]), hashes


def candidates(args):
    output, registration = path(args.output), path(args.registration)
    if output.exists() or registration.exists():
        raise FileExistsError("fresh candidate output and registration required")
    raw, audit_file = path(args.raw), path(args.audit)
    audit = read_json(audit_file)
    xstest, harmful, dolly = load_sources(raw, audit)
    length, token_hashes = tokenizer()
    selected, harmful_order, envelope, losses = candidate_pool(xstest, harmful, dolly, length)
    output.mkdir(parents=True)
    save(output / "candidates.json", {"items": selected, "order": "independent random.Random(seed).shuffle original source order; first200", "seed": SEED})
    save(output / "advbench_order.json", {"items": harmful_order, "seed": SEED, "order": "independent random.Random(seed).shuffle original source order"})
    for name, source in (("xstest_prompts.csv", raw / "xstest_prompts.csv"), ("source_audit.json", audit_file),
                         ("screening_rubric.json", path(RUBRIC)), ("protocol.md", path("docs/refusal_protocol.md"))):
        target = output / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in SOURCES:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path(name), target)
    frozen = {str(p.relative_to(output)): stream_hash(p) for p in output.rglob("*") if p.is_file()}
    manifest = {"stage": "candidates_only_before_adviser_screen", "seed": SEED, "candidate_count": 200,
        "source_pins": PINS, "raw_source_sha256": {n: audit["files"][n]["sha256"] for n in
            ("xstest_prompts.csv", "advbench_harmful_behaviors.csv", "dolly.jsonl")},
        "model_id": "Qwen/Qwen3-4B", "model_revision": MODEL_REVISION, "tokenizer_sha256": token_hashes,
        "files": frozen, "source_sha256": {n: stream_hash(path(n)) for n in SOURCES},
        "envelope": envelope, "losses": losses, "no_model_outputs": True, "no_gpu_run_registered": True,
        "reviewer": read_json(path(RUBRIC))["reviewer_policy"]}
    save(output / "manifest.json", manifest)
    save(registration, {**manifest, "manifest_sha256": stream_hash(output/"manifest.json"),
        "candidate_ids_hashes": [{k:r[k] for k in ("id", "prompt_sha256", "normalized_sha256", "raw_token_length")} for r in selected]})
    print(f"CANDIDATES_READY: 200 frozen candidates; envelope {lohi(envelope)}; registration {registration.relative_to(ROOT)}")


def lohi(envelope):
    return [envelope["dolly_integer_min"], envelope["dolly_integer_max"]]


def verify_candidates(output, registration):
    manifest = read_json(output / "manifest.json")
    if registration["manifest_sha256"] != stream_hash(output / "manifest.json"):
        raise ValueError("candidate registration manifest mismatch")
    for name, expected in manifest["files"].items():
        if stream_hash(output/name) != expected:
            raise ValueError("frozen candidate input changed")
    for name, expected in manifest["source_sha256"].items():
        if stream_hash(path(name)) != expected:
            raise ValueError("active preparation source changed since candidate registration")
    if any(registration.get(k) != v for k,v in manifest.items()):
        raise ValueError("candidate registration contents changed")
    expected_ids = [{k:r[k] for k in ("id", "prompt_sha256", "normalized_sha256", "raw_token_length")}
                    for r in read_json(output/"candidates.json")["items"]]
    if registration.get("candidate_ids_hashes") != expected_ids:
        raise ValueError("registered candidate IDs/order/hashes differ from frozen pool")
    return manifest


def validate_screen(candidates, rubric, submission, candidate_sha, rubric_sha):
    reviewer = submission.get("reviewer", {})
    if (reviewer.get("model_identifier") != "claude-opus-5-5" or reviewer.get("advisor_overlap") is not True
            or not isinstance(reviewer.get("exposed_settings"), dict)
            or submission.get("candidate_file_sha256") != candidate_sha or submission.get("rubric_sha256") != rubric_sha):
        raise ValueError("actual adviser identity/overlap/settings and exact frozen hashes required")
    labels = submission.get("labels", [])
    ids = {r["id"] for r in candidates}
    if len(ids) != len(candidates) or len(labels) != len(ids) or {r["id"] for r in labels} != ids:
        raise ValueError("complete unique candidate screening IDs required")
    for label in labels:
        if (set(label) != {"id", "label", "reason_code"} or label["label"] not in rubric["labels"]
                or label["reason_code"] not in rubric["reason_codes"][label["label"]]):
            raise ValueError("screen label/reason code outside frozen schema")
    return {r["id"]: r for r in labels}


def pair_approved(harmful, candidates, labels, limit=100):
    approved = [(i, r) for i, r in enumerate(candidates) if labels[r["id"]]["label"] == "harmless"]
    used, result, unmatched = set(), [], []
    for a in harmful:
        length = a["raw_token_length"]
        options = [(abs(b["raw_token_length"]-length), i, b) for i,b in approved
                   if b["id"] not in used and 4*length <= 5*b["raw_token_length"] <= 6*length]
        if not options:
            unmatched.append(a["id"])
            continue
        _, _, b = min(options, key=lambda item: item[:2])
        result.append({"id": f"pair-{len(result):03d}", "harmful": a, "benign": b,
                       "benign_harmful_raw_length_ratio": b["raw_token_length"]/length})
        used.add(b["id"])
        if len(result) == limit:
            break
    return result, {"maximum_pairs": limit, "actual_pairs": len(result), "approved_candidates": len(approved),
                    "unmatched_harmful_ids_before_stop": unmatched, "pair_count_is_not_a_yield_gate": True}


def xstest_split(rows):
    safe_types = {r["type"] for r in rows if r["label"] == "safe"}
    unsafe_types = {r["type"] for r in rows if r["label"] == "unsafe"}
    direct = sorted(t for t in safe_types if "contrast_"+t in unsafe_types)
    if len(direct) != 6:
        raise ValueError("exact six audited literal paired types required")
    rng = random.Random(SEED)
    rng.shuffle(direct)
    dev_types = set(direct[:3]) | {"contrast_"+t for t in direct[:3]}
    result = {"development": [], "final": []}
    for split, types, maximum in (("development", dev_types, 10), ("final", (safe_types|unsafe_types)-dev_types, 6)):
        for kind in sorted(types):
            group = [r for r in rows if r["type"] == kind]
            for r in rng.sample(group, min(maximum, len(group))):
                family = ("privacy" if kind in ("privacy_public", "privacy_fictional", "contrast_privacy") else
                          "discrimination" if kind in ("nons_group_real_discr", "real_group_nons_discr", "contrast_discr") else
                          kind.removeprefix("contrast_"))
                result[split].append({"id": "xstest-"+r["id"], "source_id": r["id"], "instruction": r["prompt"],
                    "type": kind, "family": family, "label": r["label"], "prompt_sha256": digest_text(r["prompt"])})
        rng.shuffle(result[split])
    counts = {s: dict(Counter(r["label"] for r in v)) for s,v in result.items()}
    if counts != {"development": {"safe": 30, "unsafe": 30}, "final": {"safe": 42, "unsafe": 30}}:
        raise ValueError("audited per-type counts do not produce registered sizes")
    if len({r["family"] for r in result["final"]}) != 5:
        raise ValueError("final must contain exactly five broad families")
    return result, {"seed": SEED, "development_direct_types": sorted(direct[:3]), "counts": counts,
        "final_broad_families": sorted({r["family"] for r in result["final"]}),
        "sampling": "single Random(seed): shuffle sorted direct types; sample source-ordered rows per sorted type; shuffle each completed split",
        "shared_pools_final_only": True}


def final_prepare(args):
    candidate_dir, registration_file, output = path(args.candidates), path(args.registration), path(args.output)
    if output.exists() or path(args.final_report).exists():
        raise FileExistsError("fresh final preparation output/report required")
    relative = registration_file.resolve().relative_to(ROOT)
    committed = subprocess.run(["git", "show", "HEAD:"+relative.as_posix()], cwd=ROOT, capture_output=True)
    if committed.returncode or committed.stdout != registration_file.read_bytes():
        raise ValueError("exact candidate registration must be committed before adviser screening/final preparation")
    registration = read_json(registration_file)
    manifest = verify_candidates(candidate_dir, registration)
    items = read_json(candidate_dir/"candidates.json")["items"]
    rubric = read_json(candidate_dir/"inputs/screening_rubric.json")
    submission = read_json(args.labels)
    labels = validate_screen(items, rubric, submission, stream_hash(candidate_dir/"candidates.json"),
                             stream_hash(candidate_dir/"inputs/screening_rubric.json"))
    pairs, pairing = pair_approved(read_json(candidate_dir/"advbench_order.json")["items"], items, labels)
    rows = list(csv.DictReader((candidate_dir/"inputs/xstest_prompts.csv").open(encoding="utf-8-sig", newline="")))
    splits, split_report = xstest_split(rows)
    output.mkdir(parents=True)
    save(output/"extraction.json", {"pairs": pairs, "pairing": pairing})
    for split, prompts in splits.items():
        save(output/(split+".json"), {"items": prompts})
    shutil.copyfile(path(args.labels), output/"screening_labels.json")
    save(output/"source_candidate_manifest.json", manifest)
    final = {"stage": "prepared_inputs_only_no_model_outputs", "source_pins": PINS, "seed": SEED,
        "candidate_registration_sha256": stream_hash(registration_file), "screening_submission_sha256": stream_hash(path(args.labels)),
        "reviewer": submission["reviewer"], "label_counts": dict(Counter(r["label"] for r in labels.values())),
        "pairing": pairing, "xstest_split": split_report, "tokenizer_sha256": manifest["tokenizer_sha256"],
        "files": {p.name: stream_hash(p) for p in output.iterdir() if p.is_file()},
        "extraction_ids_hashes": [{"pair_id": p["id"], **{c: {k:p[c][k] for k in ("id", "prompt_sha256", "raw_token_length")} for c in ("harmful", "benign")}} for p in pairs],
        "evaluation_ids_hashes": {s: [{k:r[k] for k in ("id", "type", "family", "label", "prompt_sha256")} for r in v] for s,v in splits.items()},
        "source_sha256": manifest["source_sha256"], "no_gpu_run_registered": True}
    save(output/"manifest.json", final)
    save(path(args.final_report), {**final, "manifest_sha256": stream_hash(output/"manifest.json")})
    print(f"Prepared {len(pairs)} extraction pairs, 60 development and 72 final prompts; no model outputs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("candidates", "final-prepare"))
    parser.add_argument("--raw", default="data/refusal-01/raw")
    parser.add_argument("--audit", default="results/refusal_data_audit.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--registration", default="results/protocols/refusal-candidates-01.json")
    parser.add_argument("--candidates", default="outputs/refusal-candidates-01")
    parser.add_argument("--labels")
    parser.add_argument("--final-report", default="results/protocols/refusal-prepared-01.json")
    args = parser.parse_args()
    if args.stage == "final-prepare" and not args.labels:
        parser.error("adviser labels required for final-prepare")
    (candidates if args.stage == "candidates" else final_prepare)(args)


if __name__ == "__main__":
    main()
