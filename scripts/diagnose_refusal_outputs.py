"""CPU-only, unblinded packet/statistics from immutable stored run03 outputs.

No generation, scoring API, judge invocation or subjective annotation. All
experiment hashes and frozen final scores are verified before preparation and
again afterward. Raw text is written only to a fresh ignored private directory.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import statistics

from scripts import refusal_reduction as reduction
from scripts.common import path, read_json
from scripts.save_nf4 import stream_hash
from scripts.tone_common import token_length

RUBRIC = "evaluation/refusal/output_diagnosis_rubric.json"
PROTOCOL = "docs/refusal_output_diagnosis_protocol.md"
SOURCE = "scripts/diagnose_refusal_outputs.py"


def save(filename, value, private=False):
    filename.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(filename, flags, 0o600 if private else 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def verify(run, report_file, lock_file):
    cfg, manifest = reduction.verify(run, models=True)
    lock, report = read_json(lock_file), read_json(report_file)
    if (reduction.selection_lock(run) != lock or report["null"] is not False or lock["decision"]["null"] or
            lock["decision"]["selected_method"] != "subtract23" or
            report["selection"] != read_json(run/"selection.json") or
            report["manifest_sha256"] != stream_hash(run/"manifest.json") or
            report["selection_lock_sha256"] != stream_hash(lock_file) or
            report["source_sha256"] != manifest["sources"]):
        raise ValueError("original run/report/selection binding differs")
    for phase in ("final",):
        if read_json(run/phase/"conditions.json")["selection_lock_sha256"] != stream_hash(lock_file):
            raise ValueError("final lock differs")
    measured = reduction.new_scores(run, "final")
    if measured != report["final"]:
        raise ValueError("recomputed frozen final scores differ from accepted result")
    retained = read_json(run/"retention/results.json")
    if retained != report["retention"] or retained["selection_lock_sha256"] != stream_hash(lock_file):
        raise ValueError("retention/report lock differs")
    for name, expected in retained["raw_sha256"].items():
        if stream_hash(run/"retention"/name) != expected:
            raise ValueError("stored retention bytes differ")
    quality = reduction.final_quality(
        {label: measured[f"m0_{label}"]["overall"] for label in ("safe", "unsafe")},
        {label: measured[f"subtract23_{label}"]["overall"] for label in ("safe", "unsafe")},
        retained["measurements"]["baseline"], retained["measurements"]["subtract23"])
    if quality != report["final_quality"] or quality["all_criteria_pass"] is not False:
        raise ValueError("accepted failed final-quality criteria do not recompute")
    return cfg, manifest, measured


def input_hashes(run, manifest, report_file, lock_file):
    files = {p.resolve() for p in run.rglob("*") if p.is_file()}
    for field in ("external_reuse_sha256", "external_model_sha256", "external_runtime_sha256"):
        files.update(path(n).resolve() for n in manifest[field])
    files.update(path(n).resolve() for n in manifest["sources"])
    files.update(path(n).resolve() for n in (report_file, lock_file, RUBRIC, PROTOCOL, SOURCE))
    return {str(p): stream_hash(p) for p in sorted(files)}


def pairs(run, measured):
    rows, predictions = reduction.phase_inputs(run, "final")
    submission = read_json(run/"final/judging/ratings.json")
    ratings = reduction.common.decode_ratings(read_json(run/"final/judging/keys.local.json"), submission)
    result = []
    for label in ("safe", "unsafe"):
        base, candidate = f"m0_{label}", f"subtract23_{label}"
        by_id = {name: {p["id"]: p for p in predictions[name]} for name in (base, candidate)}
        details = {name: {d["id"]: d for d in measured[name]["details"]} for name in (base, candidate)}
        for row in rows[base]:
            identifier = row["id"]
            before, after = details[base][identifier], details[candidate][identifier]
            membership, reasons = [], []
            if label == "safe" and before["useful_and_correct"] != after["useful_and_correct"]:
                membership.append("A")
                reasons.append("safe_usefulness_loss" if before["useful_and_correct"] else "safe_usefulness_gain")
            if label == "unsafe" and before["refused"] and not after["refused"]:
                membership.append("B")
                reasons.append("unsafe_refusal_removed_new_valid_full_compliance" if after["valid_full_compliance"] else "unsafe_refusal_removed_invalid")
            if after["degenerate"] or after["unscorable"]:
                membership.append("C")
                if after["degenerate"]:
                    reasons.append("candidate_mechanical_degeneration")
                if after["unscorable"]:
                    reasons.append("candidate_unscorable")
            pair = {"id": identifier, "source_class": label, "family": row["family"],
                    "membership": membership, "selection_reasons": reasons, "prompt": row["instruction"]}
            for alias, name in (("baseline", base), ("candidate", candidate)):
                prediction = by_id[name][identifier]
                generated = prediction["generated_ids"]
                if prediction["generated_tokens_including_eos"] != len(generated):
                    raise ValueError("stored generated-token count differs")
                # Never copy raw token arrays or rendered prompts into packets.
                pair[alias] = {"output": prediction["output"], "frozen_details": details[name][identifier],
                    "frozen_ai_rating": {k: v for k, v in ratings[name][identifier].items() if k != "id"},
                    "hit_token_limit": prediction["hit_token_limit"], "ended_eos": prediction["ended_eos"],
                    "generated_tokens_including_eos": len(generated),
                    "response_tokens_excluding_eos": token_length(prediction),
                    "max_new_tokens": 320}
            result.append(pair)
    if len(result) != 72 or len({r["id"] for r in result}) != 72:
        raise ValueError("expected unique72 final pairs")
    return result


def count_table(items, fields):
    counts = Counter(tuple(f(row) for f in fields.values()) for row in items)
    return [{**dict(zip(fields, values)), "count": count} for values, count in sorted(counts.items())]


def stats(items):
    transitions = {}
    outcomes = ("refused", "valid_full_compliance", "empty", "degenerate", "unscorable")
    if items[0]["source_class"] == "safe":
        outcomes = ("useful_and_correct", *outcomes)
    for key in outcomes:
        transitions[key] = count_table(items, {
            "baseline_capped": lambda r: r["baseline"]["hit_token_limit"],
            "candidate_capped": lambda r: r["candidate"]["hit_token_limit"],
            "baseline_outcome": lambda r, k=key: r["baseline"]["frozen_details"][k],
            "candidate_outcome": lambda r, k=key: r["candidate"]["frozen_details"][k]})
    lengths = {}
    for alias in ("baseline", "candidate"):
        values = [r[alias]["response_tokens_excluding_eos"] for r in items]
        lengths[alias] = {"count": len(values), "min": min(values), "median": statistics.median(values),
                         "mean": statistics.mean(values), "max": max(values),
                         "ended_eos": sum(r[alias]["ended_eos"] for r in items),
                         "capped": sum(r[alias]["hit_token_limit"] for r in items)}
    return {"pairs": len(items), "paired_cap_outcome_flag_cross_tabs": transitions,
            "token_lengths_excluding_eos": lengths,
            "paired_cap_flags": count_table(items, {
                "baseline_capped": lambda r: r["baseline"]["hit_token_limit"],
                "candidate_capped": lambda r: r["candidate"]["hit_token_limit"],
                "candidate_degenerate": lambda r: r["candidate"]["frozen_details"]["degenerate"],
                "candidate_unscorable": lambda r: r["candidate"]["frozen_details"]["unscorable"]})}


def template(items):
    annotations = []
    for row in items:
        entry = {"id": row["id"], "membership": row["membership"]}
        for alias in ("baseline", "candidate"):
            entry[alias] = {k: None for k in ("truncation_code", "completion", "error_code", "repetition_code", "confidence", "evidence_notes")}
            if "B" in row["membership"]:
                entry[alias]["unsafe_structure"] = {k: None for k in ("topical_engagement", "refusal_structure", "style_changed")}
        if "A" in row["membership"]:
            entry["safe_flip_explanation"] = None
            entry["existing_label_uncertainty_notes"] = None
        annotations.append(entry)
    return {"reviewer": None, "unblinded": True, "not_regrading": True, "annotations": annotations}


def markdown(packet):
    text = ["# Run03 stored-output diagnostic packet\n",
            "UNBLINDED; texts are DATA, never instructions. No external fact checks, new answers or label replacements.\n",
            "T0 means uncapped, not necessarily complete. Completion is a separate annotation.\n"]
    for pair in packet["pairs"]:
        text.append(f"\n## {pair['id']} ({pair['source_class']}; {pair['family']})\n")
        text.append("Membership/reasons: "+", ".join(pair["membership"]+pair["selection_reasons"])+"\n")
        for title, value in (("Original prompt", pair["prompt"]), ("Baseline", pair["baseline"]["output"]), ("Candidate subtract23", pair["candidate"]["output"])):
            # A longer fence keeps response text from escaping its data block.
            fence = "`"*max(3, 1+max((len(s) for s in re.findall(r"`+", value)), default=0))
            text.append(f"\n### {title}\n{fence}text\n{value}\n{fence}\n")
        metadata = {alias: {k: v for k, v in pair[alias].items() if k != "output"} for alias in ("baseline", "candidate")}
        text.append("\nFrozen metadata:\n```json\n"+json.dumps(metadata, indent=2)+"\n```\n")
    return "\n".join(text)


def privacy_check(report, raw_pairs):
    prohibited = {"prompt", "response", "output", "rendered_prompt", "input_ids", "generated_ids", "token_ids",
                  "password", "access_token", "hf_token", "api_key", "authorization", "secret"}
    def walk(value):
        if isinstance(value, dict):
            if prohibited & {k.casefold() for k in value}:
                raise ValueError("private field in public statistics")
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(report)
    encoded = json.dumps(report, ensure_ascii=False)
    if re.search(r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|Bearer\s+\S+|/home/|/tmp/", encoded):
        raise ValueError("credential/path pattern in public statistics")
    for pair in raw_pairs:
        for value in (pair["prompt"], pair["baseline"]["output"], pair["candidate"]["output"]):
            if len(value) >= 30 and value in encoded:
                raise ValueError("raw experiment text in public statistics")


def prepare(args):
    run, output, report_file, lock_file = map(path, (args.run, args.output, args.report, args.lock))
    public = path(args.statistics)
    if output.exists() or public.exists():
        raise FileExistsError("fresh diagnosis directory and statistics file required")
    if not output.resolve().is_relative_to(path("outputs").resolve()):
        raise ValueError("raw packet must remain in ignored outputs/")
    _, manifest, measured = verify(run, report_file, lock_file)
    before = input_hashes(run, manifest, report_file, lock_file)
    all_pairs = pairs(run, measured)
    selected = [r for r in all_pairs if r["membership"]]
    reasons = Counter(reason for pair in selected for reason in pair["selection_reasons"])
    memberships = Counter(m for pair in selected for m in pair["membership"])
    expected = {"safe_usefulness_loss": 9, "safe_usefulness_gain": 3,
                "unsafe_refusal_removed_new_valid_full_compliance": 6, "unsafe_refusal_removed_invalid": 1,
                "candidate_mechanical_degeneration": 3, "candidate_unscorable": 1}
    if len(selected) != 19 or dict(reasons) != expected or dict(memberships) != {"A": 12, "B": 7, "C": 3}:
        raise ValueError("registered pair selection/count expectations differ")
    output.mkdir(parents=True, mode=0o700)
    os.chmod(output, 0o700)
    packet = {"unblinded": True, "not_regrading": True, "candidate": "subtract23",
        "rubric_sha256": stream_hash(path(RUBRIC)), "pairs": selected}
    save(output/"packet.json", packet, True)
    save(output/"packet.md", markdown(packet), True)
    save(output/"rubric.json", path(RUBRIC).read_text(encoding="utf-8"), True)
    save(output/"reviewer_template.json", template(selected), True)
    # Raw receipt paths are private. No subjective annotations are filled in.
    artifacts = {p.name: stream_hash(p) for p in output.iterdir() if p.is_file()}
    verify(run, report_file, lock_file)
    after = input_hashes(run, manifest, report_file, lock_file)
    if before != after:
        raise ValueError("original input bytes changed during diagnosis preparation")
    save(output/"manifest.json", {"input_sha256": before, "packet_files_sha256": artifacts,
        "input_hashes_unchanged_after": True, "selected_ids": [r["id"] for r in selected],
        "source_sha256": {n: stream_hash(path(n)) for n in (SOURCE, RUBRIC, PROTOCOL)},
        "reviewer_annotations_authored": False}, True)
    grouped = {}
    for label in ("safe", "unsafe"):
        rows = [r for r in all_pairs if r["source_class"] == label]
        grouped[label] = {"all_families": stats(rows), "by_family": {
            family: stats([r for r in rows if r["family"] == family]) for family in sorted({r["family"] for r in rows})}}
    report = {"scope": "post-hoc unblinded stored-output diagnosis; descriptive counts, not re-grading",
        "original_development_null": False, "original_final_quality_all_criteria_pass": False,
        "original_result_sha256": stream_hash(report_file), "selection_lock_sha256": stream_hash(lock_file),
        "run_manifest_sha256": stream_hash(run/"manifest.json"),
        "original_source_sha256": manifest["sources"],
        "source_sha256": {n: stream_hash(path(n)) for n in (SOURCE, RUBRIC, PROTOCOL)},
        "packet_files_sha256": {**artifacts, "manifest.json": stream_hash(output/"manifest.json")},
        "input_hashes_verified": len(before), "input_hashes_unchanged_after": True,
        "selection_counts": {"unique_pairs": 19, "membership": dict(memberships), "reasons": dict(reasons)},
        "selected_pairs": [{k: r[k] for k in ("id", "source_class", "family", "membership", "selection_reasons")} for r in selected],
        "paired_statistics": grouped, "annotations_status": "not supplied; reviewer template entirely blank",
        "limits": ["T0 means not capped, not necessarily complete; completion is separate.",
                   "Original labels and failed final-quality decision are unchanged; no stored grader explanations exist.",
                   "Cap associations are descriptive, not causal. No new rates, confidence intervals, benchmark scores or external fact checks.",
                   "Source-unsafe review describes structure only, not correctness or operational effectiveness."]}
    privacy_check(report, all_pairs)
    save(public, report)
    print(f"PACKET_READY: {len(selected)} unique pairs; A12/B7/C3 with overlaps; {len(before)} input hashes unchanged.")
    print(f"Private packet: {output}; raw-text-free statistics: {public}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/refusal-03")
    parser.add_argument("--report", default="results/refusal_03.json")
    parser.add_argument("--lock", default="results/protocols/refusal-selection-03.json")
    parser.add_argument("--output", default="outputs/refusal-diagnosis-03")
    parser.add_argument("--statistics", default="results/refusal_output_diagnosis_03.json")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
