"""Validate stored qualitative annotations and export counts/codes, never notes.

CPU/files only. Frozen packet, preparation inputs, rubric, labels and experiment
code are read-only. This is post-hoc advisory description, not benchmark grading.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from scripts.common import path, read_json
from scripts.save_nf4 import stream_hash

SOURCE = "scripts/summarize_refusal_diagnosis.py"
CONDITIONS = ("baseline", "candidate")
CODES = ("truncation_code", "completion", "error_code", "repetition_code", "confidence")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def load_frozen(directory):
    manifest = read_json(directory/"manifest.json")
    if not manifest["input_hashes_unchanged_after"] or manifest["reviewer_annotations_authored"]:
        raise ValueError("original packet preparation must be verified and unannotated")
    for name, expected in manifest["packet_files_sha256"].items():
        if stream_hash(directory/name) != expected:
            raise ValueError("immutable packet/template/rubric/Markdown changed")
    for name, expected in manifest["source_sha256"].items():
        if stream_hash(path(name)) != expected:
            raise ValueError("immutable preparation source/rubric/protocol changed")
    packet, template, rubric = [read_json(directory/name) for name in
                                ("packet.json", "reviewer_template.json", "rubric.json")]
    if packet["rubric_sha256"] != stream_hash(directory/"rubric.json"):
        raise ValueError("packet/rubric binding differs")
    if any(value.get("unblinded") is not True or value.get("not_regrading") is not True
           for value in (packet, template)):
        raise ValueError("unblinded non-regrading scope required")
    ids = [r["id"] for r in packet["pairs"]]
    if len(ids) != 19 or len(set(ids)) != 19 or ids != manifest["selected_ids"]:
        raise ValueError("immutable packet requires19 unique selected pairs")
    if ids != [r["id"] for r in template["annotations"]]:
        raise ValueError("template/packet ID order differs")
    schema = {"version": 1, "conditions": list(CONDITIONS),
        "code_enums": {k: list(rubric["per_condition"][k]) for k in CODES},
        "safe_flip_explanation": rubric["safe_flip_explanation"],
        "unsafe_structure": {k: rubric["unsafe_structure"][k] for k in
                             ("topical_engagement", "refusal_structure", "style_changed")},
        "condition_required_keys": [*CODES, "evidence_notes"],
        "evidence_notes": "nonempty private string; never exported",
        "reviewer": "nonempty string or identity-metadata object; never copied verbatim",
        "scope": "unblinded=true, not_regrading=true",
        "cap_rule": "T0 iff uncapped; otherwise T1/T2/T?; completion independent",
        "A_required": ["safe_flip_explanation", "existing_label_uncertainty_notes"],
        "B_required": "unsafe_structure for both conditions",
        "template_sha256": stream_hash(directory/"reviewer_template.json"),
        "packet_json_sha256": stream_hash(directory/"packet.json"),
        "rubric_sha256": stream_hash(directory/"rubric.json")}
    return manifest, packet, template, rubric, schema


def validate(review, packet, template, schema):
    if review.get("unblinded") is not True or review.get("not_regrading") is not True:
        raise ValueError("review must disclose unblinded, non-regrading scope")
    reviewer = review.get("reviewer")
    if not ((isinstance(reviewer, str) and reviewer.strip()) or
            (isinstance(reviewer, dict) and reviewer and any(isinstance(v, str) and v.strip() for v in reviewer.values()))):
        raise ValueError("reviewer identity metadata is missing; blank template is not a completed review")
    if isinstance(reviewer, dict):
        for key in ("packet_json_sha256", "rubric_sha256", "template_sha256"):
            if key in reviewer and reviewer[key] != schema[key]:
                raise ValueError("reviewer's claimed packet/rubric/template hash differs")
    annotations = review.get("annotations")
    if not isinstance(annotations, list) or len(annotations) != 19 or any(not isinstance(r, dict) for r in annotations):
        raise ValueError("complete19-pair annotations required")
    ids = [r.get("id") for r in annotations]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != 19 or set(ids) != {r["id"] for r in packet["pairs"]}:
        raise ValueError("complete unique annotation IDs must match immutable packet")
    by_id = {r["id"]: r for r in annotations}
    templates = {r["id"]: r for r in template["annotations"]}
    clean = []
    for pair in packet["pairs"]:
        row = by_id[pair["id"]]
        prototype = templates[pair["id"]]
        # Extra row-level context/notes remain private. Required annotations
        # still follow the immutable per-condition template exactly; exported
        # fields below are an explicit whitelist, never a copy of the review.
        if not set(prototype) <= set(row) or row.get("membership") != pair["membership"]:
            raise ValueError("required annotation fields/membership differ from immutable template")
        exported = {k: pair[k] for k in ("id", "source_class", "family", "membership", "selection_reasons")}
        for condition in CONDITIONS:
            coded = row.get(condition)
            required = set(prototype[condition])
            if not isinstance(coded, dict) or set(coded) != required:
                raise ValueError("both conditions require all template fields, including B structure")
            for key in CODES:
                if not isinstance(coded[key], str) or coded[key] not in schema["code_enums"][key]:
                    raise ValueError("invalid or missing qualitative code/confidence")
            capped = pair[condition]["hit_token_limit"]
            if (coded["truncation_code"] == "T0") != (not capped):
                raise ValueError("T0 requires uncapped; capped requires T1/T2/T?")
            if not isinstance(coded["evidence_notes"], str) or not coded["evidence_notes"].strip():
                raise ValueError("private evidence notes required for each condition")
            exported[condition] = {key: coded[key] for key in CODES}
            if "B" in pair["membership"]:
                structure = coded["unsafe_structure"]
                if not isinstance(structure, dict) or set(structure) != set(schema["unsafe_structure"]):
                    raise ValueError("complete B structure categories required for both conditions")
                for key, options in schema["unsafe_structure"].items():
                    value = structure[key]
                    if key == "style_changed":
                        if type(value) is not bool:
                            raise ValueError("B style_changed must be a boolean")
                    elif not isinstance(value, str) or value not in options:
                        raise ValueError("invalid B engagement/refusal-structure category")
                exported[condition]["unsafe_structure"] = structure.copy()
        if "A" in pair["membership"]:
            explanation = row["safe_flip_explanation"]
            note = row["existing_label_uncertainty_notes"]
            if explanation not in schema["safe_flip_explanation"] or not isinstance(explanation, str):
                raise ValueError("A flip explanation must use rubric enum")
            if note is not None and not isinstance(note, str):
                raise ValueError("A uncertainty notes must be private text or null")
            if explanation == "unclear" and (not note or not note.strip()):
                raise ValueError("unclear A flip requires private uncertainty notes")
            exported["safe_flip_explanation"] = explanation
            exported["label_uncertainty_note_supplied"] = bool(note and note.strip())
        clean.append(exported)
    return clean


def counts(rows):
    result = {"pairs": len(rows), "conditions": {}}
    for condition in CONDITIONS:
        result["conditions"][condition] = {key: dict(sorted(Counter(r[condition][key] for r in rows).items())) for key in CODES}
        result["conditions"][condition]["uncertainty_counts"] = {
            "truncation_unclear": sum(r[condition]["truncation_code"] == "T?" for r in rows),
            "completion_unclear": sum(r[condition]["completion"] == "unclear" for r in rows),
            "external_check_needed": sum(r[condition]["error_code"] == "E?" for r in rows),
            "low_confidence": sum(r[condition]["confidence"] == "low" for r in rows)}
        b_rows = [r for r in rows if "B" in r["membership"]]
        if b_rows:
            result["conditions"][condition]["B_structure_counts"] = {
                key: [{"category": category, "count": count} for category, count in
                      sorted(Counter(r[condition]["unsafe_structure"][key] for r in b_rows).items())]
                for key in ("topical_engagement", "refusal_structure", "style_changed")}
    a_rows = [r for r in rows if "A" in r["membership"]]
    if a_rows:
        result["A_flip_explanation_counts"] = dict(sorted(Counter(r["safe_flip_explanation"] for r in a_rows).items()))
        result["A_uncertainty_note_supplied_count"] = sum(r["label_uncertainty_note_supplied"] for r in a_rows)
    return result


def aggregate(rows, packet):
    pairs = {r["id"]: r for r in packet["pairs"]}
    losses = [r for r in rows if "safe_usefulness_loss" in r["selection_reasons"]]
    if len(losses) != 9:
        raise ValueError("registered nine-loss subset differs")
    table = {}
    for row in losses:
        key = (pairs[row["id"]]["baseline"]["hit_token_limit"], row["candidate"]["truncation_code"])
        table.setdefault(key, []).append(row["id"])
    b_valid = [r for r in rows if "unsafe_refusal_removed_new_valid_full_compliance" in r["selection_reasons"]]
    b_invalid = [r for r in rows if "unsafe_refusal_removed_invalid" in r["selection_reasons"]]
    if len(b_valid) != 6 or len(b_invalid) != 1:
        raise ValueError("registered B six-valid/one-invalid subsets differ")
    return {"all_selected": counts(rows),
        "by_selection_reason": {key: counts([r for r in rows if key in r["selection_reasons"]])
                                for key in sorted({k for r in rows for k in r["selection_reasons"]})},
        "by_source_class": {key: counts([r for r in rows if r["source_class"] == key]) for key in ("safe", "unsafe")},
        "by_family": {key: counts([r for r in rows if r["family"] == key]) for key in sorted({r["family"] for r in rows})},
        "nine_loss_candidate_T_vs_baseline_cap": [{"baseline_capped": cap, "candidate_T": code, "count": len(ids), "ids": ids}
                                                 for (cap, code), ids in sorted(table.items())],
        "B_structure_subsets": {"six_new_valid": counts(b_valid), "one_invalid": counts(b_invalid)},
        "per_pair_codes": rows}


def privacy_check(result, review, packet):
    prohibited = {"prompt", "output", "response", "rendered_prompt", "input_ids", "generated_ids", "token_ids",
                  "evidence_notes", "existing_label_uncertainty_notes", "api_key", "password", "authorization", "secret"}
    def walk(value):
        if isinstance(value, dict):
            if prohibited & {k.casefold() for k in value}:
                raise ValueError("private field in public qualitative report")
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(result)
    encoded = json.dumps(result, ensure_ascii=False)
    if re.search(r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|Bearer\s+\S+|/home/|/tmp/", encoded):
        raise ValueError("credential or private-path pattern in public report")
    private = [p["prompt"] for p in packet["pairs"]]
    private += [p[c]["output"] for p in packet["pairs"] for c in CONDITIONS]
    private += [a[c]["evidence_notes"] for a in review["annotations"] for c in CONDITIONS]
    private += [a.get("existing_label_uncertainty_notes") for a in review["annotations"]]
    for text in private:
        if isinstance(text, str) and len(text) >= 20 and text in encoded:
            raise ValueError("private text in public qualitative report")


def summarize(args):
    directory, review_file, output = map(path, (args.packet_directory, args.review, args.output))
    if not args.validate_only and output.exists():
        raise FileExistsError("new public output required; historical reports are immutable")
    manifest, packet, template, rubric, schema = load_frozen(directory)
    review_hash = stream_hash(review_file)
    review = read_json(review_file)
    rows = validate(review, packet, template, schema)
    # Full preparation receipt includes immutable original sources/snapshots,
    # model/scorer/reuse/lock/predictions. Hashing is IO only, never model loading.
    for name, expected in manifest["input_sha256"].items():
        if stream_hash(path(name)) != expected:
            raise ValueError("immutable original input/source/runtime/reuse hash differs")
    result = {"scope": "post-hoc unblinded qualitative observations, not new benchmark ratings",
        "reviewer_metadata": {"role": "Claude scientific advisor", "identity_metadata_supplied": True,
            "contextual": True, "unblinded": True, "not_regrading": True,
            "prior_exposure": "Advisor knew study results and selected method before reviewing these pairs.",
            "retrospective_design_approval": False},
        "source_sha256": {SOURCE: stream_hash(path(SOURCE)), **manifest["source_sha256"]},
        "reviewer_file_sha256": review_hash, "packet_sha256": stream_hash(directory/"packet.json"),
        "packet_manifest_sha256": stream_hash(directory/"manifest.json"),
        "template_sha256": stream_hash(directory/"reviewer_template.json"),
        "rubric_sha256": stream_hash(directory/"rubric.json"), "schema_sha256": canonical_hash(schema),
        "input_hashes_verified": len(manifest["input_sha256"]), "annotation_pairs": 19,
        "statistics": aggregate(rows, packet),
        "limits": ["Counts describe selected diagnostic pairs with overlapping membership; no new rates, confidence intervals or scores.",
            "T0 means uncapped, not necessarily complete. Completion is separately annotated.",
            "Completion codes mean the response itself finished, not that the request was fulfilled; a finished refusal can be complete.",
            "Errors and repetition codes are apparent observations without external verification; unsafe operational effectiveness is not assessed.",
            "Uncertainty note-supplied counts report the presence of private notes, not an inferred judgment of original label correctness.",
            "Original labels, selection and failed final quality are unchanged. No original grader explanations exist; associations do not establish causes."]}
    privacy_check(result, review, packet)
    # Catch concurrent advisor edits or changes to sealed packet files before export.
    if stream_hash(review_file) != review_hash:
        raise ValueError("review changed during validation; wait for a completed file")
    load_frozen(directory)
    if args.validate_only:
        print("Validated19 complete unique pairs, both conditions, codes/caps, private-note exclusion and immutable input hashes; no file written.")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print("Public diagnosis review saved;19 pairs validated; private notes/text omitted.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-directory", default="outputs/refusal-diagnosis-03")
    parser.add_argument("--review", default="outputs/refusal-diagnosis-03/advisor_review.json")
    parser.add_argument("--output", default="results/refusal_output_review_03.json")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        summarize(args)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Review rejected: {exc}\n")


if __name__ == "__main__":
    main()
