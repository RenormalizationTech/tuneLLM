"""Standard-library answer/paragraph-ID metrics for the versioned science task.

Semantics reference: AllenAI QASPER v0.3 bundled qasper_evaluator.py (source
SHA256 781aba7cd8e524bef4f0a1b4bf3504e5b02cb1d8d5bf32a8f0a89dfa83e86bfe),
itself attributing normalization/token F1 to SQuAD v1.1. Independently expressed
here; synthetic primitive parity does not establish benchmark equivalence.
Evidence IDs use sets (duplicates invalid in strict JSON); upstream paragraph
F1 instead uses original list lengths, so duplicate inputs differ deliberately.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import string


def normalize_answer(answer):
    text = answer.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(?:a|an|the)\b", " ", text).split())


def answer_f1(prediction, reference):
    left, right = normalize_answer(prediction).split(), normalize_answer(reference).split()
    common = sum((Counter(left) & Counter(right)).values())
    return 2 * common / (len(left) + len(right)) if common else 0.0


def evidence_scores(prediction, reference):
    left, right = set(prediction), set(reference)
    if not left and not right:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    common = len(left & right)
    return {"precision": common / len(left) if left else 0.0,
            "recall": common / len(right) if right else 0.0,
            "f1": 2 * common / (len(left) + len(right))}


def exact_paragraph_f1(prediction, reference):
    """QASPER list-length denominator with exact string intersection (no normalization)."""
    if not prediction and not reference:
        return 1.0
    common = len(set(prediction) & set(reference))
    return 2 * common / (len(prediction) + len(reference)) if common else 0.0


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_response(text, lenient=False):
    """Strict schema, or frozen recovery of one JSON object / plain answer.

    Lenient plain text never invents evidence; only the exact normalized word
    'unanswerable' becomes abstention. No arbitrary keyword/regex extraction.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    candidate = text.strip()
    if lenient:
        fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", candidate)
        if fenced:
            candidate = fenced.group(1)
        elif not candidate.startswith("{"):
            start, stop = candidate.find("{"), candidate.rfind("}")
            if start >= 0 and stop > start:
                candidate = candidate[start:stop + 1]
    try:
        value = json.loads(candidate, object_pairs_hook=_unique_object,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, TypeError):
        if lenient and not any(c in candidate for c in "{}[]"):
            abstain = normalize_answer(candidate) == "unanswerable"
            return {"answer": "" if abstain else candidate, "unanswerable": abstain, "evidence": []}
        return None
    if not isinstance(value, dict) or set(value) != {"answer", "unanswerable", "evidence"}:
        return None
    answer, abstain, evidence = value["answer"], value["unanswerable"], value["evidence"]
    if not isinstance(answer, str) or type(abstain) is not bool or not isinstance(evidence, list):
        return None
    if lenient and abstain:
        value = {"answer": "", "unanswerable": True, "evidence": []}
        answer, evidence = "", []
    if any(not isinstance(e, str) or not re.fullmatch(r"p\d{6}", e) for e in evidence):
        return None
    if len(evidence) != len(set(evidence)) or (abstain and (answer.strip() or evidence)) or (not abstain and not answer.strip()):
        return None
    return value


def score_question(row, text, lenient=False):
    pred = parse_response(text, lenient)
    refs = row["references"]
    if not refs:
        raise ValueError("nonempty references required")
    result = {"id": row["id"], "paper_id": row["paper_id"], "valid": pred is not None,
              "answer_f1": 0.0, "evidence_f1": 0.0, "evidence_precision": 0.0, "evidence_recall": 0.0,
              "exact_evidence_text_f1": 0.0, "exact_evidence_all_f1": 0.0,
              "abstained": False, "unsupported_evidence_ids": [], "out_of_document_evidence_ids": []}
    if pred is None:
        return result
    answer = "Unanswerable" if pred["unanswerable"] else pred["answer"]
    result["answer_f1"] = max(answer_f1(answer, "Unanswerable" if r["unanswerable"] else r["answer"]) for r in refs)
    # Answer and evidence maxima are independent, as in the upstream evaluator.
    # Unmapped *text* evidence stays in the recall denominator. It cannot become
    # a perfect empty-set match. FLOAT SELECTED evidence is excluded explicitly.
    def reference_ids(ref):
        if ref["unanswerable"]:
            return []
        return ref["evidence"] + [f"unmapped-text-{i}" for i, _ in enumerate(ref.get("unmatched_text_evidence", []))]
    paragraphs = {p["id"]: p for p in row["paragraphs"]}
    predicted_ids = [paragraphs.get(e, {}).get("canonical_id", e) for e in pred["evidence"]]
    evidence = max((evidence_scores(predicted_ids, reference_ids(r)) for r in refs), key=lambda s: s["f1"])
    result.update({"evidence_" + k: v for k, v in evidence.items()})
    predicted_text = [paragraphs[e].get("text", e) if e in paragraphs else "INVALID-ID:" + e for e in pred["evidence"]]
    def reference_text(ref):
        if ref["unanswerable"]:
            return []
        if "raw_annotation" in ref:
            return ref["raw_annotation"].get("evidence", [])
        return [paragraphs[e].get("text", e) for e in ref["evidence"] if e in paragraphs] + ref.get("unmatched_text_evidence", [])
    result["exact_evidence_all_f1"] = max(exact_paragraph_f1(predicted_text, reference_text(r)) for r in refs)
    result["exact_evidence_text_f1"] = max(exact_paragraph_f1(predicted_text, [e for e in reference_text(r) if "FLOAT SELECTED" not in e]) for r in refs)
    result["abstained"] = pred["unanswerable"]
    result["unsupported_evidence_ids"] = sorted(set(pred["evidence"]) - set(row["context_ids"]))
    result["out_of_document_evidence_ids"] = sorted(set(pred["evidence"]) - {p["id"] for p in row["paragraphs"]})
    return result


def score_qasper(rows, predictions):
    """All sampled rows remain denominators; missing/invalid outputs score zero.

    Predictions map prepared IDs to raw generated text. Extra IDs are rejected.
    Scores range 0..1. No filtering on observed model outputs is performed.
    """
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate evaluation ID")
    if set(predictions) - {r["id"] for r in rows}:
        raise ValueError("unknown prediction ID")
    report = {"official_equivalence": False, "score_scale": "0..1", "sampled_questions": len(rows),
              "evidence_scope": "custom text paragraph-ID projection: whitespace-normalized and duplicate-content canonicalized; unmapped text penalizes recall",
              "exact_evidence_scope": "upstream exact-string/list-length semantics; both text-only and all-evidence views; no whole-benchmark equivalence claim",
              "missing_predictions": sum(r["id"] not in predictions for r in rows)}
    for mode in ("strict", "lenient"):
        scored = [score_question(r, predictions.get(r["id"]), mode == "lenient") for r in rows]
        groups = {"all_sampled": list(range(len(rows))), "primary": [i for i, r in enumerate(rows) if r["primary"]],
                  "evidence_strict_primary": [i for i, r in enumerate(rows) if r.get("evidence_strict_primary", False)],
                  "nontext_or_unmatched": [i for i, r in enumerate(rows) if r["nontext_or_unmatched_evidence"]],
                  "figure_table": [i for i, r in enumerate(rows) if r.get("nontext_evidence", False)],
                  "unmatched_text": [i for i, r in enumerate(rows) if r.get("unmatched_text_evidence", False)]}
        summaries = {}
        for name, indices in groups.items():
            count = len(indices)
            mean = lambda key: sum(scored[i][key] for i in indices) / count if count else None
            answerable = [i for i in indices if all(not r["unanswerable"] for r in rows[i]["references"])]
            unanswerable = [i for i in indices if all(r["unanswerable"] for r in rows[i]["references"])]
            summaries[name] = {"denominator": count, "answer_f1": mean("answer_f1"), "evidence_f1": mean("evidence_f1"),
                "exact_evidence_text_f1": mean("exact_evidence_text_f1"), "exact_evidence_all_f1": mean("exact_evidence_all_f1"),
                "full_text_in_context": sum(rows[i].get("full_text_in_context", False) for i in indices),
                "annotation_disagreement": sum(rows[i].get("annotation_disagreement", False) for i in indices),
                "missing_context_evidence": sum(any(r.get("missing_context_evidence") for r in rows[i]["references"]) for i in indices),
                "invalid_outputs": sum(not scored[i]["valid"] for i in indices),
                "abstentions": sum(scored[i]["abstained"] for i in indices),
                "answerable_denominator": len(answerable),
                "false_unanswerable_rate": sum(scored[i]["abstained"] for i in answerable) / len(answerable) if answerable else None,
                "unanswerable_denominator": len(unanswerable),
                "unanswerable_accuracy": sum(scored[i]["abstained"] for i in unanswerable) / len(unanswerable) if unanswerable else None,
                "unsupported_evidence_outputs": sum(bool(scored[i]["unsupported_evidence_ids"]) for i in indices)}
        report[mode] = {"groups": summaries, "records": scored}
    return report


def paired_paper_bootstrap(arm_a, arm_b, metric="answer_f1", draws=10000, seed=20261117):
    """Paired mean(a-b), resampling whole papers with replacement.

    The estimand is the mean over question/item records, not equal-paper means.
    A sampled paper contributes all its items (including repeated multiplicity).
    Supply identical frozen subsets to both arms; missing scores are an error.
    PRL paired-review item scores can use the same function with one row/paper.
    Percentile 95% intervals use linear interpolation; this is not a significance
    test or evidence of non-inferiority. A single paper gives a degenerate CI.
    """
    if type(draws) is not int or draws < 1:
        raise ValueError("positive bootstrap draw count required")
    left, right = {r["id"]: r for r in arm_a}, {r["id"]: r for r in arm_b}
    if len(left) != len(arm_a) or len(right) != len(arm_b) or set(left) != set(right) or not left:
        raise ValueError("nonempty identical unique item IDs required")
    groups = {}
    for key in sorted(left):
        a, b = left[key], right[key]
        if a["paper_id"] != b["paper_id"]:
            raise ValueError("paired paper identities differ")
        values = [a[metric], b[metric]]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("finite numeric paired metrics required")
        groups.setdefault(a["paper_id"], []).append(values[0] - values[1])
    clusters = [groups[k] for k in sorted(groups)]
    rng, samples = random.Random(seed), []
    for _ in range(draws):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        samples.append(sum(sum(c) for c in sampled) / sum(len(c) for c in sampled))
    samples.sort()
    def percentile(q):
        position = (len(samples) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return samples[low] + (samples[high] - samples[low]) * (position - low)
    return {"difference_a_minus_b": sum(sum(c) for c in clusters) / len(left),
            "ci95": [percentile(.025), percentile(.975)], "papers": len(clusters), "items": len(left),
            "draws": draws, "seed": seed, "estimand": "item mean; paired paper-cluster percentile bootstrap"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True, help="prepared question JSONL")
    p.add_argument("--predictions", required=True, help="JSON object mapping prepared IDs to raw text")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    prepared, predictions = Path(a.prepared).read_bytes(), Path(a.predictions).read_bytes()
    rows = [json.loads(s) for s in prepared.decode().splitlines() if s.strip()]
    report = score_qasper(rows, json.loads(predictions))
    report["input_sha256"] = {"prepared": hashlib.sha256(prepared).hexdigest(),
                              "predictions": hashlib.sha256(predictions).hexdigest()}
    report["scorer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with Path(a.output).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)


if __name__ == "__main__":
    main()
