"""Strict A-D choice scoring and separately labelled first standalone-letter fallback."""
import argparse
import re
import unicodedata
from scripts.common import read_json, write_json


def score_retention(rows, predictions):
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("unique nonempty reference IDs required")
    if len({p["id"] for p in predictions}) != len(predictions) or {r["id"] for r in rows} != {p["id"] for p in predictions}:
        raise ValueError("predictions must match reference IDs exactly")
    by_id = {p["id"]: p for p in predictions}
    details = []
    for row in rows:
        if row["answer"] not in ("A", "B", "C", "D"):
            raise ValueError("reference answer must be A, B, C or D")
        prediction = by_id[row["id"]]
        text = unicodedata.normalize("NFKC", prediction["output"]).strip()
        strict = text.upper() if text.upper() in ("A", "B", "C", "D") else None
        candidates = [strict] if strict else re.findall(r"\b[A-D]\b", text)
        fallback = candidates[0] if candidates else None
        details.append({"id": row["id"], "strict_correct": strict == row["answer"],
                        "fallback_correct": fallback == row["answer"], "invalid": strict is None,
                        "fallback_invalid": fallback is None, "fallback_ambiguous": len(set(candidates)) > 1,
                        "strict_choice": strict, "fallback_choice": fallback,
                        "hit_token_limit": prediction["hit_token_limit"]})
    totals = {key: sum(int(row[key]) for row in details) for key in
              ("strict_correct", "fallback_correct", "invalid", "fallback_invalid", "fallback_ambiguous", "hit_token_limit")}
    return {"total": len(rows), **totals, "strict_accuracy": totals["strict_correct"] / len(rows),
            "fallback_accuracy": totals["fallback_correct"] / len(rows), "details": details,
            "scoring": "NFKC + strip. Strict: entire output is A-D, case-insensitive. Fallback: strict choice if valid, otherwise first case-preserved word-boundary uppercase A-D; lowercase prose is never uppercased. Ambiguity and invalid outputs reported separately."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predictions = read_json(args.predictions)
    if isinstance(predictions, dict):
        predictions = predictions["predictions"]
    write_json(args.output, score_retention(read_json(args.fixture)["tasks"], predictions))


if __name__ == "__main__":
    main()
