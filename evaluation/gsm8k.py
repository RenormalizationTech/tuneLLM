"""Numerical GSM8K smoke scoring; strict final marker and fallback are separate."""
from decimal import Decimal, InvalidOperation
import re

NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
FINAL = re.compile(rf"####\s*({NUMBER})\s*$")
LAST = re.compile(NUMBER)


def number(value):
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None


def score_gsm8k(references, predictions):
    if not references or len({r["id"] for r in references}) != len(references):
        raise ValueError("references must be nonempty with unique IDs")
    if len({r["id"] for r in predictions}) != len(predictions) or {r["id"] for r in references} != {r["id"] for r in predictions}:
        raise ValueError("predictions must match references exactly")
    by_id = {r["id"]: r for r in predictions}
    details = []
    for row in references:
        expected = FINAL.search(row["assistant"].strip().splitlines()[-1])
        if not expected:
            raise ValueError(f"missing reference final answer: {row['id']}")
        prediction = by_id[row["id"]]
        text = prediction["output"].strip()
        final = FINAL.fullmatch(text.splitlines()[-1]) if text else None
        all_numbers = LAST.findall(text)
        target = number(expected.group(1))
        details.append({"id": row["id"], "strict_correct": bool(final and number(final.group(1)) == target),
                        "fallback_correct": bool(all_numbers and number(all_numbers[-1]) == target),
                        "format_compliant": bool(final), "hit_token_limit": prediction["hit_token_limit"]})
    totals = {key: sum(int(row[key]) for row in details)
              for key in ("strict_correct", "fallback_correct", "format_compliant", "hit_token_limit")}
    return {"total": len(details), **totals, "strict_accuracy": totals["strict_correct"] / len(details),
            "fallback_accuracy": totals["fallback_correct"] / len(details), "details": details,
            "caveat": "Small deterministic pipeline check, not evidence of improvement; last-number fallback can reward noncompliant answers."}
