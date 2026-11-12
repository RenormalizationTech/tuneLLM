"""Shared exact-answer scoring. Predictions JSON: [{id, output, ...}, ...]."""
import argparse
import unicodedata
from scripts.common import read_json, write_json


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def score(tasks, predictions):
    expected = [task["id"] for task in tasks]
    received = [prediction["id"] for prediction in predictions]
    if not tasks or len(set(expected)) != len(expected):
        raise ValueError("fixture must have nonempty, unique task IDs")
    if len(set(received)) != len(received) or set(received) != set(expected):
        raise ValueError("predictions must match fixture IDs exactly, without duplicates")
    by_id = {prediction["id"]: prediction for prediction in predictions}
    details, categories = [], {}
    for task in tasks:
        prediction = by_id[task["id"]]
        correct = normalize(prediction["output"]) in {normalize(a) for a in task["answers"]}
        details.append({"id": task["id"], "correct": correct})
        counts = categories.setdefault(task["category"], {"correct": 0, "total": 0})
        counts["correct"] += int(correct)
        counts["total"] += 1
    correct = sum(row["correct"] for row in details)
    return {"scoring": "NFKC + casefold + whitespace normalization; exact full answer",
            "correct": correct, "total": len(tasks), "accuracy": correct / len(tasks),
            "categories": categories, "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="evaluation/small_tasks.json")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_json(args.output, score(read_json(args.fixture)["tasks"], read_json(args.predictions)))


if __name__ == "__main__":
    main()
