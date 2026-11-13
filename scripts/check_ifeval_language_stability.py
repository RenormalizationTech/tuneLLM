"""Seed-sensitivity diagnostic for IFEval language clauses; never rewrite scores."""
import argparse
from collections import Counter
from importlib.metadata import version

from scripts.common import digest, path, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screens", nargs="+", required=True)
    parser.add_argument("--fixture", default="data/benchmark-screen-01/test.json")
    parser.add_argument("--output", default="results/physics_ifeval_language_stability.json")
    args = parser.parse_args()
    if path(args.output).exists():
        raise FileExistsError(args.output)
    import langdetect
    from langdetect import DetectorFactory
    fixture = read_json(args.fixture)["ifeval"]
    expected_ids = {str(row["key"]) for row in fixture}
    seeds = list(range(20))
    conditions = {}
    original_seed = DetectorFactory.seed
    try:
        for name in args.screens:
            directory = path(name)
            run = read_json(directory / "run.json")
            scores = read_json(directory / "ifeval_scores.json")
            predictions = read_json(directory / "ifeval_predictions.json")
            outputs = {row["id"]: row["output"] for row in predictions}
            records = {row["id"]: row for row in scores["records"]}
            if (not run.get("completed") or run["fixture_sha256"] != digest(args.fixture)
                    or len(outputs) != len(predictions) or len(records) != len(scores["records"])
                    or set(outputs) != expected_ids or set(records) != expected_ids):
                raise ValueError("complete, unique, fixture-aligned predictions/scores required")
            flags = {key: [list(records[key]["instruction_strict"]) for _ in seeds] for key in records}
            cases = []
            for row in fixture:
                identifier = str(row["key"])
                for index, instruction in enumerate(row["instruction_id_list"]):
                    if instruction not in {"language:response_language", "change_case:english_capital", "change_case:english_lowercase"}:
                        continue
                    text = outputs[identifier]
                    target = row["kwargs"][index]["language"] if instruction == "language:response_language" else "en"
                    precondition = (text.isupper() if instruction == "change_case:english_capital" else
                                    text.islower() if instruction == "change_case:english_lowercase" else True)
                    labels, passing = [], []
                    for seed in seeds:
                        DetectorFactory.seed = seed
                        if not precondition:
                            label = "case_precondition_failed"
                        else:
                            try:
                                label = langdetect.detect(text)
                            except langdetect.LangDetectException:
                                label = "detection_failed"
                        labels.append(label)
                        passing.append(precondition and label == target)
                        flags[identifier][seed][index] = passing[-1]
                    cases.append({"id": identifier, "instruction": instruction, "expected_language": target,
                                  "case_precondition": precondition, "label_counts": dict(Counter(labels)),
                                  "language_clause_passes": sum(passing), "language_clause_stable": len(set(passing)) == 1,
                                  "primary_clause_result": records[identifier]["instruction_strict"][index]})
            counts = [sum(all(flags[key][seed]) for key in flags) for seed in seeds]
            conditions[directory.name] = {"primary_prompt_strict_correct": scores["prompt_strict_correct"],
                "seeded_prompt_strict_counts": counts, "seeded_count_range": [min(counts), max(counts)],
                "all_seeded_prompt_counts_equal_primary": all(n == scores["prompt_strict_correct"] for n in counts),
                "cases": cases, "predictions_sha256": digest(directory / "ifeval_predictions.json"),
                "primary_scores_sha256": digest(directory / "ifeval_scores.json")}
    finally:
        DetectorFactory.seed = original_seed
    write_json(args.output, {"scope": "Supplementary language-detector seed sensitivity only; primary official scores retained unchanged",
        "primary_scores_modified": False, "seeds": seeds, "langdetect_version": version("langdetect"),
        "fixture_sha256": digest(args.fixture), "diagnostic_source_sha256": digest(__file__),
        "method": "Replay language and English-case clauses for seeds 0..19; retain other official clause results and recombine prompt-strict flags. No text or labels are altered.",
        "conditions": conditions})
    print(args.output)


if __name__ == "__main__":
    main()
