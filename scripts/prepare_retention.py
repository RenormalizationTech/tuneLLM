"""Prepare a fixed non-math MMLU retention fixture from cached parquet, offline."""
import argparse
from collections import defaultdict
import random
from scripts.common import digest, path, read_json, write_json

REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


def eligible_subject(subject):
    return "math" not in subject.casefold() and subject not in {"abstract_algebra", "high_school_statistics"}


def select_retention(rows, seed=1234, count=128):
    subjects = defaultdict(list)
    for index, row in enumerate(rows):
        if eligible_subject(row["subject"]):
            subjects[row["subject"]].append((index, row))
    order = sorted(subjects)
    if not order or count <= 0:
        raise ValueError("eligible subjects and positive count required")
    rng = random.Random(seed)
    rng.shuffle(order)
    base, remainder = divmod(count, len(order))
    selected, allocation = [], {}
    for position, subject in enumerate(order):
        allocation[subject] = base + int(position < remainder)
        for index, row in rng.sample(subjects[subject], allocation[subject]):
            if len(row["choices"]) != 4 or row["answer"] not in range(4):
                raise ValueError("expected four choices and integer answer 0..3")
            choices = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", row["choices"]))
            selected.append({"id": f"mmlu-all-test-{index:05d}", "source_row": index,
                             "subject": subject, "answer": "ABCD"[row["answer"]],
                             "user": f"{row['question']}\n\n{choices}\n\nAnswer with only the letter."})
    rng.shuffle(selected)
    return selected, {"subject_order": order, "allocation": allocation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", default="outputs/mmlu_snapshot.json")
    parser.add_argument("--output", default="data/mmlu/retention.json")
    args = parser.parse_args()
    source = read_json(args.snapshot_manifest)
    if source["dataset_id"] != "cais/mmlu" or source["revision"] != REVISION or source["config"] != "all" or source["split"] != "test":
        raise ValueError("unexpected MMLU source/config/split/revision")
    import pyarrow.parquet as pq
    parquet = path(source["snapshot"]) / "all/test-00000-of-00001.parquet"
    tasks, allocation = select_retention(pq.read_table(parquet).to_pylist())
    out = path(args.output)
    if out.exists() or out.with_name("provenance.json").exists():
        raise FileExistsError("refusing to replace fixture or provenance")
    provenance = {"source": source, "parquet_sha256": digest(parquet), "seed": 1234,
                  "total": 128, "excluded_subject_rule": "subject contains math (case-insensitive), or is abstract_algebra or high_school_statistics",
                  "selection": "sort subjects, seeded shuffle for allocation, seeded sample per subject, final seeded shuffle using one Python Random(1234) stream",
                  "selected_ids": [row["id"] for row in tasks], **allocation,
                  "purpose": "Final retention only; never training, validation, or checkpoint selection"}
    write_json(out, {"tasks": tasks, "provenance": provenance})
    write_json(out.with_name("provenance.json"), {**provenance, "fixture_sha256": digest(out)})
    print(out)


if __name__ == "__main__":
    main()
