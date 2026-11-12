"""Offline GSM8K parquet -> disjoint single-turn JSONL and provenance; no downloads."""
import argparse
import json
import random
import re
from evaluation.score import normalize
from scripts.common import digest, path, read_json, write_json

INSTRUCTION = "Solve the problem. End your response with a final line in the form #### N, where N is the numerical answer."


def prepare_records(train, test, seed, validation_size):
    """Rows retain upstream split/row IDs; held-out test is preserved in full."""
    counts = {"train_test_overlap_removed": 0, "train_duplicates_removed": 0,
              "calculator_annotations_removed": 0, "source_train_rows": len(train),
              "source_test_rows": len(test)}
    test_questions = {normalize(row["question"]) for row in test}
    if len(test_questions) != len(test):
        raise ValueError("duplicate test questions: preserve source and resolve before preparing")
    seen, cleaned = set(), []

    def convert(row, source_split, index):
        answer, removed = re.subn(r"<<.*?>>", "", row["answer"], flags=re.DOTALL)
        counts["calculator_annotations_removed"] += removed
        return {"id": f"gsm8k-main-{source_split}-{index:05d}",
                "source_id": {"split": source_split, "row": index},
                "user": row["question"].strip() + "\n\n" + INSTRUCTION,
                "assistant": answer.strip()}

    test_rows = [convert(row, "test", i) for i, row in enumerate(test)]
    for i, row in enumerate(train):
        question = normalize(row["question"])
        if question in test_questions:
            counts["train_test_overlap_removed"] += 1
        elif question in seen:
            counts["train_duplicates_removed"] += 1
        else:
            seen.add(question)
            cleaned.append(convert(row, "train", i))
    if not 0 < validation_size < len(cleaned):
        raise ValueError("validation_size must leave nonempty train and validation sets")
    random.Random(seed).shuffle(cleaned)
    splits = {"train": cleaned[validation_size:], "validation": cleaned[:validation_size], "test": test_rows}
    return splits, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", default="outputs/gsm8k_snapshot.json")
    parser.add_argument("--output", default="data/gsm8k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--smoke-train-rows", type=int, default=8)
    args = parser.parse_args()
    manifest = read_json(args.snapshot_manifest)
    if manifest["dataset_id"] != "openai/gsm8k" or manifest["revision"] != "740312add88f781978c0658806c59bc2815b9866" or manifest["config"] != "main":
        raise ValueError("unexpected GSM8K source/config/revision")
    import pyarrow.parquet as pq
    snapshot = path(manifest["snapshot"])
    raw, files = {}, {}
    for split in ("train", "test"):
        shards = sorted((snapshot / "main").glob(f"{split}-*.parquet"))
        if not shards:
            raise ValueError(f"no local {split} parquet files in {snapshot / 'main'}")
        raw[split] = []
        for shard in shards:
            files[str(shard.relative_to(snapshot))] = digest(shard)
            raw[split].extend(pq.read_table(shard).to_pylist())
    splits, counts = prepare_records(raw["train"], raw["test"], args.seed, args.validation_size)
    if not 0 < args.smoke_train_rows <= len(splits["train"]):
        raise ValueError("smoke subset must contain only available training rows")
    out = path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in {**splits, "smoke_train": splits["train"][:args.smoke_train_rows]}.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    write_json(out / "provenance.json", {
        "source": manifest, "source_parquet_sha256": files, "seed": args.seed,
        "validation_size": args.validation_size, "question_normalization": "NFKC + casefold + whitespace",
        "counts": counts, "instruction": INSTRUCTION,
        "answer_processing": "Remove <<...>> calculator annotations; retain reasoning and final #### answer",
        "split_ids": {name: [row["id"] for row in rows] for name, rows in splits.items()},
        "smoke_train_ids": [row["id"] for row in splits["train"][:args.smoke_train_rows]],
        "output_sha256": {p.name: digest(p) for p in sorted(out.glob("*.jsonl"))},
    })
    print(json.dumps({"output": str(out), "rows": {k: len(v) for k, v in splits.items()}, "counts": counts}))


if __name__ == "__main__":
    main()
