"""Freeze disjoint development/test screens from pinned local benchmark snapshots."""
from collections import defaultdict
import json
import random
import re

from scripts.common import digest, path, write_json

MMLU = "c30699e8356da336a370243923dbaf21066bb9fe"
IFEVAL = "966cd89545d6b6acfd7638bc708b98261ca58e84"
PHYSICS = {"college_physics", "high_school_physics", "conceptual_physics"}


def stratified(rows, count, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["subject"]].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    order = sorted(groups)
    rng.shuffle(order)
    chosen = []
    while len(chosen) < count and any(groups.values()):
        for subject in order:
            if groups[subject] and len(chosen) < count:
                chosen.append(groups[subject].pop())
    if len(chosen) != count:
        raise ValueError("insufficient stratified rows")
    return chosen


def multiple_choice(row, split, index):
    subject = row["subject"].replace("_", " ")
    prompt = f"The following are multiple choice questions (with answers) about {subject}.\n\n{row['question']}\n"
    prompt += "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", row["choices"]))
    prompt += "\nAnswer:"
    return {"id": f"mmlu-{split}-{index:05d}", "subject": row["subject"], "prompt": prompt,
            "choices": [" A", " B", " C", " D"], "answer_index": row["answer"],
            "source_split": split, "source_row": index}


def main():
    import pyarrow.parquet as pq
    out = path("data/benchmark-screen-01")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    snap = path(f".hf/hub/datasets--cais--mmlu/snapshots/{MMLU}/all")
    sources = []
    fixtures = {}
    for split, count_general, count_physics in [("development", 32, 32), ("test", 128, 96)]:
        parts = ["dev", "validation"] if split == "development" else ["test"]
        rows = []
        for part in parts:
            file = snap / f"{part}-00000-of-00001.parquet"
            sources.append({"file": str(file.relative_to(path('.'))), "sha256": digest(file)})
            rows += [multiple_choice(r, part, i) for i, r in enumerate(pq.read_table(file).to_pylist())]
        physics = [r for r in rows if r["subject"] in PHYSICS]
        general = [r for r in rows if r["subject"] not in PHYSICS]
        general = stratified(general, count_general, 4701)
        physics = stratified(physics, count_physics, 4702)
        gsmfile = path("data/gsm8k/validation.jsonl" if split == "development" else "data/gsm8k/test.jsonl")
        gsm = [json.loads(line) for line in gsmfile.read_text().split("\n") if line]
        random.Random(4703).shuffle(gsm)
        gsm = gsm[:32 if split == "development" else 128]
        fixtures[split] = {"general_mc": general, "physics_mc": physics, "gsm8k": gsm}
    ifeval_file = path(f".hf/hub/datasets--google--IFEval/snapshots/{IFEVAL}/ifeval_input_data.jsonl")
    ifeval = [json.loads(line) for line in ifeval_file.read_text().split("\n") if line]
    # Avoid prompts asking the offline model to fetch a URL; otherwise preserve tasks unchanged.
    ifeval = [row for row in ifeval if not re.search(r"https?://", row["prompt"])]
    random.Random(4704).shuffle(ifeval)
    fixtures["development"]["ifeval"] = ifeval[:16]
    fixtures["test"]["ifeval"] = ifeval[16:48]
    scorer = path("artifacts/tools/ifeval")
    provenance = {"mmlu_revision": MMLU, "ifeval_revision": IFEVAL,
        "ifeval_scorer": json.loads((scorer / "source.json").read_text()),
        "ifeval_source_sha256": {f.name: digest(f) for f in scorer.glob("*.py")},
        "source_files": sources + [{"file": str(ifeval_file.relative_to(path('.'))), "sha256": digest(ifeval_file)}],
        "multiple_choice_protocol": "zero-shot raw standard MMLU prompt; summed conditional log-likelihood of answer letters; custom local runner, not a published full MMLU score",
        "ifeval_protocol": "official prompt-level strict and instruction-level strict, offline non-URL subset",
        "seed": [4701, 4702, 4703, 4704], "writing": "added separately from held-out physics papers before training"}
    for name, fixture in fixtures.items():
        write_json(out / f"{name}.json", {**fixture, "provenance": provenance})
    write_json(out / "provenance.json", {**provenance, "fixture_sha256": {name: digest(out / f"{name}.json") for name in fixtures},
        "counts": {name: {task: len(rows) for task, rows in fixture.items()} for name, fixture in fixtures.items()}})
    print(out)


if __name__ == "__main__":
    main()
