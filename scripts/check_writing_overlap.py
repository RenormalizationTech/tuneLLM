"""Flag exact word-shingle overlap with the local training corpus; not a memorization test."""
import argparse
import hashlib
import re

from scripts.common import digest, path, read_json, write_json
from scripts.physics_data import read_papers


def words(text):
    return re.findall(r"[\w]+", text.casefold())


def shingles(tokens, width):
    for i in range(max(0, len(tokens) - width + 1)):
        yield hashlib.blake2b(" ".join(tokens[i:i + width]).encode(), digest_size=16).digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", default="data/physics-prl-01/train.jsonl")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if path(args.output).exists():
        raise FileExistsError(args.output)
    corpus = [words(row["text"]) for row in read_papers(args.training)]
    index = {width: {s for paper in corpus for s in shingles(paper, width)} for width in [10, 25]}
    records = []
    for row in read_json(args.predictions):
        tokens = words(row["output"])
        record = {"id": row["id"], "words": len(tokens)}
        for width, known in index.items():
            current = list(shingles(tokens, width))
            matches = sum(s in known for s in current)
            record[f"matching_{width}grams"] = matches
            record[f"fraction_matching_{width}grams"] = matches / len(current) if current else None
        records.append(record)
    write_json(args.output, {"training_sha256": digest(args.training), "predictions_sha256": digest(args.predictions),
        "normalization": "casefold; Unicode word tokens; exact within-paper shingles hashed with BLAKE2b-128",
        "outputs_with_25gram_overlap": sum(r["matching_25grams"] > 0 for r in records), "records": records,
        "limits": "Overlap may reflect common scientific language, supplied examples or pretraining. Absence of exact overlap does not establish originality; no causal memorization or plagiarism judgment."})


if __name__ == "__main__":
    main()
