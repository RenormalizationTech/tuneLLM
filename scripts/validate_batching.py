"""Compare batch sizes 1 and 8 on eight fixed validation rows at the same prepared NF4 representation."""
import argparse
import random
from evaluation.gsm8k import score_gsm8k
from scripts.batched_generation import REPRESENTATION, generate_batched, load_nf4
from scripts.common import (digest, load_config, local_snapshot, path, provenance, read_json,
                            select_gpu, setup_hf, write_json)
from scripts.training_data import load_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qlora.json")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["gpu"] = args.gpu
    select_gpu(cfg["gpu"])
    setup_hf(cfg)
    splits = load_splits(cfg["splits"])
    data_manifest = read_json(path(cfg["splits"]["validation"]).parent / "provenance.json")
    for filename in cfg["splits"].values():
        if digest(filename) != data_manifest["output_sha256"][path(filename).name]:
            raise ValueError("prepared split checksum mismatch")
    rows = list(splits["validation"])
    random.Random(42).shuffle(rows)
    rows = rows[:8]
    if len(rows) != 8:
        raise ValueError("eight validation rows required")
    out = path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    metadata = provenance(cfg)
    metadata.update({"representation": REPRESENTATION, "seed": 42,
                     "validation_sha256": digest(cfg["splits"]["validation"]),
                     "selected_validation_ids": [row["id"] for row in rows],
                     "status": "running", "completed": False,
                     "caveat": "Numerical batching effects can change greedy tokens; record mismatches rather than assuming batch invariance. Timing order is batch1 then batch8, no warmup."})
    write_json(out / "run.json", metadata)
    write_json(out / "validation_rows.json", rows)
    snapshot = local_snapshot(cfg)
    from transformers import AutoTokenizer, set_seed
    import torch
    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model = load_nf4(snapshot)
    outputs, scores = {}, {}
    for size in (1, 8):
        outputs[size] = generate_batched(model, tokenizer, rows, batch_size=size, max_new_tokens=512,
                                         seed=42, phase=f"batch{size}")
        scores[size] = score_gsm8k(rows, outputs[size]["predictions"])
        write_json(out / f"batch{size}.json", {**outputs[size], "scores": scores[size]})
    mismatches = []
    for index, row in enumerate(rows):
        one, eight = outputs[1]["predictions"][index], outputs[8]["predictions"][index]
        score_one, score_eight = scores[1]["details"][index], scores[8]["details"][index]
        ids_one, ids_eight = one["generated_ids"], eight["generated_ids"]
        first = next((i for i in range(max(len(ids_one), len(ids_eight)))
                      if (ids_one[i] if i < len(ids_one) else None) != (ids_eight[i] if i < len(ids_eight) else None)), None)
        mismatches.append({"id": row["id"], "exact_tokens_match": ids_one == ids_eight,
                           "first_token_mismatch_index": first,
                           "scoring_matches": score_one == score_eight,
                           "batch1_scores": score_one, "batch8_scores": score_eight})
    write_json(out / "comparison.json", {"details": mismatches,
        "token_mismatch_rows": sum(not row["exact_tokens_match"] for row in mismatches),
        "scoring_mismatch_rows": sum(not row["scoring_matches"] for row in mismatches),
        "timing": {str(size): outputs[size]["timing"] for size in (1, 8)}})
    metadata.update({"status": "complete", "completed": True})
    write_json(out / "run.json", metadata)
    print(out)


if __name__ == "__main__":
    main()
