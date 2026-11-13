"""Paper-level split checks, within-paper causal chunks and token-weighted NLL."""
import json
import math
from scripts.common import path


def read_papers(filename):
    return [json.loads(line) for line in path(filename).read_text(encoding="utf-8").split("\n") if line.strip()]


def validate_splits(splits, expected_counts=None):
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("train/validation/test paper splits required")
    ids, papers = set(), set()
    for split, rows in splits.items():
        if not rows or (expected_counts is not None and len(rows) != expected_counts[split]):
            raise ValueError(f"unexpected paper count: {split}")
        for row in rows:
            if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("id", "paper_id", "text")):
                raise ValueError("paper requires nonempty string id, paper_id, text")
            if row["id"] in ids or row["paper_id"] in papers:
                raise ValueError("one row per paper; duplicate ID/paper within or across splits")
            ids.add(row["id"])
            papers.add(row["paper_id"])


def chunk_paper(row, tokenizer, max_length):
    if max_length < 2 or tokenizer.eos_token_id is None:
        raise ValueError("sequence length >=2 and EOS required")
    ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
    if not ids or tokenizer.eos_token_id in ids:
        raise ValueError("paper must have tokens and contain no embedded EOS")
    ids = ids + [tokenizer.eos_token_id]
    result, start = [], 0
    while start < len(ids) - 1:
        block = ids[start:start + max_length]
        result.append({"id": f"{row['id']}::chunk{len(result):04d}", "paper_id": row["paper_id"],
                       "input_ids": block, "attention_mask": [1] * len(block),
                       "labels": [-100] + block[1:], "loss_tokens": len(block) - 1,
                       "paper_token_start": start, "paper_token_stop": start + len(block),
                       "repeated_left_context_tokens": int(start > 0)})
        start += len(block) - 1
    return result


def chunk_papers(rows, tokenizer, max_length):
    return [chunk for row in rows for chunk in chunk_paper(row, tokenizer, max_length)]


def exposure(chunks, max_length):
    return {"chunks": len(chunks), "paper_count": len({row["paper_id"] for row in chunks}),
            "loss_tokens_per_epoch": sum(row["loss_tokens"] for row in chunks),
            "input_tokens_per_epoch": sum(len(row["input_ids"]) for row in chunks),
            "repeated_context_tokens_per_epoch": sum(row["repeated_left_context_tokens"] for row in chunks),
            "remainder_chunks": sum(len(row["input_ids"]) < max_length for row in chunks),
            "policy": "Plain causal LM; no chat template or assistant mask. One token of within-paper overlap preserves every target after the first paper token. EOS only at paper end. No cross-paper packing, dropped tails or silent truncation."}


def aggregate_nll(records):
    papers = {}
    for record in records:
        count, nll = record["loss_tokens"], record["nll_sum"]
        if count <= 0 or not math.isfinite(nll) or nll < 0:
            raise ValueError("positive target count and finite nonnegative NLL required")
        paper = papers.setdefault(record["paper_id"], {"loss_tokens": 0, "nll_sum": 0.0, "chunks": 0})
        paper["loss_tokens"] += count
        paper["nll_sum"] += nll
        paper["chunks"] += 1
    if not papers:
        raise ValueError("nonempty NLL records required")
    for paper in papers.values():
        paper["nll"] = paper["nll_sum"] / paper["loss_tokens"]
    count = sum(p["loss_tokens"] for p in papers.values())
    nll = sum(p["nll_sum"] for p in papers.values()) / count
    return {"loss_tokens": count, "token_weighted_nll": nll,
            "perplexity": math.exp(nll) if nll < 709 else None,
            "mean_per_paper_nll": sum(p["nll"] for p in papers.values()) / len(papers),
            "per_paper": papers}
