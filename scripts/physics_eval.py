"""Paper-text NLL evaluation; test use requires an explicitly authorized frozen protocol."""
import argparse
import os
import sys
import time
from scripts.common import ROOT, path, read_json, select_gpu, setup_hf, write_json
from scripts.physics_data import aggregate_nll, chunk_papers, read_papers
from scripts.training_data import AssistantCollator


def evaluate_chunks(model, chunks, tokenizer, batch_size=1, device="cuda:0"):
    import torch
    import torch.nn.functional as functional
    was_training = model.training
    model.eval()
    records = []
    collator = AssistantCollator(tokenizer.pad_token_id)
    started = time.perf_counter()
    try:
        for start in range(0, len(chunks), batch_size):
            members = chunks[start:start + batch_size]
            inputs = {key: value.to(device) for key, value in collator(members).items()}
            with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type,
                                                        dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
                logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False).logits
                labels = inputs["labels"][:, 1:]
                losses = functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                                   labels.reshape(-1), ignore_index=-100, reduction="none").reshape(labels.shape)
                totals = losses.double().sum(dim=1).cpu().tolist()
                counts = (labels != -100).sum(dim=1).cpu().tolist()
            for row, nll, count in zip(members, totals, counts):
                if count != row["loss_tokens"]:
                    raise RuntimeError("evaluation target count differs from prepared chunk")
                records.append({"paper_id": row["paper_id"], "chunk_id": row["id"], "nll_sum": nll, "loss_tokens": count})
    finally:
        model.train(was_training)
    return {**aggregate_nll(records), "chunk_records": records,
            "evaluation_wall_seconds": time.perf_counter() - started,
            "scope": "GPU forward uses explicit BF16 autocast before/after training and in standalone evaluation; per-target FP32 cross entropy, FP64 sum. Plain within-paper causal chunks including final EOS, excluding padding and context-only first position"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--adapter")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from scripts.physics_train import verify_prepared, load_base
    prepared = path(args.prepared).resolve()
    cfg, _ = verify_prepared(prepared, args.protocol)
    if args.split == "test" and not cfg["allow_test_evaluation"]:
        raise ValueError("frozen protocol does not authorize test evaluation")
    output = path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if ROOT.resolve() != (prepared / "source").resolve():
        command = [sys.executable, "-m", "scripts.physics_eval", "--prepared", str(prepared),
                   "--protocol", str(path(args.protocol).resolve()), "--split", args.split, "--output", str(output)]
        if args.adapter:
            command.extend(["--adapter", str(path(args.adapter).resolve())])
        os.chdir(prepared / "source")
        os.execv(sys.executable, command)
    select_gpu("1")
    setup_hf(cfg)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(prepared / "inputs/tokenizer", local_files_only=True)
    chunks = chunk_papers(read_papers(prepared / f"inputs/{args.split}.jsonl"), tokenizer, cfg["max_sequence_length"])
    model = load_base(cfg, training=False)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, path(args.adapter), is_trainable=False, local_files_only=True)
    write_json(output, {"split": args.split, "adapter": args.adapter, "config": cfg,
                        "metrics": evaluate_chunks(model, chunks, tokenizer, cfg["eval_batch_size"])})


if __name__ == "__main__":
    main()
