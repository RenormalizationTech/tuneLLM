"""Hash-bound single-configuration science QLoRA; prepare on CPU, execute explicitly.

Selection uses token-weighted development assistant NLL. No final-test text is
loaded during training. Frozen source is replayed in a new process on execution.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from scripts.qwen38_science_runtime import sha, write, versions

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ["scripts/qwen38_train_science.py", "scripts/qwen38_science_runtime.py",
                "scripts/qwen38_nf4_smoke.py", "scripts/training_data.py", "scripts/common.py"]


def paper_subset(rows, target, seed=42):
    if target <= 0 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("positive target and unique records required")
    papers = defaultdict(list)
    for row in rows:
        papers[row["paper_id"]].append(row)
    order = sorted(papers); random.Random(seed).shuffle(order)
    selected = []
    for paper in order:
        selected.extend(sorted(papers[paper], key=lambda r: r["id"]))
        if len(selected) >= target:
            break
    if len(selected) < target:
        raise ValueError("insufficient eligible paper-grouped examples")
    return selected


def checkpoint_steps(steps, epochs, steps_per_epoch, task):
    if task == "prl":
        return [steps_per_epoch * n for n in range(1, epochs + 1)]
    return sorted({max(1, math.ceil(steps * fraction / 3)) for fraction in (1, 2, 3)})


def prepare(args):
    from transformers import AutoTokenizer
    from scripts.training_data import encode_row, OverlengthError
    out = (ROOT / args.output).resolve()
    prepared = (ROOT / args.prepared).resolve()
    if out.exists() or not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("fresh output inside project outputs required")
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest["task"] != args.task:
        raise ValueError("prepared task identity mismatch")
    base = ROOT / "artifacts/qwen38-27b-nf4-01"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    selected, encoded, excluded = {}, {}, {}
    inputs = {}
    for split, count in (("train", args.train_count), ("validation", args.validation_count)):
        source = prepared / f"{split}.jsonl"
        if sha(source) != manifest["output_sha256"][source.name]:
            raise ValueError("prepared task data changed")
        inputs[split] = sha(source)
        rows = [json.loads(line) for line in source.read_text().split("\n") if line.strip()]
        eligible, encoded_by_id, exclusions = [], {}, []
        for row in rows:
            if not row["training_eligible"]:
                exclusions.append({"id": row["id"], "reason": "grounding/annotation audit not eligible"})
                continue
            try:
                tokens = encode_row(row, tokenizer, args.max_length)
            except OverlengthError:
                exclusions.append({"id": row["id"], "reason": "full prompt+target exceeds cap"})
                continue
            eligible.append(row); encoded_by_id[row["id"]] = tokens
        selected[split] = paper_subset(eligible, count, 42)
        encoded[split] = [{"id": r["id"], "paper_id": r["paper_id"], **encoded_by_id[r["id"]]}
                          for r in selected[split]]
        excluded[split] = exclusions
    if {r["paper_id"] for r in selected["train"]} & {r["paper_id"] for r in selected["validation"]}:
        raise ValueError("training/development paper overlap")
    epochs = 1 if args.task == "qasper" else 3
    steps_per_epoch = math.ceil(len(encoded["train"]) / 8)
    steps = steps_per_epoch * epochs
    cfg = {"task": args.task, "base": str(base), "repo_root": str(ROOT), "seed": 42,
           "versions": versions(), "python_executable": sys.executable,
           "rank": 16, "alpha": 32, "dropout": 0.05, "learning_rate": 5e-5,
           "weight_decay": 0.01, "max_grad_norm": 1.0, "warmup_steps": math.ceil(steps * 0.05),
           "gradient_accumulation": 8, "epochs": epochs, "max_length": args.max_length,
           "steps_per_epoch": steps_per_epoch, "total_steps": steps, "gpu": args.gpu,
           "checkpoint_steps": checkpoint_steps(steps, epochs, steps_per_epoch, args.task),
           "max_seconds": 43200 if args.task == "qasper" else 21600,
           "optimization_weighting": "equal-example mean loss per accumulation group; partial groups divided by actual size",
           "selection": "minimum token-weighted development assistant/EOS NLL; earliest step breaks ties",
           "training_counts": {s: len(rows) for s, rows in encoded.items()},
           "prepared_manifest_sha256": sha(prepared / "manifest.json"), "prepared_input_sha256": inputs,
           "nf4_artifact_manifest": str(ROOT / "results/qwen38_27b_nf4_artifact_01.json"),
           "nf4_artifact_manifest_sha256": sha(ROOT / "results/qwen38_27b_nf4_artifact_01.json")}
    out.mkdir(parents=True)
    (out / "inputs").mkdir(); (out / "source/scripts").mkdir(parents=True)
    for split, rows in encoded.items():
        (out / f"inputs/{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for filename in SOURCE_FILES:
        shutil.copyfile(ROOT / filename, out / "source" / filename)
    (out / "source/scripts/__init__.py").write_text("")
    write(out / "config.json", cfg)
    protocol = {"config": cfg, "selected_ids": {s: [r["id"] for r in rows] for s, rows in selected.items()},
                "exclusions": excluded, "source_sha256": {f: sha(out / "source" / f) for f in SOURCE_FILES},
                "input_sha256": {s: sha(out / f"inputs/{s}.jsonl") for s in encoded},
                "target_tokens": {s: sum(sum(x != -100 for x in r["labels"]) for r in rows) for s, rows in encoded.items()},
                "test_data_loaded_by_trainer": False}
    registration = ROOT / "results/protocols" / f"{out.name}.json"
    if registration.exists():
        raise FileExistsError("training registration already exists")
    registration.parent.mkdir(parents=True, exist_ok=True)
    write(registration, protocol)
    write(out / "protocol.json", protocol)
    for file in list((out / "inputs").glob("*")) + list((out / "source/scripts").glob("*")) + [out / "config.json", out / "protocol.json"]:
        file.chmod(0o444)
    return {"prepared": str(out.relative_to(ROOT)), "protocol_sha256": sha(out / "protocol.json"),
            "counts": cfg["training_counts"], "checkpoint_steps": cfg["checkpoint_steps"]}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def validate_prepared(out):
    protocol = json.loads((out / "protocol.json").read_text())
    if json.loads((out / "config.json").read_text()) != protocol["config"]:
        raise ValueError("training configuration changed")
    for filename, expected in protocol["source_sha256"].items():
        if sha(out / "source" / filename) != expected:
            raise ValueError(f"training source changed: {filename}")
    for split, expected in protocol["input_sha256"].items():
        if sha(out / f"inputs/{split}.jsonl") != expected:
            raise ValueError(f"training inputs changed: {split}")
    cfg = protocol["config"]
    registration = Path(cfg["repo_root"]) / "results/protocols" / f"{out.name}.json"
    if registration.read_bytes() != (out / "protocol.json").read_bytes():
        raise ValueError("registered training protocol changed")
    if versions() != cfg["versions"] or sys.executable != cfg["python_executable"]:
        raise ValueError("registered runtime changed")
    if sha(cfg["nf4_artifact_manifest"]) != cfg["nf4_artifact_manifest_sha256"]:
        raise ValueError("NF4 manifest changed")
    manifest = json.loads(Path(cfg["nf4_artifact_manifest"]).read_text())
    for name, record in manifest["files"].items():
        file = Path(cfg["base"]) / name
        if file.stat().st_size != record["bytes"] or sha(file) != record["sha256"]:
            raise ValueError(f"NF4 input changed: {name}")
    return cfg


def execute(out):
    out = out.resolve()
    cfg = validate_prepared(out)
    if (out / "result.json").exists():
        raise FileExistsError("training execution already attempted; preserve its result")
    if ROOT.resolve() != (out / "source").resolve():
        command = [sys.executable, "-B", "-m", "scripts.qwen38_train_science", "--execute", str(out)]
        os.chdir(out / "source")
        os.execve(sys.executable, command, dict(os.environ, PYTHONPATH=str(out / "source"),
                                               CUDA_VISIBLE_DEVICES=cfg["gpu"], HF_HUB_OFFLINE="1",
                                               TOKENIZERS_PARALLELISM="false"))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpu"]
    import torch
    from transformers import set_seed
    from scripts.qwen38_science_runtime import load_runtime, answer_loss
    set_seed(cfg["seed"])
    start = time.monotonic()
    result = {"completed": False, "checkpoints": [], "steps": 0, "protocol_sha256": sha(out / "protocol.json")}
    write(out / "result.json", result)
    try:
        model, runtime = load_runtime(cfg["rank"], cfg["alpha"], cfg["dropout"], cfg["base"])
        result["runtime"] = {k: v for k, v in runtime.items() if k != "targets"}
        train, validation = [read_jsonl(out / f"inputs/{s}.jsonl") for s in ("train", "validation")]
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        def lr_factor(step):
            if step < cfg["warmup_steps"]:
                return (step + 1) / max(1, cfg["warmup_steps"])
            return max(0.0, (cfg["total_steps"] - step) / max(1, cfg["total_steps"] - cfg["warmup_steps"]))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        def loss(row):
            tensors = {key: torch.tensor([row[key]], dtype=torch.long, device="cuda:0") for key in ("input_ids", "labels", "attention_mask")}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return answer_loss(model, **tensors)
        def deadline():
            if time.monotonic() - start > cfg["max_seconds"]:
                raise TimeoutError("registered training/development time cap")
        def dev_nll():
            total, tokens = 0.0, 0
            model.eval()
            with torch.no_grad():
                for row in validation:
                    deadline()
                    value, n = loss(row)
                    if not torch.isfinite(value):
                        raise RuntimeError("non-finite development loss")
                    total += value.item() * n; tokens += n
            model.train()
            return {"nll": total / tokens, "loss_tokens": tokens}
        model.train()
        for epoch in range(cfg["epochs"]):
            order = list(range(len(train))); random.Random(cfg["seed"] + epoch).shuffle(order)
            for first in range(0, len(order), cfg["gradient_accumulation"]):
                group = order[first:first + cfg["gradient_accumulation"]]
                optimizer.zero_grad(set_to_none=True)
                summed, tokens = 0.0, 0
                for index in group:
                    deadline()
                    value, n = loss(train[index])
                    if not torch.isfinite(value):
                        raise RuntimeError("non-finite training loss")
                    (value / len(group)).backward()
                    summed += value.item() * n; tokens += n
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step(); scheduler.step()
                result["steps"] += 1
                with (out / "training.jsonl").open("a") as log:
                    log.write(json.dumps({"step": result["steps"], "epoch": epoch + 1,
                        "group_token_weighted_nll": summed / tokens, "target_tokens": tokens,
                        "gradient_norm": norm.item(), "elapsed_seconds": time.monotonic() - start}) + "\n")
                if result["steps"] in cfg["checkpoint_steps"]:
                    metrics = dev_nll()
                    checkpoint = out / "checkpoints" / f"step-{result['steps']:05d}"
                    model.save_pretrained(checkpoint, safe_serialization=True)
                    # Optimizer and RNG state make an incomplete run inspectable; no automatic resume.
                    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, checkpoint / "training_state.pt")
                    result["checkpoints"].append({"step": result["steps"], "path": str(checkpoint),
                        "adapter_files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                                          for p in checkpoint.glob("adapter*") if p.is_file()}, **metrics})
                result["elapsed_seconds"] = time.monotonic() - start
                write(out / "result.json", result)
        if result["steps"] != cfg["total_steps"] or len(result["checkpoints"]) != len(cfg["checkpoint_steps"]):
            raise ValueError("incomplete registered training schedule")
        result["selected"] = min(result["checkpoints"], key=lambda r: (r["nll"], r["step"]))
        result["completed"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        write(out / "result.json", result)
    return {k: v for k, v in result.items() if k != "runtime"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", type=Path)
    parser.add_argument("--task", choices=["qasper", "prl"])
    parser.add_argument("--prepared")
    parser.add_argument("--output")
    parser.add_argument("--train-count", type=int, default=480)
    parser.add_argument("--validation-count", type=int, default=150)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    if args.execute:
        result = execute(args.execute)
    else:
        if not args.task or not args.prepared or not args.output or not args.gpu.isdecimal() or args.max_length != 8192:
            parser.error("prepare requires task/prepared/output, one GPU and registered 8192 cap")
        result = prepare(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
