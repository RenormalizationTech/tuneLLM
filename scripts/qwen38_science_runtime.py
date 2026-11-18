"""Explicit text-only 27B runtime inspection and bounded backward preflight.

Default inspection creates a meta model only. --execute-preflight uses the saved
NF4 on one GPU; it never saves an adapter or evaluates a final-test question.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/qwen38-27b-nf4-01"
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
           "in_proj_b", "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "huggingface_hub")}


def inventory(rank=16, base=BASE):
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    cfg = Qwen3_5TextConfig.from_pretrained(base, local_files_only=True)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(cfg)
    targets = []
    for name, module in model.named_modules():
        if name.startswith("model.layers.") and name.rsplit(".", 1)[-1] in TARGETS:
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f"unexpected adapter target class: {name}")
            targets.append({"name": name, "in_features": module.in_features,
                            "out_features": module.out_features,
                            "lora_parameters": rank * (module.in_features + module.out_features)})
    if len(targets) != 496 or {int(r["name"].split(".")[2]) for r in targets} != set(range(64)):
        raise ValueError("expected all 496 projections across 64 language layers")
    return {"loader": "Qwen3_5ForCausalLM", "model_loaded": False, "meta_only": True,
            "versions": versions(), "rank": rank, "targets": targets,
            "target_count": len(targets),
            "lora_parameters": sum(r["lora_parameters"] for r in targets),
            "target_suffix_counts": dict(Counter(r["name"].rsplit(".", 1)[-1] for r in targets)),
            "base_config_sha256": sha(Path(base) / "config.json")}


def load_runtime(rank=16, alpha=32, dropout=0.05, base=BASE):
    import torch
    from transformers import Qwen3_5ForCausalLM
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from scripts.qwen38_nf4_smoke import validate_loading
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        base, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa", output_loading_info=True)
    validate_loading(loading)
    if any("visual" in name or "mtp" in name for name, _ in model.named_modules()):
        raise ValueError("vision/MTP modules must be absent")
    before = dict(Counter(str(p.dtype) for p in model.parameters()))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    expected = inventory(rank, base)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=rank,
                                            lora_alpha=alpha, lora_dropout=dropout,
                                            target_modules=[r["name"] for r in expected["targets"]], bias="none"))
    trained = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if any("lora_" not in name for name, _ in trained):
        raise ValueError("unexpected trainable base/state parameter")
    count = sum(p.numel() for _, p in trained)
    if count != expected["lora_parameters"]:
        raise ValueError("trainable parameter count differs from meta inventory")
    metadata = {"before_kbit_preparation_parameter_dtypes": before,
                "after_kbit_preparation_parameter_dtypes": dict(Counter(str(p.dtype) for p in model.parameters())),
                "kbit_preparation": "PEFT default floating-weight casts; baseline must use identical preparation",
                "trainable_parameters": count, "target_count": expected["target_count"],
                "targets": [r["name"] for r in expected["targets"]]}
    return model, metadata


def answer_loss(model, input_ids, labels, attention_mask):
    """Compute exactly supervised next-token CE without full-prompt vocab logits."""
    import torch
    positions = torch.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten()
    if not positions.numel() or input_ids.shape[0] != 1:
        raise ValueError("nonempty assistant loss and batch one required")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                    logits_to_keep=positions, use_cache=False)
    targets = labels[:, positions + 1]
    return torch.nn.functional.cross_entropy(outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
                                             targets.reshape(-1)), int(targets.numel())


def preflight(out, gpu, lengths, rank, max_seconds, target_tokens):
    if out.exists():
        raise FileExistsError("fresh preflight output required")
    out.mkdir(parents=True)
    artifact = json.loads((ROOT / "results/qwen38_27b_nf4_artifact_01.json").read_text())
    for name, expected in artifact["files"].items():
        p = BASE / name
        if p.stat().st_size != expected["bytes"] or sha(p) != expected["sha256"]:
            raise ValueError(f"saved NF4 identity changed: {name}")
    cfg = {"scope": "synthetic backward mechanics only; no quality, final data or persistent adapter",
           "gpu": gpu, "lengths": lengths, "rank": rank, "alpha": 2 * rank,
           "dropout": 0.05, "seed": 42, "max_seconds": max_seconds, "target_tokens": target_tokens,
           "nf4_manifest_sha256": sha(ROOT / "results/qwen38_27b_nf4_artifact_01.json"),
           "source_sha256": {name: sha(ROOT / name) for name in
                             ("scripts/qwen38_science_runtime.py", "scripts/qwen38_nf4_smoke.py")},
           "versions": versions()}
    write(out / "registration.json", cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch
    from transformers import AutoTokenizer, set_seed
    set_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable")
    start = time.monotonic()
    result = {"completed": False, "probes": [], "gpu": torch.cuda.get_device_name(0)}
    write(out / "result.json", result)
    try:
        model, metadata = load_runtime(rank, 2 * rank)
        result["runtime"] = metadata
        tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
        # Synthetic token probes have no scientific labels or checkpoint selection role.
        head = tokenizer.apply_chat_template([{"role": "user", "content": "Describe the supplied scientific text briefly."}],
                                             tokenize=True, return_dict=False,
                                             add_generation_prompt=True, enable_thinking=False)
        unit = tokenizer(" The sample measures a response under controlled conditions.", add_special_tokens=False)["input_ids"]
        answer_unit = tokenizer("A controlled response is measured.", add_special_tokens=False)["input_ids"]
        tail = (answer_unit * ((target_tokens + len(answer_unit) - 1) // len(answer_unit)))[:target_tokens - 1]
        tail.append(tokenizer.eos_token_id)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
        model.train()
        for length in lengths:
            if time.monotonic() - start >= max_seconds:
                result["stop_reason"] = "preflight time cap"
                break
            body_len = length - len(head) - len(tail)
            if body_len < 1:
                raise ValueError("preflight sequence too short for synthetic wrapper")
            ids = head + (unit * ((body_len + len(unit) - 1) // len(unit)))[:body_len] + tail
            batch = torch.tensor([ids], device="cuda:0")
            labels = batch.clone(); labels[:, :-len(tail)] = -100
            torch.cuda.reset_peak_memory_stats()
            tick = time.monotonic()
            try:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, count = answer_loss(model, batch, labels, torch.ones_like(batch))
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite preflight loss")
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                finite = all(bool(torch.isfinite(g).all()) for g in grads)
                nonzero = sum(bool(torch.count_nonzero(g)) for g in grads)
                if not finite or not nonzero:
                    raise RuntimeError("invalid or zero adapter gradients")
                optimizer.step(); torch.cuda.synchronize()
                result["probes"].append({"sequence_length": length, "loss_tokens": count,
                    "loss": loss.item(), "gradient_tensors": len(grads), "nonzero_gradient_tensors": nonzero,
                    "finite_gradients": finite, "seconds": time.monotonic() - tick,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            except torch.cuda.OutOfMemoryError as exc:
                result["probes"].append({"sequence_length": length, "oom": True, "message": str(exc)[:300]})
                result["stop_reason"] = "OOM; no automatic policy change"
                break
            write(out / "result.json", result)
        result["completed"] = len(result["probes"]) == len(lengths) and all(not p.get("oom") for p in result["probes"])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        write(out / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--execute-preflight", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lengths", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--target-tokens", type=int, default=256)
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    if not out.is_relative_to(ROOT / "outputs") or not args.gpu.isdecimal():
        parser.error("output must stay inside outputs/ and gpu must be one index")
    if (args.rank < 1 or any(n < 128 or n > 8192 for n in args.lengths)
            or not 0 < args.max_seconds <= 7200 or not 1 <= args.target_tokens <= 1024):
        parser.error("rank, lengths or resource cap outside preflight bounds")
    if args.execute_preflight:
        result = preflight(out, args.gpu, args.lengths, args.rank, args.max_seconds, args.target_tokens)
        result = {**result, "runtime": {k: v for k, v in result.get("runtime", {}).items() if k != "targets"}}
    else:
        out.mkdir(parents=True, exist_ok=False)
        result = inventory(args.rank); write(out / "inventory.json", result)
        result = {k: v for k, v in result.items() if k != "targets"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
