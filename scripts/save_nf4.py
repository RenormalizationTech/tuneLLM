"""Save the exact prepared Qwen NF4 base; verify a stored-quantization reload on GPU1."""
import argparse
import gc
import hashlib
import time
from unittest.mock import patch

from scripts.batched_generation import REPRESENTATION, load_nf4, render_prompt
from scripts.common import load_config, local_snapshot, path, provenance, read_json, select_gpu, setup_hf, write_json


def stream_hash(filename, chunk_bytes=8 * 1024 * 1024):
    value = hashlib.sha256()
    with path(filename).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            value.update(chunk)
    return value.hexdigest()


def tensor_hash(tensor, chunk_elements=1024 * 1024):
    import torch
    value = hashlib.sha256()
    flat = tensor.detach().reshape(-1)
    for start in range(0, flat.numel(), chunk_elements):
        value.update(flat[start:start + chunk_elements].to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def state_manifest(model):
    manifest = {}
    for name, tensor in model.state_dict().items():
        manifest[name] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "sha256": tensor_hash(tensor)}
    return manifest


def reload_stored_nf4(checkpoint, training=False):
    """No quantization_config override: Transformers must restore the saved bnb state."""
    import torch
    from transformers import AutoModelForCausalLM
    from peft import prepare_model_for_kbit_training
    # Fail explicitly if this load tries to quantize floating weights again.
    with patch("bitsandbytes.functional.quantize_4bit", side_effect=RuntimeError("unexpected requantization of stored NF4")):
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            device_map={"": 0}, torch_dtype=torch.float32, attn_implementation="sdpa")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = not training
    return model


def probe(model, tokenizer, rows):
    import torch
    model.eval()
    logits, records = [], []
    for row in rows:
        prompt = render_prompt(tokenizer, row)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            logits.append(model(**inputs).logits[0, -1].detach().float().cpu())
            generated = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=16,
                                       repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                                       eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                       use_cache=True)
        records.append({"id": row["id"], "input_ids": inputs["input_ids"][0].tolist(),
                        "generated_ids": generated[0, inputs["input_ids"].shape[1]:].tolist()})
    return torch.stack(logits), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--checkpoint", default="artifacts/nf4/Qwen3-4B")
    parser.add_argument("--output", default="outputs/nf4-checkpoint-01")
    parser.add_argument("--result", default="results/nf4_checkpoint.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["gpu"] = "1"
    select_gpu("1")
    setup_hf(cfg)
    checkpoint, out, result = path(args.checkpoint), path(args.output), path(args.result)
    if any(p.exists() for p in (checkpoint, out, result)):
        raise FileExistsError("refusing to overwrite NF4 checkpoint, logs, or result")
    out.mkdir(parents=True)
    metadata = provenance(cfg)
    metadata.update({"completed": False, "phase": "loading_original", "representation": REPRESENTATION,
                     "checkpoint": str(checkpoint), "gpu": "1", "seed": 42})
    write_json(out / "run.json", metadata)
    started = time.perf_counter()
    try:
        import torch
        from transformers import AutoTokenizer, set_seed
        set_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        upstream = local_snapshot(cfg)
        tokenizer = AutoTokenizer.from_pretrained(upstream, local_files_only=True, trust_remote_code=False)
        rows = [{"id": "arithmetic", "user": "What is 17 + 25? Reply with only the integer."},
                {"id": "physics", "user": "Explain conservation of energy in one sentence."}]
        model = load_nf4(upstream)
        print("Fingerprinting original packed weights, quantization state, and FP32 modules", flush=True)
        before = state_manifest(model)
        write_json(out / "original_state.json", before)
        before_logits, before_ids = probe(model, tokenizer, rows)
        torch.save(before_logits, out / "original_last_token_logits.pt")
        write_json(out / "original_probes.json", before_ids)
        metadata["phase"] = "saving"
        write_json(out / "run.json", metadata)
        checkpoint.mkdir(parents=True)
        model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(checkpoint)
        write_json(checkpoint / "provenance.json", {**metadata, "upstream_snapshot": str(upstream)})
        del model
        gc.collect()
        torch.cuda.empty_cache()
        saved_cfg = read_json(checkpoint / "config.json")
        quantization = saved_cfg["quantization_config"]
        if not quantization.get("load_in_4bit") or quantization["bnb_4bit_quant_type"] != "nf4" or not quantization["bnb_4bit_use_double_quant"]:
            raise RuntimeError("saved checkpoint does not record expected double-quantized NF4")
        metadata["phase"] = "reloading_stored_quantization"
        write_json(out / "run.json", metadata)
        reloaded = reload_stored_nf4(checkpoint)
        print("Fingerprinting stored-quantization reload (requantization forbidden)", flush=True)
        after = state_manifest(reloaded)
        write_json(out / "reloaded_state.json", after)
        after_logits, after_ids = probe(reloaded, tokenizer, rows)
        torch.save(after_logits, out / "reloaded_last_token_logits.pt")
        write_json(out / "reloaded_probes.json", after_ids)
        mismatches = [key for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]
        checks = {"all_state_tensors_equal": not mismatches,
                  "packed_weights_and_quantization_state_equal": not mismatches,
                  "fixed_last_token_logits_exact": torch.equal(before_logits, after_logits),
                  "max_abs_logit_difference": (before_logits - after_logits).abs().max().item(),
                  "fixed_generated_ids_equal": before_ids == after_ids,
                  "requantization_forbidden_during_reload": True,
                  "state_tensor_count": len(before), "mismatched_state_keys": mismatches}
        metadata.update({"checks": checks, "elapsed_wall_seconds": time.perf_counter() - started,
                         "stored_quantization_config": quantization,
                         "completed": not mismatches and checks["fixed_last_token_logits_exact"] and checks["fixed_generated_ids_equal"]})
        if not metadata["completed"]:
            raise RuntimeError(f"NF4 serialization verification failed: {checks}")
        metadata["phase"] = "complete"
        write_json(checkpoint / "provenance.json", metadata)
        files = {p.name: {"bytes": p.stat().st_size, "sha256": stream_hash(p)} for p in sorted(checkpoint.iterdir()) if p.is_file()}
        write_json(out / "checkpoint_files.json", files)
        write_json(result, {"completed": True, "model_id": cfg["model_id"], "revision": cfg["revision"],
                            "representation": REPRESENTATION, "gpu": "1", "checkpoint": str(checkpoint.relative_to(path("."))),
                            "source_commit": metadata["git_head"]["stdout"],
                            "source_sha256": metadata["source_sha256"], "packages": metadata["environment"]["packages"],
                            "checks": checks, "stored_quantization_config": quantization,
                            "files": files, "total_bytes": sum(item["bytes"] for item in files.values()),
                            "elapsed_wall_seconds": metadata["elapsed_wall_seconds"],
                            "log_directory": str(out.relative_to(path(".")))})
        print(f"NF4 checkpoint verified: {result}", flush=True)
    except BaseException as exc:
        metadata.update({"completed": False, "phase": "failed", "error": str(exc), "error_type": type(exc).__name__})
        raise
    finally:
        write_json(out / "run.json", metadata)


if __name__ == "__main__":
    main()
