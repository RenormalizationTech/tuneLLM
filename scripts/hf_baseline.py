"""Offline BF16 single-GPU baseline. Run: python -m scripts.hf_baseline --output outputs/hf-run"""
import argparse
import time
from scripts.common import (digest, load_config, local_snapshot, path, provenance,
                            read_json, select_gpu, setup_hf, validate_snapshot, write_json)
from evaluation.score import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--gpu", help="physical GPU index or UUID; overrides config")
    parser.add_argument("--output", required=True, help="new output directory; refuses overwrite")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.gpu is not None:
        cfg["gpu"] = args.gpu
    select_gpu(cfg["gpu"])
    setup_hf(cfg)
    generation = cfg["generation"]
    if generation.get("do_sample") is not False or generation.get("num_beams") != 1:
        raise ValueError("initial comparison uses greedy generation")
    if generation["max_new_tokens"] <= 0 or cfg["warmup_runs"] < 0:
        raise ValueError("invalid generation length or warmup count")
    fixture = read_json(cfg["fixture"])
    tasks = fixture["tasks"]
    # Validate IDs and fixture before allocating a model.
    score(tasks, [{"id": task["id"], "output": ""} for task in tasks])
    snapshot = local_snapshot(cfg)
    out = path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    metadata = provenance(cfg)
    metadata.update({"fixture_sha256": digest(cfg["fixture"]), "snapshot": str(snapshot),
                     "checkpoint_weight_bytes": validate_snapshot(snapshot),
                     "runtime": "transformers BF16, one GPU, batch size 1",
                     "caveat": "Greedy smoke comparison deliberately differs from model-card sampling advice; inspect repetition and truncation. Fixture is not a capability benchmark.",
                     "measurement": "Synchronized wall time around generate (prefill + decode + Python overhead); excludes tokenization/loading. Counts emitted tokens including terminal EOS. No decode-only or first-token latency claim.",
                     "first_token_latency_seconds": None})
    write_json(out / "run.json", metadata)
    write_json(out / "fixture.json", fixture)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected GPU must support CUDA BF16")
    set_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("require an existing pad token distinct from EOS")
    torch.cuda.synchronize()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation=cfg["attn_implementation"])
    model.eval()
    torch.cuda.synchronize()
    metadata.update({"load_wall_seconds": time.perf_counter() - started,
                     "parameter_count": sum(p.numel() for p in model.parameters()),
                     "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
                     "selected_physical_gpu": str(cfg["gpu"]), "logical_device": "cuda:0",
                     "gpu_name": torch.cuda.get_device_name(0), "torch_cuda_build": torch.version.cuda,
                     "loaded_allocated_bytes": torch.cuda.memory_allocated(),
                     "loaded_reserved_bytes": torch.cuda.memory_reserved()})
    write_json(out / "chat_template.json", {"chat_template": tokenizer.chat_template,
                                            "enable_thinking": False})

    def prepare(task):
        rendered = tokenizer.apply_chat_template(task["messages"], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to("cuda:0")
        return rendered, inputs

    def generate(inputs):
        with torch.inference_mode():
            return model.generate(**inputs, **generation, pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=tokenizer.eos_token_id,
                                  temperature=None, top_p=None, top_k=None,
                                  return_dict_in_generate=False)

    _, warmup = prepare(tasks[0])
    for _ in range(cfg["warmup_runs"]):
        generate(warmup)
    torch.cuda.synchronize()
    del warmup
    torch.cuda.reset_peak_memory_stats()
    predictions = []
    for task in tasks:
        rendered, inputs = prepare(task)
        torch.cuda.synchronize()
        started = time.perf_counter()
        generated = generate(inputs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        new_ids = generated[0, inputs["input_ids"].shape[1]:].tolist()
        ended_eos = bool(new_ids and new_ids[-1] == tokenizer.eos_token_id)
        predictions.append({"id": task["id"], "output": tokenizer.decode(new_ids, skip_special_tokens=True),
                            "rendered_prompt": rendered, "input_ids": inputs["input_ids"][0].tolist(),
                            "generated_ids": new_ids, "input_tokens": inputs["input_ids"].shape[1],
                            "generated_tokens_including_eos": len(new_ids), "ended_eos": ended_eos,
                            "hit_token_limit": len(new_ids) >= generation["max_new_tokens"] and not ended_eos,
                            "generation_wall_seconds": elapsed,
                            "generated_tokens_per_wall_second": len(new_ids) / elapsed})
        write_json(out / "predictions.json", predictions)
        print(f"{task['id']}: {elapsed:.3f}s, {len(new_ids)} emitted tokens", flush=True)
    elapsed = sum(row["generation_wall_seconds"] for row in predictions)
    metadata.update({"measured_generation_wall_seconds": elapsed,
                     "generated_tokens_per_wall_second": sum(len(row["generated_ids"]) for row in predictions) / elapsed,
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                     "memory_scope": "PyTorch allocator on selected GPU including loaded weights; excludes other processes and non-PyTorch CUDA allocations",
                     "completed": True})
    write_json(out / "scores.json", score(tasks, predictions))
    write_json(out / "run.json", metadata)
    print(out)


if __name__ == "__main__":
    main()
