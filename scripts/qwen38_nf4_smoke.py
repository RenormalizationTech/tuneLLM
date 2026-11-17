"""Local-only Qwen3.8 text/NF4 smoke; inspection/help never loads a model/GPU.

Official Transformers5.23 route: Qwen3_5TextConfig selects text_config and the
qwen3_5_text conversion registry removes language_model under model. No custom
weight conversion, vision loading, downloads, benchmarks or training.
Root example (new isolated env):
  python -B -m scripts.qwen38_nf4_smoke --inspect
  python -B -m scripts.qwen38_nf4_smoke --execute-gpu
  python -B -m scripts.qwen38_nf4_smoke --execute-gpu \
    --save-nf4 artifacts/qwen38-27b-nf4-01 --compare-reload
The last command frees the parent model and checks tokens in a separate process.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/qwen38_27b_nf4.json"
BASE = "Qwen/Qwen3.8-27B"
REV = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"


def local(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT/p


def read(file): return json.loads(local(file).read_text())


def sha(file):
    h = hashlib.sha256()
    with local(file).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def write(file, value):
    file = Path(file); file.parent.mkdir(parents=True, exist_ok=True)
    def runtime_value(item):
        if isinstance(item, (set, frozenset)):
            return sorted(item)
        raise TypeError(f"Unsupported runtime metadata: {type(item).__name__}")
    file.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                               default=runtime_value)+"\n")


def validate_config(cfg):
    fixed = {"model_id": BASE, "revision": REV, "loader": "Qwen3_5ForCausalLM", "dtype": "bfloat16",
             "seed": 42, "batch_size": 1, "max_new_tokens": 64, "enable_thinking": False}
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()): raise ValueError("fixed text-only smoke profile required")
    if not re.fullmatch(r"\d+", cfg["gpu"]): raise ValueError("exactly one physical GPU index required")
    if cfg["quantization"] != {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "bfloat16"}:
        raise ValueError("NF4/double-quant/BF16 compute required")
    if len(cfg["prompts"]) != 2 or any(not isinstance(p, str) or not p.strip() for p in cfg["prompts"]): raise ValueError("exactly two text prompts required")


def official_support(package_root=None):
    """Read installed official source, without importing modeling or torch."""
    if package_root is None:
        try: package_root = importlib.metadata.distribution("transformers").locate_file("transformers")
        except importlib.metadata.PackageNotFoundError: return {"available": False, "reason": "Transformers not installed"}
    root = Path(package_root)
    files = [root/"models/qwen3_5/modeling_qwen3_5.py", root/"models/qwen3_5/configuration_qwen3_5.py", root/"conversion_mapping.py"]
    if not all(p.is_file() for p in files): return {"available": False, "reason": "official Qwen3_5 source missing; use isolated Transformers5.23.0 environment"}
    model, config, conversion = [p.read_text() for p in files]
    supported = ("class Qwen3_5ForCausalLM(" in model and "config: Qwen3_5TextConfig" in model
        and 'base_config_key = "text_config"' in config
        and re.search(r'"qwen3_5_text"\s*:\s*\[PrefixChange\(prefix_to_remove="language_model",\s*model_prefix="model"\)', conversion) is not None)
    return {"available": bool(supported), "loader": "Qwen3_5ForCausalLM", "config_class": "Qwen3_5TextConfig",
            "source_sha256": {str(p): sha(p) for p in files},
            "official_key_mapping": "model.language_model.* -> model.* via PrefixChange; expected vision/MTP skipped",
            "documentation": "https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5#usage-tips-and-notes"}


def inspect(cfg):
    validate_config(cfg); source = read(cfg["reference"])["config"]; text = source["text_config"]
    if source["architectures"] != ["Qwen3_5ForConditionalGeneration"] or (text["model_type"], text["num_hidden_layers"], text["hidden_size"]) != ("qwen3_5_text", 64, 5120):
        raise ValueError("saved official full/text config differs")
    snapshot = local(cfg["snapshot"]); index_file = snapshot/"model.safetensors.index.json"
    index = read(index_file) if index_file.is_file() else {}
    keys = list(index.get("weight_map", {})); shards = sorted(set(index.get("weight_map", {}).values()))
    versions = {}
    for name in ("torch", "transformers", "accelerate", "bitsandbytes", "huggingface_hub"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return {"model_id": BASE, "revision": REV, "source_architecture": source["architectures"], "installed_versions": versions,
        "text_layers": 64, "hidden_size": 5120, "layer_types": dict(Counter(text["layer_types"])),
        "official_support": official_support(), "snapshot": str(snapshot),
        "source_key_counts": dict(Counter("language" if k.startswith("model.language_model.") else "vision" if k.startswith("model.visual.") else "mtp" if k.startswith(("mtp.", "model.mtp.")) else "other" for k in keys)),
        "shards": len(shards), "missing_shards": [n for n in shards if not (snapshot/n).is_file()],
        "inspection_only": True, "model_loaded": False}


def validate_loading(info):
    failures = {k: info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs") if info.get(k)}
    unexpected = info.get("unexpected_keys", [])
    bad = [k for k in unexpected if not k.startswith(("model.visual.", "visual.", "mtp.", "model.mtp."))]
    if failures or bad: raise ValueError("missing/mismatched/unexpected LANGUAGE weights: "+str({**failures, "unexpected": bad}))
    return {"missing_keys": [], "mismatched_keys": [], "expected_skipped_vision_mtp": unexpected,
            "note": "official ignored vision/MTP keys may be absent from loading_info; source-key counts are recorded separately"}


def artifact(value, *, fresh=False):
    p = local(value).resolve(); root = (ROOT/"artifacts").resolve()
    if p == root or not p.is_relative_to(root): raise ValueError("saved NF4 must stay inside ignored artifacts/")
    if fresh and p.exists(): raise FileExistsError("fresh NF4 destination required")
    return p


def loading_kwargs(cfg, saved=False):
    import torch
    kwargs = {"local_files_only": True, "trust_remote_code": False, "dtype": torch.bfloat16,
              "device_map": {"": "cuda:0"}, "attn_implementation": cfg["attn_implementation"], "output_loading_info": True}
    if not saved:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    return kwargs


def hybrid_check(model, inputs, cfg):
    import torch
    with torch.inference_mode():
        prefill = model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True)
        if not torch.isfinite(prefill.logits).all() or prefill.past_key_values is None: raise ValueError("invalid prefill logits/cache")
        token = prefill.logits[:, -1].argmax(-1).reshape(1, 1)
        mask = torch.cat((inputs["attention_mask"], torch.ones_like(token)), -1)
        cached = model(input_ids=token, attention_mask=mask, past_key_values=prefill.past_key_values, use_cache=True, logits_to_keep=1, return_dict=True)
        full = model(input_ids=torch.cat((inputs["input_ids"], token), -1), attention_mask=mask, use_cache=False, logits_to_keep=1, return_dict=True)
        a, b = cached.logits[:, -1].float(), full.logits[:, -1].float()
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        passed = finite and torch.allclose(a, b, **cfg["cache_check"])
        record = {"finite": finite, "allclose": bool(passed), "max_abs_delta": (a-b).abs().max().item(),
                  **cfg["cache_check"], "cache_class": type(prefill.past_key_values).__name__, "next_token_id": token.item(),
                  "decode_steps_checked": 1, "cached_top1": a.argmax(-1).item(), "full_prefix_top1": b.argmax(-1).item(),
                  "argmax_matches": bool(torch.equal(a.argmax(-1), b.argmax(-1)))}
        if not finite: raise ValueError("nonfinite hybrid cached/full-prefix logits: "+str(record))
        # Preserve a failed numerical diagnostic; it does not prevent testing
        # serialization using the same cached generation path before/after reload.
        return record


def inventory(model):
    from bitsandbytes.nn import Linear4bit
    modules = [{"name": n, "class": type(m).__name__} for n, m in model.named_modules()]
    quantized = [n for n, m in model.named_modules() if isinstance(m, Linear4bit)]
    if not quantized: raise ValueError("no actual Linear4bit modules loaded")
    special = {n: {"dtype": str(p.dtype), "shape": list(p.shape)} for n, p in model.named_parameters()
               if n.endswith((".A_log", ".dt_bias", ".conv1d.weight", ".conv1d.bias"))}
    vision = [n for n, _ in model.named_modules() if n.startswith(("model.visual", "visual"))]
    if vision: raise ValueError("text-only loader unexpectedly loaded vision")
    return {"modules": modules, "linear4bit_modules": quantized, "linear4bit_count": len(quantized),
            "parameter_dtype_counts": dict(Counter(str(p.dtype) for p in model.parameters())),
            "native_state_space_and_conv_dtypes": special, "vision_modules": vision,
            "keep_in_fp32_modules": getattr(model, "_keep_in_fp32_modules", None),
            "keep_in_fp32_modules_strict": getattr(model, "_keep_in_fp32_modules_strict", None)}


def execute(args, cfg):
    if not args.execute_gpu: raise ValueError("root authorization and --execute-gpu required")
    plan = inspect(cfg)
    if not plan["official_support"]["available"]: raise ValueError("official text loader/mapping unavailable")
    if args.compare_reload and not args.save_nf4: raise ValueError("--compare-reload requires explicit --save-nf4")
    saved = artifact(args.load_saved) if args.load_saved else None
    target = artifact(args.save_nf4, fresh=True) if args.save_nf4 else None
    if saved and target: raise ValueError("load-saved and save-nf4 are separate modes")
    run = local(args.run or cfg["run"]); output = local(args.output or cfg["result"])
    if not run.resolve().is_relative_to((ROOT/"outputs/qwen38-27b-nf4-01").resolve()): raise ValueError("logs must stay inside outputs/qwen38-27b-nf4-01")
    if run.exists() or output.exists(): raise FileExistsError("fresh smoke run/result required")
    os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=cfg["gpu"], HF_HOME=str(local(cfg["hf_home"])),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported(): raise ValueError("one explicit BF16 CUDA device required")
    snapshot = local(cfg["snapshot"])
    if not saved:
        if not plan["shards"] or plan["missing_shards"]: raise ValueError("original local snapshot incomplete; root downloads separately")
        if read(snapshot/"config.json") != read(cfg["reference"])["config"]: raise ValueError("official snapshot config differs from pinned feasibility record")
    else:
        origin = read(saved/"smoke_origin.json")
        if origin["model_id"] != BASE or origin["revision"] != REV: raise ValueError("saved NF4 origin differs")
        quant = read(saved/"config.json").get("quantization_config", {})
        if not quant.get("load_in_4bit") or quant.get("bnb_4bit_quant_type") != "nf4" or not quant.get("bnb_4bit_use_double_quant") or quant.get("bnb_4bit_compute_dtype") != "bfloat16":
            raise ValueError("stored checkpoint is not the requested NF4/double-quant/BF16")
    run.mkdir(parents=True); began = time.perf_counter()
    versions = {}
    for name in ("torch", "transformers", "accelerate", "bitsandbytes", "huggingface_hub"):
        versions[name] = importlib.metadata.version(name)
    report = {"status": "started", "model_id": BASE, "revision": REV, "python": sys.version, "packages": versions,
        "torch_cuda_build": torch.version.cuda, "physical_gpu": cfg["gpu"], "gpu_name": torch.cuda.get_device_name(0),
        "source_inspection": plan, "load_saved": str(saved) if saved else None, "quantization_config_supplied": saved is None,
        "optional_kernels": {n: importlib.util.find_spec(n) is not None for n in ("fla", "causal_conv1d", "kernels")},
        "native_positions": True, "batch_size": 1, "enable_thinking": False}
    if local(cfg["source_record"]).is_file():
        source_record = read(cfg["source_record"])
        if source_record.get("model_id", BASE) != BASE or source_record.get("revision", REV) != REV: raise ValueError("root source record identity differs")
        report["source_record"] = {"path": str(local(cfg["source_record"])), "sha256": sha(cfg["source_record"])}
    write(run/"record.json", report)
    try:
        torch.manual_seed(cfg["seed"]); torch.cuda.reset_peak_memory_stats()
        source = saved or snapshot; kwargs = loading_kwargs(cfg, saved=bool(saved))
        if not saved: kwargs["config"] = Qwen3_5TextConfig.from_pretrained(source, local_files_only=True, trust_remote_code=False)
        loaded = time.perf_counter(); model, info = Qwen3_5ForCausalLM.from_pretrained(source, **kwargs)
        model.eval().requires_grad_(False); torch.cuda.synchronize()
        report["load_seconds"] = time.perf_counter()-loaded; report["loading_info"] = validate_loading(info)
        text = model.config.get_text_config()
        if text.model_type != "qwen3_5_text" or text.hidden_size != 5120 or text.num_hidden_layers != 64: raise ValueError("loaded text config differs")
        modules = inventory(model); write(run/"modules.json", modules)
        report["module_inventory"] = {"file": str(run/"modules.json"), "sha256": sha(run/"modules.json"),
            "linear4bit_count": modules["linear4bit_count"], "parameter_dtype_counts": modules["parameter_dtype_counts"],
            "native_dtype_policy": "No custom casts of A_log/dt_bias/conv; recorded exactly as official loader produced."}
        tok = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
        report["native_token_ids"] = {"tokenizer_eos": tok.eos_token_id, "tokenizer_pad": tok.pad_token_id,
            "generation_eos": model.generation_config.eos_token_id, "generation_pad": model.generation_config.pad_token_id}
        report["generations"] = []
        for index, prompt in enumerate(cfg["prompts"]):
            rendered = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tok(rendered, add_special_tokens=False, return_tensors="pt").to("cuda:0")
            inputs = {k: inputs[k] for k in ("input_ids", "attention_mask")}
            if index == 0: report["hybrid_cache_check"] = hybrid_check(model, inputs, cfg)
            calls = [0]
            def finite(module, args_, logits):
                calls[0] += 1
                if not torch.isfinite(logits).all(): raise ValueError("nonfinite raw lm_head logits")
            handle = model.lm_head.register_forward_hook(finite); started = time.perf_counter()
            try:
                with torch.inference_mode():
                    tokens = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=64, use_cache=True,
                        temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
                torch.cuda.synchronize()
            finally: handle.remove()
            suffix = tokens[0, inputs["input_ids"].shape[1]:].tolist()
            if not suffix or len(suffix) > 64 or not calls[0]: raise ValueError("empty/invalid smoke generation")
            result = {"prompt": prompt, "input_ids": inputs["input_ids"][0].tolist(), "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "generated_ids": suffix, "text": tok.decode(suffix, skip_special_tokens=True),
                "seconds": time.perf_counter()-started, "finite_lm_head_calls": calls[0]}
            report["generations"].append(result); write(run/f"prompt-{index}.json", result); write(run/"record.json", report)
        report["memory"] = {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "model_footprint_bytes": model.get_memory_footprint()}
        if target:
            target.mkdir(parents=True); model.save_pretrained(target, safe_serialization=True); tok.save_pretrained(target)
            write(target/"smoke_origin.json", {"model_id": BASE, "revision": REV, "loader": cfg["loader"], "source_config_sha256": sha(snapshot/"config.json")})
            report["saved_nf4"] = str(target)
        del model, inputs, tokens
        import gc
        gc.collect(); torch.cuda.empty_cache()
        if args.compare_reload:
            child_run = run/"reload"; child_result = run/"reload-summary.json"
            command = [sys.executable, "-B", "-m", "scripts.qwen38_nf4_smoke", "--config", str(local(args.config)), "--execute-gpu",
                       "--load-saved", str(target), "--run", str(child_run), "--output", str(child_result)]
            with (run/"reload.log").open("x") as log: subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            reload = read(child_result)
            same = all(a["input_ids"] == b["input_ids"] and a["generated_ids"] == b["generated_ids"] for a, b in zip(report["generations"], reload["generations"])) and len(reload["generations"]) == 2
            report["separate_process_reload"] = {"result": str(child_result), "exact_tokens": same}
            if not same: raise ValueError("separate-process deterministic reload token mismatch")
        status = "complete" if report["hybrid_cache_check"]["allclose"] else "complete_with_cache_difference"
        report.update(status=status, elapsed_seconds=time.perf_counter()-began)
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc)); write(run/"record.json", report); write(output, report); raise
    write(run/"record.json", report); write(output, report)
    print(json.dumps({"status": report["status"], "result": str(output), "linear4bit_count": report["module_inventory"]["linear4bit_count"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG); parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--execute-gpu", action="store_true"); parser.add_argument("--run"); parser.add_argument("--output")
    parser.add_argument("--save-nf4"); parser.add_argument("--load-saved"); parser.add_argument("--compare-reload", action="store_true")
    args = parser.parse_args(); cfg = read(args.config); validate_config(cfg)
    if args.inspect or not args.execute_gpu: print(json.dumps(inspect(cfg), indent=2)); return
    execute(args, cfg)


if __name__ == "__main__": main()
