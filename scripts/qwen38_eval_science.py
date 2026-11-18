"""Recorded, single-GPU science generations from prepared, frozen JSONL inputs.

The base and adapter use identical PEFT preparation/casts. No task text is
truncated. Every generated output and its exact prompt/token IDs are retained.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

from scripts.qwen38_science_runtime import sha, write, versions

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/qwen38-27b-nf4-01"
PROMPTS = {
    "base": "",
    "verify": "Use only the supplied evidence. Check that every answer claim and cited paragraph is supported. Follow the requested output format exactly.\n\n",
    "schema": 'Output-format example for QA: {"answer":"the stated method","unanswerable":false,"evidence":["p000000"]}. For an unanswerable QA question use {"answer":"","unanswerable":true,"evidence":[]}. These are format examples, not evidence for the current question.\n\n',
}


def render(tokenizer, row, template):
    user = PROMPTS[template] + row["user"]
    return tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)


def execute(args):
    out = (ROOT / args.output).resolve()
    source = (ROOT / args.input).resolve()
    if out.exists() or not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("fresh project-local output required")
    rows = [json.loads(line) for line in source.read_text().split("\n") if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("nonempty unique input IDs required")
    if args.task == "prl" and args.template == "schema":
        raise ValueError("QA schema prompt is not a PRL baseline")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    prepared = []
    for row in rows:
        prompt = render(tokenizer, row, args.template)
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) + args.max_new_tokens > 8192:
            raise ValueError(f"full prompt exceeds registered budget: {row['id']}; revise inputs before model runs")
        prepared.append((row, prompt, ids))
    artifact_manifest = ROOT / "results/qwen38_27b_nf4_artifact_01.json"
    artifact = json.loads(artifact_manifest.read_text())
    for name, record in artifact["files"].items():
        file = BASE / name
        if file.stat().st_size != record["bytes"] or sha(file) != record["sha256"]:
            raise ValueError(f"NF4 base changed: {name}")
    adapter = (ROOT / args.adapter).resolve() if args.adapter else None
    if adapter:
        config = json.loads((adapter / "adapter_config.json").read_text())
        if (config.get("r") != 16 or config.get("lora_alpha") != 32
                or config.get("lora_dropout") != 0.05 or config.get("task_type") != "CAUSAL_LM"
                or config.get("bias") != "none" or config.get("base_model_name_or_path") != str(BASE)):
            raise ValueError("adapter configuration/base identity differs from the registered runtime")
        training = adapter.parents[1]
        training_result = json.loads((training / "result.json").read_text())
        training_cfg = json.loads((training / "config.json").read_text())
        if (not training_result.get("completed") or Path(training_result["selected"]["path"]).resolve() != adapter
                or training_cfg["task"] != args.task
                or training_cfg["nf4_artifact_manifest_sha256"] != sha(artifact_manifest)):
            raise ValueError("adapter must be the selected checkpoint for this task and NF4 base")
    out.mkdir(parents=True)
    cfg = {"task": args.task, "template": args.template, "template_text": PROMPTS[args.template],
           "input_sha256": sha(source), "input_path": str(source), "ids": [r["id"] for r in rows],
           "gpu": args.gpu, "max_new_tokens": args.max_new_tokens, "max_context": 8192,
           "max_seconds": args.max_seconds, "seed": 42, "batch_size": 1,
           "cached_generation": True, "enable_thinking": False, "do_sample": False,
           "base_manifest_sha256": sha(artifact_manifest), "versions": versions(),
           "source_sha256": {name: sha(ROOT / name) for name in
                             ("scripts/qwen38_eval_science.py", "scripts/qwen38_science_runtime.py",
                              "scripts/qwen38_nf4_smoke.py")},
           "adapter": str(adapter) if adapter else None,
           "adapter_files": {p.name: sha(p) for p in adapter.glob("adapter*")} if adapter else {},
           "base_preparation": "same PEFT floating casts and rank16 wrapper; disabled adapter for unadapted arm",
           "purpose": args.purpose}
    write(out / "registration.json", cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    from transformers import set_seed
    from scripts.qwen38_science_runtime import load_runtime
    set_seed(42)
    result = {"completed": False, "generations": 0, "registration_sha256": sha(out / "registration.json")}
    write(out / "result.json", result)
    start = time.monotonic()
    try:
        model, runtime = load_runtime()
        if adapter:
            from safetensors.torch import load_file
            from peft import get_peft_model_state_dict, set_peft_model_state_dict
            if not (adapter / "adapter_model.safetensors").is_file():
                raise ValueError("selected safetensors adapter required")
            state = load_file(adapter / "adapter_model.safetensors")
            report = set_peft_model_state_dict(model, state, adapter_name="default")
            if report.unexpected_keys or any("lora_" in key for key in report.missing_keys):
                raise ValueError("incomplete/mismatched adapter state")
            actual = get_peft_model_state_dict(model)
            if set(actual) != set(state) or any(not torch.equal(actual[k].detach().cpu(), state[k]) for k in state):
                raise ValueError("adapter reload tensor equality failed")
            result["adapter_tensor_equality"] = True
        model.eval()
        result["runtime"] = {k: v for k, v in runtime.items() if k != "targets"}
        context = nullcontext() if adapter else model.disable_adapter()
        with context, torch.inference_mode():
            for row, prompt, ids in prepared:
                if time.monotonic() - start > args.max_seconds:
                    raise TimeoutError("registered generation time cap reached")
                batch = torch.tensor([ids], dtype=torch.long, device="cuda:0")
                torch.cuda.reset_peak_memory_stats()
                tick = time.monotonic()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sequence = model.generate(input_ids=batch, attention_mask=torch.ones_like(batch),
                        do_sample=False, num_beams=1, use_cache=True, max_new_tokens=args.max_new_tokens,
                        temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                        logits_to_keep=1)
                torch.cuda.synchronize()
                tokens = sequence[0, len(ids):].tolist()
                eos = bool(tokens and tokens[-1] == tokenizer.eos_token_id)
                prediction = {"id": row["id"], "paper_id": row["paper_id"], "rendered_prompt": prompt,
                    "input_ids": ids, "generated_ids": tokens,
                    "output": tokenizer.decode(tokens, skip_special_tokens=True), "ended_with_eos": eos,
                    "hit_token_limit": len(tokens) == args.max_new_tokens and not eos,
                    "seconds": time.monotonic() - tick, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                with (out / "predictions.jsonl").open("a") as stream:
                    stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                result["generations"] += 1
                result["elapsed_seconds"] = time.monotonic() - start
                write(out / "result.json", result)
                print(json.dumps({"done": result["generations"], "total": len(rows), "id": row["id"]}), flush=True)
        if sha(source) != cfg["input_sha256"]:
            raise ValueError("inputs changed during generation")
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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=["qasper", "prl"], required=True)
    parser.add_argument("--template", choices=list(PROMPTS), default="base")
    parser.add_argument("--adapter")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--purpose", choices=["development", "final", "reload", "mechanics"], required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-seconds", type=int, default=43200)
    args = parser.parse_args()
    if not args.gpu.isdecimal() or not 1 <= args.max_new_tokens <= 512 or not 1 <= args.max_seconds <= 43200:
        parser.error("one GPU and bounded generation settings required")
    print(json.dumps(execute(args), indent=2))


if __name__ == "__main__":
    main()
