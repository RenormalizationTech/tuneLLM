"""Bounded base-quantization screen: no adapters and no fresh science final data.

32 general MMLU likelihoods, 32 GSM8K questions, and the frozen 35-question
QASPER development subset. HF NF4 uses native stored dtypes, not PEFT casts.
Cross-backend differences are workflow comparisons. BF16 GGUF is local only.
"""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import time

from scripts.qwen38_science_runtime import sha, write

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/qwen38-quant-screen-01"
REGISTRY = ROOT / "results/protocols/qwen38-quant-screen-01.json"


def prepare():
    if RUN.exists() or REGISTRY.exists():
        raise FileExistsError("fresh quantization screen required")
    fixture_path = ROOT / "data/benchmark-screen-01/test.json"
    fixture = json.loads(fixture_path.read_text())
    sampled = {}
    for task in ("general_mc", "gsm8k"):
        selected = list(fixture[task]); random.Random(20261117).shuffle(selected)
        sampled[task] = selected[:32]
    qasper = ROOT / "outputs/qwen38-qasper-evaluation-01/prompt_development.jsonl"
    sampled["qasper_development"] = [json.loads(line) for line in qasper.read_text().split("\n") if line.strip()]
    RUN.mkdir(parents=True); (RUN / "source").mkdir()
    write(RUN / "fixture.json", sampled)
    for name in ("qwen38_quant_screen.py", "benchmark_screen.py", "batched_generation.py", "gguf_eval.py", "qwen38_nf4_smoke.py"):
        shutil.copyfile(ROOT / "scripts" / name, RUN / "source" / name)
    shutil.copyfile(ROOT / "outputs/qwen38-qasper-evaluation-01/source/qwen38_science_score.py", RUN / "source/qwen38_science_score.py")
    protocol = {"scope": "small descriptive capability screen; no training or final QASPER/PRL generation",
        "fixture_sha256": sha(RUN / "fixture.json"), "source_fixture_sha256": sha(fixture_path),
        "qasper_input_sha256": sha(qasper), "arms": ["nf4", "q4", "bf16_gguf"],
        "counts": {task: len(rows) for task, rows in sampled.items()},
        "decoding": {"seed": 42, "batch_size": 1, "thinking": False, "greedy": True, "max_new_tokens": 512},
        "NF4_runtime": "explicit text-only loader, native saved BF16 unquantized weights; no PEFT preparation",
        "reference": "BF16 GGUF within llama.cpp; Transformers BF16 not required for this bounded screen",
        "maximum_total_gpu_hours": 6, "max_seconds_per_arm": 5400,
        "gpu_accounting": "count visible assigned GPUs times model-phase elapsed wall time",
        "source_sha256": {p.name: sha(p) for p in (RUN / "source").glob("*.py")}}
    write(RUN / "registration.json", protocol); write(REGISTRY, protocol)
    return {"prepared": str(RUN.relative_to(ROOT)), "counts": protocol["counts"]}


def execute(arm, gpus):
    protocol = json.loads(REGISTRY.read_text())
    if REGISTRY.read_bytes() != (RUN / "registration.json").read_bytes() or sha(RUN / "fixture.json") != protocol["fixture_sha256"]:
        raise ValueError("registered quantization screen changed")
    for name, expected in protocol["source_sha256"].items():
        if sha(RUN / "source" / name) != expected:
            raise ValueError("quantization screen snapshot changed")
    out = RUN / arm
    if out.exists():
        raise FileExistsError("arm already attempted")
    out.mkdir()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    from scripts.batched_generation import generate_batched
    from scripts.benchmark_screen import GGUFBackend, hf_multiple_choice
    from scripts.qwen38_nf4_smoke import validate_loading
    from evaluation.gsm8k import score_gsm8k
    fixture = json.loads((RUN / "fixture.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "artifacts/qwen38-27b-nf4-01", local_files_only=True)
    scorer_spec = importlib.util.spec_from_file_location("quant_qasper_score", RUN / "source/qwen38_science_score.py")
    scorer = importlib.util.module_from_spec(scorer_spec); scorer_spec.loader.exec_module(scorer)
    torch.manual_seed(42)
    start = time.monotonic(); backend = None
    result = {"completed": False, "arm": arm, "gpus": gpus, "scores": {}, "gpu_count": len(gpus.split(","))}
    write(out / "result.json", result)
    try:
        if arm == "nf4":
            base = ROOT / "artifacts/qwen38-27b-nf4-01"
            artifact = json.loads((ROOT / "results/qwen38_27b_nf4_artifact_01.json").read_text())
            for name, expected in artifact["files"].items():
                if sha(base / name) != expected["sha256"]:
                    raise ValueError("saved NF4 changed")
            model, info = Qwen3_5ForCausalLM.from_pretrained(base, local_files_only=True, trust_remote_code=False,
                dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa", output_loading_info=True)
            validate_loading(info); model.eval(); model.config.use_cache = True
            result["native_parameter_dtypes"] = dict(Counter(str(p.dtype) for p in model.parameters()))
        else:
            filename = "Qwen3.8-27B-text-Q4_K_M.gguf" if arm == "q4" else "Qwen3.8-27B-text-BF16.gguf"
            model_file = ROOT / "artifacts/qwen38-27b-gguf-01" / filename
            recorded = json.loads((ROOT / "outputs/qwen38-27b-gguf-01/result.json").read_text())["files"][filename]
            if sha(model_file) != recorded["sha256"]:
                raise ValueError("converted GGUF changed")
            backend = GGUFBackend(str(model_file), tokenizer, out)
        def deadline():
            if time.monotonic() - start > protocol["max_seconds_per_arm"]:
                raise TimeoutError("registered quantization arm cap")
        deadline()
        if backend:
            mc = backend.multiple_choice(fixture["general_mc"])
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mc = hf_multiple_choice(model, tokenizer, fixture["general_mc"], batch_size=1)
        write(out / "general_mc.json", mc)
        result["scores"]["general_mc"] = {"correct": sum(r["correct"] for r in mc), "total": len(mc)}
        for task in ("gsm8k", "qasper_development"):
            predictions = []
            for row in fixture[task]:
                deadline()
                if backend:
                    prediction = backend.generate([row], 512)[0]
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction = generate_batched(model, tokenizer, [row], batch_size=1, max_new_tokens=512, seed=42)["predictions"][0]
                predictions.append(prediction)
                with (out / f"{task}_predictions.jsonl").open("a") as log:
                    log.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                print(json.dumps({"arm": arm, "task": task, "done": len(predictions), "total": len(fixture[task])}), flush=True)
            if task == "gsm8k":
                scored = score_gsm8k(fixture[task], predictions)
            else:
                scored = scorer.score_qasper(fixture[task], {r["id"]: r["output"] for r in predictions})
            write(out / f"{task}_scores.json", scored)
            result["scores"][task] = {k: v for k, v in scored.items() if k not in ("details", "strict", "lenient")}
            result["elapsed_seconds"] = time.monotonic() - start
            write(out / "result.json", result)
        result["completed"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if backend:
            backend.close()
        result["elapsed_seconds"] = time.monotonic() - start
        result["gpu_hours"] = result["gpu_count"] * result["elapsed_seconds"] / 3600
        write(out / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", choices=["nf4", "q4", "bf16_gguf"])
    parser.add_argument("--gpus", default="0")
    args = parser.parse_args()
    if not args.gpus.replace(",", "").isdecimal() or len(args.gpus.split(",")) > 2:
        parser.error("one or two explicit GPUs required")
    print(json.dumps(prepare() if args.prepare else execute(args.execute, args.gpus), indent=2))


if __name__ == "__main__":
    main()
