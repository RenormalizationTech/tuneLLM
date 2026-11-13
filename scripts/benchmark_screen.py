"""Paired local MMLU-likelihood, GSM8K, official IFEval and physics-writing screens."""
import argparse
import importlib.util
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import types
from urllib.request import ProxyHandler, build_opener

from scripts.batched_generation import generate_batched, left_pad
from scripts.common import digest, load_config, local_snapshot, path, provenance, read_json, select_gpu, setup_hf, write_json
from scripts.gguf_eval import DEFAULT_SERVER, request_json, stop_child, wait_ready


def ifeval_scores(rows, predictions):
    expected = [str(row["key"]) for row in rows]
    received = [p["id"] for p in predictions]
    if not rows or len(set(expected)) != len(expected) or len(set(received)) != len(received) or set(received) != set(expected):
        raise ValueError("IFEval prediction IDs must match unique fixture keys exactly")
    package = types.ModuleType("instruction_following_eval")
    package.__path__ = [str(path("artifacts/tools/ifeval"))]
    sys.modules["instruction_following_eval"] = package
    import nltk
    nltk.data.path.insert(0, str(path("artifacts/tools/nltk")))
    from instruction_following_eval.evaluation_lib import InputExample, test_instruction_following_strict
    by_id = {p["id"]: p["output"] for p in predictions}
    records = []
    for row in rows:
        example = InputExample(**row)
        output = test_instruction_following_strict(example, {row["prompt"]: by_id[str(row["key"])]})
        records.append({"id": str(row["key"]), "prompt_strict": output.follow_all_instructions,
                        "instruction_strict": output.follow_instruction_list})
    return {"total": len(rows), "prompt_strict_correct": sum(r["prompt_strict"] for r in records),
            "instruction_strict_correct": sum(sum(r["instruction_strict"]) for r in records),
            "instruction_total": sum(len(r["instruction_strict"]) for r in records), "records": records}


def writing_checks(row, output):
    number = r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?"
    source_numbers = set(re.findall(number, row["source"]))
    generated_numbers = set(re.findall(number, output))
    cites = lambda text: set(k.strip() for group in re.findall(r"\\cite\w*\s*\{([^}]+)\}", text) for k in group.split(","))
    unsupported_numbers = sorted(generated_numbers - source_numbers)
    unsupported_cites = sorted(cites(output) - cites(row["source"]))
    return {"id": row["id"], "task": row["task"], "nonempty": bool(output.strip()),
            "new_number_flags": unsupported_numbers, "new_citation_flags": unsupported_cites,
            "word_count": len(output.split()),
            "automatic_fidelity_flag": bool(unsupported_numbers or unsupported_cites),
            "human_scientific_fidelity": None, "human_style_preference": None,
            "note": "Flags need review; absence of flags does not establish claim fidelity or writing quality."}


def choice_token_ids(tokenizer, rows):
    letters = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in [" A", " B", " C", " D"]]
    if any(len(x) != 1 for x in letters) or len({tuple(x) for x in letters}) != 4:
        raise ValueError("this benchmark requires one-token answer choices")
    ids = [x[0] for x in letters]
    for row in rows:
        if row["choices"] != [" A", " B", " C", " D"] or row["answer_index"] not in range(4):
            raise ValueError("expected four standard MMLU answer letters and answer_index0..3")
        prompt = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        for choice, token in zip(row["choices"], ids):
            if tokenizer(row["prompt"] + choice, add_special_tokens=False)["input_ids"] != prompt + [token]:
                raise ValueError("answer boundary tokenization mismatch")
    return ids


def attach_adapter(model, checkpoint):
    from peft import PeftModel
    # Keep BF16 frozen base weights BF16. K-bit preparation would upcast them.
    return PeftModel.from_pretrained(model, path(checkpoint), is_trainable=False,
                                     local_files_only=True, autocast_adapter_dtype=True)


def hf_multiple_choice(model, tokenizer, rows, batch_size=16):
    import torch
    ids = choice_token_ids(tokenizer, rows)
    result = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [tokenizer(row["prompt"], add_special_tokens=False)["input_ids"] for row in batch]
        inputs = {k: torch.tensor(v, device="cuda:0") for k, v in left_pad(prompts, tokenizer.pad_token_id).items()}
        with torch.inference_mode():
            logits = model(**inputs, logits_to_keep=1).logits[:, -1].float()
            values = torch.log_softmax(logits, dim=-1)[:, ids].cpu().tolist()
        for row, scores in zip(batch, values):
            prediction = max(range(4), key=lambda i: scores[i])
            result.append({"id": row["id"], "subject": row["subject"], "answer_index": row["answer_index"],
                           "predicted_index": prediction, "correct": prediction == row["answer_index"],
                           "choice_loglikelihood": scores})
    return result


class GGUFBackend:
    def __init__(self, filename, tokenizer, out):
        self.tokenizer = tokenizer
        self.opener = build_opener(ProxyHandler({}))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        self.log = (out / "server.log").open("w")
        self.child = None
        try:
            self.child = subprocess.Popen([str(path(DEFAULT_SERVER)), "-m", str(path(filename)), "--host", "127.0.0.1",
                "--port", str(port), "-c", "8192", "-ngl", "99", "--parallel", "1", "-t", "4", "-fa", "on"],
                stdout=self.log, stderr=subprocess.STDOUT)
            wait_ready(self.child, self.opener, self.base, 180)
        except BaseException:
            self.close()
            raise

    def close(self):
        stop_child(self.child)
        self.log.close()

    def completion(self, prompt, cap, probabilities=0):
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        actual = request_json(self.opener, self.base, "/tokenize", {"content": prompt, "add_special": False, "parse_special": True})["tokens"]
        if actual != ids:
            raise ValueError("GGUF input token IDs differ from HF")
        response = request_json(self.opener, self.base, "/completion", {"prompt": ids, "n_predict": cap,
            "temperature": 0, "seed": 42, "repeat_penalty": 1.0, "return_tokens": True,
            "n_probs": probabilities, "post_sampling_probs": False,
            "cache_prompt": False, "samplers": ["temperature"]}, timeout=300)
        if response.get("truncated") or response.get("tokens_evaluated") != len(ids):
            raise ValueError("GGUF completion context truncated or evaluated prompt count differs from verified IDs")
        return response

    def multiple_choice(self, rows):
        ids = choice_token_ids(self.tokenizer, rows)
        result = []
        for row in rows:
            response = self.completion(row["prompt"], 1, 512)
            probs = response.get("completion_probabilities", response.get("probs"))[0]
            scores = {p["id"]: p["logprob"] for p in probs["top_logprobs"]}
            if not all(i in scores for i in ids):
                response = self.completion(row["prompt"], 1, len(self.tokenizer))
                probs = response.get("completion_probabilities", response.get("probs"))[0]
                scores = {p["id"]: p["logprob"] for p in probs["top_logprobs"]}
            if not all(i in scores for i in ids):
                raise ValueError("GGUF log-probability response is missing answer choices after full-vocabulary retry")
            values = [scores[i] for i in ids]
            prediction = max(range(4), key=lambda i: values[i])
            result.append({"id": row["id"], "subject": row["subject"], "answer_index": row["answer_index"],
                           "predicted_index": prediction, "correct": prediction == row["answer_index"],
                           "choice_loglikelihood": values})
        return result

    def generate(self, rows, cap):
        from scripts.batched_generation import render_prompt
        records = []
        for row in rows:
            prompt = render_prompt(self.tokenizer, row)
            response = self.completion(prompt, cap)
            records.append({"id": row["id"], "output": response["content"], "generated_ids": response["tokens"],
                            "input_ids": self.tokenizer(prompt, add_special_tokens=False)["input_ids"],
                            "hit_token_limit": response.get("stop_type") == "limit", "rendered_prompt": prompt,
                            "server_stop_type": response.get("stop_type"), "server_reported_timings": response.get("timings")})
        return records


def run_screen(model, tokenizer, fixture, out, writing=None, gguf=None, tasks=None, diagnostics=None):
    from evaluation.gsm8k import score_gsm8k
    selected = set(tasks or ["general_mc", "physics_mc", "gsm8k", "ifeval", "writing"])
    summary = {}
    diagnostics = diagnostics if diagnostics is not None else []
    for task in ["general_mc", "physics_mc"]:
        if task not in selected:
            continue
        rows = fixture[task]
        records = gguf.multiple_choice(rows) if gguf else hf_multiple_choice(model, tokenizer, rows)
        write_json(out / f"{task}.json", records)
        summary[task] = {"total": len(records), "correct": sum(r["correct"] for r in records)}
        print(task, summary[task], flush=True)
    def generate(rows, cap, task):
        started = time.perf_counter()
        if gguf:
            records = gguf.generate(rows, cap)
            model_timing = None
        else:
            generated = generate_batched(model, tokenizer, rows, batch_size=8,
                                         max_new_tokens=cap, seed=42, phase="screen/" + task)
            records = generated["predictions"]
            model_timing = generated["timing"]
        diagnostics.append({"task": task, "rows": len(rows), "max_new_tokens": cap,
                            "pipeline_wall_seconds": time.perf_counter() - started,
                            "returned_generated_token_ids": sum(len(row["generated_ids"]) for row in records),
                            "hf_generation_timing": model_timing})
        return records
    if "gsm8k" in selected:
        rows = fixture["gsm8k"]
        predictions = generate(rows, 512, "gsm8k")
        write_json(out / "gsm8k_predictions.json", predictions)
        scores = score_gsm8k(rows, predictions)
        write_json(out / "gsm8k_scores.json", scores)
        summary["gsm8k"] = {k: v for k, v in scores.items() if k != "details"}
        print("gsm8k", summary["gsm8k"], flush=True)
    if "ifeval" in selected:
        rows = fixture["ifeval"]
        predictions = generate([{"id": str(r["key"]), "user": r["prompt"]} for r in rows], 1024, "ifeval")
        write_json(out / "ifeval_predictions.json", predictions)
        scores = ifeval_scores(rows, predictions)
        write_json(out / "ifeval_scores.json", scores)
        summary["ifeval"] = {k: v for k, v in scores.items() if k != "records"}
        print("ifeval", summary["ifeval"], flush=True)
    if writing and "writing" in selected:
        predictions = []
        for task, cap in [("title", 48), ("abstract", 256), ("introduction", 768)]:
            rows = [r for r in writing if r["task"] == task]
            if rows:
                predictions += generate(rows, cap, "writing/" + task)
        write_json(out / "writing_predictions.json", predictions)
        by_id = {p["id"]: p for p in predictions}
        checks = [writing_checks(row, by_id[row["id"]]["output"]) for row in writing]
        write_json(out / "writing_checks.json", checks)
        summary["writing"] = {"total": len(checks), "nonempty": sum(c["nonempty"] for c in checks),
                              "automatic_fidelity_flags": sum(c["automatic_fidelity_flag"] for c in checks),
                              "human_ratings": "pending", "quality_improvement_established": False}
    write_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=["nf4", "bf16", "gguf"], default="nf4")
    parser.add_argument("--adapter")
    parser.add_argument("--gguf")
    parser.add_argument("--fixture", default="data/benchmark-screen-01/development.json")
    parser.add_argument("--writing")
    parser.add_argument("--tasks", nargs="+", choices=("general_mc", "physics_mc", "gsm8k", "ifeval", "writing"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config("configs/baseline.json")
    select_gpu("1")
    setup_hf(cfg)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from scripts.save_nf4 import reload_stored_nf4
    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    fixture = read_json(args.fixture)
    writing = read_json(args.writing)["tasks"] if args.writing else None
    tokenizer = AutoTokenizer.from_pretrained(local_snapshot(cfg), local_files_only=True)
    backend = None
    started = time.perf_counter()
    metadata = {**provenance(cfg), "completed": False, "representation": args.representation, "adapter": args.adapter,
                "fixture_sha256": digest(args.fixture), "writing_sha256": digest(args.writing) if args.writing else None,
                "model_revision": cfg["revision"], "seed": 42, "thinking": False, "mc_batch": 16,
                "generation_batch": 1 if args.representation == "gguf" else 8,
                "multiple_choice": "raw standard zero-shot MMLU letter likelihood; custom local runner",
                "cross_backend_caveat": "same input tokens and settings; numerical/runtime differences remain"}
    write_json(out / "run.json", metadata)
    try:
        if args.representation == "gguf":
            if not args.gguf or args.adapter:
                raise ValueError("GGUF requires --gguf and no adapter")
            backend = GGUFBackend(args.gguf, tokenizer, out)
            model = None
        elif args.representation == "nf4":
            model = reload_stored_nf4(path("artifacts/nf4/Qwen3-4B"))
        else:
            model = AutoModelForCausalLM.from_pretrained(local_snapshot(cfg), local_files_only=True,
                torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
        if args.adapter:
            model = attach_adapter(model, args.adapter)
        if model is not None:
            model.eval()
            torch.cuda.reset_peak_memory_stats()
        diagnostics = []
        run_screen(model, tokenizer, fixture, out, writing, backend, args.tasks, diagnostics)
        generation_seconds = sum(row["pipeline_wall_seconds"] for row in diagnostics)
        generated_tokens = sum(row["returned_generated_token_ids"] for row in diagnostics)
        metadata["generation_diagnostics"] = {"groups": diagnostics, "pipeline_wall_seconds": generation_seconds,
            "returned_generated_token_ids": generated_tokens,
            "returned_token_ids_per_pipeline_wall_second": generated_tokens / generation_seconds if generation_seconds else None,
            "scope": "Generation pipeline wall includes rendering/tokenization/transfers/decoding; GGUF includes token parity HTTP requests. HF IDs include terminal EOS; GGUF returned IDs use server conventions. Diagnostic, not decode-only throughput; excludes scoring and MC likelihood."}
        metadata["peak_pytorch_allocated_bytes"] = torch.cuda.max_memory_allocated() if model is not None else None
        metadata["peak_pytorch_reserved_bytes"] = torch.cuda.max_memory_reserved() if model is not None else None
        metadata["memory_scope"] = "PyTorch allocator during screen, including loaded weights; excludes other processes and GGUF child allocations (null for GGUF)"
        metadata.update({"completed": True, "wall_seconds": time.perf_counter() - started})
    except BaseException as exc:
        metadata.update({"error": str(exc), "error_type": type(exc).__name__})
        raise
    finally:
        if backend:
            backend.close()
        write_json(out / "run.json", metadata)


if __name__ == "__main__":
    main()
