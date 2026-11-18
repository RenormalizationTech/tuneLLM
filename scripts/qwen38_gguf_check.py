"""Inspect converted text GGUF; optional bounded token-parity/generation smoke."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener

from scripts.qwen38_science_runtime import sha, write
from scripts.gguf_eval import request_json, wait_ready, stop_child

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "artifacts/tools/llama.cpp"


def metadata(model):
    sys.path.insert(0, str(TOOLS / "gguf-py"))
    from gguf import GGUFReader
    reader = GGUFReader(model)
    def contents(name):
        value = reader.fields[name].contents()
        return value.item() if hasattr(value, "item") else value
    names = [tensor.name for tensor in reader.tensors]
    if any(any(word in name.lower() for word in ("vision", "visual", "mmproj", "mtp", "nextn")) for name in names):
        raise ValueError("unexpected vision/MTP tensor in text-only GGUF")
    if contents("general.architecture") != "qwen35":
        raise ValueError("unexpected GGUF architecture")
    template = contents("tokenizer.chat_template")
    if not isinstance(template, str):
        raise ValueError("single recorded chat template required")
    return {"architecture": contents("general.architecture"), "tensor_count": len(names),
            "block_count": contents("qwen35.block_count"),
            "tensor_types": dict(Counter(str(t.tensor_type) for t in reader.tensors)),
            "no_vision_or_mtp_tensor_names": True,
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(), "chat_template": template}


def run(args):
    model = (ROOT / args.model).resolve(); out = (ROOT / args.output).resolve()
    if out.exists() or not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("fresh project output directory required")
    info = metadata(model)
    out.mkdir(parents=True)
    write(out / "metadata.json", info)
    if not args.execute_smoke:
        return {k: v for k, v in info.items() if k != "chat_template"}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "artifacts/qwen38-27b-nf4-01", local_files_only=True)
    prompts = ["What is 2 + 2? Answer with only the number.", "Name the capital of France in one short sentence."]
    prepared = []
    for prompt in prompts:
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
        prepared.append({"prompt": prompt, "rendered": rendered,
                         "ids": tokenizer(rendered, add_special_tokens=False)["input_ids"]})
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0)); port = reservation.getsockname()[1]
    server = TOOLS / "build-gcc11/bin/llama-server"
    command = [str(server), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
               "--ctx-size", "8192", "--parallel", "1", "--threads", "4", "--gpu-layers", "all", "--flash-attn", "auto"]
    if "," in args.gpus:
        command += ["--split-mode", "layer", "--tensor-split", "1,1"]
    cfg = {"scope": "two short text prompts; tokenizer parity and functional load only",
           "model": str(model.relative_to(ROOT)), "model_sha256": sha(model),
           "server_sha256": sha(server), "command": command, "gpus": args.gpus,
           "script_sha256": sha(__file__), "max_new_tokens": 32,
           "cached_generation": True, "thinking": False, "batch_size": 1}
    write(out / "registration.json", cfg)
    opener = build_opener(ProxyHandler({})); base = f"http://127.0.0.1:{port}"
    child = None; result = {"completed": False, "predictions": []}
    start = time.monotonic()
    try:
        with (out / "server.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus),
                                     stdout=log, stderr=subprocess.STDOUT)
            wait_ready(child, opener, base, 300)
            for row in prepared:
                response = request_json(opener, base, "/tokenize", {"content": row["rendered"], "add_special": False, "parse_special": True})
                if response["tokens"] != row["ids"]:
                    write(out / "token_mismatch.json", {"reference": row, "actual": response})
                    raise ValueError("HF/GGUF token parity failed before completion")
            for row in prepared:
                response = request_json(opener, base, "/completion", {"prompt": row["rendered"],
                    "n_predict": 32, "temperature": 0.0, "seed": 42, "cache_prompt": False,
                    "stream": False, "return_tokens": True, "samplers": ["temperature"],
                    "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0}, timeout=300)
                if response.get("truncated") or response.get("tokens_evaluated") != len(row["ids"]):
                    raise ValueError("GGUF prompt truncated or changed")
                result["predictions"].append({**row, "response": response})
            result["completed"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop_child(child)
        result["seconds"] = time.monotonic() - start
        write(out / "result.json", result)
    return {"completed": result["completed"], "seconds": result["seconds"],
            "outputs": [r["response"]["content"] for r in result["predictions"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/qwen38-27b-gguf-01/Qwen3.8-27B-text-Q4_K_M.gguf")
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute-smoke", action="store_true")
    parser.add_argument("--gpus", default="1")
    args = parser.parse_args()
    if not args.gpus.replace(",", "").isdecimal() or len(args.gpus.split(",")) > 2:
        parser.error("one or two explicit GPU indices required")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
