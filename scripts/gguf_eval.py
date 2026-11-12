"""Evaluate a local GGUF against saved HF raw prompts; fail before generation without token parity.

python -m scripts.gguf_eval --model artifacts/model.gguf \
    --reference-run outputs/hf-baseline-01 --gpu 1 --output outputs/gguf-run
Uses the pinned llama.cpp /health, /tokenize, /props, and raw /completion APIs.
"""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from evaluation.score import score
from scripts.common import (ROOT, command, digest, path, provenance, read_json,
                            select_gpu, write_json)

DEFAULT_SERVER = "artifacts/tools/llama.cpp/build-gcc11/bin/llama-server"
PINNED_TOOL_REVISION = "8ca5a8008b57aa39f2601546c0d07912045080a5"


def request_json(opener, base, endpoint, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    request = Request(base + endpoint, data=data, headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict) or "error" in result:
        raise RuntimeError(f"unexpected {endpoint} response: {result}")
    return result


def wait_ready(child, opener, base, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"llama-server exited with {child.returncode}; inspect server.log")
        try:
            result = request_json(opener, base, "/health", timeout=min(2, max(0.01, deadline - time.monotonic())))
            if result.get("status") == "ok" and child.poll() is None:
                return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(min(0.25, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"llama-server health deadline exceeded: {last_error}")


def stop_child(child):
    """Only signal the Popen child created by this invocation."""
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-bin", default=DEFAULT_SERVER)
    parser.add_argument("--reference-run", default="outputs/hf-baseline-01")
    parser.add_argument("--gpu", default="1", help="one physical GPU index or UUID")
    parser.add_argument("--output", required=True, help="new directory; refuses overwrite")
    parser.add_argument("--gpu-layers", default="all", help="use 0 for an explicitly recorded CPU fallback")
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=300)
    args = parser.parse_args()
    model, server, reference = path(args.model).resolve(), path(args.server_bin).resolve(), path(args.reference_run).resolve()
    if not model.is_file() or model.suffix.lower() != ".gguf":
        parser.error("--model must be an existing local GGUF")
    if not server.is_file():
        parser.error("--server-bin must be an existing local llama-server")
    if min(args.ctx_size, args.threads, args.startup_timeout, args.request_timeout) <= 0:
        parser.error("context, threads, and timeout values must be positive")
    if args.gpu_layers != "all" and (not args.gpu_layers.isdecimal()):
        parser.error("--gpu-layers must be all or a nonnegative integer")
    select_gpu(args.gpu)
    reference_run = read_json(reference / "run.json")
    if reference_run.get("completed") is not True:
        raise ValueError("reference HF run must have completed successfully")
    reference_cfg = reference_run["config"]
    generation = reference_cfg["generation"]
    if reference_cfg["enable_thinking"] is not False or generation["do_sample"] is not False or generation["num_beams"] != 1 or generation["repetition_penalty"] != 1.0:
        raise ValueError("reference must use non-thinking greedy generation with repetition_penalty=1")
    fixture = read_json(reference / "fixture.json")
    if digest(reference_cfg["fixture"]) != reference_run["fixture_sha256"] or read_json(reference_cfg["fixture"]) != fixture:
        raise ValueError("shared fixture changed since reference run")
    reference_predictions = read_json(reference / "predictions.json")
    score(fixture["tasks"], reference_predictions)  # Enforce exact, unique reference IDs.
    by_id = {row["id"]: row for row in reference_predictions}
    for row in reference_predictions:
        if not isinstance(row.get("rendered_prompt"), str) or not row.get("input_ids") or not all(type(token) is int for token in row["input_ids"]):
            raise ValueError("reference must contain rendered_prompt and input_ids for every task")
        if len(row["input_ids"]) + generation["max_new_tokens"] >= args.ctx_size:
            raise ValueError("context too small for saved prompt plus generation budget")

    # A temporary loopback bind picks an unused port without attaching to another server.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    server_command = [str(server), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
                      "--ctx-size", str(args.ctx_size), "--parallel", "1", "--threads", str(args.threads),
                      "--gpu-layers", args.gpu_layers, "--flash-attn", "auto"]
    completion_settings = {"n_predict": generation["max_new_tokens"], "temperature": 0.0,
                           "seed": reference_cfg["seed"], "repeat_penalty": 1.0,
                           "cache_prompt": False, "stream": False, "return_tokens": True,
                           "samplers": ["temperature"], "top_k": 0, "top_p": 1.0, "min_p": 0.0,
                           "presence_penalty": 0.0, "frequency_penalty": 0.0,
                           "dry_multiplier": 0.0, "mirostat": 0, "stop": []}
    resolved = {"model": str(model), "server_bin": str(server), "reference_run": str(reference),
                "gpu": args.gpu, "server_command": server_command,
                "startup_timeout": args.startup_timeout, "request_timeout": args.request_timeout,
                "completion_settings": completion_settings,
                "warmup_runs": reference_cfg["warmup_runs"], "reference_config": reference_cfg}
    out = path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    metadata = provenance(resolved)
    metadata.update({"completed": False, "status": "starting", "runtime": "llama.cpp raw /completion, one slot",
                     "model_file_bytes": model.stat().st_size, "model_mtime_ns": model.stat().st_mtime_ns,
                     "server_binary_sha256": digest(server), "expected_tool_revision": PINNED_TOOL_REVISION,
                     "reference_run_sha256": digest(reference / "run.json"),
                     "reference_predictions_sha256": digest(reference / "predictions.json"),
                     "fixture_sha256": reference_run["fixture_sha256"],
                     "caveat": "Eight-item project smoke fixture, not a capability benchmark. Greedy comparison differs from model-card sampling advice. Runtime/precision may change outputs despite matching input tokens.",
                     "measurement": "HTTP request wall time includes request encoding, loopback transport, server queuing/prefill/decode and response parsing; excludes startup, parity checks and warmups. Server timings and token counts are separately labelled and may count EOS differently from HF.",
                     "first_token_latency_seconds": None})
    tool_root = next((p for p in server.parents if (p / "tools/server/README.md").is_file()), None)
    if tool_root:
        metadata["tool_checkout_revision"] = command(["git", "-C", str(tool_root), "rev-parse", "HEAD"])
        metadata["tool_checkout_status"] = command(["git", "-C", str(tool_root), "status", "--porcelain"])
        metadata["server_api_readme_sha256"] = digest(tool_root / "tools/server/README.md")
    write_json(out / "run.json", metadata)
    write_json(out / "config.json", resolved)
    write_json(out / "fixture.json", fixture)
    opener = build_opener(ProxyHandler({}))  # Always use loopback directly, ignoring proxy environment.
    child = None
    try:
        with (out / "server.log").open("w", encoding="utf-8") as log:
            started = time.perf_counter()
            child = subprocess.Popen(server_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            metadata["child_pid"] = child.pid
            wait_ready(child, opener, base, args.startup_timeout)
            metadata["startup_wall_seconds"] = time.perf_counter() - started
            props = request_json(opener, base, "/props", timeout=args.request_timeout)
            write_json(out / "server_props.json", props)
            metadata["server_build_info"] = props.get("build_info")
            if path(props["model_path"]).resolve() != model:
                raise RuntimeError("ready server reports a different model path")
            parity = []
            for task in fixture["tasks"]:
                reference_row = by_id[task["id"]]
                actual = request_json(opener, base, "/tokenize", {
                    "content": reference_row["rendered_prompt"], "add_special": False, "parse_special": True,
                }, args.request_timeout)["tokens"]
                matched = actual == reference_row["input_ids"]
                parity.append({"id": task["id"], "matches_hf": matched,
                               "hf_input_ids": reference_row["input_ids"], "gguf_input_ids": actual})
            write_json(out / "token_parity.json", parity)
            metadata["all_input_tokens_match_hf"] = all(row["matches_hf"] for row in parity)
            if not metadata["all_input_tokens_match_hf"]:
                raise RuntimeError("GGUF/HF token parity failed; no completions requested; inspect token_parity.json")

            def complete(reference_row):
                if child.poll() is not None:
                    raise RuntimeError("llama-server exited before completion")
                started = time.perf_counter()
                response = request_json(opener, base, "/completion", {
                    **completion_settings, "prompt": reference_row["rendered_prompt"],
                }, args.request_timeout)
                wall = time.perf_counter() - started
                if response.get("tokens_evaluated") != len(reference_row["input_ids"]):
                    write_json(out / "prompt_count_failure.json", response)
                    raise RuntimeError("completion prompt count differs from verified tokens (possible automatic special token)")
                return response, wall

            warmups = []
            for _ in range(resolved["warmup_runs"]):
                response, wall = complete(by_id[fixture["tasks"][0]["id"]])
                warmups.append({"http_wall_seconds": wall, "server_reported_timings": response.get("timings")})
            write_json(out / "warmup.json", warmups)
            predictions = []
            for task in fixture["tasks"]:
                reference_row = by_id[task["id"]]
                response, wall = complete(reference_row)
                predicted = response.get("tokens_predicted")
                predictions.append({"id": task["id"], "output": response["content"],
                                    "rendered_prompt": reference_row["rendered_prompt"],
                                    "input_ids": reference_row["input_ids"], "generated_ids": response.get("tokens"),
                                    "output_matches_hf_text": response["content"] == reference_row["output"],
                                    "hit_token_limit": response.get("stop_type") == "limit",
                                    "context_truncated": response.get("truncated"),
                                    "http_wall_seconds": wall, "server_reported_predicted_tokens": predicted,
                                    "server_reported_tokens_per_http_wall_second": predicted / wall if predicted is not None else None,
                                    "server_reported_timings": response.get("timings"), "server_response": response})
                write_json(out / "predictions.json", predictions)
                print(f"{task['id']}: {wall:.3f}s HTTP wall, {predicted} server-reported predicted tokens", flush=True)
                if response.get("truncated"):
                    raise RuntimeError("server reported context truncation; inspect predictions.json")
            write_json(out / "scores.json", score(fixture["tasks"], predictions))
            total_wall = sum(row["http_wall_seconds"] for row in predictions)
            counts = [row["server_reported_predicted_tokens"] for row in predictions]
            metadata.update({"completed": True, "status": "complete", "total_http_wall_seconds": total_wall,
                             "server_reported_tokens_per_http_wall_second": sum(counts) / total_wall if all(n is not None for n in counts) else None})
    except BaseException as exc:
        metadata.update({"completed": False, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        stop_child(child)
        metadata["child_returncode_after_cleanup"] = None if child is None else child.returncode
        write_json(out / "run.json", metadata)
    print(out)


if __name__ == "__main__":
    main()
