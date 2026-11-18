"""Prepare and execute a hash-bound, CPU-only text BF16 -> Q4_K_M conversion.

Preparation hashes the official source and records exact commands before any
conversion. Execution requires that unchanged registration and a fresh artifact
directory. The BF16 GGUF is a local reference, not an additional quantization.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "artifacts/tools/llama.cpp"
TOOL_REV = "8ca5a8008b57aa39f2601546c0d07912045080a5"
SNAPSHOT = ROOT / ".hf/hub/models--Qwen--Qwen3.8-27B/snapshots/37561816e66b5b9153b1b64c10968c18c1e6f4a0"
ARTIFACT = ROOT / "artifacts/qwen38-27b-gguf-01"
RUN = ROOT / "outputs/qwen38-27b-gguf-01"
PROTOCOL = ROOT / "results/protocols/qwen38-27b-gguf-01.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tool_identity():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TOOLS, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=TOOLS, text=True)
    if revision != TOOL_REV or dirty:
        raise ValueError("unchanged pinned llama.cpp source required")
    return {"revision": revision, "converter_sha256": sha(TOOLS / "convert_hf_to_gguf.py"),
            "quantizer_sha256": sha(TOOLS / "build-cpu/bin/llama-quantize"),
            "server_sha256": sha(TOOLS / "build-gcc11/bin/llama-server")}


def source_identity():
    expected = json.loads((ROOT / "outputs/qwen38-27b-nf4-01/source.json").read_text())
    verified = {}
    for name, record in expected["files"].items():
        path = SNAPSHOT / name
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError(f"source checkpoint changed: {name}")
        verified[name] = record
    return verified


def prepare():
    if RUN.exists() or ARTIFACT.exists() or PROTOCOL.exists():
        raise FileExistsError("fresh conversion run, artifact and registration required")
    if shutil.disk_usage(ROOT).free < 100_000_000_000:
        raise ValueError("at least 100 GB free required for local BF16/Q4 conversion")
    tools = tool_identity()
    files = source_identity()
    RUN.mkdir(parents=True)
    (RUN / "source").mkdir()
    shutil.copyfile(__file__, RUN / "source/qwen38_gguf.py")
    converter = [sys.executable, str(TOOLS / "convert_hf_to_gguf.py"), str(SNAPSHOT),
                 "--outfile", str(ARTIFACT / "Qwen3.8-27B-text-BF16.gguf"),
                 "--outtype", "bf16", "--no-mtp"]
    quantizer = [str(TOOLS / "build-cpu/bin/llama-quantize"),
                 str(ARTIFACT / "Qwen3.8-27B-text-BF16.gguf"),
                 str(ARTIFACT / "Qwen3.8-27B-text-Q4_K_M.gguf"), "Q4_K_M", "16"]
    protocol = {"scope": "CPU text-only conversion and standard Q4_K_M, no imatrix, no quality claims",
                "model_id": "Qwen/Qwen3.8-27B", "revision": SNAPSHOT.name,
                "source_files": files, "tools": tools,
                "script_sha256": sha(__file__), "commands": {"bf16": converter, "q4": quantizer},
                "timeouts_seconds": {"bf16": 3600, "q4": 1800},
                "python": sys.version, "python_executable": sys.executable,
                "artifact": str(ARTIFACT.relative_to(ROOT)), "run": str(RUN.relative_to(ROOT)),
                "vision": "text model conversion; no mmproj", "mtp": "explicit --no-mtp",
                "bf16_role": "local same-backend reference; no BF16 publication selected",
                "registration": "Exact source/script/tool hashes captured before conversion; not a task-quality protocol"}
    write(RUN / "registration.json", protocol)
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    write(PROTOCOL, protocol)
    (RUN / "registration.json").chmod(0o444)
    (RUN / "source/qwen38_gguf.py").chmod(0o444)
    return {"prepared": True, "protocol": str(PROTOCOL.relative_to(ROOT)), "sha256": sha(PROTOCOL)}


def execute():
    if ARTIFACT.exists():
        raise FileExistsError("refuses existing artifact directory")
    protocol = json.loads(PROTOCOL.read_text())
    if PROTOCOL.read_bytes() != (RUN / "registration.json").read_bytes():
        raise ValueError("conversion protocol changed")
    if sha(__file__) != protocol["script_sha256"] or tool_identity() != protocol["tools"]:
        raise ValueError("conversion source/tool changed")
    if sys.executable != protocol["python_executable"]:
        raise ValueError("use the registered interpreter")
    if source_identity() != protocol["source_files"]:
        raise ValueError("official checkpoint changed")
    ARTIFACT.mkdir(parents=True)
    started = time.monotonic()
    result = {"completed": False, "protocol_sha256": sha(PROTOCOL), "stages": {}}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    try:
        for stage in ("bf16", "q4"):
            tick = time.monotonic()
            with (RUN / f"{stage}.log").open("xb") as log:
                child = subprocess.run(protocol["commands"][stage], cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       timeout=protocol["timeouts_seconds"][stage])
            result["stages"][stage] = {"returncode": child.returncode, "seconds": time.monotonic() - tick}
            write(RUN / "result.json", result)
            child.check_returncode()
        result["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                           for p in sorted(ARTIFACT.glob("*.gguf"))}
        if len(result["files"]) != 2:
            raise ValueError("expected exactly BF16 and Q4_K_M GGUF files")
        result["completed"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["quality_evaluated"] = False
        write(RUN / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare() if args.prepare else execute(), indent=2))


if __name__ == "__main__":
    main()
