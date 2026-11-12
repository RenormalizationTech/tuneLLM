"""Shared stdlib configuration, local-cache setup, and provenance."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "transformers", "accelerate", "huggingface-hub", "peft",
            "bitsandbytes", "datasets", "tokenizers", "safetensors")


def path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def read_json(filename):
    return json.loads(path(filename).read_text(encoding="utf-8"))


def write_json(filename, value):
    p = path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                 encoding="utf-8")


def digest(filename):
    return hashlib.sha256(path(filename).read_bytes()).hexdigest()


def load_config(filename):
    cfg = read_json(filename)
    if not re.fullmatch(r"[0-9a-f]{40}", cfg.get("revision", "")):
        raise ValueError("revision must be an immutable 40-character commit SHA")
    if cfg.get("enable_thinking") is not False or cfg.get("dtype") != "bfloat16":
        raise ValueError("initial baseline requires BF16 and enable_thinking=false")
    return cfg


def setup_hf(cfg, offline=True):
    home = path(cfg["hf_home"]).resolve()
    os.environ["HF_HOME"] = str(home)
    os.environ["HF_HUB_CACHE"] = str(home / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(home / "datasets")
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return home / "hub"


def select_gpu(gpu):
    gpu = str(gpu)
    if not re.fullmatch(r"\d+|GPU-[0-9a-fA-F-]+", gpu):
        raise ValueError("select exactly one physical GPU index or GPU UUID")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("this CLI supports one process on one selected GPU")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return gpu


def command(args):
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def environment(disk_path=ROOT):
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    disk = shutil.disk_usage(path(disk_path))
    return {
        "utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "executable": sys.executable, "platform": platform.platform(), "packages": packages,
        "disk": {"path": str(path(disk_path)), "total_bytes": disk.total,
                 "used_bytes": disk.used, "free_bytes": disk.free},
        "nvidia_smi": command(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used",
                               "--format=csv,noheader,nounits"]),
    }


def provenance(cfg):
    sources = {}
    for directory, pattern in (("scripts", "*.py"), ("evaluation", "*.py"),
                               ("configs", "*.json"), ("", "requirements*.txt")):
        for p in sorted((ROOT / directory).glob(pattern)):
            sources[str(p.relative_to(ROOT))] = digest(p)
    return {"config": cfg, "environment": environment(), "argv": sys.argv,
            "git_head": command(["git", "rev-parse", "HEAD"]),
            "git_status": command(["git", "status", "--porcelain"]),
            "source_sha256": sources,
            "pip_freeze": command([sys.executable, "-m", "pip", "freeze"])}


def required_weights(files):
    """Only the canonical safetensors checkpoint, never .bin alternatives."""
    files = set(files)
    if "model.safetensors.index.json" in files:
        return sorted(f for f in files if re.fullmatch(r"model-\d+-of-\d+\.safetensors", f)) + ["model.safetensors.index.json"]
    if "model.safetensors" in files:
        return ["model.safetensors"]
    raise ValueError("no canonical safetensors checkpoint found")


def validate_snapshot(snapshot):
    snapshot = Path(snapshot)
    weights = required_weights(p.name for p in snapshot.iterdir())
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not shards or any(not (snapshot / shard).is_file() for shard in shards):
            raise ValueError("local snapshot is missing indexed safetensors shards")
        weights = sorted(shards)
    if not (snapshot / "config.json").is_file() or not (snapshot / "tokenizer_config.json").is_file():
        raise ValueError("local snapshot is missing model/tokenizer configuration")
    return sum((snapshot / f).stat().st_size for f in weights if f.endswith(".safetensors"))


def local_snapshot(cfg):
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(cfg["model_id"], revision=cfg["revision"],
                                     cache_dir=str(path(cfg["hf_home"]) / "hub"),
                                     local_files_only=True))
    validate_snapshot(snapshot)
    return snapshot
