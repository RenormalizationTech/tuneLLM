"""Pure selection and read-only local snapshots for the modest training run."""
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
from scripts.common import ROOT, digest, path, write_json
from scripts.training_data import OverlengthError, encode_row


def seeded_subset(rows, count, seed, excluded_ids=()):
    excluded = set(excluded_ids)
    selected = [row for row in rows if row["id"] not in excluded]
    if len({row["id"] for row in rows}) != len(rows) or count <= 0 or len(selected) < count:
        raise ValueError("unique IDs and enough eligible rows required")
    random.Random(seed).shuffle(selected)
    return selected[:count]


def select_training_rows(rows, tokenizer, max_length, count, seed):
    eligible, excluded = [], []
    for row in rows:
        try:
            encode_row(row, tokenizer, max_length)
        except OverlengthError:
            excluded.append(row["id"])
        else:
            eligible.append(row)
    return seeded_subset(eligible, count, seed), {
        "source_rows": len(rows), "eligible_rows": len(eligible),
        "overlength_excluded_count": len(excluded), "overlength_excluded_ids": excluded,
        "policy": "Exclude only overlength train rows before seeded subset; all other encoding errors fail. Held-out rows are never filtered or truncated."}


def choose_checkpoint(history, candidate_steps=(32, 64)):
    losses = {}
    for row in history:
        if row.get("step") in candidate_steps and "eval_loss" in row:
            value = float(row["eval_loss"])
            if not math.isfinite(value) or row["step"] in losses:
                raise ValueError("candidate validation losses must be finite and unique")
            losses[row["step"]] = value
    if set(losses) != set(candidate_steps):
        raise ValueError("missing candidate validation loss")
    selected = min(candidate_steps, key=lambda step: (losses[step], step))
    return {"selected_step": selected, "candidate_validation_losses": {str(k): v for k, v in losses.items()},
            "rule": "minimum validation loss among steps 32/64 only; exact tie selects step 32; no test or generation override"}


def write_jsonl(filename, rows):
    Path(filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def snapshot_sources(destination, source_root=ROOT):
    destination = Path(destination)
    for directory in ("scripts", "evaluation", "configs", "tests"):
        for source in sorted((Path(source_root) / directory).rglob("*")):
            if source.is_file() and source.suffix in (".py", ".json"):
                target = destination / source.relative_to(source_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    for source in Path(source_root).glob("requirements*.txt"):
        shutil.copyfile(source, destination / source.name)


def seal_snapshot(out):
    """Freeze source, resolved config and selected input bytes. Mutable results stay outside."""
    out = Path(out)
    files = [out / "config.json", out / "protocol.json"]
    for directory in (out / "source", out / "inputs"):
        files.extend(p for p in directory.rglob("*") if p.is_file())
    manifest = {str(p.relative_to(out)): digest(p) for p in sorted(files)}
    write_json(out / "frozen.json", {"sha256": manifest})
    for p in files + [out / "frozen.json"]:
        p.chmod(0o444)
    for directory in (out / "source", out / "inputs"):
        for p in sorted((p for p in directory.rglob("*") if p.is_dir()), reverse=True):
            p.chmod(0o555)
        directory.chmod(0o555)
    return digest(out / "frozen.json")


def verify_snapshot(out):
    out = Path(out)
    manifest = json.loads((out / "frozen.json").read_text())
    for relative, expected in manifest["sha256"].items():
        if digest(out / relative) != expected:
            raise ValueError(f"frozen input/source changed: {relative}")
    return manifest


def verify_committed_protocol(out, protocol, repo_root):
    """Manager commits this exact label-free protocol before execution."""
    out, protocol, repo_root = Path(out), Path(protocol).resolve(), Path(repo_root).resolve()
    if protocol.read_bytes() != (out / "protocol.json").read_bytes():
        raise ValueError("provided protocol differs from immutable prepared protocol; never recompute subsets on execute")
    relative = str(protocol.relative_to(repo_root))
    committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=repo_root,
                               capture_output=True, check=True).stdout
    if committed != protocol.read_bytes():
        raise ValueError("protocol must match its committed HEAD bytes")
