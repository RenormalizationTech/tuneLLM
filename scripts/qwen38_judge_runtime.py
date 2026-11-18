"""Pin the tools-off Claude CLI before any new 27B AI data/review calls.

Uses the existing isolated temp-directory invocation and envelope/model checks.
This runtime has a separate version/hash binding from historical judge studies.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from scripts import tone_judge
from scripts.qwen38_science_runtime import sha, write

ROOT = Path(__file__).resolve().parents[1]
VERSION = "2.1.326 (Claude Code)"
BINARY_SHA = "b6b3647d981130dc44fc670846ae97b6cde531ffb667cf9c4d17cfcafb38cb37"
AUDIT = ROOT / "outputs/qwen38-judge-runtime-01"


def identity():
    binary = shutil.which("claude")
    if not binary or sha(binary) != BINARY_SHA:
        raise ValueError("Claude binary changed; preserve the run and review a new pin")
    version = subprocess.check_output([binary, "--version"], text=True, timeout=30).strip()
    if version != VERSION:
        raise ValueError("Claude version differs from new 27B runtime binding")
    return {"binary": binary, "binary_sha256": BINARY_SHA, "version": version,
            "requested_model": tone_judge.MODEL, "argv": tone_judge.argv(binary),
            "invoker_sha256": sha(ROOT / "scripts/tone_judge.py"), "wrapper_sha256": sha(__file__)}


def invoke(payload):
    binding = json.loads((AUDIT / "binding.json").read_text())
    if not json.loads((AUDIT / "audit.json").read_text()).get("passed"):
        raise ValueError("new runtime isolation audit required")
    if identity() != binding:
        raise ValueError("judge runtime identity changed")
    attempt = tone_judge.invoke(binding["binary"], payload)
    if identity() != binding:
        raise ValueError("judge binary changed during call")
    return attempt


def audit():
    if AUDIT.exists():
        raise FileExistsError("fresh judge runtime audit required")
    binding = identity()
    AUDIT.mkdir(parents=True)
    write(AUDIT / "binding.json", binding)
    payload = {"task": "Context isolation check only. No external information is supplied.",
               "question": "What is the name of the user's current project? If it is not in this JSON, return UNKNOWN.",
               "return_schema": {"project": "string"}, "required_output": "JSON only"}
    write(AUDIT / "payload.json", payload)
    attempt = tone_judge.invoke(binding["binary"], payload)
    write(AUDIT / "attempt.json", attempt)
    outer, models = tone_judge.outer_result(attempt)
    output = tone_judge.result_json(outer.get("result"))
    passed = output == {"project": "UNKNOWN"} and identity() == binding
    record = {"passed": passed, "output": output, "model_ids": models,
              "binding_sha256": sha(AUDIT / "binding.json"), "attempt_sha256": sha(AUDIT / "attempt.json"),
              "interpretation": "mechanical tools-off/temp-directory/context check; not a proof of model independence or factual correctness"}
    write(AUDIT / "audit.json", record)
    if not passed:
        raise ValueError("judge context isolation check failed")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(audit(), indent=2))


if __name__ == "__main__":
    main()
