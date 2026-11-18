"""Validate selected checkpoint hashes/config before the shared science evaluator."""
import argparse
import json
from pathlib import Path

from scripts.qwen38_eval_science import execute, ROOT
from scripts.qwen38_science_runtime import sha, write


def validate(adapter, task, template):
    if template != "base":
        raise ValueError("selected adapters must use the identical training task prompt")
    adapter = Path(adapter).resolve()
    training = adapter.parents[1]
    result = json.loads((training / "result.json").read_text())
    config = json.loads((training / "config.json").read_text())
    if not result.get("completed") or Path(result["selected"]["path"]).resolve() != adapter or config["task"] != task:
        raise ValueError("completed selected checkpoint for this task required")
    files = result["selected"].get("adapter_files")
    if not files or set(files) != {"adapter_model.safetensors", "adapter_config.json"}:
        raise ValueError("training-time hashes of exactly the saved adapter files required")
    for name, expected in files.items():
        file = adapter / name
        if file.stat().st_size != expected["bytes"] or sha(file) != expected["sha256"]:
            raise ValueError(f"selected adapter changed after save: {name}")
    return {"training_result_sha256": sha(training / "result.json"), "adapter_files": files,
            "template": template, "validation_script_sha256": sha(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=["qasper", "prl"], required=True)
    parser.add_argument("--template", default="base")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--purpose", choices=["development", "final", "reload", "mechanics"], required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-seconds", type=int, default=43200)
    args = parser.parse_args()
    if not args.gpu.isdecimal() or not 1 <= args.max_new_tokens <= 512 or not 1 <= args.max_seconds <= 43200:
        parser.error("bounded generation parameters required")
    audit = validate(ROOT / args.adapter, args.task, args.template)
    destination = ROOT / args.output
    validation_file = destination.with_name(destination.name + "-adapter-validation.json")
    if validation_file.exists():
        raise FileExistsError("adapter validation already recorded")
    write(validation_file, audit)
    print(json.dumps(execute(args), indent=2))


if __name__ == "__main__":
    main()
