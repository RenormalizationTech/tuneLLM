"""Execute only the unchanged registered quantization source and fixed GPU layout."""
import argparse
import json

from scripts.qwen38_quant_screen import execute, ROOT, RUN, REGISTRY
from scripts.qwen38_science_runtime import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=["nf4", "q4", "bf16_gguf"])
    parser.add_argument("--gpus", required=True)
    args = parser.parse_args()
    devices = args.gpus.split(",")
    wanted = 2 if args.arm == "bf16_gguf" else 1
    if len(devices) != wanted or len(set(devices)) != wanted or any(not g.isdecimal() for g in devices):
        parser.error("one GPU for NF4/Q4, two for BF16 GGUF; no alternative layouts")
    protocol = json.loads(REGISTRY.read_text())
    for name, expected in protocol["source_sha256"].items():
        file = RUN / "source" / name if name == "qwen38_science_score.py" else ROOT / "scripts" / name
        if sha(file) != expected:
            raise ValueError(f"registered live source changed: {name}")
    record = RUN / f"{args.arm}-launcher.json"
    if record.exists():
        raise FileExistsError("arm launch already recorded")
    write(record, {"arm": args.arm, "gpus": args.gpus, "source_check": "all registered live sources match",
                   "launcher_sha256": sha(__file__), "protocol_sha256": sha(REGISTRY)})
    print(json.dumps(execute(args.arm, args.gpus), indent=2))


if __name__ == "__main__":
    main()
