"""Emit GPU inventory, package/CUDA versions, and free disk without loading a model."""
import argparse
from scripts.common import environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--disk-path", default=".")
    args = parser.parse_args()
    report = environment(args.disk_path)
    try:
        import torch
        report["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available(),
                           "visible_device_count": torch.cuda.device_count()}
    except ImportError as exc:
        report["torch"] = {"error": str(exc)}
    write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
