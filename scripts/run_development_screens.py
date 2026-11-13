"""Screen each completed run's nominated epoch, then prepare a one-shot selection mapping."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.common import digest, path, read_json, write_json
from scripts.physics_selection import best_epoch, file_hash, load_epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--stage", choices=["lr_pilot", "rank"], required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    target = path(args.mapping)
    if target.exists():
        raise FileExistsError(target)
    baseline = read_json("outputs/physics-screen-nf4-dev-02/run.json")
    for source in ["scripts/benchmark_screen.py", "scripts/batched_generation.py", "scripts/save_nf4.py", "evaluation/gsm8k.py"]:
        if digest(source) != baseline["source_sha256"][source]:
            raise ValueError(f"inference dependency changed since paired baseline: {source}")
    global_protocol = read_json("results/physics_prl_protocol.json")
    fixture = path("data/benchmark-screen-01/development.json")
    writing = path("data/benchmark-screen-01/writing_development.json")
    if digest(fixture) != global_protocol["benchmark_provenance"]["fixture_sha256"]["development"]:
        raise ValueError("development fixture changed")
    wmeta = read_json("data/benchmark-screen-01/writing_provenance.json")
    if digest("data/benchmark-screen-01/writing_provenance.json") != global_protocol["writing_provenance_sha256"]:
        raise ValueError("writing provenance changed")
    if digest(writing) != wmeta["files"][writing.name]["sha256"]:
        raise ValueError("development writing fixture changed")
    candidates, jobs, exposures = [], [], []
    for directory in args.runs:
        run_dir = path(directory)
        run, cfg = read_json(run_dir / "run.json"), read_json(run_dir / "config.json")
        if not run.get("completed") or cfg["seed"] != 42 or cfg["representation"] != "nf4":
            raise ValueError("selection screen requires completed seed42 NF4 candidates")
        nominated = best_epoch(load_epochs(run_dir))
        checkpoint = path(nominated["checkpoint"])
        out = path(f"outputs/physics-screen-{run_dir.name}-e{nominated['epoch']}-dev")
        adapter_hash = file_hash(checkpoint / "adapter_model.safetensors")
        command = [sys.executable, "-u", "-m", "scripts.benchmark_screen", "--representation", "nf4",
                   "--adapter", str(checkpoint), "--fixture", str(fixture), "--writing", str(writing), "--output", str(out)]
        if out.exists():
            old = read_json(out / "run.json")
            if (not old.get("completed") or old["fixture_sha256"] != digest(fixture)
                    or old["writing_sha256"] != digest(writing) or path(old["adapter"]).resolve() != checkpoint.resolve()
                    or old["generation_batch"] != 8):
                raise ValueError("existing screen does not match nominated checkpoint and frozen fixtures")
        else:
            jobs.append({"command": command, "output": str(out), "log": str(out) + ".log"})
        exposures.append(run["exposure"]["actual_loss_tokens"])
        candidates.append({"id": run_dir.name, "rank": cfg["rank"], "learning_rate": cfg["learning_rate"],
            "run_dir": str(run_dir), "screens_by_epoch": {str(nominated["epoch"]): {"directory": str(out)}},
            "gate_metadata": {"nominated_epoch": nominated["epoch"], "checkpoint_adapter_sha256": adapter_hash,
                              "prompt_condition": "zero-shot adapter vs two-example base writing comparator",
                              "no_alternate_epoch_gate_search": True}})
    if len(set(exposures)) != 1:
        raise ValueError("candidates did not receive equal registered maximum token exposure")
    mapping = {"split": "development", "stage": args.stage,
        "baseline": {"directory": "outputs/physics-screen-nf4-dev-02"},
        "writing_baseline": {"directory": "outputs/physics-writing-nf4-fewshot-dev"},
        "candidates": candidates, "jobs": jobs, "status": "prepared",
        "fixture_sha256": digest(fixture), "writing_sha256": digest(writing),
        "helper_sha256": digest(__file__)}
    write_json(target, mapping)
    if args.execute:
        for job in jobs:
            print("screening", job["output"], flush=True)
            with open(job["log"], "x") as log:
                subprocess.run(job["command"], stdout=log, stderr=subprocess.STDOUT, check=True)
        mapping["status"] = "all_registered_screens_completed"
        write_json(target, mapping)
    else:
        print(json.dumps(jobs, indent=2))
    print(target)


if __name__ == "__main__":
    main()
