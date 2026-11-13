"""Prepare or execute a fixed final-test matrix after development selection.

Preparation is CPU-only and requires the selected configuration's completed
seed-43 repeat and BF16 control. Commit the two generated evaluation protocols
before execution. No final result is used for checkpoint selection.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.common import ROOT, digest, path, read_json, write_json
from scripts.run_support import verify_committed_protocol


def choose(selection):
    decision = selection["decision"]
    if decision["fallback_base"]:
        return min(selection["candidates"], key=lambda c: (c["best_epoch"]["token_weighted_nll"], c["rank"], c["learning_rate"]))
    return next(c for c in selection["candidates"] if c["id"] == decision["selected_id"])


def prepare(args):
    target = path(args.plan)
    if target.exists():
        raise FileExistsError(target)
    selection_file = path(args.selection)
    selection = read_json(selection_file)
    if selection.get("stage") != "rank" or selection.get("split") != "development":
        raise ValueError("fixed development rank decision required")
    selected = choose(selection)
    primary_checkpoint = path(selected["best_epoch"]["checkpoint"])
    primary = primary_checkpoint.parent.parent
    primary_protocol = read_json(primary / "protocol.json")
    primary_cfg = read_json(primary / "config.json")
    baseline_cfg = read_json("configs/baseline.json")
    if any(primary_cfg[key] != baseline_cfg[key] for key in ("model_id", "revision")):
        raise ValueError("benchmark BF16 snapshot differs from training lineage")
    rank, lr = selected["rank"], selected["learning_rate"]
    epoch, step = selected["best_epoch"]["epoch"], selected["best_epoch"]["step"]
    runs = {"nf4_adapter": primary, "nf4_repeat": path(args.repeat), "bf16_control": path(args.control)}
    checkpoints = {}
    for name, directory in runs.items():
        cfg, run = read_json(directory / "config.json"), read_json(directory / "run.json")
        protocol = read_json(directory / "protocol.json")
        representation, seed = ("bf16", 42) if name == "bf16_control" else ("nf4", 43 if name == "nf4_repeat" else 42)
        if (not run.get("completed") or cfg["rank"] != rank or cfg["learning_rate"] != lr
                or cfg["representation"] != representation or cfg["seed"] != seed or cfg["epochs"] != 3
                or run["exposure"]["actual_loss_tokens"] != 1567140):
            raise ValueError(f"incomplete or mismatched registered run: {name}")
        keys = ("model_id", "revision", "steps_per_epoch", "max_sequence_length", "warmup_steps",
                "per_device_train_batch_size", "gradient_accumulation_steps")
        if (any(cfg[key] != primary_cfg[key] for key in keys)
                or protocol["raw_corpus_sha256"] != primary_protocol["raw_corpus_sha256"]):
            raise ValueError("matched control requires identical lineage, data and schedule")
        checkpoint = directory / "checkpoints" / f"checkpoint-{step}"
        if not (checkpoint / "adapter_model.safetensors").is_file():
            raise ValueError("fixed selected-epoch checkpoint missing")
        checkpoints[name] = str(checkpoint)
    if digest(primary_checkpoint / "adapter_model.safetensors") != selected["checkpoint_adapter_sha256"]:
        raise ValueError("selected checkpoint changed")
    baseline = read_json("outputs/physics-screen-nf4-dev-02/run.json")
    source_names = ["scripts/benchmark_screen.py", "scripts/batched_generation.py", "scripts/save_nf4.py", "evaluation/gsm8k.py"]
    if any(digest(name) != baseline["source_sha256"][name] for name in source_names):
        raise ValueError("paired inference implementation changed")
    frozen = {str(selection_file): digest(selection_file),
              str(Path(__file__).resolve()): digest(__file__),
              "results/physics_lr_selection.json": digest("results/physics_lr_selection.json"),
              **{name: digest(name) for name in source_names}}
    for checkpoint in checkpoints.values():
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            filename = path(checkpoint) / name
            frozen[str(filename)] = digest(filename)
    global_protocol = read_json("results/physics_prl_protocol.json")
    writing_provenance = read_json("data/benchmark-screen-01/writing_provenance.json")
    if digest("data/benchmark-screen-01/writing_provenance.json") != global_protocol["writing_provenance_sha256"]:
        raise ValueError("registered writing provenance changed")
    for name in ("test.json", "writing_test.json", "writing_test_fewshot.json"):
        filename = path("data/benchmark-screen-01") / name
        expected = (global_protocol["benchmark_provenance"]["fixture_sha256"]["test"] if name == "test.json"
                    else writing_provenance["files"][name]["sha256"])
        if digest(filename) != expected:
            raise ValueError("registered final fixture changed")
        frozen[str(filename)] = digest(filename)
    preparations, protocols = {}, {}
    for representation in ("nf4", "bf16"):
        cfg = read_json("configs/physics.json")
        cfg["allow_test_evaluation"] = True
        cfg["fixed_rank_selection_sha256"] = digest(selection_file)
        filename = path(f"outputs/physics-final-eval-{representation}-config.json")
        if filename.exists():
            raise FileExistsError(filename)
        write_json(filename, cfg)
        directory = path(f"outputs/physics-final-eval-{representation}")
        subprocess.run([sys.executable, "-m", "scripts.physics_train", "--config", str(filename),
                        "--output", str(directory), "--representation", representation,
                        "--rank", str(rank), "--learning-rate", str(lr), "--epochs", "3"], check=True)
        protocol = path(f"results/protocols/physics-final-eval-{representation}.json")
        prepared_protocol = read_json(directory / "protocol.json")
        if (prepared_protocol["raw_corpus_sha256"] != primary_protocol["raw_corpus_sha256"]
                or read_json(directory / "config.json")["max_sequence_length"] != primary_cfg["max_sequence_length"]):
            raise ValueError("final evaluation data/context differs from primary training")
        if protocol.exists():
            raise FileExistsError(protocol)
        shutil.copyfile(directory / "protocol.json", protocol)
        preparations[representation], protocols[representation] = str(directory), str(protocol)
        frozen[str(protocol)] = digest(protocol)
    jobs, nll_files = [], {}
    for name, representation in (("nf4_base", "nf4"), ("nf4_adapter", "nf4"), ("nf4_repeat", "nf4"),
                                 ("bf16_base", "bf16"), ("bf16_control", "bf16")):
        output = path(f"outputs/physics-final-{name}-paper-nll.json")
        command = [sys.executable, "-u", "-m", "scripts.physics_eval", "--prepared", preparations[representation],
                   "--protocol", protocols[representation], "--split", "test", "--output", str(output)]
        if name in checkpoints:
            command += ["--adapter", checkpoints[name]]
        nll_files[name] = str(output)
        jobs.append({"command": command, "output": str(output), "kind": "paper_nll"})
    conditions = {}
    specs = [("nf4_base", "nf4", None), ("nf4_adapter", "nf4", None),
             ("bf16_base", "bf16", None), ("bf16_control", "bf16", None),
             ("nf4_base_fewshot", "nf4", None), ("bf16_gguf", "gguf", "BF16"),
             ("q8", "gguf", "Q8_0"), ("q4", "gguf", "Q4_K_M")]
    for name, representation, gguf in specs:
        output = path(f"outputs/physics-final-screen-{name}")
        command = [sys.executable, "-u", "-m", "scripts.benchmark_screen", "--representation", representation,
                   "--fixture", "data/benchmark-screen-01/test.json", "--output", str(output)]
        if gguf:
            filename = path(f"artifacts/gguf/Qwen3-4B-{gguf}.gguf")
            frozen[str(filename)] = digest(filename)
            command += ["--gguf", str(filename), "--tasks", "general_mc", "physics_mc", "gsm8k", "ifeval"]
        else:
            writing = "writing_test_fewshot.json" if name == "nf4_base_fewshot" else "writing_test.json"
            command += ["--writing", f"data/benchmark-screen-01/{writing}"]
            if name == "nf4_base_fewshot":
                command += ["--tasks", "writing"]
        if name in checkpoints:
            command += ["--adapter", checkpoints[name]]
        conditions[name] = {"directory": str(output), "representation": representation,
                            "checkpoint": checkpoints.get(name),
                            "checkpoint_sha256": digest(path(checkpoints[name]) / "adapter_model.safetensors") if name in checkpoints else None,
                            "prompt_condition": "two-example" if name == "nf4_base_fewshot" else "zero-shot"}
        if name in nll_files:
            conditions[name]["paper_nll"] = nll_files[name]
        jobs.append({"command": command, "output": str(output), "kind": "screen"})
    mapping = {"split": "test", "selection_fixed": True, "selection_report": str(selection_file),
               "selection_report_sha256": digest(selection_file), "conditions": conditions,
               "paired_comparisons": [["nf4_base", "nf4_adapter"], ["bf16_base", "bf16_control"],
                                      ["nf4_base", "bf16_base"], ["nf4_adapter", "bf16_control"],
                                      ["bf16_base", "bf16_gguf"], ["bf16_gguf", "q8"], ["bf16_gguf", "q4"]],
               "paper_only_conditions": {"nf4_repeat": {"paper_nll": nll_files["nf4_repeat"],
                                         "checkpoint_sha256": digest(path(checkpoints["nf4_repeat"]) / "adapter_model.safetensors"),
                                         "checkpoint": checkpoints["nf4_repeat"], "representation": "nf4"}},
               "paper_only_comparisons": [["nf4_adapter", "nf4_repeat"], ["nf4_base", "nf4_repeat"]],
               "training_runs": sorted(str(p.parent) for p in path("outputs").glob("physics-r*/run.json")
                                       if read_json(p).get("completed"))}
    write_json(target, {"status": "prepared_commit_evaluation_protocols_before_execution", "fixed_epoch": epoch,
                        "descriptive_only": selection["decision"]["fallback_base"], "frozen_sha256": frozen,
                        "preparations": preparations, "protocols": protocols, "jobs": jobs, "mapping": mapping})
    print(target)


def execute(args):
    plan = read_json(args.plan)
    for filename, expected in plan["frozen_sha256"].items():
        if digest(filename) != expected:
            raise ValueError(f"fixed final input changed: {filename}")
    for representation, directory in plan["preparations"].items():
        cfg = read_json(path(directory) / "config.json")
        verify_committed_protocol(path(directory), path(plan["protocols"][representation]), cfg["repo_root"])
    for job in plan["jobs"]:
        output = path(job["output"])
        invocation = Path(str(output) + ".invocation.json")
        expected = {"command": job["command"], "fixed_inputs": plan["frozen_sha256"]}
        if invocation.exists():
            if read_json(invocation) != expected:
                raise ValueError("existing evaluation invocation differs from fixed plan")
        elif output.exists():
            raise ValueError("existing result has no fixed-plan invocation record")
        else:
            write_json(invocation, expected)
        if output.exists():
            if job["kind"] == "screen" and read_json(output / "run.json").get("completed"):
                continue
            if job["kind"] == "paper_nll" and read_json(output).get("split") == "test":
                continue
            raise ValueError(f"existing incomplete result requires investigation: {output}")
        print("evaluating", output, flush=True)
        with Path(str(output) + ".log").open("x") as stream:
            subprocess.run(job["command"], stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
    write_json("outputs/physics_final_mapping.json", plan["mapping"])
    print("all fixed final evaluations complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="outputs/physics_final_plan.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--selection", default="results/physics_rank_selection.json")
    parser.add_argument("--repeat")
    parser.add_argument("--control")
    args = parser.parse_args()
    if args.execute:
        execute(args)
    elif args.repeat and args.control:
        prepare(args)
    else:
        parser.error("preparation requires --repeat and --control")


if __name__ == "__main__":
    main()
