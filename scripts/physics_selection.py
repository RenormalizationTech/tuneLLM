"""One-shot CPU development selection; never searches for or reads final-test results.

Mapping JSON: {"split":"development", "stage":"lr_pilot" or "rank",
 "baseline":{"directory":"outputs/definitive-base-dev02"},
 "writing_baseline":{"directory":"outputs/base-fewshot-writing"},
 "candidates":[{"id":"r32-low", "rank":32, "learning_rate":5e-5,
   "run_dir":"outputs/physics-r32-low",
   "screens_by_epoch":{"1":{"directory":"outputs/r32-low-epoch1-dev"}, ...}}]}

Screen specs also accept {"metrics":{general_mc/physics_mc/gsm8k/ifeval:
 {total,correct,truncated}, writing:{total,nonempty,flags,truncated}}}.
MC truncation is not applicable to one-token likelihood. Optional gate_metadata
is retained for audit and never overrides the numeric gates. All candidate
epoch reports and selected-epoch development screens must exist before calling.
The output is exclusive: it cannot overwrite an earlier selection decision.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re

from scripts.common import path, read_json

SEED = 20261112
RESAMPLES = 10000


def file_hash(filename):
    value = hashlib.sha256()
    with path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def paper_values(report):
    papers = report["per_paper"]
    if not papers:
        raise ValueError("nonempty per-paper NLL required")
    for paper in papers.values():
        if type(paper["loss_tokens"]) is not int or paper["loss_tokens"] <= 0 or not math.isfinite(paper["nll_sum"]) or paper["nll_sum"] < 0:
            raise ValueError("positive integer counts and finite nonnegative paper NLL sums required")
    weighted = sum(p["nll_sum"] for p in papers.values()) / sum(p["loss_tokens"] for p in papers.values())
    if not math.isclose(weighted, report["token_weighted_nll"], rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("token-weighted NLL disagrees with paper sums/counts")
    return papers


def percentile(values, probability):
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def paired_bootstrap(candidate, reference, resamples=RESAMPLES, seed=SEED):
    a, b = paper_values(candidate), paper_values(reference)
    if set(a) != set(b):
        raise ValueError("paired bootstrap requires identical paper IDs")
    ids = sorted(a)
    if any(a[identifier]["loss_tokens"] != b[identifier]["loss_tokens"] for identifier in ids):
        raise ValueError("paired paper target counts differ")
    if type(resamples) is not int or resamples <= 0:
        raise ValueError("positive integer resample count required")
    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        # Ratio of sampled sums, not an unweighted mean of per-paper NLLs.
        loss_a = sum(a[key]["nll_sum"] for key in sample) / sum(a[key]["loss_tokens"] for key in sample)
        loss_b = sum(b[key]["nll_sum"] for key in sample) / sum(b[key]["loss_tokens"] for key in sample)
        deltas.append(loss_a - loss_b)
    deltas.sort()
    lower, upper = percentile(deltas, 0.025), percentile(deltas, 0.975)
    return {"candidate_minus_reference_nll": candidate["token_weighted_nll"] - reference["token_weighted_nll"],
            "ci95": [lower, upper], "includes_zero": lower <= 0 <= upper,
            "resamples": resamples, "seed": seed, "papers_per_draw": len(ids),
            "method": "paired paper resampling with replacement; token-weighted ratio per draw; percentile95 CI with linear interpolation"}


def load_epochs(run_dir):
    run_dir = path(run_dir)
    history = read_json(run_dir / "learning_curve.json")
    by_step = {}
    for filename in sorted((run_dir / "validation").glob("step*.json")):
        match = re.fullmatch(r"step-?(\d+)", filename.stem)
        if match:
            step = int(match.group(1))
            if step in by_step:
                raise ValueError("duplicate validation step file")
            by_step[step] = filename
    epochs = []
    for row in history:
        if "eval_loss" not in row or row.get("step", 0) == 0:
            continue
        epoch = row.get("epoch")
        if epoch is None or not math.isclose(epoch, round(epoch), abs_tol=1e-8) or epoch < 1:
            raise ValueError("validation checkpoint must map to a completed integer epoch")
        filename = by_step[row["step"]]
        report = read_json(filename)
        if len(paper_values(report)) != 10:
            raise ValueError("registered development loss must contain exactly10 papers")
        if not math.isclose(row["eval_loss"], report["token_weighted_nll"], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("learning curve and saved development NLL disagree")
        checkpoint = run_dir / "checkpoints" / f"checkpoint-{row['step']}"
        if not (checkpoint / "adapter_model.safetensors").is_file():
            raise ValueError("epoch checkpoint adapter missing")
        epochs.append({"epoch": round(epoch), "step": row["step"], "checkpoint": str(checkpoint),
                       "token_weighted_nll": report["token_weighted_nll"], "report": report,
                       "validation_file": str(filename), "validation_sha256": file_hash(filename)})
    if not epochs or sorted(row["epoch"] for row in epochs) != list(range(1, len(epochs) + 1)):
        raise ValueError("epoch validation reports must be complete, unique and consecutive")
    if (run_dir / "config.json").exists() and len(epochs) != read_json(run_dir / "config.json")["epochs"]:
        raise ValueError("candidate has not completed all registered epochs")
    return sorted(epochs, key=lambda row: row["epoch"])


def best_epoch(epochs):
    return min(epochs, key=lambda row: (row["token_weighted_nll"], row["epoch"]))


def load_screen(spec, hashes):
    if "metrics" in spec:
        return spec["metrics"]
    directory = path(spec["directory"])

    def read(name):
        filename = directory / name
        hashes[str(filename)] = file_hash(filename)
        return read_json(filename)

    summary = read("summary.json")
    metadata = read("run.json")
    if metadata.get("completed") is not True or metadata.get("generation_batch") != 8:
        raise ValueError("development screens must be complete and use definitive generation batch8")
    result = {}
    for task in ("general_mc", "physics_mc"):
        if task in summary:
            result[task] = {"total": summary[task]["total"], "correct": summary[task]["correct"], "truncated": None}
    if "gsm8k" in summary:
        result["gsm8k"] = {"total": summary["gsm8k"]["total"], "correct": summary["gsm8k"]["fallback_correct"],
                           "truncated": summary["gsm8k"]["hit_token_limit"]}
    if "ifeval" in summary:
        predictions = read("ifeval_predictions.json")
        if len(predictions) != summary["ifeval"]["total"] or len({r["id"] for r in predictions}) != len(predictions):
            raise ValueError("IFEval prediction count/IDs inconsistent")
        result["ifeval"] = {"total": summary["ifeval"]["total"], "correct": summary["ifeval"]["prompt_strict_correct"],
                            "truncated": sum(int(row["hit_token_limit"]) for row in predictions)}
    if "writing" in summary:
        predictions = read("writing_predictions.json")
        if len(predictions) != summary["writing"]["total"] or len({r["id"] for r in predictions}) != len(predictions):
            raise ValueError("writing prediction count/IDs inconsistent")
        result["writing"] = {"total": summary["writing"]["total"], "nonempty": summary["writing"]["nonempty"],
                             "flags": summary["writing"]["automatic_fidelity_flags"],
                             "truncated": sum(int(row["hit_token_limit"]) for row in predictions)}
    return result


def screen_gates(candidate, baseline, writing_baseline=None, metadata=None):
    checks = {}
    for task, expected, allowed_drop in (("general_mc", 32, 4), ("physics_mc", 32, 4), ("gsm8k", 32, 4), ("ifeval", 16, 3)):
        a, b = candidate[task], baseline[task]
        if a["total"] != expected or b["total"] != expected:
            raise ValueError(f"{task} gate requires same registered{expected} items")
        if any(type(row["correct"]) is not int or not 0 <= row["correct"] <= expected for row in (a, b)):
            raise ValueError("correct counts must be in range")
        drop = b["correct"] - a["correct"]
        checks[task + "_correct_drop"] = {"value": drop, "maximum": allowed_drop, "passed": drop <= allowed_drop}
        if task in ("gsm8k", "ifeval"):
            for row in (a, b):
                if type(row["truncated"]) is not int or not 0 <= row["truncated"] <= expected:
                    raise ValueError("truncation counts must be in range")
            increase = a["truncated"] - b["truncated"]
            checks[task + "_truncation_increase"] = {"value": increase, "maximum": 2, "passed": increase <= 2}
    a, b = candidate["writing"], (writing_baseline or baseline)["writing"]
    if a["total"] <= 0 or a["total"] != b["total"]:
        raise ValueError("writing gate requires same nonempty fixture size as base fewshot")
    for row in (a, b):
        if any(type(row[key]) is not int or not 0 <= row[key] <= row["total"] for key in ("nonempty", "flags", "truncated")):
            raise ValueError("writing counts must be in range")
    checks["writing_all_nonempty"] = {"value": a["nonempty"], "required": a["total"], "passed": a["nonempty"] == a["total"]}
    for key in ("flags", "truncated"):
        increase = a[key] - b[key]
        checks["writing_" + key + "_increase"] = {"value": increase, "maximum": 2, "passed": increase <= 2}
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks,
            "gate_metadata": metadata, "human_style_improvement_claim": False,
            "interpretation": "Net correct-count gates, GSM last-number fallback, IFEval prompt strict. Writing flags need review; no human style claim. Truncation is gated by increase, not zero absolute count; MC is single-token likelihood."}


def select_candidates(candidates, stage):
    if stage not in ("lr_pilot", "rank") or not candidates:
        raise ValueError("nonempty lr_pilot/rank candidate registry required")
    if len({row["id"] for row in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    if stage == "lr_pilot" and any(row["rank"] != 32 for row in candidates):
        raise ValueError("LR pilot candidates must have rank32")
    passing = [row for row in candidates if row["gate"]["passed"]]
    if not passing:
        return {"selected_id": None, "fallback_base": True, "reason": "all candidates failed development screen gates", "comparisons": {}}
    best = min(passing, key=lambda row: (row["epoch"]["token_weighted_nll"], row["id"]))
    comparisons = {row["id"]: paired_bootstrap(row["epoch"]["report"], best["epoch"]["report"]) for row in candidates}
    if stage == "lr_pilot":
        lower = min(passing, key=lambda row: (row["learning_rate"], row["id"]))
        selected = lower if comparisons[lower["id"]]["includes_zero"] else best
    else:
        compatible = [row for row in passing if comparisons[row["id"]]["includes_zero"]]
        selected = min(compatible, key=lambda row: (row["rank"], row["id"]))
    return {"selected_id": selected["id"], "fallback_base": False, "best_passing_nll_id": best["id"],
            "selected_epoch": selected["epoch"]["epoch"], "selected_checkpoint": selected["epoch"]["checkpoint"],
            "comparisons": comparisons,
            "rule": "Epoch:minimum token-weighted development NLL,tie earlier. Filter screen gates first. LR:prefer lower passing LR if its paired CI vs best contains0; otherwise best. Rank:smallest passing rank whose paired CI vs best contains0."}


def selection_report(mapping):
    if mapping.get("split") != "development":
        raise ValueError("utility accepts development inputs only; final-test selection forbidden")
    hashes = {}
    baseline = load_screen(mapping["baseline"], hashes)
    writing_base = load_screen(mapping["writing_baseline"], hashes)
    candidates, reports = [], []
    for entry in mapping["candidates"]:
        epochs = load_epochs(entry["run_dir"])
        chosen = best_epoch(epochs)
        screen = load_screen(entry["screens_by_epoch"][str(chosen["epoch"])], hashes)
        gate = screen_gates(screen, baseline, writing_base, entry.get("gate_metadata"))
        candidate = {"id": entry["id"], "rank": entry["rank"], "learning_rate": entry["learning_rate"], "epoch": chosen, "gate": gate}
        candidates.append(candidate)
        hashes[str(path(entry["run_dir"]) / "learning_curve.json")] = file_hash(path(entry["run_dir"]) / "learning_curve.json")
        for epoch in epochs:
            hashes[epoch["validation_file"]] = epoch["validation_sha256"]
        reports.append({"id": entry["id"], "rank": entry["rank"], "learning_rate": entry["learning_rate"],
                        "best_epoch": {k: v for k, v in chosen.items() if k != "report"}, "gate": gate,
                        "epochs": [{k: v for k, v in epoch.items() if k not in ("report",)} for epoch in epochs],
                        "checkpoint_adapter_sha256": file_hash(path(chosen["checkpoint"]) / "adapter_model.safetensors")})
    decision = select_candidates(candidates, mapping["stage"])
    return {"stage": mapping["stage"], "split": "development", "decision": decision, "candidates": reports,
            "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "papers_per_draw": 10, "confidence": 0.95,
                          "caution": "Paired paper bootstrap is a development-selection heuristic, not proof of equivalence; no final-test results read."},
            "gate_metadata": mapping.get("gate_metadata"), "raw_development_sha256": hashes,
            "selection_policy": "Apply once after all registered candidate development screens; output refuses overwrite. All-fail falls back to base; no human style claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = path(args.output)
    if output.exists():
        raise FileExistsError("selection already exists; refuses to overwrite a decision")
    report = selection_report(read_json(args.mapping))
    report["mapping_sha256"] = file_hash(args.mapping)
    report["selector_source_sha256"] = file_hash(Path(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
