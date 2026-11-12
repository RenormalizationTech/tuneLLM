"""Compact, exploratory summary of a completed QLoRA run; never emits prompts or predictions."""
import argparse
import json
from math import comb, isfinite

from scripts.common import digest, path, read_json


def exact_mcnemar(fixed, broken):
    """Two-sided conditional binomial test on discordant pairs, no correction."""
    if any(type(value) is not int or value < 0 for value in (fixed, broken)):
        raise ValueError("discordant counts must be nonnegative integers")
    n = fixed + broken
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(fixed, broken) + 1)) / (2 ** n))


def indexed(rows):
    if not rows or any(not isinstance(row.get("id"), str) for row in rows):
        raise ValueError("nonempty rows with string IDs required")
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate IDs")
    return result


def learning_curve(baseline, history):
    baseline_loss = baseline["eval_loss"]
    validation = [{"step": row["step"], "validation_loss": row["eval_loss"]}
                  for row in history if "eval_loss" in row and row.get("step") in (16, 32, 48, 64)]
    if [row["step"] for row in validation] != [16, 32, 48, 64]:
        raise ValueError("validation history must include exactly steps16/32/48/64 in order")
    if not all(isfinite(value) for value in [baseline_loss] + [row["validation_loss"] for row in validation]):
        raise ValueError("validation losses must be finite")
    training = [{key: row[key] for key in ("step", "loss", "grad_norm", "learning_rate") if key in row}
                for row in history if "loss" in row]
    return {"baseline_validation_loss": baseline_loss, "validation_loss_history": validation,
            "training_log_points": training}


def verify_selection_history(selection, curve):
    by_step = {str(row["step"]): row["validation_loss"] for row in curve["validation_loss_history"]}
    if any(by_step.get(step) != value for step, value in selection["candidate_validation_losses"].items()):
        raise ValueError("selection candidate losses disagree with validation history")


def paired_counts(base_rows, adapter_rows, metric):
    base, adapter = indexed(base_rows), indexed(adapter_rows)
    if base.keys() != adapter.keys():
        raise ValueError("paired conditions have different IDs")
    counts = {"fixed": 0, "broken": 0, "both": 0, "neither": 0}
    for identifier, row in base.items():
        before, after = row[metric], adapter[identifier][metric]
        if type(before) is not bool or type(after) is not bool:
            raise ValueError("paired scores must be booleans")
        key = "both" if before and after else "broken" if before else "fixed" if after else "neither"
        counts[key] += 1
    return {**counts, "b": counts["broken"], "c": counts["fixed"],
            "total": len(base), "discordant": counts["fixed"] + counts["broken"],
            "accuracy_delta": (counts["fixed"] - counts["broken"]) / len(base),
            "mcnemar_exact_two_sided_p": exact_mcnemar(counts["fixed"], counts["broken"]),
            "interpretation": "Exploratory, unadjusted exact two-sided McNemar; b=broken=base right/adapter wrong, c=fixed=base wrong/adapter right. No continuity correction; zero discordant pairs gives p=1."}


def condition_summary(saved, expected_ids, task):
    details, predictions = indexed(saved["scores"]["details"]), indexed(saved["predictions"])
    if set(details) != set(expected_ids) or set(predictions) != set(expected_ids) or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("saved score/prediction IDs differ from protocol")
    if saved["scores"]["total"] != len(expected_ids):
        raise ValueError("saved total differs from protocol")
    summary = {"total": len(expected_ids), "strict_correct": 0, "fallback_correct": 0,
               "format_compliant": 0, "invalid": 0, "truncated": 0}
    tokens = 0
    for identifier in expected_ids:
        row, prediction = details[identifier], predictions[identifier]
        for metric in ("strict_correct", "fallback_correct", "hit_token_limit"):
            if type(row[metric]) is not bool:
                raise ValueError("saved scores must be booleans")
        if prediction["hit_token_limit"] != row["hit_token_limit"]:
            raise ValueError("prediction/score truncation mismatch")
        compliant = row["format_compliant"] if task == "gsm8k" else not row["invalid"]
        if type(compliant) is not bool:
            raise ValueError("format score must be boolean")
        summary["strict_correct"] += int(row["strict_correct"])
        summary["fallback_correct"] += int(row["fallback_correct"])
        summary["format_compliant"] += int(compliant)
        summary["invalid"] += int(not compliant)
        summary["truncated"] += int(row["hit_token_limit"])
        tokens += len(prediction["generated_ids"])
    for key in ("strict_correct", "fallback_correct"):
        if saved["scores"][key] != summary[key]:
            raise ValueError("saved aggregate disagrees with per-item scores")
    summary.update({"strict_accuracy": summary["strict_correct"] / len(expected_ids),
                    "fallback_accuracy": summary["fallback_correct"] / len(expected_ids),
                    "mean_generated_tokens_including_eos": tokens / len(expected_ids),
                    "total_generated_tokens_including_eos": tokens,
                    "invalid_definition": "missing strict final #### number line" if task == "gsm8k" else "not an exact case-insensitive single A-D letter"})
    if task == "retention":
        summary["fallback_invalid"] = sum(int(row["fallback_invalid"]) for row in details.values())
        summary["fallback_ambiguous"] = sum(int(row["fallback_ambiguous"]) for row in details.values())
    summary["timing"] = {key: saved.get("timing", {}).get(key) for key in
                         ("generation_wall_seconds", "pipeline_wall_seconds", "generated_tokens_per_generation_wall_second")}
    return summary


def summarize(run):
    run = path(run).resolve()
    metadata, protocol = read_json(run / "run.json"), read_json(run / "protocol.json")
    if metadata.get("completed") is not True or metadata.get("status") != "complete" or metadata.get("phase") != "complete":
        raise ValueError("run must be completed with complete status/phase")
    reload_checks = metadata.get("reload", {})
    for key in ("exact_state_tensors_equal", "gsm8k_first16_tokens_equal", "retention_all128_tokens_equal"):
        if reload_checks.get(key) is not True:
            raise ValueError(f"reload check not passed: {key}")
    if read_json(run / "reload_state.json").get("exact_state_tensors_equal") is not True:
        raise ValueError("saved reload tensor check failed")
    if digest(run / "protocol.json") != metadata["protocol_sha256"] or digest(run / "frozen.json") != metadata["frozen_manifest_sha256"]:
        raise ValueError("protocol/frozen manifest hash mismatch")
    cfg = read_json(run / "config.json")
    if digest(run / "config.json") != protocol["resolved_config_sha256"]:
        raise ValueError("resolved config hash mismatch")
    selection = read_json(run / "selection.json")
    if selection != metadata["selection"]:
        raise ValueError("selection metadata mismatch")
    losses = selection["candidate_validation_losses"]
    if set(losses) != {str(step) for step in cfg["candidate_steps"]} or selection["selected_step"] != min(cfg["candidate_steps"], key=lambda step: (losses[str(step)], step)):
        raise ValueError("selection violates frozen validation-only rule")
    evaluations, raw = {}, {}
    source_files = {"run.json", "protocol.json", "frozen.json", "config.json", "selection.json", "reload_state.json",
                    "baseline_validation.json", "validation_history.json", "adapter/provenance.json"}
    curve = learning_curve(read_json(run / "baseline_validation.json"), read_json(run / "validation_history.json"))
    verify_selection_history(selection, curve)
    for task, expected_ids in (("gsm8k", protocol["selected_ids"]["test"]), ("retention", protocol["retention_ids"])):
        base_name, adapter_name, reload_name = f"base_final_{task}.json", f"adapter_final_{task}.json", f"reload_{task}.json"
        source_files.update((base_name, adapter_name, reload_name))
        base, adapter, reloaded = (read_json(run / name) for name in (base_name, adapter_name, reload_name))
        expected_batches = [expected_ids[i:i + cfg["generation_batch_size"]] for i in range(0, len(expected_ids), cfg["generation_batch_size"])]
        if [b["ids"] for b in base["batches"]] != expected_batches or [b["ids"] for b in adapter["batches"]] != expected_batches:
            raise ValueError("base/adapter batches differ from protocol order")
        expected_reload = protocol["reload_gsm8k_ids"] if task == "gsm8k" else expected_ids
        after_by_id, reload_by_id = indexed(adapter["predictions"]), indexed(reloaded["predictions"])
        if set(reload_by_id) != set(expected_reload):
            raise ValueError("reload IDs differ from declared scope")
        for identifier in expected_reload:
            if after_by_id[identifier]["generated_ids"] != reload_by_id[identifier]["generated_ids"]:
                raise ValueError("saved reload generated tokens mismatch")
        details_base, details_after = indexed(base["scores"]["details"]), indexed(adapter["scores"]["details"])
        evaluations[task] = {
            "base": condition_summary(base, expected_ids, task),
            "adapter": condition_summary(adapter, expected_ids, task),
            "paired_primary_fallback": paired_counts(base["scores"]["details"], adapter["scores"]["details"], "fallback_correct"),
            "paired_secondary_strict": paired_counts(base["scores"]["details"], adapter["scores"]["details"], "strict_correct"),
            "item_deltas": [{"id": identifier,
                             "fallback_correct_delta": int(details_after[identifier]["fallback_correct"]) - int(details_base[identifier]["fallback_correct"]),
                             "strict_correct_delta": int(details_after[identifier]["strict_correct"]) - int(details_base[identifier]["strict_correct"])} for identifier in expected_ids],
        }
        raw[task] = {"reload_rows": len(expected_reload), "reload_ids": expected_reload}
    adapter_file = run / "adapter/adapter_model.safetensors"
    execution = metadata["execution_provenance"]
    retention_source = protocol.get("retention_source") or {}
    return {
        "status": "completed", "exploratory": True,
        "limitations": "One seeded modest run on fixed subsets; paired p-values are exploratory and unadjusted for multiple comparisons. No general improvement or retention equivalence claim.",
        "model_id": cfg["model_id"], "model_revision": cfg["revision"], "representation": protocol["representation"],
        "source_commit": execution["git_head"]["stdout"], "preparation_commit": metadata["git_head"]["stdout"],
        "source_sha256": protocol["source_sha256"], "protocol_sha256": metadata["protocol_sha256"],
        "frozen_manifest_sha256": metadata["frozen_manifest_sha256"],
        "dataset": {"gsm8k_revision": cfg["dataset_revision"], "retention_source": retention_source.get("source"),
                    "raw_data_sha256": protocol["raw_data_sha256"],
                    "selected_data_sha256": {name: value for name, value in protocol["input_sha256"].items() if name in ("train.jsonl", "validation.jsonl", "test.jsonl", "retention.json")}},
        "seed": cfg["seed"], "primary_metrics": protocol["primary_metrics"],
        "selection": selection,
        "training": {**curve, "train_rows": cfg["train_rows"], "validation_rows": cfg["validation_rows"],
                     "config": cfg["training"], "lora": cfg["lora"], "effective_batch_size": protocol["effective_batch_size"],
                     "actual_steps": metadata["global_step"], "actual_epoch": metadata["epoch"],
                     "warmup_steps": metadata["resolved_warmup_steps"], "wall_seconds": metadata["training_wall_seconds"],
                     "wall_time_scope": "Whole Trainer.train interval including scheduled validation and checkpoint saves; not pure optimizer compute",
                     "peak_pytorch_allocated_bytes": metadata["peak_training_allocated_bytes"],
                     "allocator_scope": "Peak PyTorch selected-GPU allocation over the whole Trainer.train interval, including loaded weights, scheduled validation and checkpoint saves; not total VRAM or other processes",
                     "metrics": {key: metadata["training_metrics"].get(key) for key in
                                 ("train_runtime", "train_loss", "train_samples_per_second", "train_steps_per_second", "epoch")}},
        "adapter": {"bytes": adapter_file.stat().st_size, "sha256": digest(adapter_file)},
        "reload": {"checks": reload_checks, "scope": raw}, "evaluation": evaluations,
        "raw_result_sha256": {name: digest(run / name) for name in sorted(source_files)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output, run = path(args.output).resolve(), path(args.run).resolve()
    if output.is_relative_to(run):
        raise ValueError("compact output must be outside the raw run directory")
    summary = summarize(run)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
