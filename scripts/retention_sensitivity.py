"""One secondary MMLU cap64 sensitivity check; primary cap8 files remain untouched."""
import argparse
import importlib.util

from scripts.common import digest, path, provenance, read_json, select_gpu, setup_hf, write_json
from scripts.run_support import verify_snapshot


def load_frozen_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def by_id(rows):
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("nonempty unique IDs required")
    return {row["id"]: row for row in rows}


def verify_settings(saved, cap, expected_ids):
    expected = {"batch_size": 8, "max_new_tokens": cap, "seed": 42,
                "do_sample": False, "num_beams": 1, "enable_thinking": False}
    if any(saved["settings"].get(key) != value for key, value in expected.items()):
        raise ValueError(f"settings must match the fixed batch8/seed42/greedy cap{cap} protocol")
    batches = [expected_ids[i:i + 8] for i in range(0, len(expected_ids), 8)]
    if [batch["ids"] for batch in saved["batches"]] != batches:
        raise ValueError("batch membership/order differs from frozen fixture")
    predictions, details = by_id(saved["predictions"]), by_id(saved["scores"]["details"])
    if set(predictions) != set(expected_ids) or set(details) != set(expected_ids):
        raise ValueError("prediction/score IDs differ from frozen fixture")


def compare_caps(primary, secondary, expected_ids):
    verify_settings(primary, 8, expected_ids)
    verify_settings(secondary, 64, expected_ids)
    scores8, scores64 = by_id(primary["scores"]["details"]), by_id(secondary["scores"]["details"])
    predictions8, predictions64 = by_id(primary["predictions"]), by_id(secondary["predictions"])
    rows = []
    fields = ("strict_correct", "fallback_correct", "invalid", "fallback_invalid", "fallback_ambiguous", "hit_token_limit")
    for identifier in expected_ids:
        row = {"id": identifier}
        for cap, scores, predictions in ((8, scores8, predictions8), (64, scores64, predictions64)):
            row[f"cap{cap}"] = {key: scores[identifier][key] for key in fields}
            row[f"cap{cap}"].update({"token_budget": cap, "generated_tokens_including_eos": len(predictions[identifier]["generated_ids"])})
        row["fallback_correct_delta"] = int(scores64[identifier]["fallback_correct"]) - int(scores8[identifier]["fallback_correct"])
        row["strict_correct_delta"] = int(scores64[identifier]["strict_correct"]) - int(scores8[identifier]["strict_correct"])
        rows.append(row)
    return {"details": rows,
            "cap8": {key: primary["scores"][key] for key in ("total", "strict_correct", "fallback_correct", "invalid", "fallback_invalid", "fallback_ambiguous", "hit_token_limit")},
            "cap64": {key: secondary["scores"][key] for key in ("total", "strict_correct", "fallback_correct", "invalid", "fallback_invalid", "fallback_ambiguous", "hit_token_limit")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True, help="new directory outside the primary run")
    args = parser.parse_args()
    run, out = path(args.run).resolve(), path(args.output).resolve()
    if out.is_relative_to(run):
        raise ValueError("secondary output must be outside the primary run")
    main_run = read_json(run / "run.json")
    if main_run.get("completed") is not True or main_run.get("status") != "complete" or main_run.get("phase") != "complete":
        raise ValueError("main run must complete before sensitivity evaluation")
    for key in ("exact_state_tensors_equal", "gsm8k_first16_tokens_equal", "retention_all128_tokens_equal"):
        if main_run.get("reload", {}).get(key) is not True:
            raise ValueError(f"main reload check did not pass: {key}")
    if read_json(run / "reload_state.json").get("exact_state_tensors_equal") is not True:
        raise ValueError("saved main reload state check failed")
    if digest(run / "frozen.json") != main_run["frozen_manifest_sha256"] or digest(run / "protocol.json") != main_run["protocol_sha256"]:
        raise ValueError("main frozen/protocol manifest changed")
    verify_snapshot(run)
    protocol, cfg = read_json(run / "protocol.json"), read_json(run / "config.json")
    used_hashes = {}
    for relative in ("scripts/batched_generation.py", "scripts/score_retention.py"):
        used_hashes[relative] = digest(run / "source" / relative)
        if used_hashes[relative] != protocol["source_sha256"][relative]:
            raise ValueError(f"frozen helper/scorer differs from original protocol: {relative}")
    generation = load_frozen_module(run / "source/scripts/batched_generation.py", "_tunellm_original_generation")
    scoring = load_frozen_module(run / "source/scripts/score_retention.py", "_tunellm_original_retention_scoring")
    fixture = run / "inputs/retention.json"
    if digest(fixture) != protocol["input_sha256"]["retention.json"]:
        raise ValueError("retention fixture changed")
    rows = read_json(fixture)["tasks"]
    expected_ids = protocol["retention_ids"]
    if len(rows) != 128 or [row["id"] for row in rows] != expected_ids:
        raise ValueError("expected same128 frozen retention rows in original order")
    primary = {condition: read_json(run / f"{condition}_final_retention.json") for condition in ("base", "adapter")}
    for saved in primary.values():
        verify_settings(saved, 8, expected_ids)
        if scoring.score_retention(rows, saved["predictions"]) != saved["scores"]:
            raise ValueError("saved cap8 scores disagree with original frozen parser")
    model_snapshot = read_json(run / "inputs/model_snapshot.json")
    snapshot = path(model_snapshot["path"])
    for name, expected in model_snapshot["files"].items():
        model_file = snapshot / name
        if model_file.stat().st_size != expected["size"] or model_file.stat().st_mtime_ns != expected["mtime_ns"] or str(model_file.resolve()) != expected["resolved"] or ("sha256" in expected and digest(model_file) != expected["sha256"]):
            raise ValueError(f"main model snapshot changed: {name}")
    adapter = run / "adapter"
    adapter_cfg = read_json(adapter / "adapter_config.json")
    if adapter_cfg["base_model_name_or_path"] != cfg["model_id"] or adapter_cfg["revision"] != cfg["revision"]:
        raise ValueError("saved adapter does not reference the original pinned model")
    select_gpu(cfg["gpu"])
    setup_hf(cfg)
    out.mkdir(parents=True, exist_ok=False)
    metadata = provenance({"primary_run": str(run), "gpu": cfg["gpu"], "max_new_tokens": 64,
                           "batch_size": 8, "seed": 42, "enable_thinking": False, "do_sample": False,
                           "model_id": cfg["model_id"], "revision": cfg["revision"]})
    metadata.update({"completed": False, "status": "running", "phase": "loading_base",
                     "purpose": "Single predeclared exploratory secondary cap64 sensitivity; main cap8 results remain primary. No training, checkpoint selection, parser or fixture changes.",
                     "primary_protocol_sha256": main_run["protocol_sha256"],
                     "primary_run_sha256": digest(run / "run.json"), "fixture_sha256": digest(fixture),
                     "original_helper_scorer_sha256": used_hashes,
                     "selected_step": main_run["selection"]["selected_step"],
                     "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
                     "adapter_provenance_sha256": digest(adapter / "provenance.json"),
                     "primary_result_sha256": {condition: digest(run / f"{condition}_final_retention.json") for condition in primary},
                     "representation": generation.REPRESENTATION, "ordered_ids": expected_ids})
    write_json(out / "run.json", metadata)
    try:
        from transformers import AutoTokenizer, set_seed
        from peft import PeftModel
        import torch
        set_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(run / "inputs/tokenizer", local_files_only=True, trust_remote_code=False)
        for row in rows:
            prompt = generation.render_prompt(tokenizer, row)
            for saved in primary.values():
                prediction = by_id(saved["predictions"])[row["id"]]
                if prediction["rendered_prompt"] != prompt or prediction["input_ids"] != tokenizer(prompt, add_special_tokens=False)["input_ids"]:
                    raise ValueError("primary saved prompt tokens differ from frozen fixture/tokenizer")
        base = generation.load_nf4(snapshot)
        comparison = {}
        for condition in ("base", "adapter"):
            metadata["phase"] = condition + "_cap64"
            write_json(out / "run.json", metadata)
            if condition == "adapter":
                model = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
            else:
                model = base
            saved = generation.generate_batched(model, tokenizer, rows, batch_size=8, max_new_tokens=64,
                                                seed=42, phase=condition + "/retention_cap64")
            saved["scores"] = scoring.score_retention(rows, saved["predictions"])
            write_json(out / f"{condition}_cap64_retention.json", saved)
            comparison[condition] = compare_caps(primary[condition], saved, expected_ids)
            write_json(out / "cap8_vs_cap64.json", comparison)
        metadata.update({"completed": True, "status": "complete", "phase": "complete"})
    except BaseException as exc:
        metadata.update({"completed": False, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        write_json(out / "run.json", metadata)
    print(out)


if __name__ == "__main__":
    main()
