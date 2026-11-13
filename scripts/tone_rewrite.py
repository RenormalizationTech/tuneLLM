"""One registered warm-rewrite fallback; original tone core remains unchanged.

Without --execute-gpu this validates local inputs only. GPU execution is for
the coordinator's explicitly released GPU0 window, never for automatic retries.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import time

from scripts.common import path, read_json
from scripts.save_nf4 import stream_hash
from scripts.tone_common import aligned
from scripts.tone_steering import committed, gpu_ready, gpu_record, load_model, save, verify_run

NEUTRAL_SHA256 = "3805d89738b7f2eb0ee34184d60cd9edfc02cd904da70b2836c7f6c6c610c441"
NEUTRAL_SYSTEM = "Answer the user's request accurately in two or three sentences. Preserve the stated facts and do not invent details."
TEMPLATE = ("Original scenario:\n{scenario}\n\nNeutral message:\n{neutral}\n\n"
            "Rewrite this message in a warm, encouraging, considerate tone. Keep every stated fact and the same requested message; "
            "add no facts, flattery or extra claims. Use at most {N} words.\nReturn only the rewritten message.")


def rewrite_rows(rows, predictions):
    aligned(rows, predictions)
    if len(rows) != 80:
        raise ValueError("all 80 original round-robin scenarios required")
    prompts, budgets = [], []
    for row, prediction in zip(rows, predictions):
        if not prediction["output"].strip():
            raise ValueError("neutral message cannot be empty")
        count = len(prediction["output"].split())
        budget = count + 3
        prompts.append({"id": row["id"], "system": NEUTRAL_SYSTEM,
            "user": TEMPLATE.format(scenario=row["user"], neutral=prediction["output"], N=budget)})
        budgets.append({"id": row["id"], "neutral_whitespace_words": count, "instructed_max_words": budget})
    return prompts, budgets


def validate_inputs(run, source_run, registration, required_neutral_sha=NEUTRAL_SHA256):
    cfg, manifest = verify_run(run)
    source_cfg, source_manifest = read_json(source_run / "config.json"), read_json(source_run / "manifest.json")
    neutral_file = source_run / "extraction/neutral.json"
    source_inputs = source_run / "inputs/extraction.json"
    rows = read_json(run / "inputs/extraction.json")
    if (cfg["gpu"] != "0" or cfg["neutral_system"] != NEUTRAL_SYSTEM or cfg["seed"] != 42
            or cfg["batch_size"] != 8 or cfg["max_new_tokens"] != 256
            or cfg["protocol"] != "docs/tone_steering_protocol_02b.md"
            or any(cfg[k] != source_cfg[k] for k in cfg if k != "protocol")):
        raise ValueError("B must preserve A config/systems and pinned GPU0 settings")
    if (stream_hash(neutral_file) != required_neutral_sha
            or registration.get("a_neutral_sha256") != required_neutral_sha
            or registration.get("rewrite_source_sha256") != stream_hash(Path(__file__))
            or registration.get("a_manifest_sha256") != stream_hash(source_run / "manifest.json")
            or registration.get("a_input_extraction_sha256") != stream_hash(source_inputs)
            or stream_hash(source_inputs) != stream_hash(run / "inputs/extraction.json")
            or registration.get("rewrite_template") != TEMPLATE):
        raise ValueError("rewrite helper/template or fixed A source registration differs")
    if source_manifest["sources"] != manifest["sources"]:
        raise ValueError("original core sources must match A exactly")
    if read_json(source_inputs) != rows:
        raise ValueError("A/B scenario order or content differs")
    source_record = read_json(source_run / "extraction/run.json")
    if (source_record.get("completed") is not True or source_record.get("representation") != "bf16"
            or source_record.get("physical_gpu") != "0" or source_record.get("gpu_uuid") != registration.get("a_gpu_uuid")):
        raise ValueError("completed BF16 A generation on the same registered GPU0 UUID required")
    # Verify original source prompts, rather than trusting only returned row IDs.
    neutral = read_json(neutral_file)
    aligned(rows, neutral["predictions"])
    from transformers import AutoTokenizer
    from scripts.batched_generation import render_prompt
    tokenizer = AutoTokenizer.from_pretrained(source_manifest["snapshot"], local_files_only=True, trust_remote_code=False)
    for row, prediction in zip(rows, neutral["predictions"]):
        prompt = render_prompt(tokenizer, {**row, "system": NEUTRAL_SYSTEM})
        if (prediction["rendered_prompt"] != prompt
                or prediction["input_ids"] != tokenizer(prompt, add_special_tokens=False)["input_ids"]):
            raise ValueError("A neutral predictions do not bind to original scenario/context")
    prompts, budgets = rewrite_rows(rows, neutral["predictions"])
    return cfg, manifest, prompts, budgets, neutral_file


def compliance(budgets, predictions):
    if [r["id"] for r in budgets] != [p["id"] for p in predictions]:
        raise ValueError("rewrite order/IDs differ")
    return [{**r, "actual_whitespace_words": len(p["output"].split()),
             "instruction_complied": len(p["output"].split()) <= r["instructed_max_words"],
             "word_budget_is_hard_eligibility_gate": False} for r, p in zip(budgets, predictions)]


def execute(args):
    run, source_run = path(args.run), path(args.source_run)
    registration = read_json(args.registration)
    cfg, manifest, prompts, budgets, neutral_file = validate_inputs(run, source_run, registration)
    if not args.execute_gpu:
        print("B inputs validated offline: 80 single rewrites; no model load or generation.")
        return
    committed(Path(__file__))
    committed("configs/tone_steering_02b.json")
    cfg, manifest = gpu_ready(args, run)
    device = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=uuid", "--format=csv,noheader"],
                            text=True, capture_output=True, check=True).stdout.strip()
    if device != registration["a_gpu_uuid"]:
        raise ValueError("GPU0 UUID differs from A; no allocation/generation")
    directory = run / "extraction"
    directory.mkdir(exist_ok=False)
    shutil.copyfile(neutral_file, directory / "neutral.json")
    if stream_hash(directory / "neutral.json") != NEUTRAL_SHA256:
        raise ValueError("neutral byte copy failed")
    shutil.copyfile(Path(__file__), directory / "rewrite_implementation.py")
    started = time.perf_counter()
    model, tokenizer = load_model(cfg, manifest, "bf16")
    from scripts.batched_generation import generate_batched
    warm = generate_batched(model, tokenizer, prompts, batch_size=8, max_new_tokens=256,
                            seed=42, phase="single registered warm rewrite")
    save(directory / "warm.json", warm)
    save(directory / "rewrite_diagnostics.json", {"items": compliance(budgets, warm["predictions"]),
        "template": TEMPLATE, "neutral_sha256": NEUTRAL_SHA256,
        "rewrite_source_sha256": stream_hash(Path(__file__)), "registration_sha256": stream_hash(path(args.registration)),
        "source_run_manifest_sha256": stream_hash(source_run / "manifest.json"),
        "asymmetry": "Warm construction conditions on the retained neutral partner; extraction re-encodes both in identical neutral context.",
        "hard_eligibility": "Original token-length window/content/task/warmer/50-pair rules unchanged; word compliance diagnostic only.",
        "attempts_per_scenario": 1})
    save(directory / "run.json", {**gpu_record(run, cfg, "bf16", started),
         "construction": "single registered B rewrite fallback", "neutral_source_sha256": NEUTRAL_SHA256,
         "rewrite_source_sha256": stream_hash(Path(__file__)), "registration_sha256": stream_hash(path(args.registration))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/tone-02b")
    parser.add_argument("--source-run", default="outputs/tone-02")
    parser.add_argument("--registration", required=True)
    parser.add_argument("--execute-gpu", action="store_true")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
