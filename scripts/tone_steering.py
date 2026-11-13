"""Small staged tone pilot. GPU stages require a separately authorized device window.

No remote judges, downloads, training or weight changes. New phase files never
overwrite old outputs. See scripts/tone_runbook.md for stage order and locks.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from scripts.common import ROOT, path, read_json, select_gpu, setup_hf, write_json
from scripts.save_nf4 import stream_hash
from scripts.tone_common import (aligned, decoded_ratings, gates, grouped_interval, literal_checks,
    make_packets, mean_direction, measurements, response_encoding, response_means, select, steer,
    token_length, validate_ratings)
from scripts.tone_fixture import validate

SOURCES = ("scripts/tone_steering.py", "scripts/tone_common.py", "scripts/tone_fixture.py",
           "scripts/common.py", "scripts/save_nf4.py", "scripts/batched_generation.py")


def save(filename, value):
    """Exclusive immutable JSON artifact; no accidental phase rerun/overwrite."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with filename.open("x", encoding="utf-8") as stream:
        stream.write(data)


def committed(filename):
    filename = path(filename).resolve()
    relative = filename.relative_to(ROOT)
    result = subprocess.run(["git", "show", f"HEAD:{relative.as_posix()}"], cwd=ROOT, capture_output=True)
    if result.returncode or result.stdout != filename.read_bytes():
        raise ValueError("reviewed protocol/lock must be committed exactly before GPU execution")


def prepare(args):
    out = path(args.run)
    if out.exists():
        raise FileExistsError("fresh run directory required")
    cfg = read_json(args.config)
    if (cfg["gpu"] not in ("0", "1") or cfg["layers"] != [11, 17, 23] or cfg["multipliers"] != [1, 2, 4]
            or cfg["retain_pairs"] != 50 or cfg["seed"] != 42
            or cfg["revision"] != "1cfa9a7208912126459214e8b04321603b3df60c"):
        raise ValueError("registered model/layer/strength/pair/seed policy required")
    fixture = path(cfg["fixture_directory"])
    validation = validate({s: read_json(fixture / f"{s}.json") for s in ("extraction", "development", "final")})
    snapshot = path(cfg["hf_home"]) / "hub/models--Qwen--Qwen3-4B/snapshots" / cfg["revision"]
    if not (snapshot / "model.safetensors.index.json").is_file():
        raise ValueError("original pinned model must already be cached; no downloads")
    nf4 = path(cfg["nf4_checkpoint"])
    report = read_json(cfg["nf4_report"])
    if report.get("completed") is not True or report.get("revision") != cfg["revision"]:
        raise ValueError("verified pinned NF4 checkpoint report required")
    external = {}
    for filename in sorted(snapshot.iterdir()):
        if filename.is_file():
            external[str(filename)] = stream_hash(filename)
    for name, record in report["files"].items():
        filename = nf4 / name
        if stream_hash(filename) != record["sha256"]:
            raise ValueError("stored NF4 checkpoint changed")
        external[str(filename)] = record["sha256"]
    out.mkdir(parents=True)
    for name in SOURCES:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path(name), target)
    save(out / "config.json", cfg)
    inputs = {f"{s}.json": fixture / f"{s}.json" for s in ("extraction", "development", "final")}
    inputs.update({"rubric.json": path(cfg["rubric"]), "protocol.md": path(cfg["protocol"])})
    for name, filename in inputs.items():
        target = out / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(filename, target)
    files = {str(p.relative_to(out)): stream_hash(p) for p in out.rglob("*") if p.is_file()}
    save(out / "manifest.json", {"files": files, "external_model_sha256": external,
         "sources": {n: stream_hash(path(n)) for n in SOURCES}, "snapshot": str(snapshot),
         "nf4_checkpoint": str(nf4), "fixture_validation": validation,
         "model_id": cfg["model_id"], "revision": cfg["revision"], "prepared_only": True,
         "semantic_fixture_audit_required": True, "no_generated_responses": True})
    print("CPU snapshot prepared. Advisor fixture audit and committed manifest are required before extraction generation.")


def verify_run(run, models=False):
    manifest = read_json(run / "manifest.json")
    for name, expected in manifest["files"].items():
        if stream_hash(run / name) != expected:
            raise ValueError("frozen tone input/source changed")
    for name, expected in manifest["sources"].items():
        if stream_hash(path(name)) != expected:
            raise ValueError("active tone/inference source differs from frozen source")
    if models:
        for filename, expected in manifest["external_model_sha256"].items():
            if stream_hash(path(filename)) != expected:
                raise ValueError("read-only base model bytes changed")
    return read_json(run / "config.json"), manifest


def gpu_ready(args, run):
    if not args.execute_gpu:
        raise ValueError("GPU stages require --execute-gpu AFTER coordinator releases the configured device")
    cfg, manifest = verify_run(run, models=True)
    if not args.registration:
        raise ValueError("committed preparation registration and semantic fixture review required")
    committed(args.registration)
    registration = read_json(args.registration)
    if (registration.get("manifest_sha256") != stream_hash(run / "manifest.json")
            or registration.get("semantic_fixture_audit_passed") is not True):
        raise ValueError("pre-generation registration differs or fixture audit not passed")
    for filename in SOURCES:
        committed(filename)
    # Compare physical UUIDs across phases before allocating any model memory.
    device = subprocess.run(["nvidia-smi", "--id=" + cfg["gpu"], "--query-gpu=uuid", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True).stdout.strip()
    for name in ("extraction/run.json", "extraction/vectors.json", "preflight/preflight.json",
                 "development/run.json", "final-bf16/run.json", "final-nf4/run.json"):
        if (run / name).is_file() and read_json(run / name).get("gpu_uuid") != device:
            raise ValueError("physical device UUID changed across comparisons")
    return cfg, manifest


def load_model(cfg, manifest, representation):
    select_gpu(cfg["gpu"])
    setup_hf(cfg, offline=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from scripts.save_nf4 import reload_stored_nf4
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count() != 1:
        raise RuntimeError("one selected physical GPU with BF16 support required")
    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(manifest["snapshot"], local_files_only=True, trust_remote_code=False)
    if representation == "nf4":
        model = reload_stored_nf4(manifest["nf4_checkpoint"])
    else:
        model = AutoModelForCausalLM.from_pretrained(manifest["snapshot"], local_files_only=True,
            use_safetensors=True, trust_remote_code=False, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": 0})
    return model.eval(), tokenizer


def generate(model, tokenizer, rows, cfg, system, cap, vector=None, layer=None, strength=0, audit=None):
    from scripts.batched_generation import generate_batched
    prompts = [{**r, "system": system} for r in rows]
    with steer(model, layer, vector, strength, audit):
        return generate_batched(model, tokenizer, prompts, batch_size=cfg["batch_size"],
                                max_new_tokens=cap, seed=42, phase="tone generation")


def gpu_record(run, cfg, representation, started):
    import importlib.metadata
    import torch
    device = subprocess.run(["nvidia-smi", "--id=" + cfg["gpu"],
        "--query-gpu=uuid,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True)
    uuid, driver = [s.strip() for s in device.stdout.strip().split(",")]
    return {"completed": True, "physical_gpu": cfg["gpu"], "representation": representation,
        "gpu_uuid": uuid, "driver_version": driver, "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "logical_device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(0), "seed": 42, "backend": "transformers sdpa",
        "dtype": "BF16 base" if representation == "bf16" else "stored NF4/BF16 compute/FP32 nonquantized",
        "packages": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "bitsandbytes")},
        "manifest_sha256": stream_hash(run / "manifest.json"), "batch_size": cfg["batch_size"],
        "elapsed_wall_seconds": time.perf_counter()-started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "memory_scope": "PyTorch allocator incl loaded weights; not total VRAM or external jobs"}


def extraction_generate(args, run):
    cfg, manifest = gpu_ready(args, run)
    directory = run / "extraction"
    directory.mkdir(exist_ok=False)
    rows = read_json(run / "inputs/extraction.json")
    started = time.perf_counter()
    model, tokenizer = load_model(cfg, manifest, "bf16")
    for condition in ("neutral", "warm"):
        generated = generate(model, tokenizer, rows, cfg, cfg[condition+"_system"], cfg["max_new_tokens"])
        save(directory / f"{condition}.json", generated)
    save(directory / "run.json", gpu_record(run, cfg, "bf16", started))


def packets(args, run):
    cfg, _ = verify_run(run)
    stage = args.stage
    directory = run / stage
    if stage == "extraction":
        rows = read_json(run / "inputs/extraction.json")
        conditions = {n: read_json(directory / f"{n}.json")["predictions"] for n in ("neutral", "warm")}
        comparisons = [("neutral", "warm")]
    else:
        if stage.startswith("final-"):
            check_lock(run, args.lock)
        split = "final" if stage.startswith("final-") else "development"
        rows = read_json(run / f"inputs/{split}.json")
        records = read_json(directory / "conditions.json")
        conditions = {n: read_json(directory / f"{n}.json")["predictions"] for n in records["conditions"]}
        comparisons = records["comparisons"]
    packet, keys = make_packets(rows, conditions, comparisons, cfg["judging_seed"])
    out = directory / "judging"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    os.chmod(out, 0o700)
    for kind in ("content", "warmth"):
        save(out / f"{kind}_packet.json", {"rubric_sha256": stream_hash(run / "inputs/rubric.json"), "items": packet[kind]})
    save(out / "keys.local.json", keys)
    os.chmod(out / "keys.local.json", 0o600)
    shutil.copyfile(run / "inputs/rubric.json", out / "rubric.json")
    source_names = (["neutral.json", "warm.json", "run.json"] if stage == "extraction" else
                    ["conditions.json", "run.json"] + [f"{n}.json" for n in conditions])
    save(out / "packet_manifest.json", {"packet_files": {p.name: stream_hash(p) for p in out.iterdir() if p.is_file()},
         "source_files": {n: stream_hash(directory / n) for n in source_names}})
    print("Give judge only rubric.json and the two *_packet.json files. Keep keys and source/results out of context.")


def import_ratings(args, run):
    verify_run(run)
    directory = run / args.stage / "judging"
    verify_packets(directory.parent)
    packet = read_json(directory / f"{args.kind}_packet.json")["items"]
    submission = validate_ratings(packet, read_json(args.ratings), args.kind)
    save(directory / f"{args.kind}_ratings.json", {**submission, "input_sha256": stream_hash(path(args.ratings)),
         "packet_sha256": stream_hash(directory / f"{args.kind}_packet.json"),
         "rubric_sha256": stream_hash(directory / "rubric.json")})


def verify_packets(directory):
    judges = directory / "judging"
    manifest = read_json(judges / "packet_manifest.json")
    for name, expected in manifest["packet_files"].items():
        if stream_hash(judges / name) != expected:
            raise ValueError("frozen judgment packet changed")
    for name, expected in manifest["source_files"].items():
        if stream_hash(directory / name) != expected:
            raise ValueError("prediction/condition source changed after packet creation")


def judged(directory):
    judges = directory / "judging"
    verify_packets(directory)
    submissions = {k: read_json(judges / f"{k}_ratings.json") for k in ("content", "warmth")}
    for kind, submission in submissions.items():
        if submission["packet_sha256"] != stream_hash(judges / f"{kind}_packet.json"):
            raise ValueError("rating import packet mismatch")
        validate_ratings(read_json(judges / f"{kind}_packet.json")["items"], submission, kind)
    return decoded_ratings(read_json(judges / "keys.local.json"), submissions["content"], submissions["warmth"])


def eligible_pairs(rows, neutral, warm, content, preferences, cfg):
    aligned(rows, neutral)
    aligned(rows, warm)
    retained, audit = [], []
    for row, a, b in zip(rows, neutral, warm):
        ratio = token_length(b) / max(1, token_length(a))
        ai = [content[n][row["id"]] for n in ("neutral", "warm")]
        fixed = [literal_checks(row, p["output"]) for p in (a, b)]
        checks = {"literal": all(r["complete"] and not r["unsupported_literal"] for r in fixed),
                  "ai_content": all(all(r["coverage"].values()) and r["task_completed"] and not r["unsupported_claims"]
                                    and not r["contradicted_claims"] for r in ai),
                  "length_ratio": cfg["pair_length_ratio"][0] <= ratio <= cfg["pair_length_ratio"][1],
                  "warmer": preferences["neutral__warm"][row["id"]] == 1.0,
                  "nonempty_uncapped": token_length(a) > 0 and token_length(b) > 0 and not a["hit_token_limit"] and not b["hit_token_limit"]}
        good = all(checks.values())
        if good and len(retained) < cfg["retain_pairs"]:
            retained.append(row["id"])
        audit.append({"id": row["id"], "family": row["family"], "eligible": good, "checks": checks, "warm_neutral_token_ratio": ratio})
    return retained, audit


def extract(args, run):
    cfg, manifest = gpu_ready(args, run)
    directory = run / "extraction"
    content, preferences = judged(directory)
    rows = read_json(run / "inputs/extraction.json")
    neutral, warm = (read_json(directory / f"{n}.json")["predictions"] for n in ("neutral", "warm"))
    retained, audit = eligible_pairs(rows, neutral, warm, content, preferences, cfg)
    attrition = {family: {"eligible": sum(r["eligible"] for r in audit if r["family"] == family),
                         "retained": sum(r["id"] in retained for r in audit if r["family"] == family)}
                 for family in sorted({r["family"] for r in rows})}
    save(directory / "eligibility.json", {"retained_ids": retained, "audit": audit, "by_family": attrition,
                                          "required_pairs": 50, "sufficient": len(retained) == 50})
    if len(retained) != 50:
        raise ValueError("fewer than 50 eligible pairs: consult advisor; never relax or fabricate")
    import torch
    started = time.perf_counter()
    model, tokenizer = load_model(cfg, manifest, "bf16")
    means = {layer: [] for layer in cfg["layers"]}
    boundaries = []
    for row, a, b in zip(rows, neutral, warm):
        if row["id"] not in retained:
            continue
        pair = []
        for prediction in (a, b):
            encoded = response_encoding(tokenizer, row, prediction["output"], cfg["neutral_system"])
            boundaries.append({"id": row["id"], "response": "neutral" if not pair else "warm",
                "masked_response_tokens": sum(encoded["response_mask"]),
                "excluded_whitespace_boundary_tokens": encoded["excluded_whitespace_boundary_tokens"],
                "boundary_method": encoded["boundary_method"]})
            inputs = torch.tensor([encoded["input_ids"]], dtype=torch.long, device="cuda:0")
            mask = torch.tensor([encoded["response_mask"]], dtype=torch.bool)
            collected = {}
            with response_means(model, cfg["layers"], mask, collected), torch.inference_mode():
                model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
            pair.append(collected)
        for layer in means:
            means[layer].append((pair[0][layer], pair[1][layer]))
    vectors, diagnostics = {}, {}
    for layer, pairs in means.items():
        vectors[str(layer)], diagnostics[str(layer)] = mean_direction(pairs)
    filename = directory / "vectors.pt"
    if filename.exists():
        raise FileExistsError(filename)
    torch.save(vectors, filename)
    save(directory / "vectors.json", {"sha256": stream_hash(filename), "layers": diagnostics,
         "boundary_audit": boundaries,
         "retained_ids": retained, "eligibility_sha256": stream_hash(directory / "eligibility.json"),
         "ratings_sha256": {n: stream_hash(directory / f"judging/{n}_ratings.json") for n in ("content", "warmth")},
         **gpu_record(run, cfg, "bf16", started)})


def vectors(run):
    import torch
    filename = run / "extraction/vectors.pt"
    if stream_hash(filename) != read_json(run / "extraction/vectors.json")["sha256"]:
        raise ValueError("extracted vector changed")
    return torch.load(filename, map_location="cpu", weights_only=True)


def preflight(args, run):
    cfg, manifest = gpu_ready(args, run)
    directory = run / "preflight"
    directory.mkdir(exist_ok=False)
    rows = read_json(run / "inputs/development.json")
    started = time.perf_counter()
    model, tokenizer = load_model(cfg, manifest, "bf16")
    initial = generate(model, tokenizer, rows, cfg, cfg["neutral_system"], 256)
    save(directory / "default_cap256.json", initial)
    hits = sum(p["hit_token_limit"] for p in initial["predictions"])
    cap = 320 if hits / len(rows) > .10 else 256
    default = initial if cap == 256 else generate(model, tokenizer, rows, cfg, cfg["neutral_system"], cap)
    save(directory / "default.json", default)
    import torch
    direction = vectors(run)["17"]
    zero_audit, nonzero_audit = {}, {}
    zero = generate(model, tokenizer, rows[:cfg["batch_size"]], cfg, cfg["neutral_system"], cap,
                    torch.zeros_like(direction), 17, 1, zero_audit)
    exact = all(a["input_ids"] == b["input_ids"] and a["generated_ids"] == b["generated_ids"]
                for a, b in zip(default["predictions"], zero["predictions"]))
    save(directory / "zero_probe.json", zero)
    smoke = generate(model, tokenizer, rows[:cfg["batch_size"]], cfg, cfg["neutral_system"], cap,
                     direction, 17, 1, nonzero_audit)
    save(directory / "middle_layer_m1_smoke.json", smoke)
    from scripts.batched_generation import left_pad, render_prompt
    ids = [tokenizer(render_prompt(tokenizer, {**r, "system": cfg["neutral_system"]}), add_special_tokens=False)["input_ids"]
           for r in rows[:cfg["batch_size"]]]
    inputs = {k: torch.tensor(v, dtype=torch.long, device="cuda:0") for k, v in left_pad(ids, tokenizer.pad_token_id).items()}
    with torch.inference_mode():
        baseline_logits = model(**inputs, use_cache=False).logits[:, -1].detach().float()
        with steer(model, 17, direction, 1):
            steered_logits = model(**inputs, use_cache=False).logits[:, -1].detach().float()
    nonzero_audit["max_abs_last_prompt_logit_delta"] = (steered_logits-baseline_logits).abs().max().item()
    hook_checks = all(a.get("prefill_calls", 0) > 0 and a.get("decode_calls", 0) > 0 for a in (zero_audit, nonzero_audit))
    hook_checks = hook_checks and nonzero_audit["max_abs_hidden_delta"] > 0 and nonzero_audit["max_abs_last_prompt_logit_delta"] > 0
    report = {"cap": cap, "initial_cap": 256, "initial_cap_hits": hits, "sample_ids": [r["id"] for r in rows],
              "batch_size": cfg["batch_size"], "seed": 42, "zero_exact": exact,
              "zero_vector_multiplier1_hook_audit": zero_audit, "middle_layer_m1_hook_audit": nonzero_audit,
              "hook_instrumentation_passed": hook_checks,
              "nonzero_smoke_scope": "Fixed layer17/m1 development-only instrumentation; tokens may remain unchanged; not a selection result.",
              "default_sha256": stream_hash(directory / "default.json"),
              "vectors_sha256": stream_hash(run / "extraction/vectors.pt"),
              "source_sha256": manifest["sources"], "manifest_sha256": stream_hash(run / "manifest.json"),
              "projected_remaining_generation_wall_seconds": default["timing"]["pipeline_wall_seconds"] * (9 + 2*60/30*3),
              "budget_scope": "Rough generation-only projection for nine dev settings plus up to three final conditions in two representations; excludes loading, extraction, teacher forcing and AI judgment. NF4 rate may differ. Preflight uses all 30 development rows; final-cap default outputs reused for selection.",
              **gpu_record(run, cfg, "bf16", started)}
    save(directory / "preflight.json", report)
    if not exact or not hook_checks:
        raise ValueError("zero vector parity or actual-hook instrumentation failed")


def budget_lock(args, run):
    if not args.budget:
        raise ValueError("committed preflight report required as --budget")
    committed(args.budget)
    registered = read_json(args.budget)
    actual = read_json(run / "preflight/preflight.json")
    if registered != actual or not actual["zero_exact"] or not actual["hook_instrumentation_passed"] or not actual["completed"]:
        raise ValueError("budget must exactly match successful preflight report")
    if stream_hash(run / "preflight/default.json") != actual["default_sha256"]:
        raise ValueError("preflight default generation changed")
    if (stream_hash(run / "extraction/vectors.pt") != actual["vectors_sha256"]
            or stream_hash(run / "manifest.json") != actual["manifest_sha256"]
            or read_json(run / "manifest.json")["sources"] != actual["source_sha256"]):
        raise ValueError("preflight vector/source/manifest changed before budget lock")
    return actual


def development(args, run):
    cfg, manifest = gpu_ready(args, run)
    budget = budget_lock(args, run)
    directory = run / "development"
    directory.mkdir(exist_ok=False)
    save(directory / "budget_lock.json", {"budget_sha256": stream_hash(path(args.budget)), "budget": budget})
    rows = read_json(run / "inputs/development.json")
    started = time.perf_counter()
    model, tokenizer = load_model(cfg, manifest, "bf16")
    directions = vectors(run)
    shutil.copyfile(run / "preflight/default.json", directory / "default.json")
    names = ["default"]
    for layer in cfg["layers"]:
        for strength in cfg["multipliers"]:
            name = f"layer{layer}_m{strength}"
            names.append(name)
            result = generate(model, tokenizer, rows, cfg, cfg["neutral_system"], budget["cap"], directions[str(layer)], layer, strength)
            save(directory / f"{name}.json", result)
    save(directory / "conditions.json", {"conditions": names, "comparisons": [["default", n] for n in names[1:]],
         "prediction_sha256": {n: stream_hash(directory / f"{n}.json") for n in names}})
    save(directory / "run.json", gpu_record(run, cfg, "bf16", started))


def select_development(args, run):
    cfg, _ = verify_run(run)
    directory = run / "development"
    rows = read_json(run / "inputs/development.json")
    content, preferences = judged(directory)
    record = read_json(directory / "conditions.json")
    predictions = {}
    for name in record["conditions"]:
        filename = directory / f"{name}.json"
        if stream_hash(filename) != record["prediction_sha256"][name]:
            raise ValueError("development predictions changed after judging")
        predictions[name] = read_json(filename)["predictions"]
    base = measurements(rows, predictions["default"], content["default"])
    candidates = []
    for layer in cfg["layers"]:
        for strength in cfg["multipliers"]:
            name = f"layer{layer}_m{strength}"
            candidate = measurements(rows, predictions[name], content[name])
            ratios = [token_length(b) / max(1, token_length(a)) for a, b in zip(predictions["default"], predictions[name])]
            scores = preferences["default__"+name]
            if set(scores) != {r["id"] for r in rows}:
                raise ValueError("incomplete warmth comparison")
            candidates.append({"condition": name, "layer": layer, "multiplier": strength, "metrics": candidate,
                               "gates": gates(base, candidate, ratios), "warmth_score": statistics.mean(scores.values())})
    save(run / "selection.json", {"split": "development", "decision": select(candidates), "default": base,
        "candidates": candidates, "manifest_sha256": stream_hash(run / "manifest.json"),
        "budget_lock_sha256": stream_hash(directory / "budget_lock.json"),
        "vectors_sha256": stream_hash(run / "extraction/vectors.pt"),
        "extraction_vectors_report_sha256": stream_hash(run / "extraction/vectors.json"),
        "development_conditions_sha256": stream_hash(directory / "conditions.json"),
        "rating_sha256": {n: stream_hash(directory / f"judging/{n}_ratings.json") for n in ("content", "warmth")},
        "final_used_for_selection": False, "interpretation": "AI rubric measurement on 30 prompts, six independent families; weak evidence, not human preference."})


def final_lock(run):
    selection = read_json(run / "selection.json")
    if selection["split"] != "development" or selection["final_used_for_selection"] is not False:
        raise ValueError("development-only fixed selection required")
    expected = {"manifest.json": selection["manifest_sha256"],
                "development/budget_lock.json": selection["budget_lock_sha256"],
                "development/conditions.json": selection["development_conditions_sha256"],
                "extraction/vectors.pt": selection["vectors_sha256"],
                "extraction/vectors.json": selection["extraction_vectors_report_sha256"],
                **{f"development/judging/{n}_ratings.json": h for n, h in selection["rating_sha256"].items()}}
    for name, expected_hash in expected.items():
        if stream_hash(run / name) != expected_hash:
            raise ValueError("selection input changed before final lock")
    verify_packets(run / "development")
    conditions = read_json(run / "development/conditions.json")
    for name, expected_hash in conditions["prediction_sha256"].items():
        if stream_hash(run / "development" / f"{name}.json") != expected_hash:
            raise ValueError("selected development predictions changed")
    files = ["manifest.json", "selection.json", "development/budget_lock.json", "development/conditions.json",
             "extraction/vectors.pt", "extraction/vectors.json", "inputs/final.json",
             "development/judging/content_ratings.json", "development/judging/warmth_ratings.json"]
    return {"selection_fixed": True, "decision": selection["decision"],
            "files": {n: stream_hash(run / n) for n in files},
            "cap": read_json(run / "development/budget_lock.json")["budget"]["cap"],
            "transfer": "exact BF16 vector, layer, multiplier and cap; no NF4 recalibration or final reselection"}


def check_lock(run, filename):
    if not filename:
        raise ValueError("committed final lock required")
    committed(filename)
    lock = read_json(filename)
    if lock != final_lock(run):
        raise ValueError("fixed final manifest or selected development inputs changed")
    return lock


def final_generate(args, run):
    cfg, manifest = gpu_ready(args, run)
    lock = check_lock(run, args.lock)  # Before opening final prompt content.
    representation = args.representation
    directory = run / ("final-" + representation)
    directory.mkdir(exist_ok=False)
    rows = read_json(run / "inputs/final.json")
    started = time.perf_counter()
    model, tokenizer = load_model(cfg, manifest, representation)
    decision, directions = lock["decision"], vectors(run)
    conditions = {"default": (cfg["neutral_system"], None, None, 0),
                  "warm_prompt": (cfg["warm_system"], None, None, 0)}
    if decision["multiplier"]:
        conditions["selected"] = (cfg["neutral_system"], directions[str(decision["layer"])], decision["layer"], decision["multiplier"])
    for name, (system, vector, layer, strength) in conditions.items():
        save(directory / f"{name}.json", generate(model, tokenizer, rows, cfg, system, lock["cap"], vector, layer, strength))
    comparisons = [["default", "warm_prompt"]]
    if "selected" in conditions:
        comparisons += [["default", "selected"], ["warm_prompt", "selected"]]
    save(directory / "conditions.json", {"conditions": list(conditions), "comparisons": comparisons,
         "selected_alias": None if decision["multiplier"] else "default",
         "final_lock_sha256": stream_hash(path(args.lock)),
         "prediction_sha256": {n: stream_hash(directory / f"{n}.json") for n in conditions}})
    save(directory / "run.json", gpu_record(run, cfg, representation, started))


def summary(args, run):
    cfg, _ = verify_run(run)
    lock = check_lock(run, args.lock)
    rows = read_json(run / "inputs/final.json")
    results = {}
    for representation in ("bf16", "nf4"):
        directory = run / ("final-"+representation)
        content, preferences = judged(directory)
        record, result = read_json(directory / "conditions.json"), {}
        if record["final_lock_sha256"] != stream_hash(path(args.lock)):
            raise ValueError("final run did not use the fixed lock")
        conditions = {}
        for name in record["conditions"]:
            filename = directory / f"{name}.json"
            if stream_hash(filename) != record["prediction_sha256"][name]:
                raise ValueError("final generation changed after judging")
            conditions[name] = read_json(filename)["predictions"]
            result[name] = measurements(rows, conditions[name], content[name])
        if "selected" in conditions:
            ratios = [token_length(b)/max(1, token_length(a)) for a, b in zip(conditions["default"], conditions["selected"])]
            result["descriptive_selected_gate"] = gates(result["default"], result["selected"], ratios)
        else:
            result["selected_alias"] = "default; identical output reused, not generated or judged twice"
        result["paired_warmth"] = {name: grouped_interval(rows, scores, cfg["bootstrap_samples"], cfg["judging_seed"])
                                    for name, scores in preferences.items()}
        result["run"] = read_json(directory / "run.json")
        result["rating_sha256"] = {k: stream_hash(directory / f"judging/{k}_ratings.json") for k in ("content", "warmth")}
        results[representation] = result
    save(path(args.output), {"completed": True, "selected": lock["decision"], "selection_fixed": True,
        "final_lock_sha256": stream_hash(path(args.lock)), "results": results,
        "limits": ["Synthetic everyday tasks, AI judgments, 12 independent final families; no human or broad capability claim.",
                   "Literal checks and AI assessments can disagree; content and warmth are measured separately.",
                   "Final gates are descriptive; never trigger reselection. NF4 uses unchanged BF16 direction and settings."]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate-fixture", "prepare", "extraction-generate", "packets", "import-ratings",
                       "extract", "preflight", "development", "select", "lock-final", "final", "summary"))
    parser.add_argument("--run", default="outputs/tone-01")
    parser.add_argument("--config", default="configs/tone_steering.json")
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--registration", help="committed prepare manifest hash + passed advisor semantic audit")
    parser.add_argument("--budget", help="committed exact preflight.json copy")
    parser.add_argument("--packet-stage", dest="packet_stage", choices=("extraction", "development", "final-bf16", "final-nf4"))
    parser.add_argument("--kind", choices=("content", "warmth"))
    parser.add_argument("--ratings")
    parser.add_argument("--lock", help="committed final lock")
    parser.add_argument("--representation", choices=("bf16", "nf4"), default="bf16")
    parser.add_argument("--output", default="outputs/tone-01/summary.json")
    args = parser.parse_args()
    run = path(args.run)
    if args.stage == "validate-fixture":
        cfg = read_json(args.config)
        print(json.dumps(validate({s: read_json(path(cfg["fixture_directory"]) / f"{s}.json")
                                   for s in ("extraction", "development", "final")}), indent=2))
    elif args.stage == "prepare":
        prepare(args)
    elif args.stage in ("packets", "import-ratings"):
        action = args.stage
        if not args.packet_stage or (action == "import-ratings" and (not args.kind or not args.ratings)):
            parser.error("packet stage, and for imports kind/ratings, are required")
        args.stage = args.packet_stage
        (packets if action == "packets" else import_ratings)(args, run)
    elif args.stage == "lock-final":
        verify_run(run)
        save(path(args.output), final_lock(run))
    else:
        {"extraction-generate": extraction_generate, "extract": extract, "preflight": preflight,
         "development": development, "select": select_development, "final": final_generate, "summary": summary}[args.stage](args, run)


if __name__ == "__main__":
    main()
