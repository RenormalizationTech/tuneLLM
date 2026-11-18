"""Fixed 27B retention: CPU prepare/score, GPU only with --execute-gpu.

Reuses benchmark_screen.choice_token_ids/ifeval_scores and evaluation.gsm8k.
No historical predictions are consumed. Native raw MMLU likelihood is distinct
from nonthinking chat generation. The retention-only reservation ledger emits
resource_usage.json for the parent's overall science-budget ledger; its budget
must fit inside, not extend, that parent budget. Never launches automatically.
"""
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VERSION = "qwen38-retention-v1"
COUNTS = {"general_mc": 128, "gsm8k": 128, "ifeval": 32, "physics_mc": 96}
CAPS = {"gsm8k": 512, "ifeval": 1024}
CHOICES = [" A", " B", " C", " D"]
GATES = {"general_mc": 5, "gsm8k": 5, "ifeval": 3}
SOURCES = ["scripts/qwen38_retention.py", "scripts/qwen38_science_runtime.py",
           "scripts/qwen38_nf4_smoke.py", "scripts/qwen38_eval_adapter.py", "scripts/qwen38_eval_science.py",
           "scripts/benchmark_screen.py", "scripts/batched_generation.py", "scripts/common.py",
           "scripts/gguf_eval.py", "evaluation/gsm8k.py"]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")


def write_status(path, value):
    Path(path).write_text(canonical(value) + "\n", encoding="utf-8")


def matched(rows, predictions, id_key="id"):
    expected = [str(r[id_key]) for r in rows]
    actual = [str(r["id"]) for r in predictions]
    if not expected or len(set(expected)) != len(expected) or len(set(actual)) != len(actual) or set(expected) != set(actual):
        raise ValueError("prediction IDs/counts must exactly match unique fixture IDs")
    return {str(r["id"]): r for r in predictions}


def validate_fixture(fixture, include_physics=False, expected_counts=None):
    from evaluation.gsm8k import FINAL
    counts = COUNTS if expected_counts is None else expected_counts
    tasks = ["general_mc", "gsm8k", "ifeval"] + (["physics_mc"] if include_physics else [])
    seen = set()
    for task in tasks:
        rows = fixture.get(task)
        if not isinstance(rows, list) or len(rows) != counts[task]:
            raise ValueError(f"{task}: exact registered fixture count required")
        for row in rows:
            key = row.get("key") if task == "ifeval" else row.get("id")
            if isinstance(key, bool) or not isinstance(key, (str, int)) or not str(key).strip() or str(key) in seen:
                raise ValueError("nonempty globally unique fixture IDs required")
            seen.add(str(key))
            if task.endswith("_mc"):
                if (not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
                        or not isinstance(row.get("subject"), str) or not row["subject"].strip()
                        or row.get("choices") != CHOICES or type(row.get("answer_index")) is not int
                        or row["answer_index"] not in range(4)):
                    raise ValueError("malformed four-choice likelihood item")
            elif task == "gsm8k":
                if (not isinstance(row.get("user"), str) or not row["user"].strip()
                        or not isinstance(row.get("assistant"), str) or not row["assistant"].strip()
                        or not FINAL.search(row["assistant"].strip().splitlines()[-1])):
                    raise ValueError("malformed GSM8K question/reference final marker")
            else:
                ids, kwargs = row.get("instruction_id_list"), row.get("kwargs")
                if (not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
                        or not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids)
                        or not isinstance(kwargs, list) or len(kwargs) != len(ids) or any(not isinstance(k, dict) for k in kwargs)):
                    raise ValueError("malformed IFEval instructions/kwargs")
    return tasks


def label_likelihood(log_scores):
    """Same argmax on logits or full-vocabulary log probabilities; first wins ties."""
    if not isinstance(log_scores, list) or len(log_scores) != 4 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in log_scores):
        raise ValueError("four finite answer-label scores required")
    return max(range(4), key=lambda i: log_scores[i])


def score_mc(rows, predictions):
    by_id = matched(rows, predictions)
    details = []
    for row in rows:
        p = by_id[str(row["id"])]
        index = label_likelihood(p["choice_loglikelihood"])
        if "predicted_index" in p and (type(p["predicted_index"]) is not int or p["predicted_index"] != index):
            raise ValueError("stored prediction disagrees with label likelihoods")
        if "answer_index" in p and p["answer_index"] != row["answer_index"]:
            raise ValueError("prediction reference label differs from fixed fixture")
        details.append({"id": str(row["id"]), "predicted_index": index, "correct": index == row["answer_index"]})
    return {"total": len(details), "correct": sum(r["correct"] for r in details), "details": details}


def verify_ifeval_assets():
    provenance_path = ROOT / "data/benchmark-screen-01/provenance.json"
    provenance = json.loads(provenance_path.read_text())
    files = {str(provenance_path): sha(provenance_path)}
    for name, expected in provenance["ifeval_source_sha256"].items():
        path = ROOT / "artifacts/tools/ifeval" / name
        if sha(path) != expected:
            raise ValueError(f"pinned official IFEval scorer changed: {name}")
        files[str(path)] = expected
    source = json.loads((ROOT / "artifacts/tools/ifeval/source.json").read_text())
    if source != provenance["ifeval_scorer"]:
        raise ValueError("IFEval official revision differs")
    resource_path = ROOT / "results/physics_nltk_resource.json"
    resource = json.loads(resource_path.read_text())
    files[str(resource_path)] = sha(resource_path)
    for name, expected in resource["files"].items():
        path = ROOT / "artifacts/tools/nltk" / name
        if sha(path) != expected:
            raise ValueError(f"pinned NLTK resource changed: {name}")
        files[str(path)] = expected
    return {"revision": source, "files": files, "nltk_version": resource["nltk_version"]}


def scorer_worker(payload):
    """CPU subprocess entry: official strict scoring only, no model libraries."""
    verify_ifeval_assets()
    random.seed(42)
    from langdetect import DetectorFactory
    DetectorFactory.seed = 42
    from scripts.benchmark_screen import ifeval_scores
    scores = ifeval_scores(payload["rows"], payload["predictions"])
    packages = {name: importlib.metadata.version(name) for name in ("nltk", "langdetect", "absl-py", "immutabledict")}
    if packages["nltk"] != "3.9.2":
        raise ValueError("pinned NLTK3.9.2 required")
    return {"scores": scores, "packages": packages, "python": platform.python_version(), "seed": 42}


def run_ifeval(rows, predictions, python=ROOT / ".venv/bin/python"):
    matched(rows, predictions, "key")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([str(Path(python).absolute()), "-B", "-m", "scripts.qwen38_retention", "--ifeval-worker"],
        input=canonical({"rows": rows, "predictions": predictions}), text=True, capture_output=True,
        cwd=ROOT, env=env, timeout=300, check=True)
    return json.loads(result.stdout)


def adapter_identity(adapter, task, base, base_manifest):
    from scripts.qwen38_eval_adapter import validate
    adapter = Path(adapter).resolve()
    audit = validate(adapter, task, "base")  # parent's training-time hashes/selection helper
    training = adapter.parents[1]
    cfg = json.loads((training / "config.json").read_text())
    saved = json.loads((adapter / "adapter_config.json").read_text())
    expected = {"r": 16, "lora_alpha": 32, "lora_dropout": .05, "task_type": "CAUSAL_LM", "bias": "none"}
    if any(saved.get(k) != v for k, v in expected.items()) or saved.get("base_model_name_or_path") != str(Path(base).resolve()):
        raise ValueError("adapter configuration/base differs from shared retention runtime")
    for key in ("use_rslora", "use_dora", "fan_in_fan_out", "lora_bias", "rank_pattern", "alpha_pattern", "modules_to_save", "layer_replication", "target_parameters"):
        if saved.get(key):
            raise ValueError(f"unsupported nondefault saved adapter setting: {key}")
    if cfg.get("task") != task or cfg.get("nf4_artifact_manifest_sha256") != sha(base_manifest):
        raise ValueError("training task/base manifest differs")
    return {**audit, "training_config_sha256": sha(training / "config.json"), "saved_config": saved}


def selected_checkpoint(training):
    result = json.loads((Path(training) / "result.json").read_text())
    if not result.get("completed"):
        raise ValueError("training must complete before preparing selected-adapter retention")
    return str(Path(result["selected"]["path"]).resolve())


def batch_plan(prepared, batch_size, pad_id, eos_id):
    """Consecutive source-order groups within each task; MC is always singleton.

    This plan is arm-independent, including padding and original token lengths.
    No sorting by length or generated outputs can affect group membership.
    """
    if type(batch_size) is not int or batch_size not in (1, 4):
        raise ValueError("retention generation batch size must be 1 or 4")
    if type(pad_id) is not int or type(eos_id) is not int or min(pad_id, eos_id) < 0 or pad_id == eos_id:
        raise ValueError("single integer EOS and distinct PAD required")
    if not prepared or len({r["id"] for r in prepared}) != len(prepared):
        raise ValueError("nonempty unique prepared IDs required")
    groups, index = [], 0
    while index < len(prepared):
        first = prepared[index]
        if first["task"] not in COUNTS:
            raise ValueError("unknown batch task")
        count = 1 if first["task"].endswith("_mc") else batch_size
        stop = index + 1
        while stop < min(index + count, len(prepared)) and prepared[stop]["task"] == first["task"]:
            stop += 1
        members = prepared[index:stop]
        cap = 0 if first["task"].endswith("_mc") else CAPS[first["task"]]
        if any(not r["input_ids"] or r["max_new_tokens"] != cap or any(type(t) is not int or t < 0 for t in r["input_ids"]) for r in members):
            raise ValueError("invalid prepared prompt/cap")
        lengths = [len(r["input_ids"]) for r in members]
        width = max(lengths)
        if width + cap > 8192:
            raise ValueError("padded prompt plus allowance exceeds8192")
        groups.append({"batch_index": len(groups), "group_id": f"{first['task']}:{len(groups):04d}",
            "task": first["task"], "ids": [r["id"] for r in members], "prompt_lengths": lengths,
            "padding_lengths": [width - n for n in lengths], "padded_prompt_width": width,
            "max_new_tokens": cap})
        index = stop
    return groups


def validate_batch_binding(binding, prepared):
    """Old registrations imply batch1; new registrations bind the complete plan."""
    if digest(prepared) != binding["prepared_sha256"]:
        raise ValueError("registered prepared token/prompt digest differs")
    protocol = binding["protocol"]
    if "batches" not in binding:
        if protocol["batch_size"] != 1:
            raise ValueError("historical registration only supports batch1")
        return
    expected = batch_plan(prepared, protocol["batch_size"], protocol["pad_token_id"], protocol["eos_token_id"])
    if (binding["batches"] != expected or protocol.get("mc_batch_size") != 1
            or protocol.get("batch_policy") != "source-order contiguous per task; MC singleton; manual left padding"):
        raise ValueError("registered batch groups/padding/semantics differ")


def generation_predictions(members, sequences, plan, eos_id, pad_id):
    """Trim batch filler after first EOS, retaining PAD emitted before EOS."""
    from scripts.batched_generation import left_pad, generated_suffix
    padded = left_pad([r["input_ids"] for r in members], pad_id)
    width = plan["padded_prompt_width"]
    if (len(sequences) != len(members) or [r["id"] for r in members] != plan["ids"]
            or width != len(padded["input_ids"][0])):
        raise ValueError("generation batch shape/group differs")
    predictions = []
    for position, (row, sequence) in enumerate(zip(members, sequences)):
        if list(sequence[:width]) != padded["input_ids"][position]:
            raise ValueError("generated sequence has wrong left-padded prefix")
        tokens = generated_suffix(sequence, width, eos_id)
        if not tokens or len(tokens) > plan["max_new_tokens"]:
            raise ValueError("generated suffix empty or exceeds cap")
        ended = tokens[-1] == eos_id
        predictions.append({"id": row["id"], "task": row["task"], "input_ids": row["input_ids"],
            "rendered_prompt": row["prompt"], "generated_ids": tokens, "ended_with_eos": ended,
            "hit_token_limit": len(tokens) == plan["max_new_tokens"] and not ended,
            "stop_reason": "eos" if ended else "token_limit" if len(tokens) == plan["max_new_tokens"] else "stopping_criteria",
            "batch_index": plan["batch_index"], "group_id": plan["group_id"], "batch_ids": plan["ids"],
            "batch_position": position, "padded_prompt_width": width,
            "padding_length": plan["padding_lengths"][position]})
    return predictions


def generate_group(model, tokenizer, members, plan, stopping_criteria=None):
    """Shared GPU mechanics for retention and the separate synthetic preflight."""
    import torch
    from scripts.batched_generation import left_pad
    pad, eos = tokenizer.pad_token_id, tokenizer.eos_token_id
    if type(pad) is not int or type(eos) is not int or pad == eos:
        raise ValueError("single integer EOS and distinct PAD required")
    inputs = {k: torch.tensor(v, dtype=torch.long, device="cuda:0")
              for k, v in left_pad([r["input_ids"] for r in members], pad).items()}
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    tick = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sequences = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=plan["max_new_tokens"],
            use_cache=True, logits_to_keep=1, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
            eos_token_id=eos, pad_token_id=pad, stopping_criteria=stopping_criteria,
            return_dict_in_generate=False)
    torch.cuda.synchronize()
    timing = {"batch_seconds": time.monotonic() - tick, "timing_scope": "shared synchronized batch generation; not exclusive per-item time",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    predictions = generation_predictions(members, sequences.tolist(), plan, eos, pad)
    for p in predictions:
        p.update(timing, output=tokenizer.decode(p["generated_ids"], skip_special_tokens=True))
        if len(members) == 1:
            p["seconds"] = timing["batch_seconds"]
    return predictions, {**plan, **timing, "generated_tokens": sum(len(p["generated_ids"]) for p in predictions)}


def validate_prediction_groups(binding, predictions, batches):
    """Require the saved batch4 execution to follow the CPU-frozen group plan."""
    expected = binding["batches"]
    if len(batches) != len(expected) or len(predictions) != sum(len(p["ids"]) for p in expected):
        raise ValueError("executed batch/prediction counts differ")
    cursor = 0
    for plan, actual in zip(expected, batches):
        if any(actual.get(k) != v for k, v in plan.items()):
            raise ValueError("executed batch groups/padding differ")
        members = predictions[cursor:cursor + len(plan["ids"])]
        if [r["id"] for r in members] != plan["ids"]:
            raise ValueError("prediction order/group differs")
        if not plan["task"].endswith("_mc"):
            for index, row in enumerate(members):
                if (row.get("group_id") != plan["group_id"] or row.get("batch_ids") != plan["ids"]
                        or row.get("padded_prompt_width") != plan["padded_prompt_width"]
                        or row.get("padding_length") != plan["padding_lengths"][index]
                        or row.get("batch_seconds") != actual.get("batch_seconds")
                        or row.get("timing_scope") != "shared synchronized batch generation; not exclusive per-item time"
                        or len(members) > 1 and "seconds" in row):
                    raise ValueError("prediction batch metadata/timing differs")
        cursor += len(members)


def reload_selected(model, adapter, audit):
    """Same exact saved-state equality checks as parent's shared evaluator."""
    import torch
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    current = model.peft_config["default"]
    saved = audit["saved_config"]
    if (current.r != saved["r"] or current.lora_alpha != saved["lora_alpha"]
            or current.lora_dropout != saved["lora_dropout"]
            or set(current.target_modules) != set(saved["target_modules"])):
        raise ValueError("loaded adapter wrapper differs from saved configuration")
    state = load_file(Path(adapter) / "adapter_model.safetensors")
    info = set_peft_model_state_dict(model, state, adapter_name="default")
    if info.unexpected_keys or any("lora_" in k for k in info.missing_keys):
        raise ValueError("incomplete/mismatched adapter reload")
    actual = get_peft_model_state_dict(model)
    if set(actual) != set(state) or any(actual[k].dtype != state[k].dtype or not torch.equal(actual[k].detach().cpu(), state[k]) for k in state):
        raise ValueError("selected adapter tensor reload equality failed")
    return {"exact_adapter_tensors": True, "tensor_count": len(state)}


def prepare(args, tokenizer=None):
    from scripts.benchmark_screen import choice_token_ids
    from scripts.qwen38_science_runtime import versions
    out, fixture_path = Path(args.output).resolve(), Path(args.fixture).resolve()
    if out.exists():
        raise FileExistsError(out)
    if not args.gpu.isdecimal() or not 0 < args.max_seconds <= 14400 or not math.isfinite(args.ledger_total_hours) or args.ledger_total_hours <= 0:
        raise ValueError("one GPU, <=4 GPU-hour arm cap and positive retention ledger budget required")
    if (args.arm == "base") != (args.adapter is None):
        raise ValueError("base has no adapter; selected task arms require their adapter")
    batch_size = getattr(args, "batch_size", 1)
    if type(batch_size) is not int or batch_size not in (1, 4):
        raise ValueError("retention generation batch size must be 1 or 4")
    fixture = json.loads(fixture_path.read_text())
    tasks = validate_fixture(fixture, args.physics)
    provenance = json.loads((ROOT / "data/benchmark-screen-01/provenance.json").read_text())
    if sha(fixture_path) != provenance["fixture_sha256"]["test"]:
        raise ValueError("fixed historical retention fixture hash differs")
    assets = verify_ifeval_assets()
    scorer = Path(args.scorer_python).absolute()
    # Blank responses exercise official input schemas, not model evaluation.
    scorer_validation = run_ifeval(fixture["ifeval"], [{"id": str(r["key"]), "output": ""} for r in fixture["ifeval"]], scorer)
    base, base_manifest = Path(args.base).resolve(), ROOT / "results/qwen38_27b_nf4_artifact_01.json"
    adapter = Path(args.adapter).resolve() if args.adapter else None
    audit = adapter_identity(adapter, args.arm, base, base_manifest) if adapter else None
    if tokenizer is None:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    if type(tokenizer.pad_token_id) is not int or type(tokenizer.eos_token_id) is not int or tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("native PAD distinct from EOS required")
    choice_ids = choice_token_ids(tokenizer, fixture["general_mc"] + (fixture["physics_mc"] if args.physics else []))
    prepared = []
    for task in tasks:
        for row in fixture[task]:
            if task.endswith("_mc"):
                prompt, cap, key = row["prompt"], 0, row["id"]
            else:
                user, key = (row["user"], row["id"]) if task == "gsm8k" else (row["prompt"], str(row["key"]))
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                    return_dict=False, add_generation_prompt=True, enable_thinking=False)
                cap = CAPS[task]
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if not ids or len(ids) + cap > 8192:
                raise ValueError(f"{key}: full prompt/output allowance exceeds8192; no truncation")
            prepared.append({"task": task, "id": str(key), "prompt": prompt, "input_ids": ids, "max_new_tokens": cap})
    source_hashes = {name: sha(ROOT / name) for name in SOURCES}
    tokenizer_hashes = {str(p): sha(p) for p in base.iterdir() if p.is_file() and (p.name.startswith("tokenizer") or p.name == "chat_template.jinja")}
    binding = {"fixture_sha256": sha(fixture_path), "tasks": tasks, "choice_ids": choice_ids,
        "prepared_sha256": digest(prepared), "source_sha256": source_hashes, "tokenizer_sha256": tokenizer_hashes,
        "batches": batch_plan(prepared, batch_size, tokenizer.pad_token_id, tokenizer.eos_token_id),
        "ifeval_assets": assets, "scorer_packages": scorer_validation["packages"],
        "versions": versions(), "base_manifest_sha256": sha(base_manifest),
        "protocol": {"batch_size": batch_size, "mc_batch_size": 1,
            "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
            "batch_policy": "source-order contiguous per task; MC singleton; manual left padding",
            "dtype": "bfloat16 autocast", "thinking": False, "seed": 42,
            "mc_prompt": "native raw fixture, full-vocabulary log probabilities of single answer letters",
            "mc_use_cache": False, "generation_use_cache": True, "generation_caps": CAPS,
            "gsm8k_gate": "fallback_correct (last-number numeric); strict_correct also reported", "gates": GATES,
            "base_preparation": "shared load_runtime with disabled adapter; same casts/wrapper as selected arms"}}
    registration = {"version": VERSION, "arm": args.arm, "adapter": str(adapter) if adapter else None,
        "adapter_audit": audit, "gpu": args.gpu, "base": str(base), "base_manifest": str(base_manifest),
        "fixture_path": str(fixture_path), "scorer_python": str(scorer), "scorer_executable_sha256": sha(scorer),
        "scorer_python_version": scorer_validation["python"], "max_seconds": args.max_seconds,
        "ledger": str(Path(args.ledger).resolve()), "ledger_total_hours": args.ledger_total_hours,
        "ledger_scope": "retention-only; parent must debit receipts from its total study budget",
        "comparison_binding": binding, "comparison_sha256": digest(binding), "python_executable": sys.executable}
    out.mkdir(parents=True)
    write_new(out / "fixture.json", {task: fixture[task] for task in tasks})
    write_new(out / "prepared.json", prepared)
    for name in SOURCES:
        destination = out / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    registration["fixture_copy_sha256"] = sha(out / "fixture.json")
    registration["prepared_copy_sha256"] = sha(out / "prepared.json")
    write_new(out / "registration.json", registration)
    return {"prepared": str(out), "counts": {t: len(fixture[t]) for t in tasks}, "comparison_sha256": digest(binding),
            "gpu_launched": False, "ifeval_validation": "pinned official scorer/kwargs validated on CPU with blank responses"}


def ledger_event(path, run_id, seconds, total_hours, finish=None):
    """Locked admission reserves worst-case arm time; unfinished runs keep reservation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not 0 < seconds <= 14400 or not math.isfinite(total_hours) or total_hours <= 0:
        raise ValueError("invalid ledger reservation")
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        text = stream.read()
        ledger = json.loads(text) if text else {"version": "qwen38-retention-ledger-v1", "total_hours": total_hours, "arms": {}}
        if ledger["version"] != "qwen38-retention-ledger-v1" or ledger["total_hours"] != total_hours:
            raise ValueError("ledger type/total cap mismatch; parent integration requires explicit retention pool")
        if finish is None:
            if run_id in ledger["arms"]:
                raise ValueError("arm already attempted/reserved; no automatic rerun")
            charged = sum(v.get("elapsed_seconds", v["reserved_seconds"]) for v in ledger["arms"].values())
            if charged + seconds > total_hours * 3600:
                raise ValueError("retention total GPU-hour budget exhausted")
            ledger["arms"][run_id] = {"reserved_seconds": seconds, "status": "reserved"}
        else:
            if run_id not in ledger["arms"] or ledger["arms"][run_id]["status"] != "reserved":
                raise ValueError("active reservation required")
            if not math.isfinite(finish["elapsed_seconds"]) or finish["elapsed_seconds"] < 0:
                raise ValueError("invalid elapsed time")
            ledger["arms"][run_id].update(finish)
        stream.seek(0); stream.truncate(); stream.write(canonical(ledger) + "\n"); stream.flush(); os.fsync(stream.fileno())
    return ledger["arms"][run_id]


def validate_run(run):
    from scripts.qwen38_science_runtime import versions
    cfg = json.loads((run / "registration.json").read_text())
    b = cfg["comparison_binding"]
    if digest(b) != cfg["comparison_sha256"] or versions() != b["versions"] or sys.executable != cfg["python_executable"]:
        raise ValueError("registered software/semantics changed")
    for name, expected in b["source_sha256"].items():
        if sha(ROOT / name) != expected or sha(run / "source" / name) != expected:
            raise ValueError(f"registered source changed: {name}")
    for name, expected in b["tokenizer_sha256"].items():
        if sha(name) != expected:
            raise ValueError("registered tokenizer changed")
    if verify_ifeval_assets() != b["ifeval_assets"] or sha(cfg["scorer_python"]) != cfg["scorer_executable_sha256"]:
        raise ValueError("registered CPU scorer changed")
    for name, key in [("fixture.json", "fixture_copy_sha256"), ("prepared.json", "prepared_copy_sha256")]:
        if sha(run / name) != cfg[key]:
            raise ValueError("prepared inputs changed")
    validate_batch_binding(b, json.loads((run / "prepared.json").read_text()))
    if sha(cfg["fixture_path"]) != b["fixture_sha256"] or sha(cfg["base_manifest"]) != b["base_manifest_sha256"]:
        raise ValueError("fixture/base identity changed")
    if cfg["adapter"] and adapter_identity(cfg["adapter"], cfg["arm"], cfg["base"], cfg["base_manifest"]) != cfg["adapter_audit"]:
        raise ValueError("selected adapter identity changed")
    return cfg


def execute(run):
    """Explicit GPU path only; this function is never called by preparation/scoring."""
    run = Path(run).resolve()
    if (run / "result.json").exists() or (run / "predictions.jsonl").exists():
        raise FileExistsError("GPU arm already attempted; preserve its outputs")
    cfg = validate_run(run)
    if not cfg["gpu"].isdecimal() or not 0 < cfg["max_seconds"] <= 14400:
        raise ValueError("registered single-GPU/time cap invalid")
    artifact = json.loads(Path(cfg["base_manifest"]).read_text())
    for name, item in artifact["files"].items():
        path = Path(cfg["base"]) / name
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError(f"NF4 artifact changed: {name}")
    run_id = str(run)
    ledger_event(cfg["ledger"], run_id, cfg["max_seconds"], cfg["ledger_total_hours"])
    result = {"completed": False, "items": 0, "registration_sha256": sha(run / "registration.json")}
    write_new(run / "result.json", result)
    os.environ.update(CUDA_VISIBLE_DEVICES=cfg["gpu"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    start = time.monotonic()
    try:
        import torch
        from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, set_seed
        from scripts.qwen38_science_runtime import load_runtime
        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.monotonic() - start >= cfg["max_seconds"]
        def check_deadline():
            if time.monotonic() - start >= cfg["max_seconds"]:
                raise TimeoutError("registered retention GPU-hour cap reached")
        set_seed(42)
        model, runtime = load_runtime(base=cfg["base"])
        if cfg["adapter"]:
            result["adapter_reload"] = reload_selected(model, cfg["adapter"], cfg["adapter_audit"])
        tokenizer = AutoTokenizer.from_pretrained(cfg["base"], local_files_only=True, trust_remote_code=False)
        prepared = json.loads((run / "prepared.json").read_text())
        protocol = cfg["comparison_binding"]["protocol"]
        if "batches" in cfg["comparison_binding"] and (tokenizer.pad_token_id != protocol["pad_token_id"] or tokenizer.eos_token_id != protocol["eos_token_id"]):
            raise ValueError("registered EOS/PAD differ from loaded tokenizer")
        plans = cfg["comparison_binding"].get("batches") or batch_plan(prepared, 1, tokenizer.pad_token_id, tokenizer.eos_token_id)
        lookup = {r["id"]: r for r in prepared}
        model.eval()
        result["runtime"] = {k: v for k, v in runtime.items() if k != "targets"}
        result["retention_batching"] = {"generation_batch_size": protocol["batch_size"], "mc_batch_size": 1,
            "groups_sha256": digest(plans), "groups": plans, "padding": "left",
            "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}
        result["device_name"] = torch.cuda.get_device_name(0)
        context = nullcontext() if cfg["adapter"] else model.disable_adapter()
        with context, torch.inference_mode():
            for plan in plans:
                check_deadline()
                members = [lookup[key] for key in plan["ids"]]
                row = members[0]
                if row["task"].endswith("_mc"):
                    batch = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda:0")
                    mask = torch.ones_like(batch)
                    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                    tick = time.monotonic()
                    prediction = {"id": row["id"], "task": row["task"], "input_ids": row["input_ids"], "rendered_prompt": row["prompt"]}
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(input_ids=batch, attention_mask=mask, logits_to_keep=1, use_cache=False).logits[0, -1].float()
                        if not bool(torch.isfinite(logits).all()):
                            raise ValueError("nonfinite label likelihood logits")
                        scores = torch.log_softmax(logits, dim=-1)[cfg["comparison_binding"]["choice_ids"]].cpu().tolist()
                        prediction.update(choice_loglikelihood=scores, predicted_index=label_likelihood(scores), hit_token_limit=False)
                    torch.cuda.synchronize()
                    prediction.update(seconds=time.monotonic() - tick, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(), batch_index=plan["batch_index"], group_id=plan["group_id"])
                    predictions = [prediction]
                    batch_record = {**plan, "batch_seconds": prediction["seconds"], "timing_scope": "exclusive singleton MC forward"}
                else:
                    predictions, batch_record = generate_group(model, tokenizer, members, plan, StoppingCriteriaList([Deadline()]))
                cap_stop = time.monotonic() - start >= cfg["max_seconds"]
                with (run / "predictions.jsonl").open("a", encoding="utf-8") as stream:
                    for prediction in predictions:
                        prediction["resource_cap_stop"] = cap_stop
                        stream.write(canonical(prediction) + "\n")
                with (run / "batches.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(canonical({**batch_record, "resource_cap_stop": cap_stop}) + "\n")
                result["items"] += len(predictions); result["elapsed_seconds"] = time.monotonic() - start
                write_status(run / "result.json", result)
                print(canonical({"done": result["items"], "total": len(prepared), "task": row["task"]}), flush=True)
                check_deadline()
        validate_run(run)
        check_deadline()
        result["completed"] = result["items"] == len(prepared)
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        if (run / "predictions.jsonl").exists():
            result["predictions_sha256"] = sha(run / "predictions.jsonl")
        if (run / "batches.jsonl").exists():
            result["batches_sha256"] = sha(run / "batches.jsonl")
        write_status(run / "result.json", result)
        receipt = {"status": "complete" if result["completed"] else "incomplete", "elapsed_seconds": result["elapsed_seconds"],
            "gpu_hours": result["elapsed_seconds"] / 3600, "gpu_count": 1, "run": run_id,
            "registration_sha256": result["registration_sha256"], "cap_seconds": cfg["max_seconds"],
            "cap_exceeded": result["elapsed_seconds"] > cfg["max_seconds"], "parent_ledger_must_debit": True}
        write_new(run / "resource_usage.json", receipt)
        ledger_event(cfg["ledger"], run_id, cfg["max_seconds"], cfg["ledger_total_hours"], receipt)
    return result


def score_predictions(fixture, predictions, ifeval_scorer):
    from evaluation.gsm8k import score_gsm8k
    tasks = list(fixture)
    if any(p.get("task") not in tasks for p in predictions):
        raise ValueError("unknown prediction task")
    report = {}
    for task in tasks:
        rows, predicted = fixture[task], [p for p in predictions if p["task"] == task]
        by_id = matched(rows, predicted, "key" if task == "ifeval" else "id")
        if task.endswith("_mc"):
            report[task] = score_mc(rows, predicted)
        else:
            if any(not isinstance(p.get("output"), str) or type(p.get("hit_token_limit")) is not bool for p in predicted):
                raise ValueError("generation output/truncation flags required")
            if task == "gsm8k":
                raw = score_gsm8k(rows, predicted)
                report[task] = {**raw, "correct": raw["fallback_correct"],
                    "details": [{**d, "correct": d["fallback_correct"]} for d in raw["details"]]}
            else:
                raw = ifeval_scorer(rows, predicted)
                records = raw["records"]
                matched(rows, records, "key")
                if any(type(r.get("prompt_strict")) is not bool for r in records):
                    raise ValueError("official strict IFEval boolean scores required")
                report[task] = {**raw, "correct": sum(r["prompt_strict"] for r in records),
                    "details": [{**r, "correct": r["prompt_strict"], "hit_token_limit": by_id[str(r["id"])]["hit_token_limit"]} for r in records]}
        report[task]["truncated"] = sum(p.get("hit_token_limit", False) for p in predicted)
    return report


def compare_scores(baseline, candidate, expected_counts=None):
    if baseline["arm"] != "base" or candidate["arm"] == "base" or baseline["comparison_sha256"] != candidate["comparison_sha256"]:
        raise ValueError("matched fixed protocol/base and selected-arm scores required")
    if set(baseline["tasks"]) != set(candidate["tasks"]):
        raise ValueError("task denominators differ")
    if not set(GATES) <= set(baseline["tasks"]):
        raise ValueError("all three retention gate tasks required")
    if set(baseline["tasks"]) - set(COUNTS):
        raise ValueError("unknown comparison task")
    expected_counts = COUNTS if expected_counts is None else expected_counts
    result = {}
    for task, base in baseline["tasks"].items():
        other = candidate["tasks"][task]
        by_id = matched(base["details"], other["details"])
        if base["total"] != other["total"] or base["total"] != len(base["details"]):
            raise ValueError("score totals differ from matched item denominator")
        if base["total"] != expected_counts[task]:
            raise ValueError("comparison denominator differs from registered task count")
        if any(type(r["correct"]) is not bool for r in base["details"] + other["details"]):
            raise ValueError("boolean per-item correctness required")
        fixed = [r["id"] for r in base["details"] if not r["correct"] and by_id[str(r["id"])]["correct"]]
        broken = [r["id"] for r in base["details"] if r["correct"] and not by_id[str(r["id"])]["correct"]]
        base_count, other_count = sum(r["correct"] for r in base["details"]), sum(r["correct"] for r in other["details"])
        if base_count != base["correct"] or other_count != other["correct"]:
            raise ValueError("aggregate correctness disagrees with item scores")
        limit = GATES.get(task)
        result[task] = {"total": base["total"], "baseline_correct": base_count, "candidate_correct": other_count,
            "fixed_ids": fixed, "broken_ids": broken, "delta_correct": other_count - base_count,
            "allowed_loss": limit, "gate_pass": other_count >= base_count - limit if limit is not None else None}
    return {"tasks": result, "retention_pass": all(r["gate_pass"] for t, r in result.items() if t in GATES),
            "interpretation": "fixed point-estimate regression limits; no non-inferiority claim; physics_mc diagnostic only"}


def score_run(run, baseline=None):
    run = Path(run).resolve()
    cfg = validate_run(run)
    result = json.loads((run / "result.json").read_text())
    if not result.get("completed") or result["registration_sha256"] != sha(run / "registration.json"):
        raise ValueError("complete registered arm required; incomplete items must not silently shrink denominators")
    if sha(run / "predictions.jsonl") != result["predictions_sha256"]:
        raise ValueError("predictions changed after execution")
    fixture = json.loads((run / "fixture.json").read_text())
    predictions = [json.loads(s) for s in (run / "predictions.jsonl").read_text().splitlines() if s.strip()]
    if cfg["comparison_binding"]["protocol"]["batch_size"] == 4 or "batches_sha256" in result:
        if sha(run / "batches.jsonl") != result["batches_sha256"]:
            raise ValueError("batch timing records changed after execution")
        batches = [json.loads(s) for s in (run / "batches.jsonl").read_text().splitlines() if s.strip()]
        validate_prediction_groups(cfg["comparison_binding"], predictions, batches)
    worker = {}
    def strict(rows, predicted):
        response = run_ifeval(rows, predicted, cfg["scorer_python"])
        if response["packages"] != cfg["comparison_binding"]["scorer_packages"]:
            raise ValueError("CPU scorer software changed")
        worker.update(response)
        return response["scores"]
    report = {"arm": cfg["arm"], "comparison_sha256": cfg["comparison_sha256"],
        "registration_sha256": sha(run / "registration.json"), "predictions_sha256": result["predictions_sha256"],
        "tasks": score_predictions(fixture, predictions, strict), "ifeval_worker": worker}
    report["batching"] = {"generation_batch_size": cfg["comparison_binding"]["protocol"]["batch_size"], "mc_batch_size": 1,
        "groups_sha256": digest(cfg["comparison_binding"].get("batches", [])),
        "timing_scope": "per-item batch_seconds shared across group; sum unique batches for arm generation time"}
    if baseline:
        baseline = Path(baseline).resolve()
        validate_run(baseline)
        base = json.loads((baseline / "scores.json").read_text())
        base_result = json.loads((baseline / "result.json").read_text())
        if (not base_result.get("completed") or base["registration_sha256"] != sha(baseline / "registration.json")
                or base_result["registration_sha256"] != base["registration_sha256"]
                or base["predictions_sha256"] != base_result["predictions_sha256"]
                or sha(baseline / "predictions.jsonl") != base["predictions_sha256"]):
            raise ValueError("baseline scoring/execution provenance differs")
        report["baseline_scores_sha256"] = sha(baseline / "scores.json")
        report["comparison"] = compare_scores(base, report)
    write_new(run / "scores.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--score", action="store_true")
    mode.add_argument("--execute-gpu", action="store_true")
    mode.add_argument("--ifeval-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--run")
    p.add_argument("--output")
    p.add_argument("--arm", choices=["base", "qasper", "prl"], default="base")
    p.add_argument("--adapter")
    p.add_argument("--training-run", help="resolve the completed run's selected checkpoint; alternative to --adapter")
    p.add_argument("--base", default=str(ROOT / "artifacts/qwen38-27b-nf4-01"))
    p.add_argument("--fixture", default=str(ROOT / "data/benchmark-screen-01/test.json"))
    p.add_argument("--physics", action="store_true")
    p.add_argument("--batch-size", type=int, choices=(1, 4), default=1,
                   help="generation only; MC stays batch1; new preparation only")
    p.add_argument("--gpu", default="0")
    p.add_argument("--max-seconds", type=int, default=14400)
    p.add_argument("--scorer-python", default=str(ROOT / ".venv/bin/python"))
    p.add_argument("--ledger")
    p.add_argument("--ledger-total-hours", type=float, default=12)
    p.add_argument("--baseline", help="completed base-arm directory for paired scoring/gates")
    a = p.parse_args()
    if a.ifeval_worker:
        result = scorer_worker(json.load(sys.stdin))
    elif a.execute_gpu or a.score:
        if not a.run:
            p.error("--run required")
        result = execute(a.run) if a.execute_gpu else score_run(a.run, a.baseline)
    else:
        if not a.output or not a.ledger:
            p.error("CPU preparation requires --output and explicit --ledger (retention pool within parent total)")
        if a.training_run:
            if a.adapter or a.arm == "base":
                p.error("--training-run requires a selected task arm and cannot accompany --adapter")
            a.adapter = selected_checkpoint(a.training_run)
        result = prepare(a)
    print(canonical(result))


if __name__ == "__main__":
    main()
