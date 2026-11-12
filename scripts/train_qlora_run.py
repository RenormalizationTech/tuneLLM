"""Prepare immutable inputs/protocol, then explicitly execute the committed modest QLoRA run.

Prepare (CPU): python -m scripts.train_qlora_run --output outputs/qlora-real-01
Copy protocol.json to a tracked path and commit it, then:
python -m scripts.train_qlora_run --execute outputs/qlora-real-01 --protocol <committed-file>
Execution re-enters the frozen source snapshot and never recomputes subsets.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

from scripts.batched_generation import REPRESENTATION, generate_batched, load_nf4
from scripts.common import (ROOT, digest, load_config, local_snapshot, path, provenance,
                            read_json, select_gpu, setup_hf, write_json)
from scripts.run_support import (choose_checkpoint, seal_snapshot, seeded_subset,
                                 select_training_rows, snapshot_sources, verify_committed_protocol,
                                 verify_snapshot, write_jsonl)
from scripts.score_retention import score_retention
from scripts.training_data import AssistantCollator, encode_row, load_splits
from evaluation.gsm8k import score_gsm8k


def prepare(config_file, output, gpu=None):
    cfg = load_config(config_file)
    if gpu is not None:
        cfg["gpu"] = gpu
    cfg["hf_home"] = str(path(cfg["hf_home"]).resolve())
    cfg["repo_root"] = str(ROOT)
    setup_hf(cfg)
    train = cfg["training"]
    effective = train["per_device_train_batch_size"] * train["gradient_accumulation_steps"]
    if cfg["candidate_steps"] != [32, 64] or cfg["train_rows"] != 1024 or effective != 16 or train["max_steps"] != 64 or train["num_train_epochs"] != 1:
        raise ValueError("fixed protocol requires 1024 rows, effective batch16, one epoch/64 steps and candidates32/64")
    if train["eval_steps"] != 16 or train["save_steps"] != 32 or train["load_best_model_at_end"]:
        raise ValueError("evaluate every16; save32/64; select only those checkpoints after training")
    if cfg["validation_rows"] != 256 or cfg["final_test_rows"] != 128 or cfg["generation_batch_size"] != 8 or cfg["reload_rows"] != 16:
        raise ValueError("fixed protocol requires val256, test128, batch8 and first16 reload rows")
    if not cfg.get("retention_fixture"):
        raise ValueError("final retention fixture required")
    source_provenance = read_json(path(cfg["splits"]["train"]).parent / "provenance.json")
    if source_provenance["source"]["revision"] != cfg["dataset_revision"]:
        raise ValueError("unexpected GSM8K revision")
    raw_hashes = {name: digest(filename) for name, filename in cfg["splits"].items()}
    for name, filename in cfg["splits"].items():
        if raw_hashes[name] != source_provenance["output_sha256"][path(filename).name]:
            raise ValueError(f"raw split checksum mismatch: {name}")
    all_rows = load_splits(cfg["splits"])
    snapshot = local_snapshot(cfg)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    training_rows, filtering = select_training_rows(all_rows["train"], tokenizer, cfg["max_length"], cfg["train_rows"], cfg["seed"])
    validation = seeded_subset(all_rows["validation"], cfg["validation_rows"], cfg["seed"])
    for row in validation:
        encode_row(row, tokenizer, cfg["max_length"])  # Fail on overlength held-out rows; never filter.
    smoke = read_json(cfg["smoke_results"])
    excluded = smoke["selected_ids"]["test"]
    if len(set(excluded)) != 16 or not set(excluded).issubset({r["id"] for r in all_rows["test"]}):
        raise ValueError("expected exactly sixteen prior smoke test IDs in original test")
    final_test = seeded_subset(all_rows["test"], cfg["final_test_rows"], cfg["seed"], excluded)
    retention = read_json(cfg["retention_fixture"])
    if len(retention["tasks"]) != 128:
        raise ValueError("128 retention rows required")
    score_retention(retention["tasks"], [{"id": r["id"], "output": "", "hit_token_limit": False} for r in retention["tasks"]])
    # Validate prompt budgets only; never generate or compute test loss in preparation.
    from scripts.batched_generation import render_prompt
    context_limit = read_json(snapshot / "config.json")["max_position_embeddings"]
    for rows, budget in ((final_test, cfg["eval_max_new_tokens"]), (retention["tasks"], cfg["retention_max_new_tokens"])):
        for row in rows:
            if len(tokenizer(render_prompt(tokenizer, row), add_special_tokens=False)["input_ids"]) + budget > context_limit:
                raise ValueError(f"held-out prompt exceeds context budget: {row['id']}; preserve held-out rows")
    if raw_hashes != {name: digest(filename) for name, filename in cfg["splits"].items()}:
        raise ValueError("raw data changed during preparation")
    out = path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    inputs.mkdir()
    selected = {"train": training_rows, "validation": validation, "test": final_test}
    for name, rows in selected.items():
        write_jsonl(inputs / f"{name}.jsonl", rows)
    write_json(inputs / "retention.json", retention)
    write_json(inputs / "gsm8k_provenance.json", source_provenance)
    write_json(inputs / "smoke_results.json", smoke)
    tokenizer.save_pretrained(inputs / "tokenizer")
    model_files = {}
    for p in sorted(snapshot.iterdir()):
        if p.is_file():
            model_files[p.name] = {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
                                  "resolved": str(p.resolve())}
            if p.suffix != ".safetensors":
                model_files[p.name]["sha256"] = digest(p)
    write_json(inputs / "model_snapshot.json", {"path": str(snapshot), "files": model_files})
    cfg["splits"] = {name: str(inputs / f"{name}.jsonl") for name in selected}
    cfg["retention_fixture"] = str(inputs / "retention.json")
    cfg["smoke_results"] = str(inputs / "smoke_results.json")
    from transformers import TrainingArguments
    warmup_args = TrainingArguments(output_dir=str(out / "checkpoints"), use_cpu=True, report_to=[],
                                    warmup_ratio=train["warmup_ratio"], warmup_steps=train.get("warmup_steps", 0))
    cfg["resolved_warmup_steps"] = warmup_args.get_warmup_steps(train["max_steps"])
    write_json(out / "config.json", cfg)
    snapshot_sources(out / "source")
    source_hashes = {str(p.relative_to(out / "source")): digest(p) for p in sorted((out / "source").rglob("*")) if p.is_file()}
    protocol = {
        "model_id": cfg["model_id"], "revision": cfg["revision"], "representation": REPRESENTATION,
        "seed": cfg["seed"], "dataset_revision": cfg["dataset_revision"],
        "raw_data_sha256": raw_hashes, "train_filtering": filtering,
        "selected_ids": {name: [r["id"] for r in rows] for name, rows in selected.items()},
        "excluded_smoke_test_ids": excluded, "smoke_results_sha256": digest(inputs / "smoke_results.json"),
        "reload_gsm8k_ids": [r["id"] for r in final_test[:cfg["reload_rows"]]],
        "retention_ids": [r["id"] for r in retention["tasks"]],
        "retention_source": retention.get("provenance"),
        "input_sha256": {str(p.relative_to(inputs)): digest(p) for p in sorted(inputs.rglob("*")) if p.is_file()},
        "source_sha256": source_hashes, "resolved_config_sha256": digest(out / "config.json"),
        "training": cfg["training"], "lora": cfg["lora"], "effective_batch_size": effective,
        "resolved_warmup_steps": cfg["resolved_warmup_steps"],
        "checkpoint_rule": "minimum validation loss among32/64 only; ties32; no generation/test override",
        "loss_scope": "Trainer assistant/EOS loss, fixed val256, same batches; not token-weighted corpus perplexity",
        "generation": {"batch_size": 8, "seed": 42, "gsm8k_max_new_tokens": 512, "retention_max_new_tokens": 8,
                       "test_timing": "all final-test generations occur after checkpoint selection",
                       "reload": "exact PEFT state-tensor equality plus first16 GSM8K token equality (same first two batches); all128 retention tokens compared"},
        "primary_metrics": {"gsm8k": "last-number fallback accuracy", "mmlu": "first case-preserved word-boundary uppercase A-D fallback (case-insensitive whole single letter accepted)"},
        "secondary_metrics": "strict accuracy, invalid/format compliance, truncation, lengths; no result used to override checkpoint selection",
    }
    write_json(out / "protocol.json", protocol)
    frozen_hash = seal_snapshot(out)
    metadata = provenance(cfg)
    metadata.update({"status": "prepared", "phase": "awaiting_committed_protocol", "completed": False,
                     "frozen_manifest_sha256": frozen_hash, "protocol_sha256": digest(out / "protocol.json")})
    write_json(out / "run.json", metadata)
    print(f"Prepared without model allocation: {out}\nCommit an exact copy of {out / 'protocol.json'}, then execute with --protocol <committed-copy>.")


def execute(out, protocol_file):
    out, protocol_file = path(out).resolve(), path(protocol_file).resolve()
    verify_snapshot(out)
    cfg = read_json(out / "config.json")
    verify_committed_protocol(out, protocol_file, cfg["repo_root"])
    metadata = read_json(out / "run.json")
    if metadata["status"] != "prepared" or metadata["frozen_manifest_sha256"] != digest(out / "frozen.json"):
        raise ValueError("run must be unstarted with its original frozen manifest; refuses rerun")
    # Replace the live interpreter so every import and source hash comes from the snapshot.
    if ROOT.resolve() != (out / "source").resolve():
        os.chdir(out / "source")
        os.execv(sys.executable, [sys.executable, "-m", "scripts.train_qlora_run", "--execute", str(out), "--protocol", str(protocol_file)])
    select_gpu(cfg["gpu"])
    setup_hf(cfg)

    def phase(name, status="running"):
        metadata.update({"phase": name, "status": status})
        write_json(out / "run.json", metadata)

    try:
        metadata["execution_provenance"] = provenance(cfg)
        metadata["committed_protocol_path"] = str(protocol_file)
        phase("verifying_frozen_model_and_inputs")
        model_snapshot = read_json(out / "inputs/model_snapshot.json")
        snapshot = Path(model_snapshot["path"])
        for name, expected in model_snapshot["files"].items():
            p = snapshot / name
            if p.stat().st_size != expected["size"] or p.stat().st_mtime_ns != expected["mtime_ns"] or str(p.resolve()) != expected["resolved"] or ("sha256" in expected and digest(p) != expected["sha256"]):
                raise ValueError(f"cached model snapshot changed: {name}")
        from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
        from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
        import torch
        set_seed(cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(out / "inputs/tokenizer", local_files_only=True, trust_remote_code=False)
        tokenizer.padding_side = "right"
        # Read only training/validation before selection. Final test is untouched until selected.
        def read_rows(name):
            return [json.loads(line) for line in Path(cfg["splits"][name]).read_text(encoding="utf-8").split("\n") if line.strip()]

        train_rows, validation_rows = read_rows("train"), read_rows("validation")
        encoded_train = [encode_row(row, tokenizer, cfg["max_length"]) for row in train_rows]
        encoded_validation = [encode_row(row, tokenizer, cfg["max_length"]) for row in validation_rows]
        phase("loading_training_model")
        model = get_peft_model(load_nf4(snapshot, training=True), LoraConfig(task_type="CAUSAL_LM", bias="none", **cfg["lora"]))
        model.peft_config["default"].base_model_name_or_path = cfg["model_id"]
        model.peft_config["default"].revision = cfg["revision"]
        if any(p.requires_grad and "lora_" not in name for name, p in model.named_parameters()):
            raise RuntimeError("unexpected trainable non-LoRA parameter")
        metadata["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        class Progress(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                metadata["last_validation"] = {"step": state.global_step, "metrics": metrics}
                write_json(out / "run.json", metadata)
                write_json(out / "validation_history.json", state.log_history)

        training_args = TrainingArguments(output_dir=str(out / "checkpoints"), bf16=True, fp16=False, tf32=False,
                                          seed=cfg["seed"], data_seed=cfg["seed"], report_to=[], push_to_hub=False,
                                          gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                                          optim="adamw_torch", label_names=["labels"], prediction_loss_only=True,
                                          **cfg["training"])
        metadata["resolved_warmup_steps"] = training_args.get_warmup_steps(cfg["training"]["max_steps"])
        if metadata["resolved_warmup_steps"] != cfg["resolved_warmup_steps"]:
            raise RuntimeError("warmup schedule differs from the frozen protocol")
        trainer = Trainer(model=model, args=training_args, train_dataset=encoded_train, eval_dataset=encoded_validation,
                          data_collator=AssistantCollator(tokenizer.pad_token_id), processing_class=tokenizer,
                          callbacks=[Progress()])
        phase("baseline_validation_loss")
        with model.disable_adapter():
            write_json(out / "baseline_validation.json", trainer.evaluate())
        phase("training")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = trainer.train()
        torch.cuda.synchronize()
        metadata.update({"training_metrics": result.metrics, "training_wall_seconds": time.perf_counter() - started,
                         "peak_training_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "global_step": trainer.state.global_step, "epoch": trainer.state.epoch})
        phase("training_complete")
        write_json(out / "validation_history.json", trainer.state.log_history)
        selection = choose_checkpoint(trainer.state.log_history, tuple(cfg["candidate_steps"]))
        checkpoint = out / "checkpoints" / f"checkpoint-{selection['selected_step']}"
        if not (checkpoint / "adapter_model.safetensors").is_file() or trainer.state.global_step != 64:
            raise RuntimeError("missing selected adapter or incomplete 64-step training")
        selection["adapter_sha256"] = digest(checkpoint / "adapter_model.safetensors")
        write_json(out / "selection.json", selection)
        metadata["selection"] = selection
        phase("checkpoint_selected")
        trainer.accelerator.free_memory()
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        # Only now read the final test/reference labels and generate any test output.
        test_rows = read_rows("test")
        retention_rows = read_json(cfg["retention_fixture"])["tasks"]

        def final_evaluation(current_model, label, gsm_rows):
            phase(label)
            gsm = generate_batched(current_model, tokenizer, gsm_rows, batch_size=8,
                                   max_new_tokens=512, seed=42, phase=label + "/gsm8k")
            gsm["scores"] = score_gsm8k(gsm_rows, gsm["predictions"])
            write_json(out / f"{label}_gsm8k.json", gsm)
            retention = generate_batched(current_model, tokenizer, retention_rows, batch_size=8,
                                         max_new_tokens=8, seed=42, phase=label + "/retention")
            retention["scores"] = score_retention(retention_rows, retention["predictions"])
            write_json(out / f"{label}_retention.json", retention)
            return gsm, retention

        phase("loading_final_base")
        base = load_nf4(snapshot)
        base_gsm, base_retention = final_evaluation(base, "base_final", test_rows)
        model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False, local_files_only=True)
        del base
        adapter = out / "adapter"
        model.save_pretrained(adapter, safe_serialization=True)
        tokenizer.save_pretrained(adapter)
        phase("selected_adapter_saved")
        after_gsm, after_retention = final_evaluation(model, "adapter_final", test_rows)
        if [b["ids"] for b in base_gsm["batches"]] != [b["ids"] for b in after_gsm["batches"]] or [b["ids"] for b in base_retention["batches"]] != [b["ids"] for b in after_retention["batches"]]:
            raise RuntimeError("base/adapter batch membership drift")
        expected_state = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}
        del model
        gc.collect()
        torch.cuda.empty_cache()
        phase("reloading_adapter")
        reloaded = PeftModel.from_pretrained(load_nf4(snapshot), adapter, is_trainable=False, local_files_only=True)
        actual_state = get_peft_model_state_dict(reloaded)
        state_equal = set(expected_state) == set(actual_state) and all(
            expected_state[key].dtype == actual_state[key].dtype and torch.equal(expected_state[key], actual_state[key].detach().cpu())
            for key in expected_state)
        write_json(out / "reload_state.json", {"exact_state_tensors_equal": state_equal, "tensor_keys": sorted(expected_state)})
        if not state_equal:
            raise RuntimeError("adapter state tensors changed on fresh-base reload")
        reloaded_gsm, reloaded_retention = final_evaluation(reloaded, "reload", test_rows[:cfg["reload_rows"]])
        gsm_equal = [r["generated_ids"] for r in after_gsm["predictions"][:cfg["reload_rows"]]] == [r["generated_ids"] for r in reloaded_gsm["predictions"]]
        retention_equal = [r["generated_ids"] for r in after_retention["predictions"]] == [r["generated_ids"] for r in reloaded_retention["predictions"]]
        if [b["ids"] for b in after_gsm["batches"][:2]] != [b["ids"] for b in reloaded_gsm["batches"]] or [b["ids"] for b in after_retention["batches"]] != [b["ids"] for b in reloaded_retention["batches"]]:
            raise RuntimeError("reload batch membership drift")
        metadata["reload"] = {"exact_state_tensors_equal": state_equal, "gsm8k_first16_tokens_equal": gsm_equal,
                              "retention_all128_tokens_equal": retention_equal}
        metadata["completed"] = gsm_equal and retention_equal
        if not metadata["completed"]:
            raise RuntimeError("generated tokens changed after adapter reload")
        phase("complete", "complete")
        write_json(adapter / "provenance.json", metadata)
    except BaseException as exc:
        metadata.update({"completed": False, "error_type": type(exc).__name__, "error": str(exc),
                         "failed_phase": metadata.get("phase")})
        phase("failed", "failed")
        raise
    print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", help="prepare only: create a new immutable run directory, no model/GPU work")
    mode.add_argument("--execute", help="execute an already prepared run using its frozen source and inputs")
    parser.add_argument("--config", default="configs/qlora_run.json")
    parser.add_argument("--gpu", help="preparation-time override only")
    parser.add_argument("--protocol", help="exact committed copy of prepared protocol.json, required for execution")
    args = parser.parse_args()
    if args.execute:
        if not args.protocol or args.gpu:
            parser.error("execution requires --protocol and forbids GPU/config drift; GPU is frozen at preparation")
        execute(args.execute, args.protocol)
    else:
        prepare(args.config, args.output, args.gpu)


if __name__ == "__main__":
    main()
