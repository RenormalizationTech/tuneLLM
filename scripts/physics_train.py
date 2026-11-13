"""Prepare a frozen paper-LM experiment; run only against the manager's committed protocol.

Prepare: python -m scripts.physics_train --output outputs/physics-r32-lr5e5 --rank 32 --learning-rate 5e-5 --epochs 3
Run: python -m scripts.physics_train --run outputs/physics-r32-lr5e5 --protocol <committed-protocol-copy>
No test loss or benchmark generation during training; checkpoint choice belongs to the registered protocol.
"""
import argparse
import gc
import json
import math
import os
import shutil
import sys
import time

from scripts.common import ROOT, load_config, local_snapshot, path, provenance, read_json, select_gpu, setup_hf, write_json
from scripts.physics_data import chunk_papers, exposure, read_papers, validate_splits
from scripts.run_support import seal_snapshot, snapshot_sources, verify_committed_protocol, verify_snapshot, write_jsonl
from scripts.save_nf4 import reload_stored_nf4, stream_hash
from scripts.training_data import AssistantCollator


def verify_nf4(cfg):
    report = read_json(cfg["nf4_report"])
    if report.get("completed") is not True or report["revision"] != cfg["revision"]:
        raise ValueError("verified pinned NF4 checkpoint required")
    for name, expected in report["files"].items():
        filename = path(cfg["nf4_checkpoint"]) / name
        if filename.stat().st_size != expected["bytes"] or stream_hash(filename) != expected["sha256"]:
            raise ValueError(f"standalone NF4 checkpoint changed: {name}")
    return report


def load_base(cfg, training):
    if cfg["representation"] == "nf4":
        return reload_stored_nf4(path(cfg["nf4_checkpoint"]), training=training)
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        local_snapshot(cfg), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    model.config.use_cache = not training
    return model


def prepare(args):
    cfg = load_config(args.config)
    cfg.update({"rank": args.rank, "learning_rate": args.learning_rate, "epochs": args.epochs,
                "seed": args.seed, "representation": args.representation, "gpu": "1", "repo_root": str(ROOT)})
    if args.sequence_length is not None:
        cfg["max_sequence_length"] = args.sequence_length
    if cfg["max_sequence_length"] not in (1024, 2048):
        raise ValueError("supported sequence lengths are1024/2048")
    for key in ("hf_home", "corpus", "nf4_checkpoint", "nf4_report"):
        cfg[key] = str(path(cfg[key]).resolve())
    setup_hf(cfg)
    nf4 = verify_nf4(cfg)
    corpus = path(cfg["corpus"])
    hashes = {name: stream_hash(corpus / f"{name}.jsonl") for name in ("train", "validation", "test")}
    papers = {name: read_papers(corpus / f"{name}.jsonl") for name in hashes}
    validate_splits(papers, cfg["expected_papers"])
    from transformers import AutoTokenizer, TrainingArguments
    tokenizer = AutoTokenizer.from_pretrained(path(cfg["nf4_checkpoint"]), local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("existing PAD distinct from EOS required")
    chunks = {name: chunk_papers(papers[name], tokenizer, cfg["max_sequence_length"]) for name in ("train", "validation")}
    counts = {name: exposure(rows, cfg["max_sequence_length"]) for name, rows in chunks.items()}
    if counts["train"]["loss_tokens_per_epoch"] > cfg["max_train_tokens_per_epoch"]:
        raise ValueError("train corpus exceeds registered per-epoch token bound; do not silently truncate")
    microbatches = math.ceil(len(chunks["train"]) / cfg["per_device_train_batch_size"])
    cfg["steps_per_epoch"] = math.ceil(microbatches / cfg["gradient_accumulation_steps"])
    cfg["max_training_steps"] = cfg["epochs"] * cfg["steps_per_epoch"]
    cfg["expected_loss_tokens"] = counts["train"]["loss_tokens_per_epoch"] * cfg["epochs"]
    cfg["expected_microbatches"] = microbatches * cfg["epochs"]
    cfg["effective_batch_size"] = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
    cfg["warmup_steps"] = TrainingArguments(output_dir="/tmp/tunellm-physics-preflight", use_cpu=True,
                                           report_to=[], warmup_ratio=cfg["warmup_ratio"]).get_warmup_steps(cfg["max_training_steps"])
    if hashes != {name: stream_hash(corpus / f"{name}.jsonl") for name in hashes}:
        raise ValueError("corpus changed during preparation")
    out = path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    inputs.mkdir()
    for name in papers:
        shutil.copyfile(corpus / f"{name}.jsonl", inputs / f"{name}.jsonl")
    shutil.copyfile(corpus / "provenance.json", inputs / "corpus_provenance.json")
    for name, rows in chunks.items():
        write_jsonl(inputs / f"{name}_chunks.jsonl", rows)
    tokenizer.save_pretrained(inputs / "tokenizer")
    write_json(inputs / "nf4_report.json", nf4)
    cfg["nf4_report"] = str(inputs / "nf4_report.json")
    write_json(out / "config.json", cfg)
    snapshot_sources(out / "source")
    input_hashes = {str(p.relative_to(inputs)): stream_hash(p) for p in sorted(inputs.rglob("*")) if p.is_file()}
    source_hashes = {str(p.relative_to(out / "source")): stream_hash(p) for p in sorted((out / "source").rglob("*")) if p.is_file()}
    protocol = {"config": cfg, "counts": counts, "raw_corpus_sha256": hashes,
                "paper_ids": {name: [row["paper_id"] for row in rows] for name, rows in papers.items()},
                "train_chunk_order_before_sampler": [row["id"] for row in chunks["train"]],
                "input_sha256": input_hashes, "source_sha256": source_hashes,
                "resolved_config_sha256": stream_hash(out / "config.json"),
                "lora": {"target_modules": "all-linear", "rank": cfg["rank"], "alpha": 2 * cfg["rank"], "dropout": 0.05},
                "exposure_policy": "Every within-paper chunk once per epoch, including every remainder; same chunks/tokens across ranks and BF16 control. No cross-paper packing. EOS at paper end only.",
                "validation_policy": "Token-weighted and per-paper validation NLL before training and each epoch; save every epoch. Register/task-dev screens and checkpoint selection are external. No automatic test evaluation.",
                "registration": "Requires the manager's committed exact protocol before run. LR pilots at rank32:5e-5/1e-4; ranks8/32/128 and seed43 repeat/BF16 control require their own prepared committed protocols."}
    write_json(out / "protocol.json", protocol)
    frozen_hash = seal_snapshot(out)
    write_json(out / "run.json", {**provenance(cfg), "completed": False, "status": "prepared", "phase": "awaiting_manager_protocol",
                                  "protocol_sha256": stream_hash(out / "protocol.json"), "frozen_manifest_sha256": frozen_hash})
    print(json.dumps({"prepared": str(out), "counts": counts, "steps_per_epoch": cfg["steps_per_epoch"],
                      "expected_loss_tokens": cfg["expected_loss_tokens"], "protocol": str(out / "protocol.json")}, indent=2))


def verify_prepared(prepared, protocol):
    verify_snapshot(prepared)
    cfg = read_json(prepared / "config.json")
    verify_committed_protocol(prepared, path(protocol), cfg["repo_root"])
    verify_nf4(cfg)
    return cfg, read_json(prepared / "protocol.json")


def replay_probe(model, token_ids, tokenizer):
    import torch
    model.eval()
    inputs = torch.tensor([token_ids], dtype=torch.long, device="cuda:0")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits[0, -1].float().cpu()
        generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=16,
                                   do_sample=False, num_beams=1, repetition_penalty=1.0, temperature=None,
                                   top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id, use_cache=True)[0, len(token_ids):].tolist()
    return logits, generated


def run(prepared, protocol):
    prepared, protocol = path(prepared).resolve(), path(protocol).resolve()
    cfg, registration = verify_prepared(prepared, protocol)
    metadata = read_json(prepared / "run.json")
    if metadata["status"] != "prepared" or metadata["frozen_manifest_sha256"] != stream_hash(prepared / "frozen.json"):
        raise ValueError("requires original unstarted prepared run; refuses overwrite/rerun")
    if ROOT.resolve() != (prepared / "source").resolve():
        os.chdir(prepared / "source")
        os.execv(sys.executable, [sys.executable, "-m", "scripts.physics_train", "--run", str(prepared), "--protocol", str(protocol)])
    select_gpu("1")
    setup_hf(cfg)

    def phase(value, status="running"):
        metadata.update({"phase": value, "status": status})
        write_json(prepared / "run.json", metadata)

    try:
        import torch
        from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
        from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
        from scripts.physics_eval import evaluate_chunks
        metadata["execution_provenance"] = provenance(cfg)
        metadata["committed_protocol"] = str(protocol)
        phase("loading")
        set_seed(cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(prepared / "inputs/tokenizer", local_files_only=True)
        train_chunks = read_papers(prepared / "inputs/train_chunks.jsonl")
        validation_chunks = read_papers(prepared / "inputs/validation_chunks.jsonl")
        model = get_peft_model(load_base(cfg, training=True), LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear",
                                                                      r=cfg["rank"], lora_alpha=2 * cfg["rank"], lora_dropout=0.05, bias="none"))
        model.peft_config["default"].base_model_name_or_path = cfg["nf4_checkpoint"] if cfg["representation"] == "nf4" else cfg["model_id"]
        model.peft_config["default"].revision = cfg["revision"]
        if any(p.requires_grad and "lora_" not in name for name, p in model.named_parameters()):
            raise RuntimeError("unexpected trainable base parameter")
        metadata["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        exposure_counts = {"actual_loss_tokens": 0, "actual_microbatches": 0}

        class PaperTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                if model.training:
                    exposure_counts["actual_loss_tokens"] += int((inputs["labels"][:, 1:] != -100).sum().item())
                    exposure_counts["actual_microbatches"] += 1
                return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

            def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
                report = evaluate_chunks(self.model, validation_chunks, tokenizer, cfg["eval_batch_size"])
                tag = f"step-{self.state.global_step:06d}"
                write_json(prepared / f"validation/{tag}.json", report)
                metrics = {"eval_loss": report["token_weighted_nll"], "eval_mean_per_paper_nll": report["mean_per_paper_nll"]}
                self.log(metrics)
                self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
                return metrics

        class Progress(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                metadata.update({"last_checkpoint_step": state.global_step, "epoch": state.epoch, "exposure": dict(exposure_counts)})
                write_json(prepared / "run.json", metadata)
                write_json(prepared / "learning_curve.json", state.log_history)

        training_args = TrainingArguments(
            output_dir=str(prepared / "checkpoints"), num_train_epochs=cfg["epochs"], learning_rate=cfg["learning_rate"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"], gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            per_device_eval_batch_size=cfg["eval_batch_size"], dataloader_drop_last=False,
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=cfg["epochs"],
            warmup_ratio=cfg["warmup_ratio"], logging_steps=1, bf16=True, fp16=False, tf32=False,
            seed=cfg["seed"], data_seed=cfg["seed"], optim="adamw_torch", report_to=[], push_to_hub=False,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, label_names=["labels"])
        trainer = PaperTrainer(model=model, args=training_args, train_dataset=train_chunks, eval_dataset=validation_chunks,
                               data_collator=AssistantCollator(tokenizer.pad_token_id), processing_class=tokenizer, callbacks=[Progress()])
        phase("baseline_validation")
        trainer.evaluate()
        phase("training")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = trainer.train()
        torch.cuda.synchronize()
        metadata.update({"training_wall_seconds": time.perf_counter() - started,
                         "peak_training_allocated_bytes": torch.cuda.max_memory_allocated(), "training_metrics": result.metrics,
                         "training_interval_scope": "Whole Trainer interval including epoch validation and checkpoint saves",
                         "exposure": exposure_counts, "global_step": trainer.state.global_step, "epoch": trainer.state.epoch})
        phase("training_complete")
        write_json(prepared / "learning_curve.json", trainer.state.log_history)
        if exposure_counts["actual_loss_tokens"] != cfg["expected_loss_tokens"] or exposure_counts["actual_microbatches"] != cfg["expected_microbatches"] or trainer.state.global_step != cfg["max_training_steps"]:
            raise RuntimeError("actual exposure/steps differ from frozen plan; do not claim equal tokens")
        adapter = prepared / "adapter"
        model.save_pretrained(adapter, safe_serialization=True)
        tokenizer.save_pretrained(adapter)
        phase("adapter_saved")
        state = {name: value.detach().cpu().clone() for name, value in get_peft_model_state_dict(model).items()}
        probe_ids = train_chunks[0]["input_ids"][:64]
        before_logits, before_ids = replay_probe(model, probe_ids, tokenizer)
        trainer.accelerator.free_memory()
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        phase("replaying_adapter")
        reloaded = PeftModel.from_pretrained(load_base(cfg, training=False), adapter, is_trainable=False, local_files_only=True)
        after_state = get_peft_model_state_dict(reloaded)
        state_equal = set(state) == set(after_state) and all(value.dtype == after_state[name].dtype and torch.equal(value, after_state[name].detach().cpu()) for name, value in state.items())
        after_logits, after_ids = replay_probe(reloaded, probe_ids, tokenizer)
        metadata["replay"] = {"exact_adapter_tensors": state_equal, "exact_logits": torch.equal(before_logits, after_logits),
                              "max_abs_logit_difference": (before_logits - after_logits).abs().max().item(), "exact_generated_ids": before_ids == after_ids,
                              "probe_scope": "first64 input tokens of first training chunk,16 new plain-text tokens; serialization check, not evaluation"}
        write_json(prepared / "replay.json", {**metadata["replay"], "input_ids": probe_ids, "before_generated_ids": before_ids, "after_generated_ids": after_ids})
        metadata["completed"] = state_equal and metadata["replay"]["exact_logits"] and metadata["replay"]["exact_generated_ids"]
        if not metadata["completed"]:
            raise RuntimeError("adapter replay differs")
        phase("complete", "complete")
        write_json(adapter / "provenance.json", metadata)
        write_json(prepared / "adapter_files.json", {p.name: {"bytes": p.stat().st_size, "sha256": stream_hash(p)} for p in sorted(adapter.iterdir()) if p.is_file()})
    except BaseException as exc:
        metadata.update({"completed": False, "failed_phase": metadata.get("phase"), "error_type": type(exc).__name__, "error": str(exc)})
        phase("failed", "failed")
        raise
    print(prepared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", help="CPU preparation only, new directory")
    mode.add_argument("--run", help="execute manager-authorized frozen preparation")
    parser.add_argument("--config", default="configs/physics.json")
    parser.add_argument("--protocol")
    parser.add_argument("--rank", type=int, choices=(8, 32, 128), default=32)
    parser.add_argument("--learning-rate", type=float, choices=(5e-5, 1e-4), default=5e-5)
    parser.add_argument("--epochs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--seed", type=int, choices=(42, 43), default=42)
    parser.add_argument("--representation", choices=("nf4", "bf16"), default="nf4")
    parser.add_argument("--sequence-length", type=int, choices=(1024, 2048))
    args = parser.parse_args()
    if args.run:
        if not args.protocol:
            parser.error("--run requires the manager's committed --protocol")
        run(args.run, args.protocol)
    else:
        prepare(args)


if __name__ == "__main__":
    main()
