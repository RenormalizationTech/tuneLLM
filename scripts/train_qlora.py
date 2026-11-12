"""Offline single-GPU NF4 + PEFT training, fixed held-out comparisons, adapter reload.

Prepare only: python -m scripts.train_qlora --smoke --prepare-only --output outputs/qlora-preflight
Train (manager): python -m scripts.train_qlora --smoke --output outputs/qlora-smoke
"""
import argparse
import gc
import random
import time
from scripts.common import (digest, load_config, local_snapshot, path, provenance,
                            read_json, select_gpu, setup_hf, write_json)
from scripts.training_data import AssistantCollator, encode_row, load_splits
from evaluation.gsm8k import score_gsm8k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qlora.json")
    parser.add_argument("--gpu")
    parser.add_argument("--output", required=True, help="new directory; refuses overwrite")
    parser.add_argument("--smoke", action="store_true", help="use prepared train-only smoke_train.jsonl; exactly 2 steps")
    parser.add_argument("--prepare-only", action="store_true", help="tokenize/inspect data on CPU; no model allocation")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.gpu is not None:
        cfg["gpu"] = args.gpu
    select_gpu(cfg["gpu"])
    setup_hf(cfg)
    if args.smoke:
        cfg["splits"]["train"] = str(path(cfg["splits"]["train"]).with_name("smoke_train.jsonl"))
        cfg["training"]["max_steps"] = 2
    rows = load_splits(cfg["splits"])
    dataset_manifest = path(cfg["splits"]["train"]).parent / "provenance.json"
    data_provenance = read_json(dataset_manifest)
    if data_provenance["source"]["revision"] != cfg["dataset_provenance"]["revision"]:
        raise ValueError("prepared data revision does not match config")
    for filename in cfg["splits"].values():
        if digest(filename) != data_provenance["output_sha256"][path(filename).name]:
            raise ValueError(f"prepared split checksum mismatch: {filename}")
    # Select once with seed; preserve all held-out files and record exact selected IDs.
    for name in ("validation", "test"):
        selected = list(rows[name])
        random.Random(cfg["seed"]).shuffle(selected)
        rows[name] = selected[:cfg["eval_rows"]]
    if cfg["eval_rows"] <= 0:
        raise ValueError("eval_rows must be positive")
    snapshot = local_snapshot(cfg)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    encoded = {name: [encode_row(row, tokenizer, cfg["max_length"]) for row in split]
               for name, split in rows.items()}
    out = path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    metadata = provenance(cfg)
    metadata.update({"dataset_provenance": data_provenance,
                     "split_sha256": {k: digest(v) for k, v in cfg["splits"].items()},
                     "selected_ids": {k: [r["id"] for r in v] for k, v in rows.items()},
                     "quantization": {"load_in_4bit": True, "type": "nf4", "double_quant": True, "compute_dtype": "bfloat16"},
                     "comparison": "same frozen NF4 base vs trained adapter; fixed held-out rows and greedy non-thinking prompts",
                     "loss_note": "Trainer evaluation loss on assistant tokens including EOS; batch/sample averaged, not a token-weighted corpus perplexity",
                     "purpose": "2-step pipeline smoke" if args.smoke else "configured training run",
                     "completed": False, "status": "running", "phase": "data_prepared"})
    write_json(out / "run.json", metadata)

    def set_phase(phase, status="running"):
        metadata.update({"phase": phase, "status": status})
        write_json(out / "run.json", metadata)

    write_json(out / "chat_template.json", {"chat_template": tokenizer.chat_template, "enable_thinking": False})
    example = encoded["train"][0]
    write_json(out / "batch_inspection.json", {
        "row_id": rows["train"][0]["id"], **example,
        "decoded_input": tokenizer.decode(example["input_ids"], skip_special_tokens=False),
        "decoded_supervised": tokenizer.decode([x for x in example["labels"] if x != -100], skip_special_tokens=False),
        "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
        "split_rows": {k: len(v) for k, v in encoded.items()},
    })
    if args.prepare_only:
        set_phase("data_prepared", status="prepared_only")
        print(f"CPU preparation complete: {out}")
        return

    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected GPU must support CUDA BF16")
    set_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False

    def load_base():
        return AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            device_map={"": 0}, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_use_double_quant=True,
                                                   bnb_4bit_compute_dtype=torch.bfloat16))

    set_phase("loading_model")
    model = get_peft_model(prepare_model_for_kbit_training(
        load_base(), use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}),
        LoraConfig(task_type="CAUSAL_LM", bias="none", **cfg["lora"]))
    model.config.use_cache = False
    model.peft_config["default"].base_model_name_or_path = cfg["model_id"]
    model.peft_config["default"].revision = cfg["revision"]
    if any(p.requires_grad and "lora_" not in name for name, p in model.named_parameters()):
        raise RuntimeError("unexpected trainable non-LoRA weights")
    metadata["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not metadata["trainable_parameters"]:
        raise RuntimeError("no trainable adapter parameters")
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"), bf16=True, fp16=False, tf32=False,
        seed=cfg["seed"], data_seed=cfg["seed"], report_to=[], push_to_hub=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", label_names=["labels"], prediction_loss_only=True,
        **cfg["training"])
    collator = AssistantCollator(tokenizer.pad_token_id)
    trainer = Trainer(model=model, args=training_args, train_dataset=encoded["train"],
                      eval_dataset=encoded["validation"], data_collator=collator,
                      processing_class=tokenizer)

    def heldout_losses(current_trainer):
        return {name: current_trainer.evaluate(data, metric_key_prefix=name)
                for name, data in encoded.items() if name != "train"}

    def generations(current_model, phase):
        set_phase(phase)
        current_model.eval()
        predictions = []
        for index, row in enumerate(rows["test"], start=1):
            messages = []
            if row.get("system"):
                messages.append({"role": "system", "content": row["system"]})
            messages.append({"role": "user", "content": row["user"]})
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                result = current_model.generate(**inputs, do_sample=False, num_beams=1,
                                                max_new_tokens=cfg["eval_max_new_tokens"],
                                                temperature=None, top_p=None, top_k=None,
                                                pad_token_id=tokenizer.pad_token_id,
                                                eos_token_id=tokenizer.eos_token_id,
                                                use_cache=True)
            ids = result[0, inputs["input_ids"].shape[1]:].tolist()
            predictions.append({"id": row["id"], "output": tokenizer.decode(ids, skip_special_tokens=True),
                                "rendered_prompt": rendered, "generated_ids": ids,
                                "hit_token_limit": len(ids) >= cfg["eval_max_new_tokens"] and ids[-1] != tokenizer.eos_token_id})
            print(f"{phase}: {index}/{len(rows['test'])} {row['id']} complete ({len(ids)} generated tokens)", flush=True)
        return predictions

    set_phase("before_loss_evaluation")
    with model.disable_adapter():
        before_loss = heldout_losses(trainer)
        before_predictions = generations(model, "before_generation")
    write_json(out / "before.json", {"losses": before_loss, "scores": score_gsm8k(rows["test"], before_predictions),
                                      "predictions": before_predictions})
    set_phase("training")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = trainer.train()
    torch.cuda.synchronize()
    metadata["training_wall_seconds"] = time.perf_counter() - started
    metadata["training_metrics"] = result.metrics
    metadata["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    set_phase("training_complete")
    set_phase("after_loss_evaluation")
    after_loss = heldout_losses(trainer)
    after_predictions = generations(model, "after_generation")
    write_json(out / "after.json", {"losses": after_loss, "scores": score_gsm8k(rows["test"], after_predictions),
                                     "predictions": after_predictions})
    adapter = out / "adapter"
    set_phase("saving_adapter")
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    set_phase("adapter_saved")
    write_json(adapter / "provenance.json", metadata)
    set_phase("reloading_adapter")
    trainer.accelerator.free_memory()
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    # Match the FP32 nonquantized modules of the prepared training base exactly.
    reload_base = prepare_model_for_kbit_training(load_base(), use_gradient_checkpointing=False)
    metadata["reload_representation"] = "NF4/double-quant BF16 compute; nonquantized floating base parameters cast to FP32 by prepare_model_for_kbit_training on both loads"
    reloaded = PeftModel.from_pretrained(reload_base, adapter, is_trainable=False,
                                        local_files_only=True)
    reloaded_predictions = generations(reloaded, "reload_generation")
    identical = [p["generated_ids"] for p in after_predictions] == [p["generated_ids"] for p in reloaded_predictions]
    write_json(out / "reload.json", {"identical_generated_ids": identical, "predictions": reloaded_predictions,
                                      "scores": score_gsm8k(rows["test"], reloaded_predictions)})
    metadata["reload_identical_generated_ids"] = identical
    metadata["completed"] = identical
    set_phase("complete" if identical else "reload_mismatch", status="complete" if identical else "failed")
    if not identical:
        raise RuntimeError("saved/reloaded adapter changed generated tokens; inspect reload.json")
    print(f"Training and fresh-base adapter reload complete: {out}")


if __name__ == "__main__":
    main()
