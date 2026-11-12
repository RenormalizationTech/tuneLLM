"""Shared Qwen NF4 representation and greedy generation. Training padding stays untouched."""
import time

REPRESENTATION = {"quant_type": "nf4", "double_quant": True, "compute_dtype": "bfloat16",
                  "nonquantized_base_dtype": "float32 (prepare_model_for_kbit_training)",
                  "attention": "sdpa"}


def render_prompt(tokenizer, row):
    messages = []
    if row.get("system"):
        messages.append({"role": "system", "content": row["system"]})
    messages.append({"role": "user", "content": row["user"]})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def left_pad(sequences, pad_id):
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("nonempty prompt sequences required")
    width = max(map(len, sequences))
    return {"input_ids": [[pad_id] * (width - len(s)) + list(s) for s in sequences],
            "attention_mask": [[0] * (width - len(s)) + [1] * len(s) for s in sequences]}


def generated_suffix(sequence, padded_width, eos_id):
    """Keep through first generated EOS; without EOS every emitted token is real.

    Tokens after EOS are batch filler. A PAD token emitted before EOS (including
    at the token cap) is an actual prediction and must remain in token comparisons.
    """
    result = list(sequence[padded_width:])
    return result[:result.index(eos_id) + 1] if eos_id in result else result


def load_nf4(snapshot, training=False):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected GPU must support CUDA BF16")
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        device_map={"": 0}, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = not training
    return model


def generate_batched(model, tokenizer, rows, batch_size=8, max_new_tokens=512, seed=42,
                     device="cuda:0", phase="generation"):
    import torch
    if batch_size <= 0 or max_new_tokens <= 0 or not rows:
        raise ValueError("positive batch/length and nonempty rows required")
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("generation requires pad distinct from EOS")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate generation IDs")
    torch.manual_seed(seed)
    model.eval()
    is_cuda = torch.device(device).type == "cuda"

    def synchronize():
        if is_cuda:
            torch.cuda.synchronize(device)

    predictions, batches = [], []
    pipeline_start = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        members = rows[start:start + batch_size]
        prompts = [render_prompt(tokenizer, row) for row in members]
        sequences = [tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]
        padded = left_pad(sequences, tokenizer.pad_token_id)
        width = len(padded["input_ids"][0])
        context_limit = getattr(model.config, "max_position_embeddings", None)
        if context_limit and width + max_new_tokens > context_limit:
            raise ValueError("prompt plus generation budget exceeds model context; no truncation allowed")
        inputs = {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in padded.items()}
        synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                                    temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                    use_cache=True, return_dict_in_generate=False)
        synchronize()
        elapsed = time.perf_counter() - started
        output = output.tolist()
        batch_ids = [row["id"] for row in members]
        batches.append({"batch_index": len(batches), "ids": batch_ids, "padded_prompt_width": width,
                        "generation_wall_seconds": elapsed})
        for position, (row, sequence) in enumerate(zip(members, output)):
            ids = generated_suffix(sequence, width, tokenizer.eos_token_id)
            ended_eos = tokenizer.eos_token_id in ids
            predictions.append({"id": row["id"], "output": tokenizer.decode(ids, skip_special_tokens=True),
                                "generated_ids": ids, "rendered_prompt": prompts[position],
                                "input_ids": sequences[position], "batch_index": len(batches) - 1,
                                "batch_position": position, "batch_ids": batch_ids,
                                "padded_prompt_width": width, "generated_tokens_including_eos": len(ids),
                                "ended_eos": ended_eos,
                                "hit_token_limit": len(ids) >= max_new_tokens and not ended_eos})
        print(f"{phase}: batch {len(batches)} completed ({min(start + batch_size, len(rows))}/{len(rows)} rows)", flush=True)
    generation_wall = sum(batch["generation_wall_seconds"] for batch in batches)
    return {"predictions": predictions, "batches": batches,
            "settings": {"batch_size": batch_size, "max_new_tokens": max_new_tokens, "seed": seed,
                         "do_sample": False, "num_beams": 1, "enable_thinking": False,
                         "generation_padding": "manual left padding; tokenizer.padding_side unchanged"},
            "timing": {"generation_wall_seconds": generation_wall,
                       "pipeline_wall_seconds": time.perf_counter() - pipeline_start,
                       "generated_tokens_per_generation_wall_second": sum(len(p["generated_ids"]) for p in predictions) / generation_wall,
                       "scope": "Synchronized batched generate wall time; includes prefill/decode; excludes tokenization, device transfers and decoding. Pipeline time includes these. Count through first EOS inclusive; without EOS keep all emitted tokens, including PAD. No warmup; diagnostic timing."}}
