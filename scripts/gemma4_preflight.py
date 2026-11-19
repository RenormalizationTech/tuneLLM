"""Gemma 4 text-only compatibility and synthetic QLoRA feasibility checks.

No task training, final-test inputs or judge calls. GPU execution is explicit;
the adapter saved by the synthetic steps is a mechanics artifact, not a release.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/gemma4-preflight-01"
PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def snapshot():
    return Path((RUN / "snapshot_path.txt").read_text().strip())


def versions():
    return {p: importlib.metadata.version(p) for p in
            ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "huggingface_hub")}


def encode(tokenizer, user, assistant, system=None):
    """Use the inference prefix, including its forced empty thought channel.

    Gemma's complete-conversation template omits that empty channel. Explicitly
    retaining the generation prefix prevents a training/inference boundary drift.
    Supervise the response and exactly one native end-of-turn token (106), not
    the generic end-of-document EOS (1); prompt and empty thought stay masked.
    """
    if not isinstance(user, str) or not user.strip() or not isinstance(assistant, str) or not assistant.strip():
        raise ValueError("nonempty user and assistant text required")
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": user})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer = tokenizer(assistant, add_special_tokens=False)["input_ids"]
    if not answer or set(answer) & set(tokenizer.all_special_ids):
        raise ValueError("response must be plain text without chat control tokens")
    turn_end = tokenizer.convert_tokens_to_ids("<turn|>")
    if turn_end != 106 or turn_end == tokenizer.eos_token_id:
        raise ValueError("unexpected native turn/EOS IDs")
    ids = tokenizer(prompt + assistant + "<turn|>", add_special_tokens=False)["input_ids"]
    if ids[:len(prefix)] != prefix or ids[len(prefix):] != answer + [turn_end]:
        raise ValueError("tokenizer merges at the supervised boundary")
    return {"input_ids": ids, "labels": [-100] * len(prefix) + answer + [turn_end],
            "attention_mask": [1] * len(ids), "prompt_tokens": len(prefix),
            "response_tokens": len(answer) + 1}


def response_loss(model, ids, labels, mask):
    import torch
    positions = torch.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten()
    if ids.shape[0] != 1 or not positions.numel():
        raise ValueError("nonempty supervised response and batch one required")
    output = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=positions)
    target = labels[:, positions + 1]
    return torch.nn.functional.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]),
                                             target.reshape(-1))


def cpu_prepare():
    import torch
    from transformers import AutoTokenizer, Gemma4ForCausalLM, Gemma4TextConfig
    tokenizer = AutoTokenizer.from_pretrained(snapshot(), local_files_only=True)
    config = Gemma4TextConfig.from_dict(json.loads((snapshot() / "config.json").read_text())["text_config"])
    with torch.device("meta"):
        model = Gemma4ForCausalLM(config)
    targets = [{"name": n, "in_features": m.in_features, "out_features": m.out_features}
               for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and n.rsplit(".", 1)[-1] in PROJECTIONS]
    if len(targets) != 410 or config.num_hidden_layers != 60:
        raise ValueError("official pinned dense text geometry differs")
    coverage = {}
    for task, folder in {"qasper": "prepared-qasper-tokenizer-01", "prl": "prepared-prl-audited-01"}.items():
        coverage[task] = {}
        for split in ("train", "validation"):
            source = ROOT / "outputs/qwen38-science-01" / folder / (split + ".jsonl")
            rows = [json.loads(line) for line in source.read_text().split("\n") if line.strip()]
            items, examples, excluded = [], [], 0
            for row in rows:
                if not row.get("training_eligible", False):
                    excluded += 1
                    continue
                value = encode(tokenizer, row["user"], row["assistant"], row.get("system"))
                items.append({"id": row["id"], "length": len(value["input_ids"]),
                              "prompt": value["prompt_tokens"], "response": value["response_tokens"]})
                if len(examples) < 3:
                    examples.append({"id": row["id"], "prompt_suffix": tokenizer.decode(
                        value["input_ids"][max(0, value["prompt_tokens"] - 20):value["prompt_tokens"]]),
                        "supervised_prefix": tokenizer.decode(value["input_ids"][value["prompt_tokens"]:][:20]),
                        "supervised_final_id": value["input_ids"][-1], "prompt_masked": all(
                            v == -100 for v in value["labels"][:value["prompt_tokens"]])})
            lengths = sorted(i["length"] for i in items)
            if not lengths:
                raise ValueError("no eligible examples")
            coverage[task][split] = {"source_relative_path": source.relative_to(ROOT).as_posix(),
                "source_sha256": sha(source), "eligible": len(items), "ineligible": excluded,
                "median": lengths[len(lengths) // 2], "p95": lengths[min(len(lengths)-1, int(.95*len(lengths)))],
                "maximum": max(lengths), "over_8192": sum(n > 8192 for n in lengths),
                "total_tokens": sum(lengths), "response_tokens": sum(i["response"] for i in items),
                "maximum_response_tokens": max(i["response"] for i in items), "examples": examples,
                "length_rows": items}
            print(task, split, len(items), "eligible; max", max(lengths), "over8192", sum(n > 8192 for n in lengths), flush=True)
    result = {"scope": "train/development tokenizer coverage and meta inventory only; no test split opened",
        "repo_id": "google/gemma-4-31B-it", "revision": snapshot().name, "versions": versions(),
        "text_config": config.to_dict(), "target_count": len(targets), "targets": targets,
        "target_suffix_counts": dict(Counter(i["name"].rsplit(".", 1)[-1] for i in targets)),
        "rank": 16, "alpha": 32, "dropout": 0.0,
        "trainable_parameters": sum(16*(i["in_features"]+i["out_features"]) for i in targets),
        "training_format": "native no-thinking generation prefix + plain response + one native turn-end; prompt masked",
        "coverage": coverage, "source_sha256": sha(__file__)}
    write(RUN / "cpu_prepare.json", result)


def load_base():
    import torch
    from transformers import BitsAndBytesConfig, Gemma4ForCausalLM, Gemma4TextConfig
    config = Gemma4TextConfig.from_dict(json.loads((snapshot() / "config.json").read_text())["text_config"])
    model, info = Gemma4ForCausalLM.from_pretrained(snapshot(), config=config,
        key_mapping={r"^model\.language_model\.": "model."},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa",
        local_files_only=True, trust_remote_code=False, output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("text weights missing or mismatched: " + str(info))
    extra = info.get("unexpected_keys", [])
    if any(not (n.startswith("model.vision_tower.") or n.startswith("model.embed_vision.")) for n in extra):
        raise ValueError("unexpected non-vision weight: " + str(extra[:5]))
    if any("vision" in n or "audio" in n for n, _ in model.named_parameters()):
        raise ValueError("text-only loader instantiated a multimodal component")
    return model, {"loader": "Gemma4ForCausalLM", "excluded_vision_weights": len(extra),
                   "missing_text_weights": [], "attn_implementation": model.config._attn_implementation,
                   "key_mapping": {r"^model\.language_model\.": "model."}}


def synthetic(tokenizer, length, target_tokens=256):
    import torch
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Describe the supplied scientific observations."}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    if not isinstance(prompt, list) or not all(isinstance(i,int) for i in prompt):
        raise TypeError("flat native prompt IDs required")
    unit = tokenizer(" A controlled sample shows a measured response.", add_special_tokens=False)["input_ids"]
    target = tokenizer("The controlled sample shows a measured response.", add_special_tokens=False)["input_ids"]
    answer = (target * ((target_tokens + len(target) - 1) // len(target)))[:target_tokens-1] + [106]
    # Insert the long synthetic body inside the user turn, before its closing
    # marker, retaining the exact inference-time assistant prefix at the boundary.
    boundary = prompt.index(106)
    head, suffix = prompt[:boundary], prompt[boundary:]
    count = length - len(prompt) - len(answer)
    if count < 1:
        raise ValueError("synthetic length too short")
    ids = head + (unit * ((count+len(unit)-1)//len(unit)))[:count] + suffix + answer
    labels = [-100] * (length-len(answer)) + answer
    return (torch.tensor([ids], device="cuda:0"), torch.tensor([labels], device="cuda:0"),
            torch.ones((1, length), dtype=torch.long, device="cuda:0"))


def generation(model, tokenizer, prompt, cap=64, input_ids=None, force_cap=False):
    import torch
    ids = (tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=False).to("cuda:0")
            if input_ids is None else input_ids)
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("one unpadded prompt tensor required")
    torch.cuda.synchronize(); start = time.monotonic()
    with torch.no_grad():
        output = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
            max_new_tokens=cap, min_new_tokens=cap if force_cap else 0, repetition_penalty=1.0,
            use_cache=True, eos_token_id=[1, 106], pad_token_id=0)
    torch.cuda.synchronize(); seconds = time.monotonic()-start
    tail = output[0, ids.shape[-1]:].tolist()
    return {"prompt": prompt, "prompt_tokens": ids.shape[-1], "cap": cap,
            "force_cap_for_throughput_only": force_cap, "output_ids": tail,
            "text": tokenizer.decode(tail, skip_special_tokens=False), "generated_tokens": len(tail),
            "seconds_including_prefill": seconds, "tokens_per_second_including_prefill": len(tail)/seconds,
            "ended_at_native_turn": bool(tail and tail[-1] == 106)}


def execute(long_response_only=False):
    import torch
    from transformers import AutoTokenizer, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from safetensors.torch import save_file
    prepared = json.loads((RUN / "cpu_prepare.json").read_text())
    if prepared["source_sha256"] != sha(__file__):
        raise ValueError("preflight source differs from CPU preparation")
    result = {"scope": "synthetic mechanics only; no real task optimization, test generation or judge calls",
              "status": "running", "versions": versions(), "revision": snapshot().name, "probes": [],
              "source_sha256": sha(__file__), "cpu_prepare_sha256": sha(RUN / "cpu_prepare.json"),
              "long_response_only": long_response_only}
    write(RUN / "gpu_result.json", result)
    start = time.monotonic(); set_seed(42)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one explicit CUDA device required")
        result["GPU"] = {"name": torch.cuda.get_device_name(0), "total_bytes": torch.cuda.get_device_properties(0).total_memory,
                         "uuid": str(torch.cuda.get_device_properties(0).uuid)}
        tick = time.monotonic(); model, loading = load_base(); torch.cuda.synchronize()
        result["load_seconds"] = time.monotonic()-tick; result["loading"] = loading
        result["memory_after_NF4_load"] = {"allocated_bytes":torch.cuda.memory_allocated(),
                                          "reserved_bytes":torch.cuda.memory_reserved()}
        tokenizer = AutoTokenizer.from_pretrained(snapshot(), local_files_only=True)
        model.eval()
        result["base_generation_smokes"] = [generation(model, tokenizer, p) for p in
            ("What is two plus three? Answer briefly.", "Describe one simple scientific observation in one sentence.")]
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32,
            lora_dropout=0.0, target_modules=[r["name"] for r in prepared["targets"]], bias="none"))
        trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if any("lora_" not in n for n,p in trainable) or sum(p.numel() for _,p in trainable) != prepared["trainable_parameters"]:
            raise ValueError("trainable inventory mismatch")
        result["trainable_parameters"] = sum(p.numel() for _,p in trainable)
        result["trainable_tensors"] = len(trainable)
        result["parameter_dtype_counts"] = dict(Counter(str(p.dtype) for p in model.parameters()))
        result["floating_weight_recipe"] = "PEFT default FP32 preparation, including tied embedding/head; match this in later base/adapter comparisons"
        result["memory_after_PEFT_preparation"] = {"allocated_bytes":torch.cuda.memory_allocated(),
                                                  "reserved_bytes":torch.cuda.memory_reserved()}
        if not long_response_only:
            model.eval(); long_ids, _, _ = synthetic(tokenizer,8192,256)
            result["long_context_generation_throughput"] = generation(model,tokenizer,
                "Synthetic repeated scientific text; mechanics throughput only",cap=256,
                input_ids=long_ids[:,:-256],force_cap=True)
            del long_ids
        model.eval(); ids, labels, mask = synthetic(tokenizer, 64, 8)
        with torch.no_grad():
            full = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
            reference = torch.nn.functional.cross_entropy(full[:,:-1].reshape(-1,full.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100)
            compact = response_loss(model, ids, labels, mask)
        result["response_only_CE"] = {"full": reference.item(), "compact": compact.item(),
            "absolute_difference": abs(reference.item()-compact.item()),
            "close": bool(torch.allclose(reference,compact,atol=1e-4,rtol=1e-4))}
        if not result["response_only_CE"]["close"]:
            raise ValueError("response-only CE does not match the full-logit loss")
        del full, reference, compact, ids, labels, mask
        optimizer = torch.optim.AdamW([p for _,p in trainable], lr=5e-5)
        model.train()
        probes = [(8192,512)] if long_response_only else [(2048,256),(4096,256),(8192,256)]
        for length, response_tokens in probes:
            ids, labels, mask = synthetic(tokenizer, length,response_tokens)
            optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); tick = time.monotonic()
            probe = {"sequence_length": length, "response_tokens": response_tokens}
            try:
                loss = response_loss(model,ids,labels,mask); loss.backward()
                grad = [(n,p.grad) for n,p in trainable if p.grad is not None]
                finite = all(bool(torch.isfinite(g).all()) for _,g in grad)
                nonzero = sum(bool(torch.count_nonzero(g)) for _,g in grad)
                if len(grad) != len(trainable) or not finite or not nonzero or not torch.isfinite(loss):
                    raise ValueError("missing, nonfinite or all-zero adapter gradients")
                norm = torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.0)
                optimizer.step(); torch.cuda.synchronize()
                probe.update(loss=loss.item(), gradient_tensors=len(grad), nonzero_gradient_tensors=nonzero,
                    finite_gradients=finite, gradient_norm_before_clip=norm.item(), seconds=time.monotonic()-tick,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    device_free_bytes=torch.cuda.mem_get_info()[0], status="passed")
                print("PROBE",json.dumps(probe),flush=True)
                result["probes"].append(probe);write(RUN / "gpu_result.json",result)
            except torch.cuda.OutOfMemoryError as error:
                probe.update(status="out_of_memory",seconds=time.monotonic()-tick,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    error=str(error));result["probes"].append(probe);raise
            finally:
                optimizer.zero_grad(set_to_none=True)
        model.eval(); model.gradient_checkpointing_disable(); ids,_,mask = synthetic(tokenizer,64,8)
        with torch.no_grad():
            ref = model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1).logits.cpu().contiguous()
        save_file({"input_ids":ids.cpu(),"attention_mask":mask.cpu(),"logits":ref},str(RUN / "reload_reference.safetensors"))
        model.save_pretrained(RUN / "synthetic_adapter",safe_serialization=True)
        result["synthetic_steps"] = len(result["probes"])
        result["synthetic_adapter_sha256"] = {p.name:sha(p) for p in (RUN / "synthetic_adapter").glob("*") if p.is_file()}
        result["reload_reference_sha256"] = sha(RUN / "reload_reference.safetensors")
        result["status"] = "mechanics_passed_pending_fresh_reload"
    except Exception as error:
        result.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        result["elapsed_seconds_including_load"] = time.monotonic()-start
        write(RUN / "gpu_result.json",result)


def reload_check():
    import torch
    from transformers import set_seed
    from peft import PeftModel, prepare_model_for_kbit_training
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file
    start = time.monotonic(); set_seed(42)
    original=json.loads((RUN/"gpu_result.json").read_text())
    if original["status"] != "mechanics_passed_pending_fresh_reload":
        raise ValueError("synthetic mechanics did not pass")
    for name,value in original["synthetic_adapter_sha256"].items():
        if sha(RUN / "synthetic_adapter" / name) != value:
            raise ValueError("synthetic adapter changed")
    model,_=load_base()
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    model=PeftModel.from_pretrained(model,RUN/"synthetic_adapter",is_trainable=False)
    model.eval(); saved=load_file(str(RUN/"synthetic_adapter/adapter_model.safetensors"))
    actual=get_peft_model_state_dict(model)
    tensors_equal=set(saved)==set(actual) and all(torch.equal(saved[k],actual[k].cpu()) for k in saved)
    reference=load_file(str(RUN/"reload_reference.safetensors"))
    with torch.no_grad():
        logits=model(input_ids=reference["input_ids"].cuda(),attention_mask=reference["attention_mask"].cuda(),
                     use_cache=False,logits_to_keep=1).logits.cpu()
    equal=torch.equal(logits,reference["logits"])
    result={"scope":"fresh-process NF4 base + synthetic adapter mechanics only","adapter_tensors_equal":tensors_equal,
        "adapter_tensor_count":len(saved),"logits_torch_equal":equal,
        "max_absolute_logit_difference":(logits.float()-reference["logits"].float()).abs().max().item(),
        "status":"passed" if tensors_equal and equal else "failed", "seconds_including_load":time.monotonic()-start,
        "source_sha256":sha(__file__),"original_result_sha256":sha(RUN/"gpu_result.json")}
    write(RUN/"fresh_reload.json",result);print(json.dumps(result,indent=2),flush=True)
    if result["status"] != "passed":
        raise ValueError("fresh-process reload mismatch")


def main():
    global RUN
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","execute","reload"])
    parser.add_argument("--run",default=str(RUN),help="Separate preflight output, with pinned snapshot metadata")
    parser.add_argument("--long-response-only",action="store_true",help="A separate full-length 512-target synthetic check")
    args=parser.parse_args()
    RUN=Path(args.run).resolve()
    if not RUN.is_relative_to(ROOT/"outputs"):
        raise ValueError("preflight output must be inside project outputs")
    if args.stage == "execute":
        if (RUN/"gpu_result.json").exists():
            raise FileExistsError("preserve prior GPU evidence; select a fresh --run")
        execute(args.long_response_only)
    else:
        {"prepare":cpu_prepare,"reload":reload_check}[args.stage]()


if __name__ == "__main__":
    main()
