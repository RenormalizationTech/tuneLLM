# SPDX-License-Identifier: Apache-2.0
"""Standalone fixed experimental refusal shift; no tuneLLM imports/downloads."""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
from pathlib import Path

import torch

BASE = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
PT_SHA256 = "671c6a839cd01a3767d6455b9a1b59a37059f712097ff1f8142324dfccbae28e"
TENSOR_SHA256 = "1d2b29bc37636b26c6ba803467052e31fa4758e9928ea0f9fc8183c9791d646b"


def sha256(file):
    digest = hashlib.sha256()
    with Path(file).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()


def validate_direction(value):
    if (not isinstance(value, torch.Tensor) or value.shape != (3584,) or value.dtype != torch.float32
            or value.device.type != "cpu" or not torch.isfinite(value).all() or not torch.isfinite(value.norm()) or value.norm() == 0):
        raise ValueError("exact finite nonzero CPU FP32 [3584] raw direction required")
    if hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest() != TENSOR_SHA256:
        raise ValueError("frozen raw tensor content differs")
    return value


def load_direction(root=None, *, use_pt=False):
    """Prefer safetensors key 'direction'; exact .pt provenance is a ROOT tensor."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    if use_pt:
        file = root/"direction.pt"
        if sha256(file) != PT_SHA256: raise ValueError("frozen .pt file hash differs")
        value = torch.load(file, map_location="cpu", weights_only=True)
    else:
        from safetensors.torch import load_file
        values = load_file(str(root/"direction.safetensors"), device="cpu")
        if set(values) != {"direction"}: raise ValueError("exact safetensors direction key required")
        value = values["direction"]
    return validate_direction(value).detach().clone()


def validate_model(model, *, model_id=BASE, revision=REVISION):
    cfg = model.config
    if model_id != BASE or revision != REVISION or getattr(cfg, "_commit_hash", None) not in (None, REVISION):
        raise ValueError("exact base model ID and revision required")
    if (cfg.model_type, cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads) != ("qwen2", 28, 3584, 28, 4) or len(model.model.layers) != 28:
        raise ValueError("pinned Qwen2 architecture/config required")


@contextmanager
def steering(model, raw_direction, *, strength=1.0, model_id=BASE, revision=REVISION):
    """Fixed -0.25raw at block18 OUTPUT, including prefill and cached decode.

    strength=0 is an identity check; strength=1 is the only experimental setting.
    Preserve the original two power-of-two scalings and FP32 arithmetic exactly.
    No input tensor, weight, output tuple tail or cache object is mutated.
    """
    validate_model(model, model_id=model_id, revision=revision)
    if type(strength) not in (int, float) or not math.isfinite(strength) or strength not in (0, 1):
        raise ValueError("only identity0 or fixed experimental strength1 supported")
    raw = validate_direction(raw_direction.detach().cpu()).clone()
    effective = .5*raw if strength else torch.zeros_like(raw)
    local = {}
    def change(hidden):
        if hidden.ndim != 3 or hidden.shape[-1] != 3584 or not hidden.is_floating_point():
            raise ValueError("floating [batch,sequence,3584] block output required")
        if hidden.device not in local: local[hidden.device] = effective.to(hidden.device)
        return (hidden.float()-.5*local[hidden.device]).to(hidden.dtype)
    def hook(module, args, output):
        return (change(output[0]), *output[1:]) if isinstance(output, tuple) else change(output)
    handle = model.model.layers[18].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def validate_snapshot(snapshot, root):
    """CLI verifies pinned metadata AND weight hashes before model allocation."""
    snapshot, root = Path(snapshot), Path(root)
    metadata = json.loads((root/"base_model.json").read_text())
    if metadata["model_id"] != BASE or metadata["revision"] != REVISION:
        raise ValueError("base metadata identity differs")
    for name, expected in metadata["files_sha256"].items():
        if Path(name).name != name or sha256(snapshot/name) != expected:
            raise ValueError("local pinned base asset differs: "+name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="Existing local pinned base snapshot; never downloads")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--steer", action="store_true", help="Opt in to fixed experimental -0.25raw shift")
    parser.add_argument("--device", default="cpu", help="cpu (default) or an explicitly selected CUDA device")
    args = parser.parse_args(); root = Path(__file__).resolve().parent
    validate_snapshot(args.snapshot, root)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
    if (tok.eos_token_id, tok.pad_token_id) != (151645, 151643): raise ValueError("native tokenizer EOS/PAD differ")
    model = AutoModelForCausalLM.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": args.device}).eval()
    validate_model(model)
    prompt = tok.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    torch.manual_seed(42)
    context = steering(model, load_direction(root)) if args.steer else nullcontext()
    with torch.inference_mode(), context:
        output = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=320,
            temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
            eos_token_id=151645, pad_token_id=151643, use_cache=True)
    print(tok.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__": main()
