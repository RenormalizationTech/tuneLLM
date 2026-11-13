---
license: apache-2.0
base_model: Qwen/Qwen3-4B
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen3
- bitsandbytes
- 4-bit
- nf4
---
# Qwen3-4B tuneLLM NF4 base

Standalone **unadapted** NF4 checkpoint for the tuneLLM learning workflow. This
repository is private. No LoRA adapter or physics training is included here.

The source is [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), pinned to
`1cfa9a7208912126459214e8b04321603b3df60c`. We loaded it with bitsandbytes NF4,
double quantization and BF16 computation, then applied PEFT k-bit preparation.
Nonquantized floating modules are FP32, so the checkpoint is about **3.21 GiB**,
rather than four bits for every model parameter.

We saved this base before attaching the physics-writing LoRA adapters. A fresh
stored-quantization reload, with requantization explicitly forbidden, matched all
**1,659 state tensors**, two fixed last-token-logit probes and their generated
IDs exactly. See `verification.json` for file hashes, settings and the precise
scope. The original local manifest includes an omitted raw provenance file;
`export_manifest.json` covers the actual release payload, excluding itself.
These are serialization checks, not broad capability claims.

## Load

The verified environment uses PyTorch 2.6.0+cu124, Transformers 4.57.6,
bitsandbytes 0.49.2 and PEFT 0.18.1 on an RTX A6000. A compatible CUDA GPU is
required for this verified recipe.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "RenormalizationTech/Qwen3-4B-tuneLLM-NF4"
# Pin revision to the publication commit recorded in the project report.
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(
    repo, device_map={"": 0}, torch_dtype=torch.float32,
    attn_implementation="sdpa",
).eval()
```

Load the stored quantization configuration; do not apply a second quantization.
For QLoRA, prepare this base for k-bit training and attach the separately saved
adapter. This NF4 representation is distinct from our llama.cpp Q4_K_M GGUF.
The original model license is Apache 2.0; its license is included.
