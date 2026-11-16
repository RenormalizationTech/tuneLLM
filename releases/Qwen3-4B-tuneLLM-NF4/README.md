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

Standalone **unadapted** NF4 checkpoint for the tuneLLM learning workflow.
No LoRA adapter or physics training is included here.

This repository was initially published privately. Historical cards,
publication receipts, reproduction notes, provenance and manifests retain their original private
status as records of the earlier publication; they do not describe current
repository visibility.

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
revision = "e9af208e44e083b77702a71d8aa65b7d97f78ce6"
tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
model = AutoModelForCausalLM.from_pretrained(
    repo, revision=revision, device_map={"": 0}, torch_dtype=torch.float32,
    attn_implementation="sdpa",
).eval()
```

Load the stored quantization configuration; do not apply a second quantization.
For QLoRA, prepare this base for k-bit training and attach the separately saved
adapter. This NF4 representation is distinct from our llama.cpp Q4_K_M GGUF.
The original model license is Apache 2.0; its license is included.

## Completed fixed-subset screen

The additive `benchmark_summary.json` records the later 384-item screen and
paired results. Historical verification, provenance and manifests are unchanged.

| Representation | General MMLU /128 | Physics MMLU /96 | GSM8K /128 | IFEval /32 |
| --- | ---: | ---: | ---: | ---: |
| Saved NF4 base | 85 | 63 | 117 | 24 |
| BF16 Transformers base | 92 | 63 | 119 | 24 |

General/physics use custom zero-shot MMLU letter likelihood, GSM8K uses numeric
last-number fallback, and IFEval uses official prompt-level strict scoring.
Generation is greedy, non-thinking, seed 42, batch eight, with 512/1024-token
GSM8K/IFEval budgets; MMLU likelihood uses batch 16. Dataset/scorer revisions,
source commits, hashes, strict math counts and truncation are in the new report.
The saved NF4 representation uses FP32 nonquantized modules; the BF16 control
keeps frozen base weights BF16. These are workflow representations, not an
isolated quantization-only intervention. Small subsets, runtime effects and
IFEval language-detection random-state sensitivity limit interpretation. Paired
tests are exploratory and unadjusted; neither equal counts nor nonsignificance
establish equivalence or a universal quantization verdict.
