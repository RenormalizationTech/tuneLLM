# 27B quantization: NF4 and Q4_K_M

The text-only Qwen3.8-27B NF4 and Q4_K_M packages are public and verified.
A small fixed screen compares them with a local BF16 GGUF reference.
These are descriptive workflow measurements, not full benchmark results or
evidence that the representations are equivalent.

The [quantization background](quantization_foundations.md) explains the formats
and error accounting. The separate [27B QLoRA results](qwen38_qlora_results.md)
evaluate adapters trained from the NF4 base.

## Representations and storage

All representations derive from `Qwen/Qwen3.8-27B` revision
`37561816e66b5b9153b1b64c10968c18c1e6f4a0`. Vision and MTP components are
excluded; these packages do not provide multimodal inference.

| Representation | Weight file size, decimal GB | Runtime and role |
| --- | ---: | --- |
| NF4 | 17.65 | Native Transformers/bitsandbytes; published base and QLoRA starting point |
| Q4_K_M GGUF | 16.74 | llama.cpp; published text-only base |
| BF16 GGUF | 53.81 | llama.cpp; local reference, not a published package |

Q4_K_M was converted from the original higher-precision source, **not the NF4
checkpoint**, using standard quantization without an importance matrix.
The [NF4 smoke](../results/qwen38_27b_nf4_execution_01.json) measured 18.02 GB
peak PyTorch allocated memory on one A6000 for short prompts. This is distinct
from weight-file size, total device memory and long-context/training requirements.
The [GGUF card](../releases/Qwen3.8-27B-tuneLLM-GGUF/README.md) records the
conversion, backend and excluded components.

## Fixed-subset capability screen

The [registered screen result](../results/qwen38_27b_quantization_screen_01.json)
contains 32 general MMLU items, 32 GSM8K questions and 35 QASPER development
questions. The QASPER complete-text primary subset contains 27 questions;
its answer and evidence F1 below use that denominator.

| Representation | MMLU /32 | GSM8K fallback /32 | GSM8K strict /32 | QASPER answer F1 /27 | QASPER evidence F1 /27 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NF4 | 30 | 26 | 26 | 0.344 | 0.524 |
| Q4_K_M | 29 | 27 | 26 | 0.334 | 0.483 |
| BF16 GGUF | 29 | 26 | 25 | 0.403 | 0.513 |

All three GSM8K arms have six token-limit hits out of 32. Last-number fallback
can reward format-noncompliant answers; strict scoring is shown separately.
The QASPER scores use custom text-only inputs and paragraph-ID evidence,
not the full official benchmark or the adapters' final evaluation set.

The screen uses greedy batch-one generation with a 512-token cap. Recorded
process charges, including loading, were **0.88 GPU-hours for NF4**,
**0.18 for Q4_K_M** and **0.75 for BF16 GGUF**. BF16 used two GPUs, while
the other arms used one. Different backends, hardware use and concurrent jobs
limit timing comparisons; these are not optimized throughput or isolated
time-to-first-token benchmarks. The [timing summary](../results/qwen38_timing_summary_01.json)
separates conversion, screen and adapter work.

## What this comparison establishes

NF4 versus Q4_K_M compares complete workflows across Transformers and llama.cpp;
backend differences prevent attributing every change to quantization. BF16 GGUF
provides a same-backend reference for Q4_K_M. There is **no BF16 Transformers
control**, so the NF4 representation's isolated effect is not measured.

The native NF4 screen keeps floating modules in BF16. The matched PEFT adapter
baselines prepare those modules in FP32. Do not pool their scores or interpret
their differences as adapter gains. Small public subsets may have appeared in
pretraining; close counts do not establish equivalence or broad capability
retention.

The original strict NF4 cached/full-prefix logit comparison failed, with maximum
difference 0.125. Short token replay and successful reload do not resolve that
numerical-equivalence failure. The recorded generation paths remain fixed.

## Public packages and reproduction

- [NF4 base](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4),
  current revision `586bb02218d196f485c321bbde3348677c63fc6a`;
  [anonymous download and GPU verification](../results/qwen38_nf4_public_verification_01.json).
- [Q4_K_M GGUF](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-GGUF),
  current revision `b4351e7d6f521b71cb3842116cdff3cae00d7317`;
  [anonymous download and GPU verification](../results/qwen38_gguf_public_verification_01.json).

The [release catalog](../releases/catalog.json) distinguishes current card
revisions from the earlier GPU-verified revisions; runtime hashes remain
verified. Use the [reproduction guide](reproduce.md#separate-qwen38-27b-nf4-environment)
for the separate NF4 environment and each published card for backend commands.
The [historical implementation plan](qwen38_release_plan.md) preserves the
original decisions, while the [project closeout](../results/qwen38_project_closeout_02.json)
records the completed scope. No additional quantization is scheduled.
