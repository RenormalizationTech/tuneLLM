---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
- experimental
- activation-steering
- refusal-steering
library_name: transformers
---
# Qwen2.5-7B refusal steering experiment

## At a glance

Experimental activation-direction subtraction on Qwen2.5-7B-Instruct. In a
selected 13-prompt benign diagnostic, refusals fell **3→1** and useful valid full
answers rose **9→11**, with **two useful rescues and one repetitive output**.
This is a case-level demonstration; the original development decisions remain
null and no confirmatory final/retention evaluation ran.

This artifact is part of **[tuneLLM](https://github.com/RenormalizationTech/tuneLLM)**, a public research portfolio
covering quantization, LoRA/QLoRA, refusal steering and trait steering.
[Results and saved examples](https://github.com/RenormalizationTech/tuneLLM/blob/main/docs/showcase.md) ·
[Reproduction and environment setup](https://github.com/RenormalizationTech/tuneLLM/blob/main/docs/reproduce.md) ·
[Methods and references](https://github.com/RenormalizationTech/tuneLLM/blob/main/docs/references.md).

Related releases: [GGUF](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-GGUF),
[NF4 base](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-NF4),
[physics QLoRA](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA),
[trait controls](https://huggingface.co/RenormalizationTech/qwen2.5-7b-gcad-steering-controls),
[refusal experiment](https://huggingface.co/RenormalizationTech/qwen2.5-7b-refusal-steering-experiment).
Each release has its own verified environment; activation steering and adapter
training use different mechanisms.

Separate 27B work: the **text-only NF4 load/save/reload smoke is complete** on
Qwen3.8-27B, using two simple prompts. It is a functional check, not a quality
benchmark. A one-step cache check exceeded its numerical tolerance; that result
is retained. The NF4 checkpoint remains local; 27B sarcasm steering and QLoRA
training are planned. [Recorded scope](https://github.com/RenormalizationTech/tuneLLM/blob/main/results/qwen38_27b_nf4_execution_01.json).

## Artifact and limitations

Completed experimental demonstration with case-level effects, **not a validated
reliable controller**. No confirmatory final or retention run qualified or ran.
This contains a raw activation direction and runtime, not base-model weights,
an adapter, or a selectively safe refusal policy.

Code is Apache-2.0. Vector artifacts are conservatively CC BY-SA 3.0; the card's
Apache metadata describes code, not an override of vector/dataset terms.
See VECTOR_LICENSE.md and DATA_SOURCES.md for separate terms and attribution.

## Fixed mechanism

Pinned base Qwen/Qwen2.5-7B-Instruct at
`a09a35458c702b33eeacc393d103063234e8bc28`, native BF16/SDPA, 28 blocks, 3584 hidden dimensions.
Raw FP32 harmful-minus-benign means use 100 AdvBench/Dolly pairs, zero-based block 18 output
at the last real prompt token. The example intervention subtracts 0.25 × the raw vector from
that same block's output at every position, including prompt prefill and cached
decoding. FP32 arithmetic then restores activation dtype. It reproduces the old
−0.5 hook applied to half the raw vector. No weight changes or training.
All-block projection was also measured; it is reported below but is not a
retained runtime option. This is an adapted block-output workflow, not a claim
of exact reproduction of the original author pipeline.

## All development arms

30 benign and 30 source-unsafe prompts per arm. Useful/correct is an AI rubric
field; source-unsafe full compliance measures willingness, not answer quality.

| Arm | Benign refusals /30 | Benign useful /30 | Benign degenerate | Unsafe valid full /30 | Unsafe refusals /30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 3 | 20 | 0 | 2 | 28 |
| Subtract −0.5 | 0 | 13 | 1 | 13 | 17 |
| All-block projection | 2 | 19 | 0 | 4 | 26 |
| Subtract −0.25 | 0 | 21 | 0 | 4 | 26 |

Baseline labels were reused from the original joint 180-item grading packet;
the −0.25 candidate was judged in a separate packet. Grading context is an
additional comparison limitation.

The original −0.5 study closed null due to usefulness loss and degeneration.
All-block projection did not reach the unsafe-contrast gain gate. The single
−0.25 follow-up also closed null under that original gate. These decisions are
preserved. Stronger subtraction reduced unsafe refusals too: no selective
safety claim is supported. No final/retention results are implied.

## Selected 13-case benign diagnostic

These 13 were selected from earlier 4B refusal cases after excluding four prompts
seen during dose development. They are not an unbiased or confirmatory held-out
benchmark. Four seen pairs are reported separately in the frozen JSON, not pooled.

Baseline→−0.25: refusals **3→1**, AI useful-field counts
**10→12**. The stricter conjunction
valid full compliance AND useful was **9→11**; this differs from the useful-field
count because an answer can include a refusal. Paired useful rescues were
`xstest-151` and `xstest-168`, with no newly refused formerly compliant case.
**One repetitive output (`xstest-153`) appeared**, so refusal reduction alone
does not establish quality preservation. Both conditions hit the cap on 7/13.
Judgments use one AI judge family, not human factual verification; robustness,
general reliability and deployment quality are unestablished. One source label was flagged as questionable
and retained unchanged in the frozen results. Original null records remain intact.

## Offline example (baseline by default)

To download this small bundle with pinned file verification, follow the public
[artifact setup guide](https://github.com/RenormalizationTech/tuneLLM/blob/main/docs/reproduce.md#download-the-published-refusal-experiment-small-files-only).
Then use an already cached pinned base snapshot. The CLI performs no model download.
The CLI checks all pinned metadata/weight hashes before loading; allow time for
local disk reads. The example is single-prompt, not an exact replay of the original batched runs (batch sizes 8/5).

```bash
python refusal_runtime.py --snapshot /path/to/pinned/snapshot --prompt "Explain a simple board-game strategy."
python refusal_runtime.py --snapshot /path/to/pinned/snapshot --prompt "Explain a simple board-game strategy." --steer
```

CPU is the default device; `--device cuda:0` explicitly opts into a CUDA device.
Greedy seed 42, cap 320, repetition penalty 1.0, native default-system template,
tokenizer EOS 151645 / PAD 151643. Checkpoint defaults dual EOS / repetition penalty 1.05 are overridden
to match the experiment. No thinking-mode claim. Base weights are BF16.

Python use: `from refusal_runtime import load_direction, steering`; load the
pinned base, then wrap inference in `steering(model, load_direction())`.
Safetensors key is `direction`. `load_direction(use_pt=True)` safely loads the
byte-identical frozen `.pt` with `weights_only=True`; that file stores a root
Tensor, not a dictionary. File/tensor hashes and geometry are verified.

`environment.json` records the current installed environment; requirements pin
the Python packages. Tested Torch build is 2.6.0+cu124 (CUDA 12.4); choose the
appropriate Torch 2.6.0 wheel for the execution device. Release checks were CPU
mechanics/serialization checks, not a new 7B inference or GPU reload test.

Frozen setting/results are in `results/`. No raw prompts, answers, reasons,
grading keys, model weights or credentials are included. See `provenance.json`
for original source hashes and `base_model.json` for pinned base asset hashes.
