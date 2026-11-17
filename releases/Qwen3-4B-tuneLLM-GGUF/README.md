---
base_model: Qwen/Qwen3-4B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - quantization
  - tunellm
---

# Qwen3-4B tuneLLM GGUF

## Portfolio summary

**Quantization experiment:** converted the pinned Qwen3-4B checkpoint to BF16
GGUF, Q8_0 and Q4_K_M, then measured file size, generation throughput and fixed
capability subsets. Weight sizes were 7.50 / 3.99 / 2.33 GiB; the short A6000
generation benchmark measured 79 / 131 / 186 tokens/s. These are microbenchmark
results, not production latency. Capability counts and reproduction instructions
below define the evaluation scope. No weights changed in this documentation update.

This release is part of **tuneLLM**, a research portfolio covering quantization,
LoRA/QLoRA, refusal steering and trait steering. Related public artifacts:
[GGUF](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-GGUF),
[NF4](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-NF4),
[physics QLoRA](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA),
[trait controls](https://huggingface.co/RenormalizationTech/qwen2.5-7b-gcad-steering-controls), and
[refusal experiment](https://huggingface.co/RenormalizationTech/qwen2.5-7b-refusal-steering-experiment).
The GitHub project is currently private; each HF release provides its own public
usage/evaluation material. Refusal and trait steering are activation interventions,
not adapter training. Qwen3.8-27B quantization, sarcasm and QLoRA are planned,
not completed results.


An unofficial learning release of Qwen3-4B converted to BF16 GGUF and quantized
to Q8_0 and Q4_K_M. These files exercise the conversion and evaluation workflow;
they do not introduce new training or claim improved model capability.

Upstream model: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), revision
`1cfa9a7208912126459214e8b04321603b3df60c`.
Original model and copyright belong to their upstream authors. The upstream
Apache-2.0 license is included in `LICENSE`. Changes by tuneLLM consist of
GGUF conversion and numerical quantization using llama.cpp commit
`8ca5a8008b57aa39f2601546c0d07912045080a5`. No importance matrix was used.

The implementation and reproduction notes are included in
`reproduction-dab1dae.zip`, a source archive of local workflow commit `dab1dae`.
This makes the workflow available independently of whether that commit has been
pushed to the [project repository](https://github.com/RenormalizationTech/tuneLLM).
The manifest records the full implementation commit, source-archive hash, model
file hashes and evaluation settings. Training datasets and intermediate model
checkpoints are not included.

## Files and initial measurements

| File | Size (GiB) | Generation tokens/s | Small-text perplexity |
| --- | ---: | ---: | ---: |
| `Qwen3-4B-BF16.gguf` | 7.498 | 79.1 | 2.6299 |
| `Qwen3-4B-Q8_0.gguf` | 3.986 | 130.5 | 2.6282 |
| `Qwen3-4B-Q4_K_M.gguf` | 2.326 | 186.0 | 2.6608 |

Throughput is from a short llama-bench test on one NVIDIA RTX A6000: three
repetitions, 64 generated tokens, full GPU offload, flash attention enabled and
four CPU threads. Prompt processing was measured separately with 128 prompt
tokens. These figures describe this microbenchmark, not production throughput,
chat latency, or a comparison with Transformers.

Perplexity used llama-perplexity on the same text for every precision: the first
32 prepared GSM8K test question/solution pairs, context 512, four chunks,
batch/microbatch 512. The tool-reported uncertainties were about 0.175–0.177.
This small sample does not establish general capability preservation. GSM8K is
an MIT-licensed dataset by Cobbe et al. (2021),
[Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168).
Dataset revision: `740312add88f781978c0658806c59bc2815b9866`.

The observed perplexity differences are smaller than the tool's reported
uncertainty on the individual estimates. These estimates do not establish a
capability ranking or equivalence between formats.

All three formats passed exact input-token parity against Transformers on an
eight-item project-authored smoke fixture. CPU scores were 7/8, 7/8 and 8/8
respectively. A one-item difference is not evidence that Q4 is more capable.
BF16 and Q8_0 produced token-identical responses to Transformers on this fixture.
Complete compact measurements are included as `evaluation.json`.

## Completed fixed-subset benchmark screen

The later screen evaluated the **same published weight bytes** on 384 items per
condition. This is separate from the initial eight-item smoke check and tiny
perplexity/microbenchmark above; the historical `evaluation.json` and provenance
remain unchanged. The additive `benchmark_summary.json` records the new screen,
paired counts, evaluated source commits and scorer hashes.

| GGUF condition | General MMLU /128 | Physics MMLU /96 | GSM8K /128 | IFEval /32 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 92 | 62 | 119 | 25 |
| Q8_0 | 89 | 61 | 121 | 25 |
| Q4_K_M | 89 | 61 | 116 | 24 |

General/physics are custom zero-shot MMLU four-choice likelihood subsets, not
full standard leaderboard scores. GSM8K uses last-number fallback correctness;
strict final-line correctness and truncation are reported separately. IFEval
uses its pinned official prompt-level strict scorer. Dataset revisions:
MMLU `c30699e8356da336a370243923dbaf21066bb9fe`,
GSM8K `740312add88f781978c0658806c59bc2815b9866`, and
IFEval `966cd89545d6b6acfd7638bc708b98261ca58e84`; official scorer source
`f8d6d14a1fc6eaef606d7d1e1fa4e96ee09afb81`.

The screen uses llama.cpp `8ca5a8008b57aa39f2601546c0d07912045080a5`,
greedy non-thinking decoding, seed 42, batch one, and output budgets of 512 tokens
for GSM8K and 1024 for IFEval. Input-token parity is checked against the pinned
upstream tokenizer. The implementation commits and source SHA256s are in the
new report; the original reproduction archive still describes the initial cycle.

| Published weight | SHA256 |
| --- | --- |
| `Qwen3-4B-BF16.gguf` | `5dbbfbb5ccea81a811ef8346248bbf8f808c94477feb84415699d632798925b9` |
| `Qwen3-4B-Q8_0.gguf` | `4b2b96c36712c42d902fb23cb361ed8ebf838a4342f77cedb87afe99e291775f` |
| `Qwen3-4B-Q4_K_M.gguf` | `f7a3480949df3ae406a9661aae19e34b6c6ae344e7b6b8edd38b6185b786f339` |

These are small fixed subsets with unknown base-pretraining exposure. The paired
tests are exploratory and unadjusted; equal totals or nonsignificant differences
do not establish equivalence, preserved general capability, or a universal
quantization ranking. IFEval language-detection random-state sensitivity limits
close comparisons. No full benchmark or production-performance claim is made.

## Use

Use a llama.cpp build that supports Qwen3 and these GGUF formats. For example:

```bash
llama-cli -m Qwen3-4B-Q4_K_M.gguf --single-turn --reasoning off \
  --chat-template-kwargs '{"enable_thinking":false}' \
  -p "Explain why the sky is blue."
```

The pinned build above supports these flags. Other releases may expose different
flags; consult their help. The saved chat template comes from the upstream
tokenizer. Non-thinking mode must be explicit for comparisons with the included
evaluations.

The experiments used greedy decoding for a reproducible pipeline comparison.
The upstream model card recommends sampling and warns that greedy generation
can repeat. Consult the upstream card for normal conversational generation
settings, supported languages and intended use.

## Limitations

This is a small educational quantization experiment. It does not establish
accuracy on comprehensive benchmarks, robustness across prompts, safety
performance, or suitability for a particular application. No domain adaptation
or activation steering was applied. Quantization and runtime changes can alter
individual answers even when input token IDs match.

The full source model's capabilities and limitations also apply. Report this
release as an unofficial derivative of the pinned Qwen model rather than an
official Qwen checkpoint.
