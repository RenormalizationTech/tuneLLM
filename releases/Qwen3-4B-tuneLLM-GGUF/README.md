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
