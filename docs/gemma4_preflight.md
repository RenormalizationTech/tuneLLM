# Gemma 4: QLoRA feasibility preflight

**The text-only Gemma 4 31B preflight passed on one RTX A6000.** An 8,192-token
synthetic training step with 512 supervised response tokens fits with Adam's
state tensors already allocated. A saved synthetic adapter reloads with exact
tensor and reference-logit equality in a fresh process.

This establishes implementation compatibility and measured memory feasibility,
not a QASPER or PRL quality result. **No Gemma task adapter has been trained or
published.** The owner authorized the [short demonstration](short_demonstration.md)
and this preflight first; the next task-training decision remains open.

## Model and training profile

The model is Google's dense instruction-tuned
[`gemma-4-31B-it`, revision `842da3794eaa0b77d5f08bae87a17459d91ff475`](https://huggingface.co/google/gemma-4-31B-it/tree/842da3794eaa0b77d5f08bae87a17459d91ff475).
The anonymous download fetched 62.55 GB of original weights, and both shard
SHA256 hashes match the published LFS metadata. Only the text decoder is loaded;
vision components are excluded.

The profile uses NF4 with double quantization, BF16 compute, SDPA, batch one,
non-reentrant gradient checkpointing, rank 16, alpha 32, dropout zero and Torch
AdamW. All **122,429,440 trainable parameters** are LoRA parameters, across
**410 decoder projections**. The ten global-attention blocks share their K/V
projection, so there are 50 distinct `v_proj` modules rather than 60.

The installed isolated science environment was reused without upgrading it:
Torch 2.12.0+cu130, Transformers 5.23.0, PEFT 0.21.0, bitsandbytes 0.49.2 and
Accelerate 1.12.0. PEFT's preparation casts non-quantized floating weights,
including the tied embedding/output head, to FP32. These choices define the
measured profile; Google's [QLoRA guide](https://gemma.ai/tuning/qlora/) documents
the supported method, while our [shared background](qlora_foundations.md)
explains the low-rank update.

## Measured mechanics

Memory below uses decimal GB, separately from the card's nominal 48 GB capacity.
Allocator-reserved memory is not total device use. Timings are individual
synthetic update observations, not an optimized training benchmark.

| Check | Result |
| --- | --- |
| Resident allocation after NF4 load | 17.97 GB |
| Resident allocation after FP32 preparation and LoRA attachment | 21.28 GB |
| 2,048 tokens / 256 supervised tokens | 11.15 s; 25.92 GB peak allocation |
| 4,096 / 256 | 28.81 s; 32.04 GB peak allocation |
| 8,192 / 256 | 86.24 s; 41.33 GB allocated, 43.43 GB reserved |
| 8,192 / 512, first optimizer step in a fresh run | 84.23 s; 39.85 GB allocated, 42.41 GB reserved |
| 8,192 / 512, with Adam moments already resident | 84.61 s; **40.83 GB allocated, 42.79 GB reserved**, 7.24 GB device memory free after the step |
| Gradients | All 820 adapter gradient tensors finite; only LoRA parameters trainable |
| Fresh-process adapter reload | All 820 saved tensors equal; reference logits `torch.equal`, maximum difference 0 |

The larger-response checks exceed the longest eligible training target (432
tokens). The steady-state check allocates **979,435,520 bytes of FP32 Adam
moments** before the measured full-length update. Different first-step/steady
states and workloads explain why these measurements should not be read as a
controlled response-length scaling comparison.

Response-only logits reduce vocabulary-sized intermediates. On the short
real-model check, compact and full-logit cross-entropy differ by only
0.00000525, within the declared tolerance. Seven CPU tests cover native
formatting, control-token rejection and compact/full loss-and-gradient parity.

## Tokenization and data coverage

Gemma's no-thinking generation prefix includes an empty thought channel that
its complete-conversation formatter omits. The preflight retains that generation
prefix, masks it entirely, then supervises the plain answer and one native
end-of-turn token (106). It does not append generic end-of-document EOS (1).

Only the existing annotation/grounding-eligible **training and development**
inputs were tokenized. No test split was opened by the Gemma preflight. The
separate demonstration displays already-completed Qwen evaluation examples.

| Existing prepared split | Eligible records | Median sequence length | Maximum | Over 8,192 |
| --- | ---: | ---: | ---: | ---: |
| QASPER train | 1,940 | 5,726 | 7,091 | 0 |
| QASPER development | 105 | 5,519 | 6,906 | 0 |
| PRL train | 57 | 6,155 | 6,609 | 0 |
| PRL development | 9 | 6,019 | 6,349 | 0 |

These are the same prepared input bytes used in the Qwen work, retokenized for
Gemma. Existing excerpts were not expanded to exploit the longer advertised
context window. Eligibility here is an input annotation rule, not a Gemma
performance gate.

## Throughput and planning scenarios

A synthetic 7,936-token prompt plus 256 forced new tokens took **72.22 seconds**,
or **3.54 generated tokens/second including prefill**. Forcing the length makes
this a mechanics throughput probe, not an answer-quality example. Two short
ordinary smoke prompts stopped naturally at the native end-of-turn token.
The completed long-generation measurement comes from preserved attempt 01;
the short smoke tests and saved adapter used for fresh reload come from attempt 02.
Their separate executed-source hashes are recorded in the report.

Using the measured 8K training token rate as a simple constant-rate extrapolation:

| Possible training workload | Approximate training-only GPU-hours |
| --- | ---: |
| Match the previous 482 QASPER training examples, one epoch | **About 7** |
| Match the 57 PRL training papers, three epochs | **About 3** |
| All 1,940 eligible QASPER training examples, one epoch | About 29–30 |

These are workload scenarios, **not wall-time forecasts, confidence intervals or
bounds**. Real sequence/response lengths, gradient accumulation, attention
scaling and warmup differ from these synthetic updates. Validation, checkpoint
writing, final generation, retention checks and AI review are excluded. The
preflight does not estimate end-to-end evaluation cost or predict quality gains.

The shorter probes processed about 184 tokens/second at 2K and 142 at 4K, versus
95–97 at 8K. Applying the 8K rate can overstate compute for shorter examples;
other real-run overheads remain. For context, the
[previous Qwen ledger](../results/qwen38_timing_summary_01.json) records QASPER
evaluation at 10.56 GPU-hours versus 5.11 for training, and PRL evaluation at 4.52
versus 2.26. Those are not Gemma estimates; total study work can be substantially
larger than the training-only scenarios.

## Retained evidence and next decision

The first process received external `SIGTERM` (exit 143) during its 512-target
check, after three completed updates. Its cause is unestablished; it is neither
an observed OOM nor a passed larger-response check. Its source, log and partial
receipt are preserved. A shorter independent process completed the 512-target
update and fresh reload, followed by the separate steady-state measurement.

The [compact report](../results/gemma4_preflight_01.json) binds the checkpoint,
input/source hashes, per-probe measurements and preserved receipts. The
[main preflight](../scripts/gemma4_preflight.py),
[steady-state check](../scripts/gemma4_steady_memory.py) and
[report builder](../scripts/summarize_gemma4_preflight.py) retain the exact code.
Raw logs, synthetic adapter weights and original model files remain local.

The engineering result supports considering a bounded **31B QASPER comparison
first**, followed by an independent PRL adapter if chosen. Future comparisons
need Gemma's own unadapted and development-selected prompt baselines. Reused
project evaluation sets must be labelled as such. The completed Qwen studies
and their outcomes remain unchanged; no Gemma task-training run is queued.
Any chosen training run should retain batch one and response-only logits while
measuring validation/save peaks and long-run allocator behaviour separately.
