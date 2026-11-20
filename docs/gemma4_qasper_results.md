# Gemma 4 31B: QASPER QLoRA results

**The QASPER adapter passes every registered task and retention gate on the
project's custom text-only evaluation.** Strict primary Answer F1 is **0.640**
against a **0.000 format floor**. In the lenient diagnostic, Answer F1 improves
from **0.334 to 0.640**, with gain interval **[0.251, 0.362]**.
This is a reused cross-family project evaluation, not a fresh held-out study or
the full official QASPER benchmark.

The adapter release is being packaged and independently reload-verified. The
completed quality result below is separate from that release verification.
The [recorded demonstration](short_demonstration.md#gemma-qasper-the-same-recorded-question)
already shows the same saved question used for Qwen, with no live inference.

## Format compliance and answer quality

All **403 baseline answers** used Markdown code fences. The frozen strict parser
rejects that format, giving a strict Answer F1 of zero. That is a format floor,
not evidence that the unadapted model cannot answer paper questions. The lenient
diagnostic recovers those answers and separates part of the content comparison
from format compliance.

| Primary subset, 309 questions | Unadapted base | Development-selected prompt | Adapter |
| --- | ---: | ---: | ---: |
| Strict Answer F1, registered primary metric | 0.000 | 0.000 | 0.640 |
| Lenient Answer F1, diagnostic | 0.334 | 0.334 | 0.640 |
| Lenient Evidence F1, diagnostic | 0.723 | 0.723 | 0.717 |

The three fixed prompts tied at strict development F1 zero on 35 questions. The
registered tie rule selected the base prompt; the prompted final arm is therefore
an alias of the baseline, with no duplicate final generation.
Three frozen prompts do not establish the best achievable Gemma-specific
prompting baseline.

The strict Answer F1 gain is **0.640**, with 95% paired paper-bootstrap CI
**[0.588, 0.692]**. The lenient gain is **0.306**, CI **[0.251, 0.362]**. Both
use 10,000 resamples of the 109 papers carrying the 309 primary questions.
The lenient comparison is descriptive and cannot replace a failed registered
gate. Here the registered gates passed. **Evidence superiority was not
established.** On the 267-question evidence-strict subset, lenient Evidence F1
is 0.734 for the baseline and 0.747 for the adapter; the primary-subset evidence
comparison above is slightly lower for the adapter.
Answers became much shorter: the median non-abstaining answer length fell from
35 to 3 words. Overlap gains can reflect concision and reference matching;
improved underlying comprehension was not established.

The adapter produced **5 strict-invalid answers out of 403** (1.24%, below the
registered 2% limit). It marked 4 of 293 answerable primary questions
unanswerable (1.37%, within the registered increase limit). No final QASPER
answer reached the 512-token cap. These counts, parser definitions and intervals
come from the [canonical result](../results/gemma4_qasper_final_01.json) and were
independently reproduced in the [outcome audit](../results/reviews/gemma4_qasper_final_audit_review_02.md).

## Retention checks

| Fixed task | Base correct | Adapter correct | Allowed net loss | Gate |
| --- | ---: | ---: | ---: | --- |
| General multiple choice, 128 items | 108 | 105 | 5 | Pass |
| GSM8K numeric fallback, 128 items | 126 | 126 | 5 | Pass |
| IFEval strict, 32 items | 26 | 26 | 3 | Pass |

These are fixed point-estimate regression limits on small samples, not
non-inferiority tests or proof that general ability is unchanged. GSM8K strict
format diagnostics are retained in the original score artifacts; the registered
gate uses numeric fallback, matching the earlier project method.

## Model and training

The base is Google's instruction-tuned
[`gemma-4-31B-it`, revision `842da3794eaa0b77d5f08bae87a17459d91ff475`](https://huggingface.co/google/gemma-4-31B-it/tree/842da3794eaa0b77d5f08bae87a17459d91ff475).
Only its text decoder is loaded; vision components are excluded. The shared
[LoRA/QLoRA background](qlora_foundations.md) explains the low-rank update and
frozen quantized base.

| Setting | Executed study |
| --- | --- |
| Base representation | On-load NF4 with double quantization and BF16 compute |
| Train / development | Same 482 / 101 selected examples as the Qwen task study |
| Adaptation | 410 decoder projections; rank 16, alpha 32, dropout 0.05; 122,429,440 trainable LoRA parameters |
| Optimization | One epoch, 61 steps; microbatch 1, accumulation 8, final group 2; AdamW, learning rate 5e-5, weight decay 0.01 |
| Checkpoint selection | Lowest token-weighted development NLL at steps 21 / 41 / 61; selected **61** |
| Precision | PEFT FP32 preparation and Adam states, outer BF16 autocast, FP32 cross-entropy |
| Runtime | SDPA, one GPU per job, batch 1, greedy cached generation, repetition penalty 1.0 |
| Boundaries | Native no-thinking prefix, response-only supervision, one terminal turn token 106; generation stops on 1 or 106 |
| Context and answer cap | At most 8,192 total tokens; QASPER generation cap 512; no truncation |
| Software | Torch 2.12.0+cu130, Transformers 5.23.0, PEFT 0.21.0, bitsandbytes 0.49.2, Accelerate 1.12.0 |

Training used **4.84 GPU-hours**. The complete job ledger sums to **15.60
GPU-hours**, with about **8.96 hours wall time** using two RTX A6000 GPUs.
Times include model loads and coexistence with the recorded small idle CUDA
context from another project; they are not exclusive-device benchmarks.
The [timing ledger](../results/gemma4_qasper_timing_01.json) binds each completed
job's elapsed-time record. Wall time ends at the original analysis stop; the
later CPU-only reporting correction is excluded.

The earlier [feasibility preflight](gemma4_preflight.md) used dropout zero and
no outer autocast. Its synthetic memory/timing observations describe that
earlier profile. The actual study's no-update longest-row check used 7,044
tokens, took 49.12 seconds and had all 820 adapter gradients finite; peak
allocation/reservation was 36.97 / 39.31 decimal GB, before Adam moments were
allocated. The separate steady-state preflight measured those moments.

Short native full/compact forward logits matched exactly. Shared-logit loss and
indexing-gradient checks also matched. Native BF16 full-versus-compact parameter
gradients differed, while repeating the compact path gave identical gradients.
That diagnostic is consistent with shape-dependent BF16 rounding; no native
gradient-equivalence claim is made. The
[registered plan](gemma4_qasper_plan.md) and
[source/mechanics review](../results/reviews/gemma4_qasper_source_review_01.md)
retain the failed early check and the validation redesign before training.

## Use the adapter

Release verification is in progress. The verified adapter revision, manifest
hash and tested download/loading commands will be recorded here and in the
[release catalog](../releases/README.md) when the public checks are complete.
The scientific results above are complete; inference-package verification is
a separate check. The package will contain the adapter and its local helper;
obtain the pinned Google base separately. It does not retrieve papers.

## Evidence, correction and scope

The [original registration](../results/protocols/gemma4-qasper-01.json) preceded
training and generation. All GPU jobs completed on 2026-11-20. The first final
analysis stopped because it expected an outdated multiple-choice method label,
although both retention runs correctly included Gemma's native BOS token.
The [registered CPU-only correction](../results/protocols/gemma4-qasper-analysis-amendment-01.json)
changed that expected description only; every source, job output, statistical
method and threshold from the study remains preserved. Training and generation
were not repeated. The [amendment review](../results/reviews/gemma4_qasper_analysis_amendment_review_01.md)
and [final independent audit](../results/reviews/gemma4_qasper_final_audit_review_02.md)
bind the corrected report and original failure evidence.

The primary subset contains 309 questions with complete text, from the 403
sampled questions; 267 form the evidence-strict subset. These questions were
previously used in Qwen evaluations. Public papers may be in pretraining. There
is one model configuration and seed, and only 16 primary questions are
unanswerable. Answer F1 rewards reference overlap and concision; supported extra
details can lower it. The selected examples are illustrations, not representative
samples or factuality validation. This study supports a within-Gemma adaptation
result under its registered method, not a broad Gemma-versus-Qwen ranking.

Gemma PRL adaptation is a separate possible follow-up; it was not part of this
completed QASPER study.
