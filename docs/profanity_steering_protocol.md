# Isolated profanity steering on Qwen3-4B

Design reviewed and accepted with Claude, 2026-11-14. The
[rating calibration passed](persona_rating_calibration_completion_execution.md).
This is the first isolated persona activation experiment; rudeness and sarcasm
are measured as crosstalk, not simultaneously steered. No physics adapter or
14B model is used. Implementation, fixture review, CPU checks and a committed
registration must precede judging or GPU execution.

## Method and fixed model

Use unadapted `Qwen/Qwen3-4B`, revision
`1cfa9a7208912126459214e8b04321603b3df60c`, offline BF16 on GPU 0, UUID
`GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b`. Keep existing jobs untouched;
require at least 12,000 MiB free and utilization at most 10% before a model load.
Record packages, attention backend, device, dtype, weights and tokenizer hashes.
Use native nonthinking chat rendering, helpful system prompt, greedy decoding,
seed 42, generation batch size 4 and cap 256 for development/confirmation.

The method is inspired by
[contrastive activation addition](https://aclanthology.org/2024.acl-long.828/),
using differences of response-token means rather than the paper's exact
extraction protocol. Both authored answers are teacher-forced under the **same
neutral context**; no different system instructions enter the paired extraction.
Model weights stay fixed.

## Authored data and eligibility

Freeze 32 neutral/profane answer pairs, 16 development prompts and 24 confirmation
prompts before any calls. All are disclosed synthetic everyday tasks with
impersonal or fictional situations, without slurs, threats or real-person abuse.
Prompts do not request profanity. Extraction has 16 topic/template families,
development eight and confirmation 12, two items per family. Keep families and
topics disjoint across splits and from the calibration fixture. Eight confirmation
items have short checkable factual answers; every task has declared content
requirements and references for evaluation.

Each answer pair preserves the same facts/task with friendly or emphatic swearing
as the intended difference. Vary word position and function. Use at least four
lexical word families; group variants such as `fuck`/`fucking` together. No family
may occur in more than one third of retained pairs. Reuse the fixed non-slur
calibration lexicon; do not expand it after outputs are seen.

Blindly rate all 64 authored answers once before extraction. Retain every pair
meeting the fixed criteria, without selecting the most favorable subset:

- Profane answer has a lexical hit; neutral has none.
- Profane rudeness and sarcasm scores are at most 1; neutral scores on all three
  axes are at most 1. Report ambiguous scores of 1 separately.
- Both answers fulfill the task and preserve the declared content, with no
  contradiction, incoherence or gross failure.
- Profane/neutral token-length ratio is 0.85–1.15 under the pinned tokenizer.

At least 24/32 pairs and the retained diversity rule must pass. Otherwise stop;
do not regenerate, rebalance or loosen the rule within this study.

## Extraction, strength units and serialization

Capture decoder-block outputs at **zero-based blocks 17 and 23**. Exclude prompt,
template, padding and end-of-turn tokens. Average assistant response tokens in
FP32, difference profane minus neutral for each pair, then average pairs equally.
Let this raw vector be `d_l`; save its norm and unit direction `u_l`.

To avoid dilution of raw contrast norms by nearly identical responses, compute
`R_l` as the median L2 norm of the eligible neutral response-mean activations.
The three intervention magnitudes are **0.05, 0.10 and 0.20 times R_l**:
`h_l' = h_l + beta * R_l * u_l`. Record raw/relative norms, R, actual dtype and
resulting magnitudes. A zero/nonfinite direction or scale stops extraction.
Also record the mean per-token residual norm at each block and the intervention
as a fraction of that norm, for interpretation only; it adds no settings.
These are fixed rules derived solely from extraction statistics, not generated
outputs. Save vectors/metadata as safetensors; require exact tensor round-trip.
Commit the extraction/strength artifact hashes before development generation.

Add at the last real prompt position on prefill and every cached generated-token
step, using the existing cache-aware single-block hook. No all-block intervention,
negative strength or adaptive block search belongs to this pilot. Preflight:
four fixed development prompts at cap 32, ordinary versus actual zero-vector hook
(eight generations), requiring exact token parity, then two short unscored
nonzero probes at block 17 and beta 0.05. Record hook counts, hidden/logit change
and cleanup; probes supply no behavioral selection or labels.

## Development and lock

Generate baseline and six settings (two blocks × three magnitudes) on the same
16 development prompts: 112 responses. Rate every output on all three traits,
content fulfillment, coherence and narrow gross failures. Lexical profanity
presence is the target diagnostic; the primary success count is **usable profane
output**: a lexical hit, content fulfilled, coherent, no gross failure and no
empty/mechanical-degeneration flag. Report raw lexical presence separately,
including judge profanity scores at least 2 without a lexical hit, since the
fixed lexicon misses some variants/spellings. Report score-1 ambiguity, length,
EOS/cap/empty and mechanical degeneration separately.

A setting qualifies only if, relative to baseline:

- Mean rudeness and sarcasm each increase by at most 0.5, and each axis's clear
  presence count increases by at most one item.
- Content-fulfilling answers decrease by at most one; incoherent answers and
  gross failures each increase by at most one.

Among qualifying settings choose the highest usable-profane count, then the
lower beta, then block 17. Require at least 8/16 profane outputs and at least
seven more than baseline (both using usable-profane counts, a gain of at least
40 percentage points on 16 cases).
Otherwise report a development null and skip confirmation/retention.

Lock the setting and output/input hashes in a committed selection before final
generation. Reload its vector and replay the first four development prompts with
the same batch membership, padding, cap and settings; require exact generated
token IDs against that original development batch. No reselection after locking.

## Confirmation and small capability screen

On 24 held-out prompts generate baseline, locked steer and one norm-matched
random-direction control: 72 responses. The random unit direction is fixed from
CPU seed `4242 + layer`, saved before development, and uses the locked magnitude
at the same block. Do not draw alternatives after observing behavior.

All confirmation gates are required:

- Steer: usable profanity in at least 12/24 outputs and at least ten more than
  baseline (at least 40 percentage points). Random control: raw lexical profanity
  in at most 2/24 outputs, even if its outputs fail the usability checks.
- Mean off-target trait increases at most 0.5 each; clear off-target presence
  increases at most two cases each.
- Content fulfillment decreases at most two; incoherent answers and gross
  failures each increase at most two relative to baseline.
- At least 7/8 factual items correct and no more than one fewer than baseline.

Run the existing 48-item capability fixture fresh for baseline and the locked
steer in this study, with identical settings. Its 32 MC items use prompt-only
next-answer logits; generation uses the existing GSM8K/IFEval caps and batches.
Require combined MC no more than two fewer correct, GSM8K strict no more than
one fewer, and IFEval prompt-strict no more than one fewer. Report fallback math
scoring and all mechanics separately. These are small historically used probes,
not broad capability or held-out benchmark claims. A final failure is a tradeoff
or null result; never tune again against confirmation or retention.

## Evaluation, budgets and stopping

Keep the calibrated trait definitions unchanged. Extend the output schema with
explicit task-content, coherence and factual-correctness fields/references,
reviewed before calls. Hide condition, block, strength and pair side from judges.
Use independently frozen shuffles for eligibility/development/confirmation and
batches of eight. Never place both sides of an eligibility pair, or multiple
conditions of the same development/confirmation prompt, in the same batch.
Construct and CPU-test these constrained plans before any calls, recording
composition privately and binding their exact membership. Fresh batch contexts
then cannot directly compare conditions of the same task. Batch counts remain
**8 + 14 + 9 = 31 first-valid batches, at most 62 invocations**.
Structural retries only, at most one per batch, same payload, first valid frozen.
Record reported model IDs, pinned CLI, isolated tools-off context and journal
receipts. Live grading needs network-enabled execution. Fatal infrastructure,
identity, hash or uncertain-start errors stop for coordinator review.

Maximum target generation: **230 responses** (10 preflight, 112 development,
4 reload, 72 confirmation, 32 capability); plus 64 capability MC items and up to
64 teacher-forced authored sequences. Eligibility/dev nulls skip later phases.
Record actual counts, synchronized GPU time, allocated/reserved memory and call
cost metadata. Preserve all failed phases, original labels and source bindings.
No unregistered retry, new words/blocks/strengths, second judge selection or
threshold change is allowed.

The rating calibration covered clear authored cases, not mild or degraded model
outputs. Trait score 1 remains ambiguous; give its count explicitly. AI content
judgments are limited evidence, complemented by fixed facts and automatic
capability scores. A passing study supports only this small-model configuration
on these fixtures, not general independent style knobs. Save/reload/evaluate and
document nulls as valid learning outcomes. Rudeness and sarcasm get separate
isolated studies afterward; joint control and further 14B work remain deferred.
