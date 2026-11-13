# Qwen3-4B written-tone steering pilot

Status: design reviewed with Claude on 2026-11-13; implementation and CPU
validation precede inference. This document alone does not authorize final-test
execution: fixture, source, rubric and timing-budget hashes must be registered
and committed first. Original quantization and QLoRA milestones are complete.

## Scope and responsibilities

The user approved reporting cleanup, then a small tone-steering experiment, then
a separately reviewed refusal study. Discuss management planning with Claude
before acting; the coordinating Codex records decisions and reviews results, and
the dedicated Codex worker implements and tests code. Ask the user when a
decision requires their input, while completing independent authorized work.

First update benchmark reports and Hugging Face documentation. Preserve model
weights, tokenizer/config files, privacy and immutable dependency pins. Add a
new compact report rather than rewriting historical evaluation files or export
manifests. Verify the remote parent commit and unchanged existing files. Keep the
public GGUF report self-contained, without paper text or private/local paths.

Then compare three ways of producing warm, encouraging written responses:
ordinary generation, a fixed warm system prompt, and inference-time activation
addition. Start from the unadapted BF16 model, not the physics adapter. No new
LoRA training is required. JSON extraction remains optional backlog. GGUF
steering, classifiers, tone LoRA and further physics training are outside this
pilot. Changes in generated tone do not establish subjective feelings.

## Model and resource policy

- Source: `Qwen/Qwen3-4B`, revision
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- NF4 dependency: `RenormalizationTech/Qwen3-4B-tuneLLM-NF4`, revision
  `e9af208e44e083b77702a71d8aa65b7d97f78ce6`.
- Reuse the project environment and saved NF4 checkpoint; prohibit a second
  quantization. Record actual package versions, model/fixture/source hashes,
  GPU, backend, dtype, batching and timing with every run.
- Physical GPU 1 only, sequential jobs. Both GPUs currently have unrelated
  workloads; do not stop them, switch devices, or assume memory headroom grants
  permission to compete. Complete CPU preparation first and obtain an available
  window before inference. Model weights remain unchanged.

## Data and separation

Use AI-authored safe everyday scenarios, disclosed as synthetic, with two to
four literal required facts per scenario. Include checkable names, weekdays,
times or other literal values. Prompts themselves contain no requested tone.
Define a scenario family as a fine-grained task template, not a broad topic.
Balance broad domains across splits while keeping whole families disjoint:

- Extraction: at most 80 candidate scenarios, retaining the first 50 eligible
  warm/neutral response pairs in a fixed order.
- Development: 30 prompts for timing, selection and gates.
- Final: 60 held-out prompts, untouched until selection is locked.

Use 16 extraction families, six development families and 12 final families,
with five scenarios per family. The resulting 34 family templates must be
distinct across splits; audit paraphrased-template leakage before generation.

The worker authors the fixture, required fact keys, allowed numeric/name values
and group labels. Freeze the fixture before model generation. Literal matching
is a limited diagnostic: failure can reflect paraphrase, and a keyword appearing
does not prove factual correctness. Pair it with separately labeled AI content
assessment. Never silently relax the required pair count or eligibility rules.
Record attrition and consult Claude if fewer than 50 eligible pairs remain.

Generate extraction responses with BF16 under neutral and warm system prompts.
Keep a pair only when both cover required facts, there are no unsupported or
contradicted facts according to the fixed checks/rubric, the warm/neutral token
length ratio is 0.85–1.25, and a blinded AI comparison judges the warm response
warmer. These response texts must not become development or final prompts.

## Activation extraction and intervention

This is a response-token mean difference variant inspired by
[contrastive activation addition](https://aclanthology.org/2024.acl-long.828/),
not an exact reproduction of that paper's extraction protocol.

Re-encode both responses under identical neutral system/user context. Extract
decoder-block outputs at zero-based indices 11, 17 and 23 (one-based blocks 12,
18 and 24), averaging assistant response tokens only, excluding template,
padding and end-of-turn tokens. Verify token boundaries explicitly. For each
layer, average the per-pair warm-minus-neutral differences to obtain a vector.
Record its norm and ratio to the median response activation norm.

Candidate multipliers are 1, 2 and 4 times the raw vector; zero is the control.
Add at the selected block output on the last real prompt token during prefill,
then on every generated-token decode step. Cache-aware hooks must handle the
prefill/decode distinction and padding correctly, and always detach afterward.
Zero strength must reproduce ordinary output. Never modify model weights.

## Timing and decoding

Use greedy decoding, `enable_thinking=false`, seed 42, identical prompts and
generation settings within each comparison. Begin with a 256-token cap. Time
default generation on development prompts only; if more than 10% of those
outputs hit the cap, choose 320 before running the actual development study.
Freeze the final cap, batching, exact preflight sample IDs, projected job budget
and code hashes before selection. Preflight outputs remain development data;
report their reuse. Do not use final prompts to tune speed or output limits.

## Development gates and selection

Compare the nine nonzero layer/multiplier settings against default generation
on the same 30 development prompts. A candidate passes only if:

- Complete required-fact coverage count decreases by at most one prompt.
- Responses with unsupported or contradicted facts increase by at most one.
- Median token-length ratio to default lies between 0.8 and 1.5.
- Token-cap hits increase by at most one.
- Empty output never occurs, and degenerate-output count does not increase.
  Define degeneration as excess repeated four-grams divided by total
  four-grams above 0.20, for responses of at least 20 whitespace words.

Use randomized A/B pairwise AI warmth judgments against default, ties counting
as half. Keep warmth and content rubrics separate. Among passing candidates
reaching a two-thirds warmth preference score, choose the smallest multiplier,
then highest score, then the layer nearest 17, then the lower index. If none
reaches two-thirds, choose the highest-scoring passing candidate only if its
score is above 0.5; break ties by smallest multiplier, then layer as above, and
label it exploratory. Otherwise retain zero steering. On 30 prompts, a small
observed preference is weak evidence. Preserve all development outcomes.

## AI judgments

Freeze a rubric and structured output schema before judgments. Randomize A/B
order and shuffle conditions within packets; keep condition names, strengths,
file paths and answer keys away from the judge. Save keys locally in ignored
storage. Prefer a fresh Claude context seeing only rubric and anonymous packets.
If the advisor session also judges, disclose that overlap. Record the judge's
reported model identifier, exposed settings, rubric hash and judgments. The
advisor reported `claude-opus-5-5`; a dated snapshot and reasoning effort were
not exposed. This does not substitute for the actual judging session's identity.

Content judgments operate on one anonymous response at a time, with the scenario
and required fact key: record coverage for each fact and counts of unsupported
and contradicted claims. Use a separate packet/schema for pairwise warmth
preference, tie, and separate 1–5 warmth/content rubric diagnostics. The worker
must validate complete IDs and schema before using any ratings for selection.

AI judgments are measurements under a rubric, not human ratings or ground truth.
Report disagreements with literal checks and uncertainty. A second Codex judge
is optional, not another model-selection loop. A pretrained classifier and base
perplexity are optional diagnostics, not standalone tone/fluency validation.

## Final evaluation and NF4 transfer

Lock selection before final generation. Evaluate default, warm system prompt
and the selected steering setting once on the 60 final prompts. If zero wins,
reuse the identical default condition rather than duplicate generation. Report
paired warmth scores and confidence intervals, content-check results, lengths,
cap hits, degeneration and actual resource costs. Include comparison against
prompting, not only against default. Bootstrap prompts with scenario-family
grouping; disclose the number of independent groups and small-sample limits.

The primary comparison is selected steering versus default; secondary
comparisons are steering versus warm prompting and warm prompting versus
default. Judge each pair independently with randomized order on the same
scenarios. Report final content-gate quantities descriptively, including whether
the selected setting would meet the development thresholds; this is a final
outcome, never a trigger for reselection.

Repeat the three-condition comparison on the saved NF4 base with its own
unsteered reference. Transfer the exact BF16 vector, layer and multiplier
unchanged as the primary experiment. Any NF4 recalibration requires a separate
development-only protocol; final outcomes never trigger reselection. Optionally
compare BF16/NF4 extracted-vector cosine similarity using extraction data only.
If zero steering was selected, run only NF4 default and warm prompting; there
is no distinct transferred-steering condition.

A small existing general/physics/math/instruction retention screen is a
separately budgeted diagnostic, not a reason to repeat the old eight-condition
comparison. No broad capability-preservation claim follows from tone tasks
alone. Failure to steer or failure of a content gate is a reportable result.

Before the subsequent refusal phase, consult Claude again and register data,
licenses, scope and two-sided measurements: benign over-refusal, appropriate
refusal and answer usefulness. No refusal execution is registered here.
