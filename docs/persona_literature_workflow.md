# Literature-first isolated behavior steering

User-selected policy, 2026-11-14: first follow an established workflow and
released author data/code where available. Establish and document each isolated
trait before consolidating its reusable workflow. Combined control is a separate
experimental topic for later discussion and literature review, not an automatic
stage after isolated results. Finish the selected 4B work before further 14B
experiments. Physics LoRA remains an independent learning exercise.

| Trait | Reference | Scope |
| --- | --- | --- |
| Rudeness | [Persona Vectors](https://arxiv.org/html/2507.21509v1), official `impolite` artifacts at [b8e0f04](https://github.com/safety-research/persona_vectors/tree/b8e0f044fe2410a6fad579f38324f03f13b4e917) | Model-generated opposing persona responses, source filtering and response-mean raw vectors. Published impoliteness is broader than our calibrated rudeness. |
| Sarcasm | [SAPS](https://aclanthology.org/2026.findings-emnlp.1594/), author code/data at [c56fb28](https://github.com/SharathHN/saps-steering/tree/c56fb28ac0d14c8fd70035e035c5452e3c6c6ccc) | Literal-to-sarcastic paraphrasing, final-prompt-token contrast, fixed versus autoregressive projection steering. Generic assistant transfer is a different question. |
| Profanity | Persona Vectors general trait-artifact procedure, extended to casual non-slur swearing | New artifacts require a separate frozen generator record and review. This is a method extension, not released author profanity data or detoxification. |

The [source review](../results/reviews/persona_literature_source_review.md) records
what was inspected: 32 small files verified against canonical Git blobs and
SHA-256s, with no dependency or model installation during the audit.

## Reusable procedure

1. Pin author source, datasets, notices and model. Inspect actual extraction,
   token/layer conventions, injection scope, scoring and selection code.
2. Write a bounded adaptation protocol before outputs. Explain changes in model,
   runtime, thinking mode, precision, sampling, evaluator and budget; changed
   experiments are not exact reproductions of published scores.
3. Verify operations against author primitives on identical tensors and a tiny
   model before GPU release. Extraction and intervention must use one residual
   basis. Response means, prompt-token vectors and adaptive projection targets
   are different methods, not interchangeable labels.
4. Establish prompted elicitation and evaluator headroom. Prompting here creates
   extraction/control data; it does not replace activation steering.
5. Select on fixed development data, lock once, report on an untouched partition.
   Per-test-example oracle layers are not deployment performance. Keep reference
   metrics separate from narrower target scores, content, cross-trait effects
   and capability checks.
6. Record save/reload, actual hook coverage, tokens/caps, resources, hashes and
   grading receipts. Preserve failures and first-valid labels. Use one source
   record and stage input/output records; do not recursively hash unrelated
   historical outputs inside every read operation.
7. Close and consolidate the isolated workflow with results and limits. Nulls
   do not establish impossibility. Further experiments need a scientific decision
   and a new protocol. Joint work remains deferred for later discussion.

## Current sequence

The earlier [three custom pilots](isolated_persona_traits_execution.md) remain
closed development nulls and were not author replications. The later
stronger-magnitude proposal was not executed and is superseded by this decision.

Rudeness is first under the [PV adaptation protocol](persona_reference_protocol.md),
then separate [sarcasm](sarcasm_reference_protocol.md) and
[profanity](profanity_reference_protocol.md) protocols. The source audit and
Claude's implementation reviews are complete. The frozen runners passed 34 PV,
24 SAPS, 21 artifact-generation, 31 grid-extension and 27 SAPS-continuation CPU checks. The original
rudeness two-stage search closed as selection_null. The original SAPS study
closed at its added positive-control quality gate, before actual steering.
Profanity elicitation passed. A separately registered
[symmetric PV grid](persona_pv_grid_protocol.md) now addresses the rudeness search
limitation and prospectively replaces profanity's unrun selection stages.
The full rudeness grid also closed selection_null (best quality-eligible gain
1.2/100); profanity's fresh 280-output grid closed selection_null with zero gain
and no lexical matches. The separate [SAPS continuation](persona_saps_continuation_protocol.md)
completed author prompt-based extraction, preserved the original failed added
gate and passed its registered held-out reference/capability criteria. Fixed
block 23/alpha 1 produced27/100 reaching the author sarcasm threshold ≥3 paraphrases versus 0 baseline
and random, with a meaning-preservation cost. The separate contextual subset
found 4/20 clear sarcasm and content preservation 13/20 versus 18/20 baseline.
This is exploratory task-specific evidence, not generic persona validation.
Original results remain unchanged. See the
[execution record](persona_reference_execution.md) and
[sarcasm closeout](persona_saps_sarcasm_continuation_closeout.md).

## Consolidated method contracts

All current studies use the original, unadapted Qwen3-4B BF16 checkpoint.
Physics LoRA, quantization comparisons and refusal vectors are separate inputs
and results, not components of these trait experiments.

For PV rudeness and profanity, construct opposing prompted answers on matched
questions. Filter using the reference trait and coherence rules, teacher-force
retained answers under their original systems, then average
each retained answer's native response-token activations. At each candidate
layer the vector is the mean positive activation minus the mean negative
activation, without unit normalization. The coefficient multiplies that raw
vector. Use decoder-block outputs, with author layer L mapping to block L−1,
excluding the final-normalized state. Apply the shift at the last real prompt
position and subsequent generated
tokens. A coefficient has no universal strength across traits: record the actual
shift norm relative to the residual. This project's registered grid covers nine sites
and coefficients0.5,1,2; it extends the author's staged search. Do not extend a
search within the same protocol after a null; a further extension needs a
separate registration and preserves the original result. Greedy single rollouts,
cap512 and integer co-rated Claude scores are disclosed project adaptations,
not author defaults.

For SAPS sarcasm, use the released literal-to-sarcastic paraphrase task. Learn
unit directions from plain versus styled final-prompt-token activations, and
learn the styled projection target in a separate prompt pass. Fixed steering
uses each input's baseline projection to compute one constant correction, added
at every forwarded position (whole prefill, then each cached decode token);
adaptive steering recomputes the correction from the current last activation.
The author's adaptive operation broadcasts that correction over the prefill,
then acts at each cached token. Both controllers here require batch size1;
these broadcasts must not be assumed to work per example in a larger batch.
These semantics differ from the PV response-only
hook and must remain separate implementations. The sarcasm result concerns
paraphrasing; a generic sarcastic assistant would need its own transfer study.

Persist the direction tensors before scored steering. Verify exact save/reload,
zero-intervention parity, nonzero activation changes, token scope, cache handling
and hook cleanup. Keep these mechanics checks separate from evidence that the
trait is controllable. Lock development selection before any held-out generation;
a null ends that protocol without opening its report questions. Retain private
raw answers, grader receipts and provenance; publish aggregate results and costs.
The per-method runbooks specify exact commands and registration dependencies:
[PV grid](../scripts/persona_pv_grid_runbook.md) and
[SAPS continuation](../scripts/persona_saps_continuation_runbook.md).

These runners are study-specific: model identity, 36-block indexing, paths and
budgets are fixed to this Qwen3-4B experiment. The method contracts are reusable;
a new model requires a separately reviewed configuration/port and mechanics
checks, rather than changing a model argument in a frozen run.

A usable isolated control has not yet been established for rudeness or profanity.
Sarcasm has an exploratory paraphrase-scoped reference pass, with quality
limits. Those PV execution contracts are
documented; they are not recommended effective
settings. Joint control remains an unstarted experimental topic for later
literature review and discussion.
