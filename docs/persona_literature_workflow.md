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
Claude's PV implementation review are complete; the frozen PV runner passed34
CPU checks. Exact initial registration was committed before GPU0 elicitation
started. The worker is implementing the separate SAPS port. See the
[execution record](persona_reference_execution.md) for live status and results.
