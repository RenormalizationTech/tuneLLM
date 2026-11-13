# Stored-output diagnosis: review provenance

Date: 2026-11-13. The user authorized diagnosing the current steering problems
and clearly documenting both interventions. The coordinator discussed the
read-only scope with the fresh Claude advisor before preparing the packet.
The protocol and rubric were fixed before inspecting the paired texts.

The dedicated Codex worker prepared 19 unique pairs: 12 safe usefulness
changes, seven refusal-to-nonrefusal transitions, and three mechanical-flag
pairs overlapping those groups. Preparation verified the original results,
locks and 315 input hashes. The protocol, rubric, preparation source and
private packet bindings were frozen in commit `227f27e`.

Claude's advisory conversation performed one contextual, unblinded review.
The coordinator spot-checked four safe pairs; this is not human validation
or an independent blinded replication. Source-unsafe review described
response structure without verifying operational correctness or effectiveness.
No raw prompts, responses or private evidence notes are published.

The initial review placed some structural fields at pair level. A follow-up
conformed those fields to the per-condition template and clarified that
completion means the response finished, not that the request was fulfilled.
The coordinator verified that T/E/R/completion/confidence codes, evidence
notes and safe-change explanations were unchanged. Both versions remain
private. Their hashes are recorded in the diagnostic report. This clarification
was not a second independent review. Neither apparent grading inconsistency
nor task effectiveness is inferred from the structural annotations.

The worker's separate validator rejected the blank template, accepted the
completed schema and exported only allowed codes and counts. Its final
read-only audit confirmed all 315 original input hashes and four immutable
packet files still matched. Both public JSON summaries passed checks for raw
text, private notes, token arrays and credential/private-path strings.

Claude's final read-only review accepted the narrative counts, selected-sample
limitations, completion/fulfillment distinction and methods documentation.
The methods guide includes the reviewed extraction blocks, separate direction
and KV cache descriptions, orthogonal-subspace terminology and exact-arithmetic
redundancy of later input projections. The coordinator checked the cited
author method, including Appendix I.1, against the implementation description.

No Qwen generation, GPU job, external fact checking, replacement benchmark
labels or new model weights were produced. The original final-quality failure
is preserved. The next-study suggestions are hypotheses, not a registered
experiment or a claim that a new configuration works.

See [diagnostic report](../../docs/refusal_output_diagnosis.md),
[methods guide](../../docs/activation_steering_methods.md) and
[validated review aggregate](../refusal_output_review_03.json).
