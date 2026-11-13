# Qwen3-4B refusal-reduction research

Status: implementation frozen after 26 new CPU checks passed, including actual
tiny-Qwen3 cached/full-prefix tests. No decrease-refusal model outputs or
experimental ratings exist yet. The exact design is in
[the protocol](refusal_reduction_protocol.md). Earlier positive-intervention
pilots remain unchanged and did not test refusal reduction.

The user authorized proceeding if Claude declined scientific review. Claude
declined the core design and did not complete a limited measurement review.
The coordinating Codex owns acceptance, and the dedicated Codex worker
implements/tests. No Claude design approval is claimed.

The two fixed methods are block-23 raw-vector subtraction at coefficient -0.5
and all-block removal of the projection onto the same unit direction. Both
act at all token positions. The latter covers pre-block residuals and
attention/MLP residual writers; a block-output-only hook would miss the
intermediate residual input to the MLP. These methods differ in operation and
coverage, so their comparison is not a factorial isolation of those factors.

The public author reference was inspected at commit
`9d852fae1a9121c78b29142de733cb1340770cc3`,
`pipeline/utils/hook_utils.py`, SHA-256
`0fa5de8d6a739f9f3ecfcb462d027d8ade269f6ce469e320c3238ef4a1d0cd56`.
The installed Qwen3 implementation has 36 blocks with hidden size 2560.
The coordinator independently checked the hook mapping and reviewed
projection arithmetic, scope and cleanup. Per-device vector caching and
sampled numerical probe diagnostics avoid repeated transfers and per-token
CPU synchronization. Normal generation does not collect those diagnostics.

The original BF16 base, extracted layer-23 vector and source data are reused.
The 60 development baseline predictions and their existing labels are reused
without re-generation or re-judging. The new candidate packet will contain
120 items; baseline labels came from a separate older packet. There is only
1/30 benign baseline refusals, so that fixture mainly measures quality
retention. Valid full compliance on the source-unsafe class is measured
separately; it means willingness to answer, not correctness or successful
execution. Invalid output cannot count as an improvement.

The fixed rule selects one method or null. A null skips final generation and
retention. Otherwise only baseline and the selected method run on 72 final
prompts and a tiny 48-item general/physics/math/instruction check. No final
result can reselect a method. Raw prompts, outputs, keys and vectors remain
ignored/private; compact records, source and registrations belong in Git.

The exact input/source/runtime snapshot is registered in
`results/protocols/refusal-run-03.json`, before model execution. Source,
scorer resources, prior artifacts, baseline labels and retention IDs are bound
by hash. The unchanged judge adapter passed its existing audit/version
binding checks; no extra dummy model call was made.
