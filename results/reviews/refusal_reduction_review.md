# Refusal-reduction pilot 03: review provenance

Date: 2026-11-13. The user clarified that the target is refusal reduction,
with direct persona style as a separate objective. The prior positive probes
remain unchanged and do not establish whether reduction works.

The user asked the Claude advisor to reconsider participation given the
fundamental LLM research context, and explicitly authorized proceeding if it
still declined. The coordinator explained the concrete local comparison,
public benchmark classes, quality measurements and absence of a release in
this study. Claude declined the core method/selection review because it
regarded that scope as safeguard removal. It also did not complete a limited
benign-measurement review. No Claude approval is claimed for this new design.
No further attempts to obtain that review are planned.

The user prefers the terminology refusal-reduction and activation-steering
research. The methods and experimental scope are described explicitly; the
terminology change did not alter them. The coordinating Codex owns scientific
acceptance and reviews code; the dedicated Codex worker implements and tests.

The coordinator fixed two interventions before new outputs: raw layer-23
subtraction at coefficient -0.5 and all-block removal of the projection onto
the same direction, both at all token positions. Valid full compliance, rather
than one minus refusal, prevents invalid responses from counting as successful
reduction. Per-class quality gates, a locked development decision and one
final comparison with a tiny automatic retention screen limit the scope.
The exact rules are in `docs/refusal_reduction_protocol.md`.

At this review stage there are no reduction-model outputs or experimental
response judgments. Code, CPU checks, manifests and exact registrations must
be accepted before GPU execution. The original private raw artifacts and
source pins are reused without new curation or re-judging baseline outputs.

## Implementation acceptance before model outputs

The coordinator independently checked the author's pinned hook source
(`9d852fae1a9121c78b29142de733cb1340770cc3`, hook-file SHA-256
`0fa5de8d6a739f9f3ecfcb462d027d8ade269f6ce469e320c3238ef4a1d0cd56`)
and reviewed the saved new hooks, runner, config and tests. New hooks use
FP32 projection and per-device vector caching, preserve activation dtype,
cover all positions and clean up on exceptions. Probe diagnostics sample the
first prefill/decode at each node while counting all calls.

All 26 new CPU checks passed after the final reporting fix, including actual
tiny-Qwen3 cache/full-prefix comparisons, projection algebra and block/writer
coverage, identity and cleanup, immutable label reuse, invalid-answer scoring,
selection boundaries, mutation rejection and null-stage guards. No old source
was edited. The exact run registration is committed before GPU probes.
