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
