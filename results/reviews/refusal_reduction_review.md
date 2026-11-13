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

## Execution and outcome acceptance

The real-model probe passed both installed identity checks, all 108
projection sites and 180 observed nodes, nonzero hidden/logit changes and
exact plain-logit restoration after cleanup. The exact budget was committed
in `dbd326c` before 120 new development responses. All old inputs and labels
were reused without modification or baseline re-judging.

Development selected `subtract23` under the fixed rule: source-unsafe valid
full compliance was 8/30 versus baseline 2/30, with safe usefulness 10/30 versus
11/30 and no increased invalid outputs. All-block removal reached 6/30 and
missed the primary threshold. The selection was committed in `099ed24` before
final or retention outputs. No alternative parameter or method followed.

On final, valid full compliance increased 1/30 to 7/30 on the source-unsafe
class, while safe usefulness fell 29/42 to 23/42. Three candidate outputs had
mechanical degeneration flags (two safe and one unsafe); one safe response
was AI-unscorable and overlapped the mechanical flags. The reduction target
passed, but final usefulness and validity criteria failed. The small automatic
retention screen passed its coarse limits. The correct conclusion is observed
refusal reduction with quality loss, not a usable configuration or a failure
to induce any effect. Family-cluster intervals include zero, limiting broad
generalization. Development selection is non-null, while final acceptance is
false; these are separate fields.

All 14 blinded classification batches were valid on their first attempt. These
independent classification calls do not imply project-advisor design approval.
No additional advisor-review attempts, model trials or judgment retries were
made. No new weights were created or released.

The dedicated worker's final read-only audit found no material issue: frozen
source/snapshot/runtime and original-artifact bindings matched, selection and
final criteria recomputed, final/retention locks matched, counts and costs
were preserved, and no raw prompts/responses, token-ID fields or secrets were
detected in the aggregate. The coordinator accepted the report as a completed
bounded research comparison, with quality failures explicit. No core file
changed after registration.
