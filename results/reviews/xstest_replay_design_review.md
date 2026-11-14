# Final 4B replay: design review

Date: 2026-11-14. The existing Claude scientific advisor accepted
`docs/xstest_replay_protocol.md` before generation. This is management/design
review, not independent grading or validation of unseen results.

The accepted scope is one regeneration of the full published XSTest-safe250
baseline, with original inputs, batch membership, decoding settings and software
bindings retained. Exact token/byte/flag comparisons and new actual-mask checks
against a derived parent reference are predeclared. The parent did not capture
mask tensors, so no old actual-mask equality is claimed.

Three clarifications were applied before preparation: the first generation call
starts the single attempt; prelaunch environment drift does not authorize package
changes; and original-artifact integrity must pass before an equality result is
issued. A mismatch, interruption or integrity failure cannot trigger a retry,
regrading, setting change or inherited labels on changed outputs.

The 4B learning study closes after the observed replay result. It does not depend
on achieving a stronger refusal effect. Existing interventions retain their
partial-effect/quality-tradeoff conclusion. Exact replay adds no label evidence
or claim of prompt robustness or a larger model being more stable.

Code, CPU checks and the exact prepared manifest require review/registration
before the coordinator releases GPU execution. No additional 4B study follows.
