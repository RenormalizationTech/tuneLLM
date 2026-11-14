# Persona calibration infrastructure completion: design review

Claude reviewed the completion on 2026-11-14 after the first calibration stopped
with a sandbox DNS error and no accepted model ratings. It accepted a new run
with unchanged scientific inputs, preserving the parent stopped run and its
failed receipt. The coordinator owns the network-enabled execution release.

The review tightened the budget to the original cumulative ceiling: charge the
parent invocation as one and permit at most 23 new invocations, checked before
each call/start. Reuse of the old 24-call binding upper bound is acceptable only
with a separate bound record spelling out parent 1, new 23, cumulative 24.
Original path guards must not be monkeypatched. Parent hashes and no-rating
failure preconditions must be verified before and after the new run.

This is an infrastructure completion, not a post-outcome fixture/rubric revision.
No ratings were available to drive selection. Only one such completion is
planned; another infrastructure failure stops for user input. Code, tests and
exact registration still require review before execution.
