# Persona rating calibration 01: incomplete infrastructure result

Registered in commit `de82153` before judgments. The fixture/rubric and code were
reviewed with Claude, and all 45 focused CPU tests passed. The frozen manifest
is `65f838f17aa1e13f3d71bad47b83562bd6cb43001b09dd99122dc4804dc9905f`,
binding 17 snapshot files and 1,065 protected files.

The first native Claude invocation ran inside the network-restricted sandbox.
It returned code 1 with an API connection/DNS error (`EAI_AGAIN`),
`is_error: true` and empty service-reported `modelUsage`. No valid ratings or
accepted pass were obtained. The runner stopped with `judging_fatal`, as
registered, and made no structural retry or later batch call.

This was an avoidable coordinator launch error, not a trait-rating failure.
The [aggregate result](../results/persona_rating_calibration_01.json) remains
incomplete, with one charged CLI invocation. Preserve the raw attempt, start,
receipt, logs, manifest and stop marker in the ignored
`outputs/persona-rating-calibration-01/` directory. No target-model generation,
activation vector or GPU experiment ran.

A read-only DNS check outside the sandbox succeeded. Claude reviewed a
[separate completion](persona_rating_calibration_completion_protocol.md), with
unchanged scientific inputs and at most 23 new invocations, preserving the
original cumulative ceiling of 24. That completion requires its own reviewed
implementation and committed registration; this stopped run is not resumed or
silently converted into a complete result.
