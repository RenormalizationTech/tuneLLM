# Persona rating calibration 01: design review

The existing Claude scientific advisor reviewed the bounded Stage 0 design on
2026-11-14 and accepted the corrected
[protocol](../../docs/persona_rating_calibration_protocol.md) for freeze before
fixture judgments. The coordinating Codex owns execution acceptance; the
dedicated Codex worker implements and tests.

The reviewed design uses 48 authored texts, six per binary trait combination,
with separate ordinal profanity/rudeness/sarcasm ratings. Two independently
shuffled blinded passes, fixed per-trait agreement and repeatability gates, a
word-boundary lexical measure, and first-valid attempt journals precede any
target-model generation. The ceiling is 24 calls including structural retries.

The advisor identified an ambiguity in "gross failure": intended profanity or
rudeness could otherwise be penalized. The corrected protocol explicitly
excludes the traits themselves from that flag, defines the narrow failures,
requires the reported judge model identity on every attempt, and makes missing
or mismatched identities fatal. Per-cell outcomes and batch composition are
disclosed without exposing cells or author labels to the judge.

The advisor confirmed these corrections and accepted the protocol. This is
design acceptance only; fixture and implementation review, focused CPU checks,
frozen preparation and a committed registration are required before any calls.
No steering, model training, GPU experiment or 14B work is authorized by this
calibration. A pass establishes limited repeatability on clear authored cases,
not human validity or sensitivity to subtle generated style.
