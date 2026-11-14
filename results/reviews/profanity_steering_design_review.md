# Isolated 4B profanity steering: design review

Claude reviewed the conditional authored-pair proposal and the complete protocol
on 2026-11-14, after the three-trait measurement calibration passed. It accepted
the corrected design for implementation before any pilot output or judgment.

The design uses 32 authored neutral/profane pairs under identical neutral context,
with at least 24 eligible pairs and declared lexical diversity. It extracts
response-pooled mean differences at fixed blocks 17 and 23 and uses fixed
fractions of an extraction-only neutral activation scale. Mean per-token norms
are recorded as diagnostics, with no additional grid. This is CAA-inspired,
not an exact paper replication.

The material review correction was to prevent same-prompt comparisons within a
judge batch: pair sides and repeated conditions of one task never co-occur.
Constrained membership must be frozen, bound and CPU-tested. The accepted primary
target is usable profanity, requiring content and coherence alongside lexical
presence, with raw presence and ambiguous ratings reported separately. A random
direction is gated on raw presence to detect generic disruption.

The advisor confirmed the finite budgets and gates: 112 development responses,
72 confirmation responses, four exact same-batch reload replays, 10 preflight
responses and 32 capability generations, totaling at most 230; 64 capability MC
items and at most 64 teacher-forced authored sequences; 31 first-valid judge
batches with at most 62 invocations. Selection precedes confirmation. A fresh
capability baseline matches the selected intervention's evaluation settings.

The core learning goal is a saved/reloaded/evaluated vector pipeline, with nulls
or quality tradeoffs valid outcomes. All work stays on the unadapted 4B model;
physics LoRA, other isolated traits, joint control and further 14B work remain
separate. Fixture and code review, CPU checks and exact committed registrations
are required before live execution. No pilot model output or judgment exists
at this design review.
