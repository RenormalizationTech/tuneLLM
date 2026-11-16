# Refusal headroom check: implementation review

Reviewer: Claude advisor, read-only. Scope: `scripts/refusal_headroom.py` and
`configs/refusal_headroom.json`, read in full; `scripts/refusal_headroom_runbook.md`
and `tests/test_refusal_headroom.py`, read in part. Checked against
[the protocol](../../docs/refusal_headroom_protocol.md) and
[the design review](refusal_headroom_design_review.md). I ran no tests and made
no calls. The root independently ran the ten focused tests: all passed in 4.77 s.

## Verdict

**Accepted. No blockers; the source can be frozen.**

## Checks against the protocol

- **C/P/I mapping.** The payload and validator use the strict schema from
  `xstest_baseline.py`: C is full compliance, P partial refusal, I full
  refusal. `parse_attempt` passes authentication, model and isolation errors
  through as fatal, not as schema retries. Headroom is nonempty P + I.
- **Split.** One dedicated `random.Random(42)`, types in sorted order, rows in
  source order before each shuffle; the first 10 go to development and the
  remaining 15 are reserved. Prepare checks for 250 rows, 10 types of 25 and no
  overlap. `verify()` recomputes the split from the pinned CSV and compares it
  with the frozen copy, so it is deterministic.
- **Only the 100 development prompts can be generated.** Generation iterates
  only the development rows. A start file is written before each call and is
  capped at 100. Prediction checks require exactly the 100 development IDs in
  order. There is no stage or command for the reserved prompts; their rendered
  prompts are frozen only as inputs.
- **Native budget.** Membership is frozen at prepare as 13 batches (12 × 8 + 4).
  Empty answers are left out of payloads, and an all-empty batch is skipped. A
  check before each call stops at 26 starts, and `read_grades` rejects more
  than 26 starts or unexpected attempt files. Only the first valid attempt is
  kept, and re-rating after a valid attempt raises an error.
- **Fatal uncertainty.** A start file without its attempt file raises
  `UncertainAttempt`. An existing generation directory blocks a rerun. Any
  armed live failure writes the permanent stop, and a stopped run cannot
  export a report.
- **No source changes or global patches.** The model must have no hooks before
  generation, and cleanup is checked after each call. The only global state
  touched is `torch.manual_seed(42)`, which has no effect under greedy
  decoding. Sources are hashed in the closure and snapshotted, with entry and
  exit integrity checks.
- **Denominator and report.** The fraction is (P + I) / 100 overall and out of
  10 per type. Empty answers are counted, never graded, and kept out of the
  numerator. Caps, EOS, degeneration and prefix results are reported
  separately. The decision labels are `insufficient_observed_headroom` and
  `observed_headroom_requires_separate_registered_plan`, with
  `no_automatic_followup`. There is no correctness or helpfulness rating.
- **Rendering.** The chat template is applied to the user message alone, so
  Qwen2.5 inserts its default system message, as the protocol requires. The
  template is re-checked against the frozen render before each call.

## Notes (no code change needed)

1. **Prefix comparison.** The disagreement check is binary: whether the grade
   is P or I, compared with whether the author prefix matcher detects a
   refusal. A P answer the matcher flags as a refusal therefore agrees. The
   real limitation is that the matcher cannot tell partial from full refusal,
   and it may miss partial refusals that don't open with a refusal phrase. It
   stays secondary and never overrides the semantic labels.
2. **Empty answers are handled differently from before.** The earlier Qwen3-4B
   XSTest baseline counted empty outputs as full refusal; this protocol leaves
   them out of the P + I numerator and reports them separately. If any occur,
   P + I + empty may be reported as a descriptive figure. It is not a gate and
   does not change the headroom decision.
3. **Batches are grouped by type,** because development order follows the
   sorted types. This is acceptable for absolute C/P/I labels and matches the
   earlier baseline's source-order batching.
4. **Dependency on the sarcasm module.** `generate` is imported from the active
   sarcasm study's module. It is hashed in the closure and only read. Any later
   change to that module needs a new registration here, not a silent rebind.
