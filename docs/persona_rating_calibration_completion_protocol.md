# Persona rating calibration 01: infrastructure completion

Reviewed with Claude on 2026-11-14. The original
[calibration](persona_rating_calibration_protocol.md) stopped before any valid
ratings: its first native CLI invocation ran in the network-restricted sandbox
and returned code 1 with `EAI_AGAIN`, an API connection error, `is_error: true`
and empty `modelUsage`. The failed invocation is charged as one used call; no
claim of zero external requests is needed. The coordinator's launch choice was
an infrastructure error, not a calibration result.

Preserve `outputs/persona-rating-calibration-01/` and its incomplete public
report unchanged. Do not remove its stop marker, reset a receipt or resume it.
Only one separate completion attempt is permitted, in
`outputs/persona-rating-calibration-completion-01/`. The worker implements a
small new wrapper; the original runner/config/protocol/fixtures stay unchanged.

## Unchanged scientific inputs

Reuse exactly the original 48 texts, author intentions, rubric, lexicon, seeds
421/422, batch memberships, scoring/structural rules, model/native binary,
audited tools-off invocation and pass/fail gates. No model ratings were seen
before this decision; no inputs or criteria were selected using an outcome.
Both fresh passes still require all 48 ratings. Calibration still establishes
limited repeatability on clear authored examples, not human validity.

## Budget and launch

- Parent charged calls: exactly 1, with no accepted ratings.
- Completion: at most 23 new native CLI invocations, at most two per batch,
  always preserving the first structurally valid result.
- Cumulative ceiling: 24, including the failed parent invocation.
- Enforce the remaining cumulative budget immediately before every possible
  new invocation and before writing a new start marker. Replayed receipts do
  not spend another call. If insufficient budget remains, report incomplete.
- The unchanged old binding format may retain its `maximum_total_calls: 24`
  upper bound. A separate immutable bound budget record must explicitly state
  parent 1, new maximum 23 and cumulative maximum 24; do not imply 24 new calls.
- The coordinator launches live grading with the tool's network-enabled
  execution permission. A read-only DNS check outside the sandbox succeeded;
  that check made no model request and consumed no grading call.

Infrastructure/auth/model-identity/uncertain-start/hash failures are still
fatal. If this completion has another infrastructure failure, stop and ask the
user; do not create another completion or change infrastructure repeatedly.
Declines and malformed rating payloads retain the original bounded structural
retry rule. No semantic repair, prompt rewrite or result substitution is allowed.

## Integrity, review and export

Bind the entire failed parent directory, its public incomplete result and the
original code/inputs by hash. Require the exact failure preconditions above,
exactly one parent start, no accepted pass and no rating JSON. Verify parent
integrity before and after completion. New sources, runtime/budget records,
protocol/review, fixture plans and packages require a frozen manifest and
committed registration before calls. Reuse pure scoring/journal helpers where
appropriate; do not monkeypatch the original runner's path guard or modify old
sources to accommodate the new directory.

Focused mocked CPU checks must cover cumulative exhaustion before a start file,
replay without spending budget, first-valid labels, rejected parent preconditions
and read-only parent protection. Existing calibration tests remain applicable
to the unchanged math, fixture, schema and journal foundation.

Export a separate aggregate-only completion report, preserving the parent's
incomplete report. Include parent/new/cumulative call counts and budget, status,
first-valid provenance, three-trait agreement/kappa/lexical gates, per-cell
counts, frozen batch composition, hashes and the original limitations. Do not
publish raw contexts, texts, individual ratings or credentials. No GPU/model
generation, vector extraction or 14B experiment belongs to this completion.
