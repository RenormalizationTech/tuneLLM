# Persona rating calibration 01: preregistered design

Reviewed with Claude on 2026-11-14 before implementation. This is the next
bounded step in the [4B-first persona pipeline](project_tracks.md). It tests
the measurement procedure, not activation steering or a model improvement.
No target-model generation, extraction, training, GPU use or 14B work occurs.

## Fixed authored fixture

The worker authors 48 short texts with enough context to interpret the delivery.
There are six texts in each of the eight binary profanity/rudeness/sarcasm
combinations, giving 24 intended-present and 24 intended-absent cases per trait.
The fixture and intended labels are frozen before any judge call. These are
author intentions on deliberately clear synthetic cases, not human ground truth.

Use impersonal situations or fictional targets. Include friendly emphatic
swearing without rudeness or sarcasm, dismissiveness without swearing, contextual
irony, and mixed cases. No slurs, threats, real-person targets or masked profanity.
Include innocent lexical traps such as `assess`, `class`, `hello` and `scrapbook`
in profanity-absent texts. Every item has an opaque identifier with no trait or
cell encoded in it. Context describes events and referents, without announcing
that a speaker is mocking, dismissive or otherwise displaying the target trait.
Rate the delivery in the text; use context to resolve its meaning, not as a
trait label. Review balance, context and intended distinctions before
judging; do not rewrite cases after seeing results.

## Frozen measurements and blinding

Score each trait independently on integers 0–3: absent, ambiguous evidence,
clear presence, strong presence. Presence means a score of at least 2; mild but
unambiguous swearing still counts as clear presence. Friendly swearing alone is not
rudeness; rudeness needs no profanity; sarcasm requires ironic or mocking intent
in context. A separate boolean gross-failure flag means slurs, threats,
harassment of real people or protected groups, sexual content, or incoherent/
unreadable text. Profanity, rudeness and sarcasm themselves are not gross failures.
The exact operational definitions and output schema are frozen in the rubric.

A fixed case-insensitive whole-word profanity lexicon supplies a separate
presence/count measure. Enumerate supported variants in advance; exclude masked
forms from this fixture. A lexical hit is a profanity diagnostic and does not
establish rudeness or sarcasm.

Judge all items twice, using independently shuffled, frozen orders from seeds
421 and 422, six batches of eight per pass. Payloads contain only opaque IDs,
context/text and the rubric, without author labels, cells, pass outcomes or
earlier ratings. Every batch is a fresh isolated tools-off invocation using the
existing audited native Claude foundation. The second pass tests repeatability;
it cannot replace or repair the first pass. The advisor reviews the design and
fixture, not the hidden judge outputs in advance.

## Fixed gates

All of the following are required:

- Complete coverage of all 48 items in each pass, with no gross-failure flags.
- At least 41/48 agreements with intended presence for **each trait in each
  pass**. The six agreement tests are separate.
- Quadratic weighted Cohen's kappa at least 0.7 between the two ordinal rating
  passes for **each trait**, using fixed categories 0, 1, 2, 3. Undefined kappa
  fails. Weights are squared category distance divided by nine.
- Lexical profanity presence agrees with intended profanity on at least 46/48
  items. Counts and lexical false positives/negatives are retained separately.

If any axis fails, stop this calibration and do not start the profanity pilot:
all three measurements are needed to detect crosstalk. Original outcomes remain
unchanged. Any altered fixture, rubric or gate is a new reviewed study.

## Calls, integrity and reporting

There are 12 first-valid batches and at most one structural retry per batch:
24 calls maximum. A structurally valid result is accepted once and never rerated.
Missing/duplicate IDs, schema/type errors, out-of-range scores or a declined
rating count as structural failures under the same retry budget. No semantic
retry, label repair or prompt rewrite is allowed. Fatal infrastructure errors
stop the run; they do not grant extra calls. Resumption preserves the journal,
first-valid results and original attempt budget.

Prepare a fresh run directory with frozen source, fixture, rubric, config,
pass/batch memberships and invocation provenance. Bind the native binary hash,
version, audited wrapper and package versions. Request the audited
`claude-opus-5-5` model and record service-reported model usage IDs on every
attempt. A missing or mismatched model identity is fatal, not a structural retry.
Review implementation and CPU
checks, then commit the registration and source before any judgments. Fail on
binding changes. Keep all raw payloads, responses, labels and attempt journals
locally under ignored `outputs/`; export only known aggregate schemas and hashes.
The report must distinguish complete pass/fail from incomplete measurement.
Include per-cell trait-presence counts in each pass and disclose the frozen
per-batch cell composition, without sending cell information to the judge.

The worker implements and tests; the coordinating Codex reviews and runs the
registered stages. Tests should cover fixture balance and blinding, lexical
boundaries, exact gate boundaries, kappa degeneracy, invalid types/duplicate IDs,
binding tampering and budget-preserving resume with mocked calls. No API calls
are needed for CPU tests.

## Interpretation and next stage

Clear authored examples assess basic trait separation. Passing does not prove
sensitivity to mild/graded model outputs, human agreement or general validity.
In this fixture, non-rude sarcasm concerns objects or situations; playful teasing
between people is not tested.
Two calls to one model family establish limited repeatability, not independent
human validation. The scientific reviewer and judge share a model family, and
the server-side model snapshot is not fully pinned even when the CLI is pinned.

A pass permits preparing a separately reviewed **isolated profanity-on-4B**
protocol. It does not authorize an unregistered block/strength search or
simultaneous steering. Rudeness and sarcasm follow as isolated experiments;
joint tests and eventual 14B transfer remain later stages.
