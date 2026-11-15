# Persona Vectors isolated reference adaptation: design review

The user authorized literature-first isolated profanity, rudeness and sarcasm
work on Qwen3-4B, with combined control explicitly deferred. Claude reviewed the
source audit and the concrete first PV `impolite` protocol before live calls.
The coordinator accepts the corrected design for implementation.

The review confirmed source instruction/data/rubric preservation, original
positive/negative systems, raw response-mean vectors, cached last-position scope,
explicit layer mapping, the fixed 10/10 development/report split, asserted pair
keys and filtering, positive controls, reference/custom score separation,
first-valid budgets and no report-set oracle selection.

Two corrections were made before freeze:

- If coefficient 2.0 leaves no quality-eligible layer, a single prespecified
  nine-layer fallback at 1.0 runs. Baseline and existing strengths are reused;
  there is no adaptive grid expansion. With fallback, the chosen layer already
  has strengths 1 and 2, so only 0.5, 3 and 4 are generated subsequently.
- Teacher forcing uses the saved native prompt/generated IDs under the original
  system, with special/EOS/pad masking. Source-style retokenization differences
  remain a disclosed diagnostic and method adaptation.

Claude confirmed exact worst-case accounting: 502 generations, 200 teacher-forced
responses, 64 MC items, 64 first-valid grading batches and at most 128 starts.
Without fallback: 422 generations, 53 batches, at most 106 starts. Earlier
semantic "unscorable" wording was removed: author trait is numeric or explicit
semantic refusal; coherence is always numeric. Schema failures retain their
structural retry/fatal handling.

Additional limits are explicit: cross-session reuse of earlier development
ratings, custom-rubric use beyond its short calibration examples, absolute and
relative shift norms as diagnostics only, integer rather than probability-weighted
reference scores, and limited sample size/single-judge uncertainty. This is a
bounded method-transfer screen, not a precise effect estimate or exact published
score replication. No cause of the earlier custom nulls is established.

The coordinator chose local reuse of author-released SAPS transformed strings
with repository attribution and provenance uncertainty disclosed; raw third-party
source corpora are not redistributed. Profanity will use the published PV
general artifact procedure as explicitly new data under a later registration.
Claude reported no blocker to those source choices.

Code acceptance, passing CPU checks and a committed exact manifest still precede
GPU or grading release. This review does not authorize mixed-trait experiments,
an environment replacement or further 14B work.

## Implementation review

Claude subsequently inspected the new runner, pure operations and tests against
the archived author source and protocol. It reported no blocking scientific,
source-fidelity or protocol mismatch, and accepted the disclosed adaptation.
Its own execution was limited to the CPU plan builder; it did not run the test
suite and did not inspect the runbook in detail. Worker CPU test results are
recorded separately, not attributed to Claude.

The review checked original artifact bytes and system strings, native token
pooling under the respective systems, intermediate-block indexing, raw vectors,
cached response-hook scope, filtering, conditional fallback, reused strengths,
held-out selection locks, random/control arms and exact judging budgets. Two
nonblocking report-wording changes were requested: explicitly name integer
versus probability-weighted scores, co-rating, the required brief reason and
one greedy versus ten sampled rollouts. A required reason can itself influence
judgments. These are limitations, not source-equivalence claims.
