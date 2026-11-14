# Persona rating calibration: completion passed

Completed 2026-11-14 under the
[completion protocol](persona_rating_calibration_completion_protocol.md),
registered in `c095b09` before judgments. The
[aggregate report](../results/persona_rating_calibration_completion_01.json)
contains gates, counts, kappa, first-valid provenance, batch composition and
hashes. The [original incomplete run](persona_rating_calibration_execution.md)
remains unchanged.

## Measurements

| Trait | Intended-presence agreement, pass 1 | Pass 2 | Quadratic weighted kappa |
| --- | ---: | ---: | ---: |
| Profanity | 48/48 | 48/48 | 0.9922 |
| Rudeness | 48/48 | 48/48 | 0.9851 |
| Sarcasm | 48/48 | 48/48 | 0.9685 |

All six agreement gates passed the fixed 41/48 threshold; all three kappa values
exceeded 0.7. Neither pass had a gross-failure flag. The whole-word lexical
counter agreed with intended profanity on 48/48 texts, with zero false positives
or negatives, passing 46/48. Each trait was intended present in 24 examples and
absent in 24; all eight binary combinations had six examples.

All 12 batches passed on their first attempts, without structural retries.
The completion charged 12 new slots and started 12 calls. Including the failed
parent invocation, cumulative usage was 13, within the original ceiling of 24.
The separately bound allowance was parent 1, at most 23 new, at most 24 total.
Live completion used network-enabled execution, with the same pinned native
CLI, model request, tools-off isolation, inputs, rubric, seeds and batch plans.

## Integrity and scope

The completion manifest is
`a61d737f0906a5bd92947b1cbdbb8c5adb5a5e69250c38709d9f93e436be5815`:
20 snapshot files and 1,100 protected bindings. Verification passed before and
after judging and during aggregate export, including the failed parent's raw
attempt, receipts, stop marker, incomplete report and original source. The
worker's 14 completion CPU checks and the foundation's 45 checks passed before
execution. Claude accepted the management design; its code review was
interrupted by its classifier, so code acceptance belongs to the coordinator,
as recorded in the [review](../results/reviews/persona_rating_calibration_completion_code_review.md).

These results support basic trait separation and repeatability on deliberately
clear authored examples. Author intentions are not human ground truth. The two
passes and advisor share a model family, with no fully pinned server snapshot.
This does not validate subtle or degraded model-generated styles. Non-rude
sarcasm concerns objects/situations; playful teasing of people is untested.
The worker independently reproduced all counts, gates and kappas from the saved
journals and verified receipts, privacy and parent/source hashes. Claude reviewed
the aggregate interpretation and emphasized the clear-example ceiling: ratings
of 1 on future outputs remain ambiguous and should be reported separately.

No target-model inference, GPU use, activation vector or steering experiment ran
in this calibration. Raw texts/ratings, payloads and journals remain local under
ignored `outputs/persona-rating-calibration-completion-01/`. The next step is a
separately reviewed isolated profanity protocol on Qwen3-4B, followed by isolated
rudeness and sarcasm. Joint steering and further 14B work remain deferred.
