# Saved profanity direction: completed behavioral test

The existing direction was applied and tested on Qwen2.5-7B-Instruct. It did
**not produce profanity in this test**. All six strengths had zero observed
profanity gain; no strength qualified for the planned held-out comparison.
This is a completed negative calibration result, not another extraction stop.

[Aggregate result](../results/author_profanity_direction_test_01.json),
[frozen protocol](author_profanity_direction_test_protocol.md),
[implementation review](../results/reviews/author_profanity_direction_test_design_review.md),
and [independent result audit](../results/reviews/author_profanity_direction_test_result_review.md).
Private raw outputs, token IDs, judgments, vectors and integrity records:
`outputs/author-profanity-direction-test-01/`.

## Actual behavior

Each condition answered the same 32 neutral development questions, greedily,
with a 48-token limit. Profanity and toxicity use the paper's 0–10 local
unsteered self-judge. Swearing counts use the previously fixed 11-word lexicon.

| Strength | Profanity /10 | Toxicity /10 | Responses with lexical swearing /32 | Mean distinct bigram | Answers changed versus baseline /32 |
| --- | --- | --- | --- | --- | --- |
| Baseline | 0 | 0 | 0 | .9910 | — |
| 1 | 0 | 0 | 0 | .9908 | 18 |
| 2 | 0 | 0 | 0 | .9901 | 24 |
| 3 | 0 | 0 | 0 | .9915 | 29 |
| 4 | 0 | 0 | 0 | .9924 | 29 |
| 6 | 0 | 0 | 0 | .9950 | 32 |
| 8 | 0 | 0 | 0 | .9924 | 32 |

No profanity score was missing. All 224 responses were nonempty, reached the
48-token limit and had distinct-bigram scores above .6. That repetition
measure is not semantic coherence, factual accuracy or helpfulness. The
answers inspected remained ordinary helpful openings with wording changes;
we did not establish preserved task accuracy or quality.

The registered selection required profanity gain >= 1 and mean distinct
bigram >= .6, choosing the qualifying gain closest to 2. Every gain was 0;
therefore selection is `calibration_null`. No final coefficient, held-out
comparison, random-control generation or secondary grading was selected or
run. The 32 held-out questions remain unused. There was no expanded grid,
layer sweep or third dataset.

## Intervention verified

The exact parent raw vector file was reused byte for byte, with positive
addition at hidden index 7 / decoder block 6 on every forwarded position.
Mechanics passed all seven gates: zero-token/logit parity, nonzero-logit and
hidden-state changes, saved/reloaded identity and prefill/decode coverage.
Each steered calibration condition recorded 4 prefill and 188 decode hook
calls, with cleanup afterward. Nominal shift norm rose from 3.060 at strength 1
to 24.476 at strength 8. The latter is approximately 55 percent of the sampled
prefill residual norm and 61 percent at first decode, so the null cannot be
explained simply as no intervention being applied. These are sampled,
report-only magnitudes, not proof of a correctly isolated causal trait.

The local judge also distinguished the supplied extraction examples:
present profanity 4.25/toxicity 3.75 versus absent 0/0, without missing profanity
scores. This limited check does not validate every semantic judgment.

Fresh completed work: 234 target generations (10 mechanics + 224 calibration),
248 local judgments (24 validity + 224 calibration),2 logit forwards, 0 teacher-
forced responses and 0 native judge calls. GPU-stage recorded time totals
approximately 6.12 minutes, excluding preparation, integrity hashing and review.
Only GPU 0 was used. Existing historical studies and artifacts remain unchanged.

## Interpretation and review limits

The saved direction separates the reconstructed pairs under the previously
known paired-question validation, but that did not translate into profanity
on neutral questions. Decodability alone did not establish behavioral control.
Both earlier row-validation nulls remain null; their outcomes were not changed.

This does not reproduce the paper's positive profanity effect. Author
contrastive data/code were unavailable; our 12 short-answer pairs are a
reconstruction. Fixed questions, one block, a narrow word list, self-judging,
and universal 48-token truncation limit the conclusion. No claim is made that
profanity steering in general is impossible or that another method will work.

The continuation plan had prior Claude review. A fresh advisor request
encountered an expired login; final fresh Claude review was unavailable.
The Codex worker independently audited the completed intervention and
calibration arithmetic. No Claude secondary call was needed after the null.
No additional experiment is authorized by this closeout; further scientific
choices should be discussed with Claude after authentication is restored.
