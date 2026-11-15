# Original-Qwen GCAD first-reply result

The registered first-reply experiment reproduced the published qualitative
pattern under a disclosed Claude evaluator substitution: both interventions
strongly increased the author's broad impoliteness construct; attention-level
GCAD preserved much more coherence than residual addition. This is a successful
bounded method reference, not an exact numerical reproduction or validation of
narrow rudeness control in ordinary conversations.

| Condition | Impoliteness /100 | Coherence /100 | Mean tokens | Cap hits /60 | EOS /60 | Degenerate /60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1.62 | 82.68 | 459.95 | 36 | 24 | 0 |
| Residual addition | 84.52 | 7.83 | 487.33 | 57 | 3 | 37 |
| GCAD | 77.37 | 75.90 | 48.95 | 0 | 60 | 0 |

Each condition has60 saved answers. Every rating was numeric; there were no
grader REFUSAL labels, so refusal-excluded and conservative trait means match.
GCAD increased impoliteness75.75 points over baseline and lost6.78 coherence
points. Its coherence exceeded residual addition by68.07 points. All three
predeclared30-point contrasts passed. All60 GCAD answers ended with EOS.
Cap/degeneration differences are part of the result, not reasons to discard
answers. These are descriptive scores, without a broad capability claim.

GCAD makes answers much shorter and curt: about49 tokens versus460 for
baseline. The author rubric rewards curtness and missing pleasantries; brevity
may contribute to both higher impoliteness and easier coherence preservation.
No helpfulness or task-fulfilment measure was applied. Coherence does not
establish that the model still answers the question adequately. The residual
collapse concerns the author all-position variant only, not every residual
steering method.

The source's20 sampled ten-question groups were frozen before truncating to
first replies, with three samples per group. This produced13 unique first
questions among60 conversations. Repeated questions and stochastic samples
are dependent;60 is not a count of60 independent question topics. There is no
matched GCAD random control and no cross-turn experiment.

The paper reports GCAD89.3 impoliteness/67.5 coherence and residual97.3/6.3
for first replies. Our different evaluator, newly pinned checkpoint revision,
per-arm seed reset and structured grading envelope prevent score-equivalence
claims. The original broad `impolite` rubric also differs from the earlier
custom rudeness rubric. These results do not retroactively change the4B nulls.

## Execution and review

The original Qwen2.5-7B checkpoint and author V/K/demand tensors were used in
BF16 with the actual author generation and hook code. Source/runtime details
and evaluator changes are recorded in the
[protocol](author_gcad_impolite_protocol.md),
[implementation review](../results/reviews/author_gcad_impolite_design_review.md)
and [runbook](../scripts/author_gcad_impolite_runbook.md).

31 CPU tests and all6 GPU mechanics checks passed. Compute was190 target
generations including ten probes, two bounded logit forwards, no teacher-forced
extraction and no multiple-choice benchmark. Main generation took72.27 seconds;
its stage including model loading took82.46 seconds. Integrity hashing and
grading add wall time. Separate impoliteness/coherence contexts used46 native
first-valid batches, with46 starts and no retries. Raw questions, answers,
reasons, token IDs and journals remain in ignored
`outputs/author-gcad-impolite-01/`; only aggregate results are public.

The coordinator independently recomputed all arm sums from saved accepted
ratings, checked46 first-valid batches and stage-file hashes, and verified
all three contrasts using exact integer arithmetic. Claude independently recomputed the saved-stream arithmetic and accepted the
qualitative interpretation, requiring the length and helpfulness limitations
above. See the
[aggregate result](../results/author_gcad_impolite_01.json).

This closes the bounded GCAD reference. Next work is the separately reviewed
profanity paper method and original-Mistral SAPS reference; no stronger GCAD
sweep, ten-turn follow-up or joint steering is automatically released.
