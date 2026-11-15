# Sarcasm SAPS continuation closeout

The separately registered continuation passed its held-out reference criteria.
This is exploratory evidence for sarcastic paraphrasing on one 100-item subset
of Qwen3-4B outputs, with reduced meaning preservation. It does not establish
generic assistant-persona control or transfer to another model.

The separate contextual review shows a content-preservation cost. The compact
off-task capability checks passed; their sample is too small to establish broad
retention.

| Held-out arm | Sarcasm /5 | Sarcasm ≥3 /100 | Meaning preservation /5 | Grammar /5 | Clarity /5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.00 | 0/100 | 4.47 | 4.84 | 4.69 |
| fixed | 1.31 | 27/100 | 3.44 | 4.77 | 4.04 |
| saps | 1.16 | 26/100 | 3.70 | 4.67 | 4.15 |
| random | 0.00 | 0/100 | 4.37 | 4.75 | 4.60 |
| prompt | 3.08 | 67/100 | 2.66 | 4.61 | 3.56 |

The author threshold ≥3 includes partial or clear inversion; clear, self-contained
sarcasm begins at 4 in that rubric. Meaning preservation labels the author's
entailment scale. It differs from our contextual clear-presence threshold below.

The fixed block 23/alpha 1 controller was locked in ca547ac before any test
generation. Both baseline and norm-matched random scored zero; fixed steering
gained 1.31 points and met all registered quality floors. All 500 test answers
ended normally, with zero caps, empty or degenerate outputs. The random arm controls the fixed winner only;
adaptive's direction specificity was not separately controlled. Random/selected
per-position norm ratios were 1.00005 in prefill and 0.99925 in decoding.

Adaptive block 23/alpha 1 has slightly lower judged sarcasm but higher meaning
preservation and clarity than fixed on this subset. These are descriptive
single-judge comparisons, not a general ranking. Explicit prompting produces
more sarcasm but lower meaning preservation. The reference gain rule tests the
development-selected fixed winner; adaptive is a separately reported arm.


## Separate contextual review

On the preselected 20-question subset of saved test answers, the broader custom
construct gave the following counts. These labels were graded separately from
the author scale and were not used for selection or the reference gate.

| Arm | Clear sarcasm /20 | Content preserved /20 | Incoherent /20 |
| --- | ---: | ---: | ---: |
| Baseline | 0 | 18 | 0 |
| Fixed | 4 | 13 | 1 |
| Adaptive | 3 | 16 | 2 |
| Random | 0 | 19 | 0 |
| Prompted | 20 | 11 | 0 |

All arms had zero gross failures and zero clear profanity. Prompting had one
clearly rude answer; other arms had zero. Ambiguous labels remain in the
aggregate. These counts show a style–content tradeoff in a small selected subset,
not generic persona effectiveness or independent three-trait control. The 20-item
subset and contextual scale differ from the 100-item sentence-alone reference.

## Method and provenance

Prompt-activation extraction used all 500 released training strings: 500 plain
plus 500 styled contrast forwards and 500 separate target forwards. Frozen draws,
native BF16 residual basis, unit directions, styled projection targets and
random units were saved before intervention. Exact zero parity, nonzero delta,
cache scope and cleanup passed ten generation/two logit probes.

Four blocks and three controller settings formed twelve 50-item development conditions.
Blocks 11, 15, 19 gave essentially zero style; 23 was best of the registered sites,
not an optimum over the model. Fixed 23/alpha 1 won at 1.40 versus baseline 0;
adaptive 23/alpha 1 scored 1.08. Adaptive 23/alpha 1.4 scored 2.56 but failed meaning
preservation 2.82<3. No quality rule was relaxed and no held-out reselection ran.

The original parent remains elicitation_null: prompted clarity 3.46 failed its
added 3.5 initial gate. This continuation separately registered author-style
prompt extraction without that extra pre-extraction gate. Parent 100 answers,
five first-valid grading batches and validation h0 were reused unchanged. The
held-out baseline captured fresh test-only h0; it never reused validation h0.


## Compact off-task capability and costs

The registered 48-item fixture was freshly evaluated under baseline and the
locked fixed controller, with input-specific baseline projections. General MC
was 11/16 and physics MC 7/16 for both. Strict math was 8/8 for both; strict
instruction following was 6/8 baseline and 7/8 selected. All registered retention
tolerances passed. This small deterministic screen is not evidence of improved
capability or broad retention, and it does not cancel the paraphrase content cost.

Fresh child cost: 1,142 generations, 1,500 prompt forwards, two logit forwards,
64 MC items and 68 native grading starts (30 development, 25 reference, 13 custom),
all first-attempt valid. Reused parent work was 100 generations and five calls;
combined 1,242 generations and 73 native starts. No reuse was counted as new work.
Development generation took 617.93 seconds; held-out generation 544.87 seconds.
The original parent gate and result remain unchanged. Private raw answers,
tensors, grade receipts and logs are retained in the ignored run directory.

The worker's final independent audit passed with no actionable findings:
matched 48-case captures/controllers/cache and cleanup, official scores, no caps,
reference/custom labels, all 68 first-valid child receipts, exact costs, immutable
parent bindings, aggregate recomputation and export privacy. Capability took
341.25 seconds. Claude reviewed the completed closeout and source contracts;
the author threshold and fixed-winner-only random-control scope are explicit.

## Limits

Sarcasm is judged from the sentence alone in the published task; our custom
contextual construct differs. Inputs contain profanity and hostility, so custom
off-target scores can reflect source content. This is paraphrasing, not an
open-ended persona demonstration. Meaning preservation is a judge score, not
independent factual validation. One co-rating Claude evaluator, integer ratings,
reconstructed instruction draws and native HF/nonthinking BF16 runtime differ
from the author setting. Development comparisons reuse an older baseline
grading session; the held-out arms were freshly graded together.

The 100-item test and one seed provide no human validity or broad generalization
guarantee. Fixed/adaptive alpha and layer ranges are narrow; only one random
direction controls perturbation size. These findings do not establish simultaneous
trait control. Joint work and further 14B experiments remain deferred.

See the [continuation protocol](persona_saps_continuation_protocol.md),
[original protocol](sarcasm_reference_protocol.md),
[workflow](persona_literature_workflow.md),
[aggregate](../results/persona_saps_sarcasm_continuation_01.json) and
[selection lock](../results/protocols/persona-saps-sarcasm-continuation-selection-01.json).
