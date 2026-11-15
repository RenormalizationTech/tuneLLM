# Original-Mistral sarcasm reference

The released fixed-projection and adaptive SAPS methods both induced judged
sarcasm from a near-zero baseline on Mistral-7B-Instruct-v0.1. At the same
alpha1, fixed projection produced more sarcasm; adaptive SAPS preserved meaning,
grammar and clarity better. Raising adaptive alpha to1.4 increased sarcasm at
a larger cost in faithfulness. This establishes a bounded task-level reference
under the disclosed Claude evaluator substitution.

All metrics use the original0–5 rubrics; each arm has450 released test inputs.

| Condition | Sarcasm | Meaning | Grammar | Clarity | Clear sarcasm≥4 /450 | Caps /450 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (SAPS0) | 0.004 | 4.11 | 4.60 | 4.45 | 0 | 0 |
| Fixed1 | 2.62 | 3.46 | 4.34 | 3.98 | 202 | 6 |
| SAPS1 | 2.17 | 3.66 | 4.48 | 4.17 | 179 | 2 |
| SAPS1.4 | 3.62 | 2.98 | 4.14 | 3.65 | 276 | 22 |

The counts at>=3 are0,277,218,397 respectively. Level3 permits partial irony;
clear, self-contained sarcasm starts at4. Do not call all397 high-strength
outputs clearly sarcastic. All three prospective descriptive checks passed:
SAPS1 sarcasm gain>=0.5, SAPS1 grammar>=Fixed1, and SAPS1.4 sarcasm>SAPS1.
These checks neither select settings nor release another experiment.

## Style and quality trade-offs

At alpha1, adaptive versus fixed changes mean sarcasm−0.45, meaning+0.20,
grammar+0.13 and clarity+0.19. Equal alpha is not equal shift energy or equal
achieved sarcasm. This comparison does not establish adaptive superiority at
matched style strength.

Increasing SAPS1 to1.4 adds1.45 sarcasm points but loses0.68 meaning points.
At1.4,127/450 entailment scores are<=2; baseline has8/450. Baseline itself
scores4.11 rather than5 for faithfulness, so interpret losses relative to it.
The high-strength arm demonstrates control with substantial content drift,
not a universally preferable rewrite model.

There are no empty or mechanically degenerate outputs. Mean generation length
rises from27.64 tokens in baseline to34–39 for steering. Two capped SAPS1.4
outputs scored4 for sarcasm; Claude's read-only review confirmed both end in
complete sentences, so the rubric's incomplete-sentence constraint is not
violated. Both use multiple sentences despite the one-sentence instruction,
showing a task-compliance gap not systematically scored by the author rubric.
No ratings were changed or rerun.

## Reproduction scope and evidence

Actual pinned author helpers, original TransformerLens2.17 runtime, FP16,
folding/centering defaults and observed two-BOS tokenization were retained.
Direction learning used500 released training sentences, random instruction
mode and source seed order. Block12 and all four conditions were fixed before
target generation; each arm loaded the model separately. There was no layer
oracle, sweep, random-direction control or capability benchmark.

The authors' Table1 reports fixed1 sarcasm3.12/grammar4.50 and SAPS1.4
sarcasm3.89/grammar4.60. These absolute results are not reproduced. Our
high-strength SAPS grammar is lower than Fixed1, despite better grammar at
matched alpha1. Released published logs have unresolved data/prompt lineage
inconsistencies, and this study uses one Claude instead of three author
judges. These limitations prevent attributing score differences to one cause.
No generic-assistant persona control, multi-turn stability or broad
trait-specificity claim follows from this complaint-rewriting task.

All19 CPU tests and six real-model mechanics gates passed. The worker audited
all four completed arms independently; the coordinator and Claude independently
recomputed saved rating arithmetic. There are1800 unique arm/sentence mappings,
7200 scalar ratings and90 first-valid batches from90 starts, with no retries.
See [execution review](../results/reviews/author_saps_mistral_execution_review.md),
[protocol](author_saps_method_protocol.md),
[runbook](../scripts/author_saps_method_runbook.md) and
[aggregate result](../results/author_saps_mistral_01.json).

Compute totals:1810 target generations including ten probes,1500 extraction
prompt forwards,450 fixed-arm captures, four mechanics captures, two logit
forwards and zero MC. Recorded GPU stages total64.47 minutes, including53.18
minutes in generation; source hashing, environment setup and native grading
add wall time. Raw answers, token traces and rating journals remain in ignored
`outputs/author-saps-mistral-01/`.

This closes the registered original-Mistral reference. Both profanity
row-validation nulls remain preserved, with the separately proposed grouped
continuation awaiting the user's choice. No joint steering or larger-model
transfer is released by these results.
