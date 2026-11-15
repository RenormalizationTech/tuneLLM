# Profanity method transfer: completed comparison

**GCAD strength 5 passed the registered held-out criteria and is retained for
Qwen2.5-7B-Instruct.** Fixed projection and adaptive SAPS produced no measured
profanity at any of their three development strengths. The earlier raw-vector
literature-method null remains unchanged.

[Aggregate result](../results/profanity_method_transfer_01.json),
[frozen protocol](profanity_method_transfer_protocol.md),
[design review](../results/reviews/profanity_method_transfer_design_review.md),
[retained profile](../configs/profanity_retained_controller.json).

## Held-out confirmation

Each arm answered the same 32 previously unused neutral questions, greedily
with a 160-token cap. Profanity and coherence were scored in separate blinded
Claude contexts on 0–100 scales. Lexical swearing uses the fixed eleven-word list.

| Arm | Profanity /100 | Coherence /100 | Responses with lexical swearing /32 | Invalid /32 | Capped /32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.00 | 85.34 | 0 | 0 | 32 |
| GCAD, strength 5 | 31.13 | 82.69 | 14 | 0 | 16 |
| Random GCAD delta | 0.00 | 85.69 | 0 | 0 | 29 |

GCAD gained 31.13 profanity points over both baseline and its random control.
Coherence fell 2.66 points from baseline, within the predeclared tolerance. All
seven confirmation gates passed. Swearing occurred in 14/32 responses, matching
the development frequency; this setting does not make every answer profane.
Profanity was higher on 27/32 paired questions and tied on five, with no lower
scores. Median profanity was 35/100; nine answers scored at least 50. See the
[descriptive distributions](../results/diagnostics/profanity_method_transfer_distributions_01.json).

The control replaces the profanity attention delta with seeded random directions
of the same per-layer norms, retaining the keys, demand, gate and sites. It
checks delta content at approximately comparable energy/site; it does not isolate the gate or
establish that GCAD is superior to every alternative method. Actual mean
per-position displacement norms were 7.610 for GCAD and 6.584 for random,
13.47 percent lower for random because gate activations and BF16 rounding
change the realized shifts; raw per-layer delta norms were matched.

## Full development comparison

| Setting | Profanity /100 | Coherence /100 | Lexical swearing /32 | Capped /32 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.00 | 80.88 | 0 | 32 |
| Fixed 1 | 0.00 | 82.69 | 0 | 25 |
| Fixed 2 | 0.00 | 86.34 | 0 | 5 |
| Fixed 4 | 0.00 | 39.09 | 0 | 0 |
| SAPS 1 | 0.00 | 83.97 | 0 | 25 |
| SAPS 2 | 0.00 | 88.47 | 0 | 12 |
| SAPS 4 | 0.00 | 62.41 | 0 | 8 |
| GCAD 3.5 | 1.94 | 82.50 | 1 | 25 |
| GCAD 5 | 28.44 | 80.72 | 14 | 10 |
| GCAD 7 | 59.94 | 59.34 | 32 | 1 |

GCAD 5 was the lowest strength satisfying all development gates. Strength 7
increased swearing but failed relative coherence: 59.34 was below the required
60.875. It was not taken into held-out testing. Fixed and SAPS had zero semantic
profanity and zero lexicon hits throughout, so they did not qualify. Every
development condition had zero empty/repetition-invalid outputs; that diagnostic
alone did not establish answer quality. No setting was changed after evaluation.

## Extraction and implementation evidence

We reused the approved five instruction pairs and twenty question texts, with
no normalized overlap against the 64 evaluation questions. New Qwen2.5-7B
elicitation produced 100 positive and 100 clean responses: 94 versus zero had
lexicon hits. Positive/negative mean profanity was 66.75/0.22 and coherence
83.07/85.95. The fixed filter retained 80 pairs spanning all 20 questions.

Fixed and SAPS shared a new last-prompt-state direction and styled target fit
from 300 prompt forwards. GCAD used the pinned author system-attention delta,
positive post-RoPE key and mean-demand extractors on the eligible native-model
responses, with 240 eager teacher-forced forwards. The SDPA generation loader
was preserved, with a separate registered eager extraction load.

All 33 focused CPU tests passed. Sixteen live mechanics generations and two
explicit logit forwards passed all fifteen gates, including read-only capture
parity, exact zero parity, nonzero effects, prefill/cache coverage and cleanup.
Exact source/input/runtime, mechanics and selected-control registrations were
committed before the corresponding model calls.

## Retained setting and artifacts

The retained profile identifies the exact checkpoint, fresh profanity bank,
source/runtime/input bindings and inference settings. Use GCAD at decoder blocks
9–19 inclusive, scale 1.5, base coefficient 5, with this bank. The working
reference uses ordinary assistant replies, seed 42, greedy decoding and cap 160.
See the [workflow](profanity_controller_workflow.md) for subsequent use.

Private trajectories, token traces, judgments, vectors, controls and integrity
records remain under `outputs/profanity-method-transfer-01/`. Compact aggregate
results, the profile and registrations are public. The checkpoint weights were
not fine-tuned or modified, and no LoRA adapter was used.

## Scope and costs

This is a successful GCAD profanity transfer on 32 held-out everyday questions,
not reproduction of the unavailable original profanity contrastive dataset. The
input systems/rubric were reconstructed and originally generated with Claude,
which also served as the single evaluator. There is no human-rating or broad
benchmark claim. Coherence does not measure factual correctness or helpfulness.
Held-out scoring used filtered predeclared batches averaging 3–4 items, versus
eight in development; within-held-out arms shared the same scoring design.

Rudeness and content drift were visible in inspected examples, especially at
strength 7; neither was systematically scored as a separate axis here. Passing
at strength 5 does not establish independent profanity/rudeness/sarcasm control.
The 160-token cap still truncated half the selected answers and all baseline
answers; interpretation concerns these saved response segments.

Completed work: 632 target generations (200 elicitation, 16 mechanics, 320
development, 96 held-out), 300 prompt forwards, 240 GCAD teacher-forced forwards,
two explicit logits, zero MC items. There were 186 first-valid native grading
batches from 187 starts, including one structural-format retry with the identical
payload and no semantic rerating. Recorded GPU stages totaled 52.43 minutes;
source hashing, preparation, code work and native grading add wall time. Only
GPU 0 was used; other workloads were left running.

The worker independently audited extraction, mechanics, development, selection
and held-out traces. Final arithmetic/privacy and Claude scientific review are
recorded in the [execution review](../results/reviews/profanity_method_transfer_execution_review.md).
No additional grid, dataset revision, joint control or larger-model run follows
this closeout.
