# Isolated rudeness and sarcasm on Qwen3-4B: execution

The [protocol](isolated_persona_traits_protocol.md) keeps the model, sites,
strength grid and shared tasks fixed across separately registered studies.
The preceding [profanity study](profanity_steering_execution.md) closed with a
development null. Rudeness and sarcasm have also closed with development nulls.
These are inference-time activation interventions, separate from physics
LoRA training. Further 14B work remains deferred.

Claude accepted both corrected extraction fixtures and found no blocking issues
in the shared implementation. The coordinator reviewed the code and completed
runbook. All 16 focused CPU checks passed before preparation. See the
[fixture](../results/reviews/isolated_persona_traits_fixture_review.md) and
[code](../results/reviews/isolated_persona_traits_code_review.md) reviews for
corrections, coverage and limits. Both corpora were frozen before new outputs.

## Rudeness

Registered in `2cfb68d` before calls, manifest
`64e542e53d7c1afc4e7c7a5f7b6f65a1e8326ebea21f1c5619028daed30e7b86`:
48 snapshot files, 1,401 protected records and 12 model files. Official Qwen3-4B
BF16 revision `1cfa9a7208912126459214e8b04321603b3df60c` was used unadapted on
guarded GPU 0. Existing jobs were left untouched.

Eligibility retained 31/32 authored pairs; one target was ambiguous, with no
replacement or rebalancing. All eight declared style families remained, with
9/11/11 opening/middle/closing positions. Closely worded families do not imply
independent semantic coverage. Only helpful system and user text entered the
model; evaluator context described a fictional single-turn exchange.

Equal-pair FP32 response-token mean differences produced directions at zero-based
blocks 17 and 23. Raw norms were 4.0291 and 11.0171; median neutral response-mean
norms R were 35.8986 and 70.1214. The fixed shifts were beta 0.05/0.10/0.20 times
R along each unit direction. The largest magnitudes were about 15.1% and 14.5%
of mean per-token residual norms. Serialization was exact, zero-vector generation
matched baseline tokens, and nonzero prefill/decode hooks changed hidden states
and logits and detached correctly. Strengths were locked in `1fc839f` before
development; unscored mechanical probes did not select settings.

| Setting | Usable rudeness | Content fulfilled | Token sequences changed versus baseline |
| --- | ---: | ---: | ---: |
| Baseline | 0/16 | 15/16 | — |
| Block 17, beta 0.05 | 0/16 | 16/16 | 7/16 |
| Block 17, beta 0.10 | 0/16 | 16/16 | 8/16 |
| Block 17, beta 0.20 | 0/16 | 15/16 | 12/16 |
| Block 23, beta 0.05 | 0/16 | 14/16 | 7/16 |
| Block 23, beta 0.10 | 0/16 | 15/16 | 12/16 |
| Block 23, beta 0.20 | 0/16 | 14/16 | 14/16 |

No development answer had clear or ambiguous rudeness, clear profanity/sarcasm,
incoherence or gross-failure flags. No setting met the target gate. Confirmation,
repeat confirmation rating, selected-vector generation reload, random-control
generation and capability comparison were skipped under the stopping rule.
Mechanical changes establish that the intervention affected outputs; they do not
establish target control or rule out other strengths, sites or representations.

All 22 grading batches passed first attempt: eight eligibility and 14 development,
176 rated texts. Actual work was 62 teacher-forced sequences and 122 generations
(10 preflight + 112 development), with no capability MC items. Model phases took
8.40/10.51/67.41 seconds, about 86.33 seconds total, with peak allocated memory
8.157 GB. These synchronized phase times include model loads but exclude CPU
verification, preparation and grading. Full verification hashes roughly 18.33 GB
of protected paths, plus 8.06 GB of model paths when requested; overlapping model
aliases increase this overhead. No checks were relaxed during the study.

The [aggregate report](../results/rudeness_steering_01.json) preserves provenance,
costs, vector metadata, metrics and the null. Raw outputs, grades and vectors stay
ignored under `outputs/rudeness-steering-01/`. A small historically used screen
and same-family AI ratings limit interpretation; this is no general capability
or human-validity claim.

The worker's independent read-only audit passed: all 1,401 protected bindings
and model/source hashes matched; journals, selection, vector provenance, hook
audits and public aggregates reconstructed correctly. The freshly generated
baseline matched all 16 earlier profanity-baseline answers exactly in tokens,
inputs, batch layout, flags and output bytes. This is a descriptive cross-check;
no prior generation or grade was substituted. Local audit note:
`outputs/workspace-maintenance-20261114/rudeness_postrun_verification.md`.

## Sarcasm

Registered in `6b08dbb` before calls, manifest
`fe43f452be3d2ab4501e7263f6e7cefa45e1bd85f4016b32d44314d2f0dbb12f`:
48 snapshot files, 1,401 protected records and 12 model files. It uses the same
official unadapted BF16 checkpoint, guarded GPU 0 and frozen settings.

All 32 authored pairs passed eligibility without replacement, retaining all
eight declared style families and 10/11/11 span positions. Scenario facts needed
for the irony are in the user prompt, visible to both model and evaluator.
The raw direction norms were 3.7318 and 8.9843 at blocks 17 and 23; R values
were 36.1693 and 70.6807. Tensor round-trip and actual zero/nonzero checks passed.
The fixed vectors/strengths were committed in `3684f8f` before development.

| Setting | Usable sarcasm | Content fulfilled | Token sequences changed versus baseline |
| --- | ---: | ---: | ---: |
| Baseline | 0/16 | 15/16 | — |
| Block 17, beta 0.05 | 0/16 | 15/16 | 7/16 |
| Block 17, beta 0.10 | 0/16 | 15/16 | 10/16 |
| Block 17, beta 0.20 | 0/16 | 13/16 | 11/16 |
| Block 23, beta 0.05 | 0/16 | 14/16 | 6/16 |
| Block 23, beta 0.10 | 0/16 | 13/16 | 8/16 |
| Block 23, beta 0.20 | 0/16 | 13/16 | 12/16 |

All three trait scores were zero throughout development: no clear or ambiguous
target/off-target behavior. There were no incoherence or gross-failure flags.
The registered target gate failed at every setting, and selection was null.
Confirmation, repeated confirmation rating, selected-vector generation replay,
random-control generation and capability comparison were skipped.

All 22 judging batches passed first attempt, rating 176 texts. Actual work was
64 teacher-forced sequences and 122 generations (10 preflight + 112 development),
with no capability MC items. Model phases took 8.56/10.59/83.71 seconds, about
102.86 seconds total, with peak allocated memory 8.157 GB. As above, these times
exclude preparation, repeated hash verification and grading. The
[aggregate report](../results/sarcasm_steering_01.json) records metrics and
provenance; raw outputs, judgments and vectors remain ignored under
`outputs/sarcasm-steering-01/`.

The worker's independent read-only audit passed: all 1,401 protected bindings,
model/source hashes, journals, vector/probe registrations, metrics and export
privacy matched. The fresh baseline matched all 16 answers from both earlier
studies exactly in tokens, inputs, layout, flags and output bytes; no stored
inference or ratings were substituted. Local note:
`outputs/workspace-maintenance-20261114/sarcasm_postrun_verification.md`.

## Stored-vector geometry: descriptive only

After sarcasm closed, the worker computed the
[reviewed diagnostic](../results/diagnostics/persona_vector_cosines_01.json)
from the saved FP32 unit vectors, using FP64 cosine calculations with a separate
`math.fsum` cross-check. Tensor, metadata, manifest, eligibility, strength and
report hashes matched before and after. No model or grading calls were added.

| Pair | Block 17 cosine | Block 23 cosine |
| --- | ---: | ---: |
| Profanity / rudeness | 0.554 | 0.693 |
| Profanity / sarcasm | 0.358 | 0.502 |
| Rudeness / sarcasm | 0.557 | 0.591 |

These saved directions are positively aligned in activation space. The
2560-dimensional isotropic random reference scale is about 0.020; observed
trait-versus-saved-random cosines range from -0.021 to 0.051. The saved random
units are exactly shared across traits within each block (seeds 4259/4265),
not six independent draws. The random scale is not a calibrated significance
threshold for learned directions.

Alignment could reflect shared authoring, inserted evaluative passages or
response-mean pooling as well as trait overlap. These small corpora do not
establish causal independence or entanglement, and cosine cannot explain a
behavioral null on its own. No cross-block comparisons, orthogonalization,
strength selection or subsequent experiment followed from this diagnostic.

## Joint stage and scope

All three bounded isolated studies closed with no validated setting in the
registered range. This does not establish that 4B cannot express the traits or
that other strengths, blocks or extraction methods would fail. Shared neutral
tasks and the sarcasm corpus's specific irony genre limit generalization.

Joint steering requires at least two supported isolated directions and a
separate reviewed protocol; full three-way control requires all three. No joint
experiment has run, and none of the three traits has a qualifying setting.
Closing the bounded studies does not achieve the
user's full 4B persona-control objective. Under development nulls, implementation
CPU checks and the executed early phases do not validate skipped confirmation,
selected-vector generation replay or capability retention.

The [next research choice](persona_next_decision.md), discussed with Claude,
is awaiting the user: one bounded stronger-magnitude 4B diagnostic, a revised
contrast/extraction design, or pausing after this negative outcome. No new
study, joint intervention or further 14B experiment is running.
