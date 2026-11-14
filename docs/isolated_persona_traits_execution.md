# Isolated rudeness and sarcasm on Qwen3-4B: execution

The [protocol](isolated_persona_traits_protocol.md) keeps the model, sites,
strength grid and shared tasks fixed across separately registered studies.
The preceding [profanity study](profanity_steering_execution.md) closed with a
development null. Rudeness has also closed with a development null; sarcasm is
next. These are inference-time activation interventions, separate from physics
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

## Sarcasm and joint stage

Sarcasm uses its already accepted separate corpus and the same fixed method.
Its registration and execution remain pending here. Joint steering requires at
least two supported isolated directions and a separate reviewed protocol; full
three-way control requires all three. No joint experiment has run and no useful
profanity or rudeness setting has been validated by these studies.
