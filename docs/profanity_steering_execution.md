# Isolated Qwen3-4B profanity steering: development null

The bounded pilot is closed under its registered stopping rule. All six settings
produced **0/16 usable profane responses** on development, matching baseline
0/16. No setting qualified; confirmation, random-control generation, selected
vector-generation reload and the capability screen were skipped. This does not
establish an inability of the model to swear or a benefit from a larger model.

## Method and technical checks

The [protocol](profanity_steering_protocol.md) and exact inputs/source were
registered in `0b18182`, manifest
`a460e717117dede42bbe685a73fba3fcc04f640ad49dcbbfef418b29729fe917`.
It binds 38 snapshot files, 1,206 protected files and 12 model files. The worker's
29 focused CPU tests passed before preparation. Claude reviewed the design and
initial fixture/code; its final fixture re-review was unavailable after a
classifier interruption, so the coordinator accepted the corrected fixture and
implementation as documented in the reviews.

All 32 authored neutral/profane pairs passed blinded eligibility: seven word
families, 14 phrases, and 10/11/11 opening/middle/closing positions. Both sides
were teacher-forced under identical neutral system/user context. Equal-pair FP32
response-token mean differences supplied directions at zero-based blocks 17 and
23 of official unadapted Qwen3-4B BF16, revision
`1cfa9a7208912126459214e8b04321603b3df60c`. No model weights or physics adapter
were trained or changed.

Raw vector norms were 2.0703 and 6.0037. Median neutral response-mean norms R were
36.9209 and 69.4858. The fixed strengths were beta 0.05/0.10/0.20 times R along
each unit direction. Safetensors tensor round-trip was exact. The actual
zero-vector hook reproduced baseline tokens exactly; nonzero prefill/decode
hooks changed hidden states and logits and detached correctly. Vector/scale
hashes were committed in `62b91f7` before development.

## Development results

| Setting | Usable profanity | Content fulfilled | Token sequences changed vs baseline |
| --- | ---: | ---: | ---: |
| Baseline | 0/16 | 15/16 | — |
| Block 17, beta 0.05 | 0/16 | 14/16 | 6/16 |
| Block 17, beta 0.10 | 0/16 | 14/16 | 9/16 |
| Block 17, beta 0.20 | 0/16 | 13/16 | 13/16 |
| Block 23, beta 0.05 | 0/16 | 14/16 | 6/16 |
| Block 23, beta 0.10 | 0/16 | 13/16 | 11/16 |
| Block 23, beta 0.20 | 0/16 | 13/16 | 14/16 |

Raw lexical hits were also zero throughout. All three judged trait scores were
zero on all 112 development outputs: no ambiguous score-1 cases or judge-clear
profanity missed by the lexicon. All outputs ended with EOS; no cap hits, empty
answers, mechanical degeneration or gross-failure flags occurred. Stronger
settings changed more token sequences and tended to shorten responses, while
the desired trait remained absent. Token-change counts are a retrospective
mechanical diagnostic, not a new selection measure or validated behavior.
The [stored-output divergence record](../results/diagnostics/profanity_development_divergence_01.json)
also preserves first-divergence/length summaries and effective magnitude units;
it changes no original metric, selection or stopping rule.

All 22 grading batches (eight eligibility, 14 development) passed on their first
attempts, covering 176 texts. The fixed quality tolerances and primary target
gate were preserved. No stronger setting, alternative block, new extraction
method or negative coefficient was tried within this run.

## Cost and conclusion

The model phases took 8.43 seconds for extraction, 10.47 for mechanical preflight
and 82.26 for development, totaling about 101 seconds including model loads,
excluding CPU verification, preparation and grading. Peak allocated memory was
8.157 GB, including resident weights rather than total VRAM usage. Actual work:
64 teacher-forced sequences and 122 generations (10 preflight + 112 development).
No final generations or capability MC items ran. GPU 0 was used and existing
jobs were left untouched.

The [aggregate report](../results/profanity_steering_01.json) preserves counts,
costs, provenance, vectors and the development-null decision. Raw outputs and
candidate vectors remain ignored under `outputs/profanity-steering-01/`.
The implementation applied the intervention and changed outputs, but this
representation/site/strength grid did not demonstrate profanity control.
The study did not test prompted profanity or every possible steering method.
Small authored cases and same-family AI judgments limit interpretation.

The next isolated studies are rudeness and sarcasm on 4B, with separately
reviewed protocols and no tuning of this completed result. There is no validated
profanity setting for a joint control. Further 14B experiments remain deferred.
