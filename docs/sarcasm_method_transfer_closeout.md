# Original Qwen7 sarcasm transfer closeout

The registered study closed **development_null**: no SAPS or GCAD setting met both the sarcasm and coherence requirements. Extraction and mechanics passed. This outcome is preserved in the [complete result](../results/sarcasm_method_transfer_01.json) and [execution review](../results/reviews/sarcasm_method_transfer_execution_review.md).

The pinned Qwen2.5-7B BF16 model answered32 calibration questions at greedy seed42, batch1, cap160. Target gain had to be>=20, at least8/32 answers had to score>=50, coherence had to be>=50 and within20 of baseline, and at most3 answers could be mechanically invalid. Baseline coherence85.875 made the effective floor65.875.

| Arm | Mean sarcasm | Mean coherence | SARC>=50 /32 | Outcome |
| --- | ---: | ---: | ---: | --- |
| Baseline | 0 | 85.875 | 0 | Reference |
| Prompted | 58.125 | 75.34375 | 25 | Positive reference |
| SAPS1 | 0 | 87.0625 | 0 | Target criteria fail |
| SAPS2 | 0 | 89.28125 | 0 | Target criteria fail |
| SAPS4 | 54.3125 | 43.75 | 18 | Coherence criteria fail |
| GCAD3.5 | 87.125 | 52.40625 | 32 | Relative coherence criterion fails |
| GCAD5 | 92.84375 | 38.84375 | 32 | Coherence criteria fail |
| GCAD7 | 92.0625 | 25.53125 | 32 | Coherence criteria fail |

All arms had zero mechanical invalid answers and zero fixed-lexicon profanity hits. Prompting elicited the target style; several interventions also increased sarcasm, but their quality scores failed the registered requirements. Coherence is not an accuracy or helpfulness guarantee.

Completed work:200 elicitation+10 mechanics+256 development=466 generations;300 SAPS prompt forwards,261 GCAD teacher-forced forwards,2 explicit logits and0 MC items. All115 native grading/artifact calls were first-valid. Recorded GPU stage wall time was2304.121 seconds (38.402 minutes), excluding grading and command-boundary hashing; generation wall time was2156.340 seconds.

No method was selected; no held-out or retained-controller result exists. The predeclared [H14 exclusion sensitivity](../results/diagnostics/sarcasm_h14_sensitivity_01.json) is not_applicable. No held-out answers were generated for that analysis.

The study uses reconstructed inputs and one Claude evaluator family, with generator/reviewer/rater overlap. Baseline32/32 answers reached cap160. These limits and the ordinary-reply task transfer constrain interpretation; this is a completed bounded negative selection result, not a claim that the target style is absent or universally uncontrollable. The [original protocol](sarcasm_method_transfer_protocol.md), frozen data, labels and decisions remain unchanged.
