# Sarcasm GCAD lower-strength closeout

**Retain GCAD base coefficient 2 on Qwen2.5-7B-Instruct for this bounded ordinary-reply task.** All registered heldout gates pass, with a **substantial 18.875-point coherence loss** and only **1.125 points of margin** inside the relative-coherence gate. This does not show that response quality is solved.

The [complete result](../results/sarcasm_gcad_low_strength_01.json), [execution review](../results/reviews/sarcasm_gcad_low_strength_execution_review.md) and [scientific review](../results/reviews/sarcasm_gcad_low_strength_scientific_review.md) preserve the measured scope. The [static controller profile](../configs/sarcasm_retained_controller.json) pins the local bank/settings; [the workflow](sarcasm_controller_workflow.md) describes reuse.

## What was tested

One prospective lower GCAD grid, coefficients 1/2/3, reusing the exact original 87-pair SARC bank and 64 baseline/prompted development answers. Those reused answers and 96 new steered answers were graded together in fresh blinded contexts. The original parent grid 3.5/5/7 remains a [closed development null](../results/sarcasm_method_transfer_01.json). Neither its result nor its grades were revised.

Native BF16 Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`, SDPA, default helpful system, greedy seed 42, batch 1 and cap 160 were fixed. GCAD uses the pinned author hook at zero-based blocks 9–19, scale 1.5, `inverse=False`, raw unnormalized SARC deltas.

## Development selection

| Arm | SARC mean | ≥50 /32 | Coherence | Capped /32 | Mean tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.00000 | 0 | 85.75000 | 32 | 160.00000 |
| prompted | 61.65625 | 27 | 76.43750 | 10 | 124.75000 |
| gcad_1 | 0.15625 | 0 | 86.06250 | 31 | 159.37500 |
| gcad_2 | 49.75000 | 16 | 74.68750 | 29 | 157.62500 |
| gcad_3 | 83.40625 | 32 | 61.31250 | 22 | 146.84375 |

GCAD 1 has insufficient sarcasm; GCAD 3 coherence 61.3125 falls below the relative floor 65.75. GCAD 2 is the only and lowest qualifier. Its selection, control tensors and metadata were committed before heldout generation.

## Heldout result: 32 questions per condition

| Arm | SARC mean | ≥50 /32 | Coherence | Capped /32 | Mean tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.06250 | 0 | 89.59375 | 32 | 160.00000 |
| prompted | 66.46875 | 29 | 77.12500 | 10 | 120.43750 |
| selected_gcad | 63.78125 | 29 | 70.71875 | 27 | 156.96875 |
| random_gcad | 0.00000 | 0 | 87.71875 | 32 | 160.00000 |

Selected sarcasm increases 63.71875 points over baseline and 63.78125 over random; 29/32 answers score at least 50. Every arm has 0 empty/invalid answers and 0 fixed 11-word profanity-lexicon hits. The selected coherence 70.71875 is above 69.59375, but only narrowly. Prompting achieves similar sarcasm with higher coherence 77.125 on these same questions; it is a descriptive reference, not a selection gate.

The selected arm reaches the 160-token cap on 27/32 answers; baseline/random reach it on 32/32. These are bounded outputs, not evidence of complete or uncapped answers.

## Coupling and random-control limits

| Condition | Descriptive impoliteness mean |
| --- | ---: |
| Baseline | 1.625 |
| Prompted | 49.90625 |
| GCAD 2 | 38.75 |
| Random | 1.65625 |

Within-selected sarcasm/impoliteness Pearson r=0.6370922702. Pooled r=0.9318638164 is **confounded by condition differences**. These descriptive correlations neither establish mechanism nor independent trait control. Off-axis scores were never a gate. Zero hits from a narrow profanity lexicon do not establish absence of all profanity.

The random control matches raw delta norms per layer and preserves keys, demands, sites and the gate formula. Actual averaged per-position shift norms were 4.0366705750 selected versus 3.1004033846 random; realized shifts were unequal, with no energy-matching claim.

## H14 sensitivity

Extraction x18 and heldout H14 overlap topically. Excluding H14 gives selected SARC 63.8387096774, coherence 70.6774193548 and 28/31 clear-sarcasm answers; baseline SARC 0.0645161290/coherence 89.4838709677. This predeclared 31-item figure is descriptive: no gate, selection or primary 32-item decision changes. Semantic independence of all prompts is not established.

## Cost and scope

Fresh work: **230 generations + 2 explicit logit forwards; 0 prompt/teacher-forced/MC work ; 88 first-valid native batches in 88 starts**. The 232 model-start files include the two logits. GPU stage wall time totals 1090.976 seconds, with 1055.486 seconds inside generation calls; peak PyTorch allocated memory 15,259,298,816 bytes. Stage wall excludes hashing, CPU preparation, human review and native grading; allocator peak is not total VRAM. Parent costs and 64 reused answer timings are separate historical work.

The artifact was reconstructed and generated/reviewed/graded within one Claude family. Questions were reused project material, held out only for this sarcasm selection; the held-out answers were generated afresh. This 32-question result measures sarcasm/coherence on bounded ordinary replies, not factual accuracy, capability, universal robustness, joint control or disentanglement. The [original-Mistral SAPS reference](../results/author_saps_mistral_01.json) uses a different model, representation and complaint-rewriting task; it remains a separate result.
