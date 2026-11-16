---
base_model: RenormalizationTech/Qwen3-4B-tuneLLM-NF4
base_model_relation: adapter
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags: [qwen3, lora, qlora, physics, experimental]
---

# Qwen3-4B tuneLLM PRL Writing QLoRA

**Experimental adapter selected by the registered development heuristic. Human writing-quality improvement is not established.**

**Measured math regression:** numeric GSM8K accuracy fell from 117/128 for the saved NF4 base to 111/128 for this adapter. The final results did not trigger reselection. Strict-format correct counts were 97 and 106, respectively.

This artifact records continued causal language-model training on cleaned paper LaTeX. It does not establish improved PRL acceptance, novelty, originality, human preference, or scientific correctness.

This repository was initially published privately. Historical cards, publication receipts, reproduction notes, provenance and manifests retain their original private status as records of the earlier publication; they do not describe current repository visibility.

Requires `RenormalizationTech/Qwen3-4B-tuneLLM-NF4` at revision `e9af208e44e083b77702a71d8aa65b7d97f78ce6`. The exported PEFT config pins that NF4 repository commit, not the upstream Qwen commit.
Lineage: `Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c`. Packed NF4 weights are restored from stored quantization; FP32 nonquantized modules match PEFT preparation. Adapters are FP32.

All-linear LoRA rank 128, alpha 256, dropout 0.05; LR 5e-05; seed 42; sequence length 1024.
Exported epoch 1, step 69: 522380 loss tokens. Full run: 3 epochs, 207 steps, 1567140 loss tokens. These budgets are distinct.

## Objective final measurements

| Condition | Task | Correct or writing diagnostic |
| --- | --- | --- |
| nf4_base | general_mc | 85/128 |
| nf4_base | physics_mc | 63/96 |
| nf4_base | gsm8k | 117/128 |
| nf4_base | ifeval | 24/32 |
| nf4_base | writing | 60/60 nonempty; 0 automatic fidelity flags; 0 generation caps; zero-shot prompts |
| nf4_adapter | general_mc | 93/128 |
| nf4_adapter | physics_mc | 67/96 |
| nf4_adapter | gsm8k | 111/128 |
| nf4_adapter | ifeval | 23/32 |
| nf4_adapter | writing | 60/60 nonempty; 3 automatic fidelity flags; 1 generation caps; zero-shot prompts |
| bf16_base | general_mc | 92/128 |
| bf16_base | physics_mc | 63/96 |
| bf16_base | gsm8k | 119/128 |
| bf16_base | ifeval | 24/32 |
| bf16_base | writing | 60/60 nonempty; 0 automatic fidelity flags; 0 generation caps; zero-shot prompts |
| bf16_control | general_mc | 95/128 |
| bf16_control | physics_mc | 64/96 |
| bf16_control | gsm8k | 114/128 |
| bf16_control | ifeval | 25/32 |
| bf16_control | writing | 60/60 nonempty; 7 automatic fidelity flags; 0 generation caps; zero-shot prompts |
| nf4_base_fewshot | writing | 60/60 nonempty; 0 automatic fidelity flags; 0 generation caps; two-example prompts |
| bf16_gguf | general_mc | 92/128 |
| bf16_gguf | physics_mc | 62/96 |
| bf16_gguf | gsm8k | 119/128 |
| bf16_gguf | ifeval | 25/32 |
| q8 | general_mc | 89/128 |
| q8 | physics_mc | 61/96 |
| q8 | gsm8k | 121/128 |
| q8 | ifeval | 25/32 |
| q4 | general_mc | 89/128 |
| q4 | physics_mc | 61/96 |
| q4 | gsm8k | 116/128 |
| q4 | ifeval | 24/32 |

| Condition | Held-out paper NLL | Perplexity |
| --- | ---: | ---: |
| nf4_base | 1.974938 | 7.206173 |
| nf4_adapter | 1.733950 | 5.662979 |
| bf16_base | 1.929116 | 6.883423 |
| bf16_control | 1.713637 | 5.549108 |
| nf4_repeat | 1.733926 | 5.662842 |

Scores include all reported conditions and any math regressions or BF16-control differences; see `evaluation.json` and `selection.json` for paired results, gate failures, paper NLL, and limits. A BF16 control uses BF16 frozen weights; NF4 uses prepared FP32 nonquantized modules. This compares workflow representations.
MMLU uses custom local zero-shot likelihood subsets, GSM8K uses numeric fallback, and IFEval uses its pinned official strict scorer. These are not full standard leaderboard scores. Lower paper NLL measures fit, not writing quality or benchmark capability.

## Data, attribution and reproduction

The exact prepared 70/10/20 paper splits are in `reproduction_data/`. All 100 paper attributions, versioned source URLs, PRL DOIs and verified CC BY4.0 licenses are in `paper_attributions.json`. Cleaning removed comments/hidden blocks, author/funding/back matter and preserved scientific equations, citations and used macros; consult corpus provenance and the source archive for exact processing. Source versions may precede final PRL wording. Cleaned text is not guaranteed to be compilable standalone LaTeX.
Public release exposes the prepared test split and frozen evaluation fixtures. Their historical held-out measurements remain as reported, but future tests on these disclosed items are no longer clean held-out evaluations; new independent data are needed for future generalization claims.

Paper text is CC BY4.0; model Apache2.0 does not override that license. MMLU and GSM8K dataset cards specify MIT; IFEval specifies Apache2.0. These three benchmark datasets are evaluation-only, not physics training data. `data_licenses.json` and `benchmark_dataset_licenses.json` record the separate terms and pinned verification. Git history, credentials, raw source archives and base-model weights are not in this adapter payload.
`reproduction.zip` contains tracked code/config/docs without history; executable source is unchanged and verified against measured modules. Metadata/documentation path redaction is recorded. Start from the pinned NF4 dependency, exact prepared data and frozen fixtures; do not re-query live paper selection. See `REPRODUCE.md` and `environment.lock.txt`.

Human writing assessment is deferred; it is not a required pending human action for completion of the computational workflow. No human quality-improvement claim is made. `human_review/packet.json` remains an optional blinded comparison of the selected zero-shot adapter with the saved NF4 base given two training-paper examples. A/B identities are hidden separately for each pair. Its random answer key is retained locally outside this payload; provenance records its SHA256 commitment. Automated flags cannot establish human preference or scientific fidelity. Do not treat absence of flags or insignificant paired tests as equivalence.

The additive `benchmark_summary.json` provides compact objective scores and paired comparisons. Historical `evaluation.json`, provenance, selection records and release manifests remain unchanged, including their original pending-review status labels. The earlier GSM8K adapter’s 121→101/128 result used a different protocol and is not directly comparable with this experiment’s 117→111/128 result. These limited screens do not establish a universal quantization verdict or writing-quality improvement.

## Development gate outcomes

### Learning-rate pilot

- physics-r32-lr5e5: passed; failed checks: none recorded. GSM8K net correct-count drop vs base: 2.
- physics-r32-lr1e4: failed; failed checks: writing_truncated_increase. GSM8K net correct-count drop vs base: 4.

### Adapter ranks

- physics-r8-lr5e5: passed; failed checks: none recorded. GSM8K net correct-count drop vs base: 2.
- physics-r32-lr5e5: passed; failed checks: none recorded. GSM8K net correct-count drop vs base: 2.
- physics-r128-lr5e5: passed; failed checks: none recorded. GSM8K net correct-count drop vs base: 1.


## Reported limitations

- One small paper corpus; arXiv versions can precede final PRL wording; existing pretraining exposure unknown.
- No matched PRB corpus control, so lower text loss is not evidence of PRL-specific style learning.
- Paper splits are disjoint but some authors overlap; theory labels were AI-reviewed.
- Loss uses identical1024-token within-paper chunks, not whole-paper context.
- Small benchmark subsets and unadjusted exploratory paired tests do not establish broad capability equivalence.
- GGUF and HF inference use different backends and batch sizes; decoding settings and input IDs are checked.
- Automatic fidelity flags and exact text overlap are diagnostics, not human scientific or style judgments.
- Ranks receive equal maximum training budgets; their nominated checkpoints can come from different epochs.
- Rank8's minimum occurred at the three-epoch budget boundary; convergence beyond that budget was not tested.
- The BF16 control uses the NF4-selected rank, learning rate and epoch; it is not independently optimized.

- Frozen corpus QA records 5 replacement characters across 2 papers. The proper-name review and unchanged split hashes are in `corpus_quality.json`; prepared scientific data remain unchanged.
