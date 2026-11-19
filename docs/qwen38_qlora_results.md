# 27B QLoRA: QASPER and PRL adapters

Two independent QLoRA adapters were trained from the same Qwen3.8-27B NF4
base. **QASPER passed its registered custom-subset task and retention gates.**
**PRL failed its abstract-quality gates** and is published as an experimental
negative result. The unadapted base is preferred for that PRL task.

The [QLoRA background](qlora_foundations.md) explains low-rank adaptation;
the [27B quantization results](qwen38_quantization_results.md) describe the base.
Training changes adapter weights; the separate [sarcasm controls](qwen38_sarcasm_results.md)
change activations during inference.

## Shared training design

The adapters are neither mixed nor trained sequentially onto each other.
Each uses one configuration and seed, rank 16 and alpha 32, with checkpoint
selection by minimum development response loss. Lower loss selects a checkpoint;
it does not establish better task answers.

| Adapter | Objective | Optimizer steps | Selected checkpoint | Training GPU-hours |
| --- | --- | ---: | ---: | ---: |
| QASPER | Paper question answering, abstention and paragraph evidence IDs in JSON | 61 | 61 | 5.11 |
| PRL | Grounded abstract from paper-body excerpts | 24 | 24 | 2.26 |

The [QASPER training record](../results/qwen38_qasper_training_01.json) and
[PRL training record](../results/qwen38_prl_training_01.json) retain the exact
targets, data, losses and resource accounting. Prompt text is masked from the
supervised response loss. The PRL support audit retained 57/70 training papers
and 9/10 development papers, without quality-driven data or parameter sweeps.

## QASPER: gain beyond prompting

The primary final analysis uses **309 questions** from a custom text-only
subset, with **403 questions** in the full sampled evaluation. A schema-example
prompt was selected using development data before final generation.

| Final arm | Strict primary Answer F1, n=309 |
| --- | ---: |
| Unadapted base | 0.457 |
| Development-selected prompt | 0.567 |
| QASPER adapter | **0.690** |

The adapter gains **0.123** over the prompt, with paired paper-bootstrap 95%
interval **[0.083, 0.165]**. Lenient parsing also passes: gain 0.106,
interval [0.068, 0.147]. Evidence F1 on the registered evidence-strict subset
(267 questions) rises from 0.699 for the better baseline to 0.759 for the adapter.
There are **zero invalid JSON outputs over all 403 questions**.

The [task report](../results/qwen38_qasper_final_science_01.json) retains subgroup
results and every gate. The [combined final report](../results/qwen38_qasper_final_01.json)
adds completed retention checks. Recommendation is limited to this registered
custom task; it is not a full official QASPER benchmark or general scientific
comprehension claim.

The baseline phase originally stopped at 395/403 questions when its time cap
expired. A [prospective completion amendment](../results/protocols/qwen38-qasper-baseline-completion-amendment-01.json)
allowed the eight never-evaluated questions to complete once, in 244.55 seconds.
The original 395 predictions were retained byte-for-byte. Complete final
coverage does not mean that the original uninterrupted phase met its cap.

## PRL: quality gates failed

The final comparison uses **40 papers** and **80 valid blinded AI review calls**,
two passes per paper. Adapter preference is **0.175**, with 95% paper-bootstrap
interval **[0.075, 0.2875]**. The registered abstract-quality gates fail despite
lower development loss and higher reference overlap. Retention passing does
not rescue this result.

The [science report](../results/qwen38_prl_final_science_01.json) retains preference,
support labels and overlap diagnostics; the [combined final report](../results/qwen38_prl_final_01.json)
adds completed retention. Support labels and preference share AI responses and
are correlated; they are not independent human factual validation. Test papers
were new to this project, not guaranteed absent from pretraining.

Development judging was interrupted by a usage quota. The
[registered fallback](../results/protocols/qwen38-prl-development-fallback-01.json)
selected the base prompt and aliased the second baseline to it. A stronger-prompt
advantage was not established; the two baseline names are not independent arms.
Final review later completed with the quality gates unchanged.

## Matched retention checks

Both adapters pass the registered point-estimate regression limits:

| Arm | General MC /128 | GSM8K numeric /128 | IFEval strict /32 |
| --- | ---: | ---: | ---: |
| Matched unadapted base | 110 | 114 | 23 |
| QASPER adapter | 109 | 119 | 26 |
| PRL adapter | 110 | 120 | 24 |

These are fixed loss limits, **not statistical noninferiority findings** or
broad capability guarantees. Matched retention uses FP32 preparation of the
floating base modules, unlike the native-BF16 quantization screen. A
[prospective batch-four amendment](../results/protocols/qwen38-retention-batch-amendment-01.json)
binds identical grouping across arms; two of four mechanics probes differed
from batch one, so no token-equivalence claim follows. Science generation
remains batch one. The [development transfer probe](../results/qwen38_transfer_analysis_01.json)
is descriptive: only cross-task cells concern transfer; same-task cells overlap
checkpoint-selection data.

## Public adapters and reproducibility

- [QASPER adapter](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-QASPER-QLoRA),
  current revision `95359154d2b9d94914977462e559c409c1f549c6`;
  [anonymous download and GPU reload](../results/qwen38_qasper_public_verification_01.json).
- [PRL abstract adapter](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-PRL-Abstract-QLoRA),
  current revision `3bb77b924555bdcc9f92b9c4aa6cf32d63340fbc`;
  [anonymous download and GPU reload](../results/qwen38_prl_public_verification_01.json).

The [catalog](../releases/catalog.json) records both current and GPU-verified
revisions, with runtime-file hashes preserved across card updates. The cards
give the supported NF4 base, runtime, prompts and data terms. No merged adapter
GGUF is released. The [reproduction guide](reproduce.md) links the separate
environment and registered launchers.

The [quantization/QLoRA closeout](../results/qwen38_science_summary_01.json)
records **24.44 GPU-hours** across its shared quantization, training, evaluation
and mechanics pools; this excludes the later sarcasm studies. The
[historical implementation plan](qwen38_release_plan.md) retains the original
decisions, and the [final project closeout](../results/qwen38_project_closeout_02.json)
records all five public 27B packages. No additional training is scheduled.
