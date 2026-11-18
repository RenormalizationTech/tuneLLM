# tuneLLM research documentation

**Quantization, LoRA/QLoRA and activation steering of open-source language models.**

tuneLLM is an AI-assisted research portfolio studying compression, domain
adaptation, refusal and written traits. The experiments compare interventions
against fixed baselines and report quality costs, unsuccessful transfers and
retained experimental references together.

Start with the [results and saved examples](showcase.md), then use the
[reproduction guide](reproduce.md) to inspect an artifact or run the portable
CPU checks. [GitHub](https://github.com/RenormalizationTech/tuneLLM) contains the
code and compact results; [Hugging Face](https://huggingface.co/RenormalizationTech)
hosts nine published artifacts, including the evaluated 27B NF4 and Q4_K_M bases.

## Research tracks

| Track | Question | Main evidence |
| --- | --- | --- |
| Quantization | What changes when a model is compressed? | 4B BF16/Q8/Q4 measurements and [fixed-subset capability comparisons](benchmark_summary.md). |
| Physics QLoRA | How does domain-text adaptation affect fit and general tasks? | Lower paper-text NLL with a numeric math regression and better answer formatting; [training and evaluation](physics_prl_execution.md). |
| Refusal steering | How do subtraction and projection affect willingness and usefulness? | A dose/quality tradeoff and a selected benign case demonstration; [7B execution record](refusal_7b_transfer_execution.md). |
| Trait steering | Can internal interventions change style while preserving coherent answers? | GCAD/SAPS references with quality costs and coupling; [consolidated evidence](steering_reference.md). |

![7B refusal steering development counts](assets/refusal_tradeoff.svg)

The figure separates refusal reduction from answer usefulness. These are
AI-judged development counts; the weaker-dose candidate was graded in a separate
packet from the reused baseline. The later selected-case diagnostic is a
separate result. [Figure data, interpretation and examples](showcase.md).

## A short reading path

1. [Results and examples](showcase.md): the main tradeoff and two selected saved responses.
2. [Activation steering methods](activation_steering_methods.md): the intervention equations and implementation choices.
3. [Four-behavior evidence](steering_reference.md): model-specific results, controls and limits.
4. [Reproduce](reproduce.md): portable CPU tests, pinned inference bundles and separate environments.
5. [Methods and references](references.md): published approaches and source attribution.
6. [File organization and storage rules](file_organization.md): where new files belong and how history is retained.

## Published artifacts

- [Qwen3-4B GGUF](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-GGUF): BF16, Q8_0 and Q4_K_M inference representations.
- [Qwen3-4B NF4 base](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-NF4): saved unadapted quantized base and reload checks.
- [Physics QLoRA adapter](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA): experimental domain-text training with evaluation and prepared data.
- [7B trait controls](https://huggingface.co/RenormalizationTech/qwen2.5-7b-gcad-steering-controls): activation banks and a standalone runtime.
- [7B refusal experiment](https://huggingface.co/RenormalizationTech/qwen2.5-7b-refusal-steering-experiment): frozen direction, runtime and selected-case findings.

The [release index](../releases/README.md) records current HF revisions and local
card/payload locations. Steering bundles use separately obtained upstream base
weights; they are distinct from LoRA adapters.

## Current scope and historical records

The original 4B/7B studies are complete. A text-only Qwen3.8-27B NF4
load/save/reload smoke has completed on two simple prompts, with exact token
replay after reload and a failed strict one-step numerical cache comparison.
It is a functional check, not a quality benchmark. The subsequent
[27B release plan](qwen38_release_plan.md) is complete: NF4, Q4_K_M and separate
QASPER QA/evidence and PRL grounded-abstract adapters are public. QASPER passed
the fixed custom text-only task and retention gates. PRL passed retention but
failed abstract-quality gates and is experimental; the unadapted base is preferred
for that task. The separate sarcasm study is deferred to reserve GPU capacity.
See the [current roadmap](project_tracks.md).

All Markdown documents under `docs/` are available through this site. Protocols,
closeouts and execution notes preserve their historical decisions and wording;
they are not all current setup instructions. Raw run directories and machine
paths mentioned there may require retained local artifacts. Use the current
reproduction guide for the public entry points.

For source and data terms, read [third-party notices](../THIRD_PARTY_NOTICES.md).
The [project README](../README.md#implementation-and-attribution) describes the
owner's direction and the roles of Codex and Claude.
