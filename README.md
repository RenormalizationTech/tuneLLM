# tuneLLM

A reproducible LLM research portfolio covering **quantization, LoRA/QLoRA,
refusal steering and trait steering**. Experiments on Qwen3-4B and
Qwen2.5-7B-Instruct examine compression, adaptation and inference-time behavior
control, with explicit baselines and response-quality measurements.

The goal is to demonstrate practical LLM research: implement methods, reproduce
results, compare alternatives and explain limitations. Positive effects, failed
transfers and quality tradeoffs are all retained. This is an experimental
portfolio; the results do not establish production-ready behavioral controls.

## Results at a glance

| Area | What was implemented | Main measured result | Interpretation |
| --- | --- | --- | --- |
| Quantization | BF16 GGUF conversion; Q8_0/Q4_K_M quantization; saved NF4 base | GGUF weights: **7.50 → 3.99 → 2.33 GiB**; short A6000 generation benchmark: **79 → 131 → 186 tokens/s** | Compression and throughput gains in a fixed microbenchmark; capability checked separately on small subsets. |
| Physics QLoRA | NF4 frozen base, LoRA training on 100 licensed physics papers, checkpoint selection and reload | Held-out paper-text NLL **1.975 → 1.734**; GSM8K **117/128 → 111/128** | Better fit to paper text with a math regression; improved writing quality is unestablished. |
| Refusal steering | Contrastive direction extraction; constant subtraction versus projection removal on 4B/7B | 7B subtraction −0.5: unsafe refusals **28/30 → 17/30**, benign usefulness **20/30 → 13/30**. Selected benign diagnostic at −0.25: refusals **3/13 → 1/13**, useful valid full answers **9/13 → 11/13** | A causal intervention effect with dose/quality tradeoffs; the 13 selected cases are a narrow demonstration, not a general success rate. |
| Trait steering | Literature-based GCAD/SAPS references; random controls; model transfer; joint-control study | 7B GCAD sarcasm: **29/32** answers scored ≥50/100 versus **0/32** baseline; coherence **89.59 → 70.72**. Joint candidates passing all gates: **0/5** | Individual effects can be strong while quality and cross-trait coupling limit combined control. |

These rows use different tasks, models and evaluation designs. They are not a
single leaderboard. Behavioral scores use an AI judge; coherence does not imply
factual correctness. Detailed counts, controls, decoding limits and provenance
are linked below.

## Published artifacts

| Artifact | Contents | Start here |
| --- | --- | --- |
| [Qwen3-4B GGUF](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-GGUF) | BF16, Q8_0 and Q4_K_M weights, measurements and reproduction notes | [Quantization record](results/initial_quantization.json) |
| [Qwen3-4B NF4](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-NF4) | Saved unadapted bitsandbytes checkpoint with verified reload | [Model workflow](docs/model_workflow.md) |
| [Physics QLoRA](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA) | Experimental adapter, prepared data, evaluation and reproduction package | [Physics study](docs/physics_prl_execution.md) |
| [7B trait-steering controls](https://huggingface.co/RenormalizationTech/qwen2.5-7b-gcad-steering-controls) | Profanity, impoliteness and sarcasm banks; standalone runtime; joint-study guide | [Four-behavior evidence](docs/steering_reference.md) |
| [7B refusal-steering experiment](https://huggingface.co/RenormalizationTech/qwen2.5-7b-refusal-steering-experiment) | Frozen refusal direction, inference runtime, experimental setting and results | [7B refusal study](docs/refusal_7b_transfer_execution.md) |

The refusal direction reduces refusal generally, including on unsafe contrasts;
it is not a benign-only refusal correction. The selected 13 cases came from
earlier 4B refusal cases, and the −0.25 diagnostic produced one repetitive output.

The steering releases contain activation interventions, not modified base-model
weights or LoRA adapters. Their base checkpoints must be obtained separately.
The earlier GSM8K adapter was archived and its HF repository deleted; its
[negative learning result](results/qlora_real_01.json) remains documented.

## Research questions and findings

**Compression:** what changes when the same checkpoint is represented in BF16,
Q8, Q4 or NF4? The [fixed-subset benchmark](docs/benchmark_summary.md) separates
accuracy measurements from size/speed tests. Backend and batching differences
limit cross-runtime timing comparisons; close scores do not prove equivalence.

**Adaptation:** does domain-text training improve the desired task? The physics
adapter reduced paper-text loss but did not establish improved scientific
writing. Its math regression is reported alongside the loss gain. See the
[protocol](docs/physics_prl_protocol.md) and [results](results/physics_prl_final.json).

**Refusal control:** how do constant vector subtraction and projection removal
affect willingness and answer quality? The [method guide](docs/activation_steering_methods.md)
explains both interventions. Stronger subtraction reduced more refusals in our
fixed settings, while introducing quality costs. The original acceptance
criteria were not met; no confirmatory final/retention run followed the 7B
transfer. The later selected-case diagnostic preserves those decisions and
reports two useful rescues plus one repetitive output. See the
[execution record](docs/refusal_7b_transfer_execution.md).

**Trait control:** can an extracted representation induce sarcasm, impoliteness
or profanity while retaining coherent replies? Individual 7B references show
measured effects with different quality costs. Sarcasm narrowly passed its
coherence-loss criterion and also increased impoliteness. The
[joint study](docs/gcad_and_combined_control.md) found strong coupling and no
candidate passing every criterion. The [fixed 4B transfer](docs/qwen3_4b_fixed_transfer_closeout.md)
induced traits but failed quality gates; checkpoint differences do not isolate
a model-size effect.

## Reproduce and inspect

Start with a released artifact's model card and its pinned revisions; each
release documents its own environment. The training, GCAD and baseline
experiments used different dependency versions, so there is no single verified
environment for every track.

| Track | Runnable entry point / reproduction guide |
| --- | --- |
| GGUF | [Released reproduction guide](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-GGUF/blob/main/REPRODUCE.md) |
| QLoRA | [Adapter reproduction guide](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA/blob/main/REPRODUCE.md) and [training code](scripts/physics_train.py) |
| Refusal | Released `refusal_runtime.py` CLI; [original runbook](scripts/refusal_7b_benign_transfer_runbook.md) for historical context |
| Traits | Released `runtime.example` and joint guide; [sarcasm workflow](docs/sarcasm_controller_workflow.md) |

For the original local 4B baseline, after the setup in
[the execution notes](docs/execution.md):

```bash
.venv/bin/python -m scripts.env_check --output outputs/environment.json
.venv/bin/python -m scripts.download_model --config configs/baseline.json
.venv/bin/python -m scripts.hf_baseline --output outputs/my-baseline
```

Historical runbooks document completed runs and may require preserved local
artifacts; they are not instructions to overwrite or repeat frozen studies.
The public test fixtures retain their historical held-out status only for the
reported runs. Future generalization studies need independent data.

## Next: a bounded Qwen3.8-27B extension

The selected larger-model direction is **sarcasm steering, quantization and one
QLoRA exercise**. Reuse the existing methods, corpus and evaluation structure
where compatible; verify architecture/runtime support and resource needs first.
The official BF16 checkpoint has approximately 55.6 GB of weight files; an
approximately 18 GB download refers to a quantized representation. Quantized
inference and QLoRA can use a smaller-memory path than the BF16 reference.
Report each intervention against its own baseline. These are planned tasks,
not completed results. ARA, broad parameter sweeps, joint-controller retuning
and production deployment are deferred.

See [the current roadmap](docs/project_tracks.md), [model workflow](docs/model_workflow.md)
and [release index](releases/README.md). Older protocol documents remain as
records of the experiments as registered; current priorities supersede their
pending-expansion language.

## Implementation and attribution

Python, PyTorch, Transformers, PEFT, bitsandbytes, llama.cpp and Hugging Face Hub
support the workflows. Protocols, configurations and compact results live in
`docs/`, `configs/` and `results/`; scripts and evaluation fixtures accompany
them. Weights, caches and large raw outputs remain in ignored directories.
Source and dataset attribution are recorded per experiment and release.

Planning and implementation used Codex with Claude scientific review; evaluation
records disclose the AI-judge adaptations and limitations. This project applies
and evaluates existing methods rather than claiming to have invented them.
