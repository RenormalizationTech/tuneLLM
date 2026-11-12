# tuneLLM: LLM Quantization and Fine-Tuning Learning Lab

## 1. Project Objective

The primary goal is hands-on familiarity with open-source LLM training and
tuning. The first deliverables are a quantized model and a working QLoRA
fine-tuning workflow, with reproducible evaluation and eventual Hugging Face
releases. Research-level behavioral steering is an optional extension.

Priority order:

1. Load and evaluate a small open-source development model.
2. Convert and quantize it for inference; compare against a baseline.
3. Fine-tune with QLoRA on a small, clearly licensed dataset, initially
   physics-oriented if a suitable dataset is readily available.
4. Evaluate, reload, document, and release selected artifacts on Hugging Face.
5. Consider larger models and behavioral representation research afterward.

Learning the Hugging Face, PyTorch, PEFT/QLoRA, GGUF, and llama.cpp workflows
is itself a successful outcome. Novel research findings are not required to
complete the initial project.

## Current status and execution priorities (2026-11-12)

- GitHub repository: created; this working directory is the project repository.
- Hugging Face organization: `RenormalizationTech` (confirmed by user).
- Hugging Face artifact repositories: not created; no artifacts uploaded.
- Environment verified: Python 3.10, PyTorch 2.6.0+cu124, both A6000 GPUs visible.
- Small-model BF16 baseline: two successful Qwen3-4B runs with identical generated tokens;
  7/8 on the basic smoke fixture. See `docs/execution.md` for evidence and limitations.
- Team: coordinating Codex manages scope and acceptance; Claude reviews planning
  and scientific assumptions; a separate Codex implements and validates code.

The stage numbers below organize topics; the first work follows Stages 0–3.
Stages 4–10 are an optional research backlog and do not gate quantization,
QLoRA, or the initial artifact releases. Qwen3-32B is a possible later scale-up,
not an initial requirement. Any follow-up model ID and architecture must be
verified before selection.

### Initial acceptance criteria

- Baseline: a revision-pinned small model loads and generates from saved prompts;
  record environment, chat template, generation settings, and thinking mode if applicable.
- Quantization: convert the upstream checkpoint to GGUF and produce Q8_0 and
  Q4_K_M first. Confirm both reload; compare size, memory, throughput, fixed
  responses, and a small scored evaluation. Q5/Q6 and a larger model are optional.
- QLoRA: perform a short training smoke run before a longer run; use frozen
  quantized base weights with trainable adapters. Keep train/validation/test
  splits separate, compare held-out domain and general behavior against the
  untuned base, and verify the saved adapter reloads with its recorded base.
  Successful execution and honest evaluation are required; improvement is measured,
  not assumed. Statistical research claims are not an initial requirement.
- Release: publish selected evaluated outputs with a model card, upstream model
  revision, license information, dataset provenance, reproduction instructions,
  and measured limitations. Start with GGUF files and an adapter; merging the
  adapter and exporting a tuned GGUF are optional subsequent exercises.

GGUF inference quantization and the bitsandbytes 4-bit/NF4 loading used for
QLoRA are separate workflows. Do not use the inference GGUF as the QLoRA
training checkpoint; both paths start from the recorded upstream model.
Keep runtime differences explicit when interpreting performance comparisons.

### Local storage

Use ignored project-root directories for large or generated files:

- `artifacts/`: downloaded model snapshots, adapters, and exports.
- `data/`: downloaded or processed datasets.
- `outputs/`: training checkpoints, raw generations, and run logs.
- `.hf/`: local Hugging Face cache, configured explicitly when running tools.

Keep code, configurations, small evaluation fixtures, and compact result summaries
in Git. Before downloads/conversion, check disk space for source weights,
high-precision GGUF, quantized outputs, and temporary files. Avoid duplicate
base-model copies and keep credentials out of version control.

### Immediate work order

1. Verify the environment, GPUs, available disk, and upstream model/tool support.
2. Select one small model and pin its revision; establish minimal fixed evaluation.
3. Run baseline inference and record measurements.
4. Convert and quantize to Q8_0 and Q4_K_M; reload and compare.
5. Prepare a small licensed dataset and run QLoRA; reload and evaluate the adapter.
6. Create artifact repositories and model cards under the existing Hugging Face
   organization, then upload selected validated outputs.

Model cards live as `README.md` files in the artifact repositories. The exact
organization ID, artifact names, and visibility must be resolved before upload.

### Minimal implementation approach (reviewed with Claude)

Complete both quantization and QLoRA on the same small model, approximately 4B,
using one available GPU initially. Qwen/Qwen3-4B is a candidate with an upstream
model card; pin its revision and verify compatibility with the chosen package
versions and llama.cpp commit before downloading. Larger models are optional.

Use Python, PyTorch and Transformers for inference and Transformers Trainer for
training; bitsandbytes supplies NF4 loading and PEFT supplies LoRA adapters.
Use datasets for data handling, huggingface_hub for model transfers, and
Accelerate as required by Trainer. Use the pinned llama.cpp conversion,
quantization, inference and benchmark tools for GGUF. Keep its conversion
dependencies separate if they conflict with the training environment.

Prefer a handful of readable scripts and saved configurations. Begin with local
JSON/CSV logs, a fixed prompt set, and a small scored task set with shared scoring
rules across Hugging Face and llama.cpp. Keep chat templates, thinking mode,
generation settings, evaluation data and model revisions explicit. Within
llama.cpp, compare high-precision and quantized GGUFs using the same settings;
investigate large discrepancies against the Hugging Face baseline.

Before QLoRA training, inspect a decoded batch to verify assistant-token loss
masking, padding, and end-of-sequence handling. Run a tiny training smoke test,
then train on a small licensed dataset with separate validation and test data.
Save and reload the adapter, and report held-out domain performance and general
retention without assuming improvement. Save run metadata with each result.

Multi-GPU training, TRL, DeepSpeed, hosted tracking, a web interface, additional
quantization formats, and custom training loops can wait until a concrete need
arises. Merging and requantizing the tuned model is a follow-up exercise.

### Candidate next training experiment (not yet run)

After the two-step smoke and adapter reload pass, use a fixed 1,024-row subset
of the prepared GSM8K training split for one epoch. Claude suggested an
effective batch of 16 (64 optimizer steps), attention LoRA rank 8, learning rate
2e-4 with a short warmup and decay, and checkpoints at steps 32 and 64.
Measure actual memory before increasing the per-device batch from the smoke
setting; gradient accumulation can provide the same effective batch.

Use validation data for checkpoint decisions. Reserve a fixed final test subset
that excludes the 16 smoke-test rows and do not use it to tune prompts or training
settings. Compare the NF4 base and selected adapter on the same examples.
Record strict-format and numeric-answer scores separately, output length,
truncation, and paired fixed/broken answers. A small trained-row probe and the
validation loss help distinguish memorization from held-out fit. Reference loss
or format improvement alone does not establish better reasoning. This proposal
is a follow-up learning experiment, not a completed result or a release claim.

------------------------------------------------------------------------

## 2. Hardware

Primary compute:

-   2 x NVIDIA RTX A6000
-   48 GB VRAM per GPU
-   96 GB aggregate VRAM across the two GPUs

This should comfortably support development and inference with
27B-32B-class models, quantization experiments, and QLoRA training,
subject to sequence length, batch size, optimizer state, and multi-GPU
configuration.

------------------------------------------------------------------------

## 3. Model Strategy

### Development model

Use a smaller Qwen-family model, roughly 4B-8B, for:

-   debugging scripts;
-   testing Hugging Face loading;
-   testing GGUF conversion;
-   testing activation hooks;
-   testing evaluation code;
-   rapid LoRA/QLoRA experiments.

Do not debug the entire pipeline on a 30B-class model.

### Primary experimental model

**Qwen3-32B**

Reasons:

-   conventional dense Transformer architecture;
-   suitable for activation extraction and representation-engineering
    experiments;
-   large enough to be behaviorally interesting;
-   manageable on 2 x RTX A6000;
-   mature Hugging Face ecosystem;
-   Apache 2.0 licensing.

### Optional follow-up architecture

Select a different architecture only after the small-model learning loop and
any larger-model experiment are stable. The original draft mentioned
Qwen3.8-27B, but that exact model ID and its architectural claims have not been
verified. It is not an execution target. Confirm the upstream model card,
license, architecture, and tool support before selecting a follow-up.

The project should avoid unnecessary assumptions specific to one model.

------------------------------------------------------------------------

## 4. Repository Strategy

Use **GitHub for continuous development** and **Hugging Face for
validated model/data artifacts**.

### GitHub

GitHub is the primary research repository and contains:

-   source code;
-   experiment configurations;
-   evaluation scripts;
-   training scripts;
-   steering code;
-   documentation;
-   benchmark results;
-   reproducibility metadata.

Frequent commits are expected.

### Hugging Face

Hugging Face contains publishable artifacts:

-   LoRA/QLoRA adapters;
-   merged models when useful;
-   GGUF releases;
-   datasets;
-   model cards;
-   benchmark metadata.

Do not upload every intermediate training checkpoint.

General rule:

> GitHub commits = development and experiments.\
> GitHub tags/releases = reproducible milestones.\
> Hugging Face commits = meaningful artifact updates.\
> Hugging Face repositories = models/datasets worth sharing.

------------------------------------------------------------------------

## 5. Proposed GitHub Layout

``` text
llm-behavior-lab/
├── README.md
├── plan.md
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── models/
│   ├── quantization/
│   ├── qlora/
│   ├── steering/
│   └── releases/
│
├── scripts/
│   ├── download_model.py
│   ├── benchmark.py
│   ├── convert_gguf.py
│   ├── quantize_gguf.py
│   └── train_qlora.py
│
├── quantization/
│   ├── evaluate.py
│   └── compare_outputs.py
│
├── finetuning/
│   ├── physics_dataset.py
│   └── train_qlora.py
│
├── steering/
│   ├── collect_activations.py
│   ├── extract_direction.py
│   ├── refusal.py
│   ├── profanity.py
│   ├── rudeness.py
│   ├── sarcasm.py
│   └── steering.py
│
├── evaluation/
│   ├── perplexity.py
│   ├── task_eval.py
│   ├── behavior_eval.py
│   └── representation_similarity.py
│
├── notebooks/
│
└── results/
```

Large checkpoints and generated model binaries should not be committed
to GitHub.

------------------------------------------------------------------------

## 6. Proposed Hugging Face Organization

Create a Hugging Face organization if the project is intended to become
public and collaborative.

Example artifact structure:

``` text
<organization>/
├── Qwen3-32B-Physics-LoRA
├── Qwen3-32B-Physics
├── Qwen3-32B-Physics-GGUF
├── Qwen3-32B-Behavior
├── physics-training-dataset
└── behavior-contrastive-dataset
```

Do not duplicate the upstream base model unnecessarily. Reference it in
model cards, for example:

``` yaml
base_model: Qwen/Qwen3-32B
license: apache-2.0
```

Each released artifact should record:

-   upstream model and exact revision when possible;
-   GitHub repository;
-   Git commit/tag used to generate it;
-   training/quantization configuration;
-   dataset/version;
-   evaluation results;
-   intended use and limitations.

------------------------------------------------------------------------

# 7. Research Roadmap

## Stage 0 - Infrastructure

### Goals

-   Create GitHub repository.
-   Create Hugging Face account/organization structure.
-   Set up Python environment.
-   Verify both RTX A6000 GPUs.
-   Establish configuration-driven experiments.
-   Add experiment logging.

### Success criterion

A small Qwen model can be loaded and queried reproducibly from a clean
environment.

------------------------------------------------------------------------

## Stage 1 - Hugging Face Baseline

Start with the development model, then move to Qwen3-32B.

Tasks:

-   download model/tokenizer;
-   inspect architecture and parameter dtypes;
-   run inference;
-   measure memory usage;
-   benchmark tokens/second;
-   test single-GPU versus multi-GPU loading where appropriate;
-   establish a fixed evaluation prompt set.

Record the exact upstream model revision.

### Baseline measurements

At minimum:

-   parameter count;
-   checkpoint size;
-   VRAM usage;
-   generation throughput;
-   first-token latency;
-   perplexity on a fixed evaluation set;
-   selected downstream task scores.

------------------------------------------------------------------------

## Stage 2 - GGUF and Quantization

Learn the complete path:

``` text
Hugging Face checkpoint
        |
        v
FP16/BF16 GGUF
        |
        +--> Q8_0
        +--> Q6_K
        +--> Q5_K_M
        +--> Q4_K_M
```

Additional quantizations may be explored later.

### Measurements

For each representation measure:

-   file size;
-   RAM/VRAM usage;
-   loading time;
-   tokens/second;
-   perplexity change;
-   task-performance change;
-   qualitative response changes.

Construct a benchmark table such as:

  Model      Size   Memory   tok/s   Perplexity   Task score
  -------- ------ -------- ------- ------------ ------------
  BF16
  Q8_0
  Q6_K
  Q5_K_M
  Q4_K_M

### Research question

How aggressively can the model be quantized before measurable capability
degradation becomes significant?

------------------------------------------------------------------------

## Stage 3 - Physics Domain Adaptation with QLoRA

Use physics-oriented training material to learn the modern PEFT/QLoRA
workflow.

Conceptually:

``` text
quantized/frozen base model + trainable LoRA adapters
```

or

\[ W\_{`\mathrm{effective}`{=tex}} = W\_{`\mathrm{base}`{=tex}} + BA. \]

### Tasks

-   build a legally usable physics training corpus;
-   define training/validation splits;
-   establish a physics evaluation set before training;
-   configure QLoRA;
-   train adapters;
-   compare checkpoints locally;
-   select a validated adapter;
-   publish the selected adapter to Hugging Face;
-   optionally merge/export a standalone model;
-   optionally generate GGUF variants.

### Evaluation

Measure both:

1.  **Domain improvement**
    -   physics QA/problem-solving;
    -   terminology;
    -   scientific reasoning;
    -   relevant held-out tasks.
2.  **General capability retention**
    -   baseline task performance;
    -   general perplexity;
    -   catastrophic forgetting or unwanted style changes.

------------------------------------------------------------------------

## Optional Stage 4 - Refusal Representation Baseline

Before studying more complicated behavioral traits, reproduce a known
representation-engineering phenomenon.

The initial target is **refusal behavior on benign over-refusal
examples**.

Construct contrastive examples and extract candidate activation
differences:

\[ `\Delta `{=tex}h_i = h_i\^{`\mathrm{refuse}`{=tex}} -
h_i\^{`\mathrm{answer}`{=tex}} \]

and candidate refusal direction:

\[ v\_{`\mathrm{refusal}`{=tex}} = `\frac{1}{N}`{=tex}`\sum`{=tex}\_i
`\Delta `{=tex}h_i. \]

Test causal intervention:

\[ h_l' = h_l - `\alpha `{=tex}v\_{`\mathrm{refusal}`{=tex}}. \]

The purpose of this stage is to establish that the activation-extraction
and steering infrastructure works reproducibly before moving to less
well-defined traits.

### Questions

-   At which layers is refusal most separable?
-   Is one direction sufficient?
-   How stable is the direction across prompt subsets?
-   Does intervention vary smoothly with steering strength?
-   Does steering damage unrelated capabilities?
-   Does the representation survive quantization?

Use benign over-refusal evaluation as the primary benchmark rather than
treating safeguard bypass as the project objective.

------------------------------------------------------------------------

## Stage 5 - Behavioral Directions

After validating the refusal pipeline, investigate three increasingly
complex traits:

1.  **Profanity**
2.  **Rudeness**
3.  **Sarcasm**

The expected progression in difficulty is approximately:

\[ `\mathrm{refusal}`{=tex} `\rightarrow`{=tex}
`\mathrm{profanity}`{=tex} `\rightarrow`{=tex} `\mathrm{rudeness}`{=tex}
`\rightarrow`{=tex} `\mathrm{sarcasm}`{=tex}. \]

Profanity has strong lexical manifestations, while sarcasm depends
substantially on context and pragmatics.

For each trait construct matched/contrastive data and estimate candidate
directions:

\[ v_t = `\mathbb{E}`{=tex}\[h_t\^+ - h_t\^-\]. \]

Do not assume beforehand that every trait is truly one-dimensional.

------------------------------------------------------------------------

## Stage 6 - Behavioral Geometry

Study the relationship among the learned directions.

For directions (v_i) and (v_j), calculate:

\[ `\cos`{=tex}`\theta`{=tex}\_{ij} = `\frac{v_i\cdot v_j}`{=tex}
{\|v_i\|\|v_j\|}. \]

Questions include:

-   Are profanity and rudeness strongly correlated?
-   Is sarcasm more independent?
-   Do directions change across layers?
-   Are the representations genuinely one-dimensional?
-   Can multiple human-defined traits be described by a smaller latent
    behavioral space?

Construct a matrix of contrastive activation differences and use
PCA/SVD:

\[ `\Delta `{=tex}H = U`\Sigma `{=tex}V\^`\top`{=tex}. \]

Estimate the effective dimensionality of behavioral representations.

Orthogonalization may be explored if correlated directions prevent
independent control, but the original directions should always be
retained for scientific comparison.

------------------------------------------------------------------------

## Stage 7 - Continuous Inference-Time Behavioral Controls

The long-term practical target is a parameterized model:

\[ h_l' = h_l
-`\alpha`{=tex}*{`\mathrm{refusal}`{=tex}}v*{`\mathrm{refusal}`{=tex}}
+`\alpha`{=tex}*p v*{`\mathrm{profanity}`{=tex}} +`\alpha`{=tex}*r
v*{`\mathrm{rudeness}`{=tex}} +`\alpha`{=tex}*s
v*{`\mathrm{sarcasm}`{=tex}}. \]

An eventual API might resemble:

``` python
generate(
    prompt,
    refusal=0.0,
    profanity=0.2,
    rudeness=0.5,
    sarcasm=1.0,
)
```

The goal is not merely binary persona switching. Test whether each
parameter behaves approximately continuously:

``` text
alpha = 0.0  -> baseline
alpha = 0.5  -> moderate effect
alpha = 1.0  -> strong effect
```

Also test negative coefficients where meaningful.

### Key requirement

Changing one parameter should be evaluated for unintended movement along
the other behavioral dimensions.

------------------------------------------------------------------------

## Stage 8 - Behavioral Robustness Under Quantization

Connect the representation-engineering work back to the quantization
study.

For example compare:

\[ v\_{`\mathrm{sarcasm}`{=tex}}\^{BF16},
v\_{`\mathrm{sarcasm}`{=tex}}\^{Q8},
v\_{`\mathrm{sarcasm}`{=tex}}\^{Q6},
v\_{`\mathrm{sarcasm}`{=tex}}\^{Q5},
v\_{`\mathrm{sarcasm}`{=tex}}\^{Q4}. \]

Measure representation similarity where technically comparable:

\[ S = `\cos`{=tex}`\left`{=tex}( v_t\^{BF16}, v_t\^{Q4}
`\right`{=tex}). \]

More importantly, test causal behavior:

> Does an equivalent steering intervention still produce the intended
> behavioral change after quantization?

### Main research question

**How robust are controllable behavioral representations under numerical
compression?**

A working hypothesis worth testing is that simpler/stronger features
such as profanity may survive aggressive quantization better than
context-dependent features such as sarcasm.

------------------------------------------------------------------------

## Stage 9 - Interaction with Physics QLoRA

Study whether domain adaptation changes behavioral geometry.

Compare:

\[ v_t\^{`\mathrm{base}`{=tex}}
`\quad`{=tex}`\text{vs}`{=tex}`\quad`{=tex}
v_t\^{`\mathrm{physics}`{=tex}}. \]

Questions:

-   Does physics QLoRA alter refusal behavior?
-   Does it rotate behavioral directions?
-   Does domain specialization affect tone?
-   Can the same steering vectors transfer between the base and adapted
    models?
-   Are changes concentrated in specific layers?

This connects all three themes:

\[ `\boxed{\text{quantization}}`{=tex} +
`\boxed{\text{domain adaptation}}`{=tex} +
`\boxed{\text{behavioral representation}}`{=tex}. \]

------------------------------------------------------------------------

## Stage 10 - Cross-Architecture Validation

After the pipeline is stable on Qwen3-32B, repeat selected experiments
on Qwen3.8-27B.

Do not necessarily repeat every experiment.

Focus on questions such as:

-   Does refusal remain low-dimensional?
-   Are profanity/rudeness/sarcasm directions similarly separable?
-   Is effective behavioral dimensionality architecture-dependent?
-   Does the newer hybrid architecture respond differently to
    quantization?
-   Do layer-wise representation patterns differ qualitatively?

------------------------------------------------------------------------

# 8. Evaluation Philosophy

Every modification should be compared against a fixed baseline.

Avoid relying solely on subjective chatbot impressions.

Use three levels of evaluation:

### Numerical

-   model size;
-   memory;
-   throughput;
-   latency;
-   perplexity.

### Capability

-   general benchmark subset;
-   physics-specific evaluation;
-   reasoning/coding tests where relevant.

### Behavioral

-   refusal/over-refusal rate;
-   profanity score;
-   rudeness score;
-   sarcasm score;
-   cross-trait interference;
-   steering-strength response curves.

For behavioral evaluation, combine automated scoring with manually
inspected fixed examples.

------------------------------------------------------------------------

# 9. Reproducibility

Each important experiment should save:

-   model name;
-   model revision/hash;
-   tokenizer revision;
-   Git commit;
-   configuration file;
-   random seed;
-   CUDA/PyTorch/Transformers versions;
-   GPU information;
-   quantization method;
-   dataset version;
-   training hyperparameters;
-   evaluation results.

Prefer configuration files over hard-coded experimental parameters.

Example:

``` yaml
base_model: Qwen/Qwen3-32B
experiment: physics_qlora_v1

training:
  method: qlora
  rank: 64
  learning_rate: ...
  epochs: ...

quantization:
  formats:
    - Q8_0
    - Q6_K
    - Q5_K_M
    - Q4_K_M
```

For every Hugging Face release, record the Git commit/tag that generated
the artifact.

------------------------------------------------------------------------

# 10. Artifact Publication Policy

During experimentation, keep intermediate checkpoints local.

Example:

``` text
/data/models/
├── checkpoint-100/
├── checkpoint-200/
├── checkpoint-300/
└── checkpoint-400/
```

After evaluation:

``` text
training
   |
   v
local checkpoints
   |
   v
evaluation
   |
   v
select validated artifact
   |
   v
Hugging Face release
```

Publish artifacts when they are:

-   useful to others;
-   reproducible;
-   evaluated;
-   meaningfully distinct from an existing release.

Use HF repository revisions for updates to the same artifact. Create
separate repositories when the artifact has a genuinely different
purpose, such as a LoRA adapter versus GGUF inference files.

------------------------------------------------------------------------

# 11. Milestones

## M0 - Environment

-   [x] Create GitHub repository
-   [x] Decide project name: tuneLLM
-   [x] Create Hugging Face organization (reported by user)
-   [x] Configure environment
-   [x] Verify both A6000 GPUs
-   [x] Run development model (Qwen3-4B, twice)

## M1 - Quantization

-   [x] Run BF16/FP16 baseline
-   [x] Convert model to GGUF
-   [x] Generate Q8_0
-   [ ] Generate Q6_K
-   [ ] Generate Q5_K_M
-   [x] Generate Q4_K_M
-   [x] Benchmark initial BF16, Q8_0, Q4_K_M variants (small comparison)
-   [ ] Optional Q5/Q6 comparisons
-   [ ] Publish useful GGUF artifacts/results

## M2 - Physics QLoRA

-   [x] Prepare first licensed dataset: GSM8K; physics specialization deferred
-   [x] Establish held-out smoke evaluation (16 validation/test rows each)
-   [x] Run two-step QLoRA smoke and verify fresh-base adapter reload
-   [ ] Run a larger bounded training experiment
-   [ ] Benchmark domain improvement
-   [ ] Check general capability retention
-   [ ] Publish selected adapter

## M3 - Refusal Baseline

-   [ ] Construct benign refusal contrastive dataset
-   [ ] Collect layer activations
-   [ ] Extract refusal direction/subspace
-   [ ] Identify useful intervention layers
-   [ ] Demonstrate causal steering
-   [ ] Measure side effects

## M4 - Behavioral Directions

-   [ ] Profanity dataset/direction
-   [ ] Rudeness dataset/direction
-   [ ] Sarcasm dataset/direction
-   [ ] Layer-wise analysis
-   [ ] Cross-direction similarity analysis
-   [ ] PCA/SVD analysis

## M5 - Behavioral Control API

-   [ ] Continuous profanity control
-   [ ] Continuous rudeness control
-   [ ] Continuous sarcasm control
-   [ ] Measure cross-trait interference
-   [ ] Evaluate negative and positive steering strengths

## M6 - Quantization x Behavior

-   [ ] Compare behavioral representations across precision levels
-   [ ] Compare steering effectiveness across precision levels
-   [ ] Measure effective dimensionality
-   [ ] Analyze which traits are most robust

## M7 - Domain Adaptation x Behavior

-   [ ] Compare base and physics-adapted behavioral geometry
-   [ ] Test steering-vector transfer
-   [ ] Measure QLoRA-induced behavioral changes

## M8 - Cross-Architecture

-   [ ] Repeat selected experiments on Qwen3.8-27B
-   [ ] Compare conventional Transformer and hybrid architecture
-   [ ] Document architecture-dependent findings

------------------------------------------------------------------------

# 12. Immediate Next Steps

Do these first:

1.  Use the existing tuneLLM repository.
2.  Record the existing Hugging Face organization ID before publication.
3.  Create artifact repositories/model cards when validated outputs are ready.
4.  Set up the Python/CUDA environment on the A6000 machine.
5.  Select a small Qwen development model.
6.  Write a minimal inference script.
7.  Verify both the Hugging Face/PyTorch and llama.cpp paths.
8.  Establish the fixed baseline evaluation suite **before** modifying
    the model.
9.  Run the first GGUF conversion.
10. Quantize the development model to Q8_0 and Q4_K_M and verify the
    complete benchmark pipeline.

After the small-model quantization pipeline is reproducible, complete a small
QLoRA workflow. Scaling to Qwen3-32B and research experiments is optional.

------------------------------------------------------------------------

# 13. Central Research Questions

The following optional research questions connect later extensions. They do not
replace the initial learning goals or gate quantization and QLoRA deliverables.

### Quantization

**How much numerical compression can an LLM tolerate before capability
and behavior change substantially?**

### Domain adaptation

**How effectively can QLoRA introduce physics-domain specialization
while preserving general capability?**

### Representation engineering

**Can known and increasingly complex behaviors be identified as
low-dimensional activation representations and causally controlled at
inference time?**

### Unified question

**How robust and low-dimensional are controllable behavioral
representations in LLMs under quantization and domain adaptation?**

This unified question connects the practical engineering stages into one
coherent research program.
