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

## Current scope clarification (2026-11-14)

Physics LoRA/QLoRA demonstrates the ability to fine-tune on domain data. It is
independent of the behavioral program: refusal activation steering, followed by
persona/tone research. Persona prompting supplies an inference-time reference;
it does not train an adapter or apply an activation vector. Physics/math
questions in a behavioral evaluation are content checks, not physics training.

See [project tracks and selected persona targets](docs/project_tracks.md).
That map supersedes older default trait priorities below and historical
transition language in immutable study records. The physics learning
milestone and refusal studies are complete; the 4B persona pipeline is open.
P1 was a prompt-only reference with
a failed acceptance gate. Persona activation studies first test profanity,
rudeness and sarcasm individually on Qwen3-4B. Simultaneous control is deferred
for later discussion and literature review. Complete the 4B pipeline before further 14B work.
Each live study requires its own committed registration; the current status
below distinguishes completed runs from accepted designs.

## Current priority (2026-11-15): original-model references

The user selected original open author models and published steering methods
before further4B adaptations or method comparisons. See
[the source and execution plan](docs/author_model_reproduction_plan.md) and
[source audit](results/reviews/author_model_source_audit_01.md). The first bounded
reference is GCAD impoliteness on Qwen2.5-7B, with the author's residual baseline
and an unsteered control. SAPS sarcasm uses Mistral-7B-v0.1. Profanity follows the
paper procedure with disclosed reconstructed inputs because supplementary data
could not be verified. No new OpenAI API setup; existing Claude evaluation is an
explicit substitution. Old4B results remain unchanged; joint control and14B are
deferred. The [GCAD first-reply reference](docs/author_gcad_impolite_closeout.md)
completed with all three registered contrasts passing: GCAD impoliteness77.37
and coherence75.90 versus baseline1.62/82.68; residual84.52/7.83. Historical
status below is retained for provenance.
The [profanity paper-method study](docs/author_profanity_method_closeout.md)
closed before steering: reconstructed minimal-edit data failed the registered
decodability gate. A revised dataset or validation scheme needs a new study;
the user approved one [controlled data revision](docs/author_profanity_revision_protocol.md).
The [original-Mistral SAPS reference](docs/author_saps_mistral_closeout.md)
completed all1800 rewrites and90 first-valid grading batches. SAPS1.4 reached
276/450 clear-sarcasm ratings vs0 baseline, with meaning4.11→2.98/5. Both
methods show a style/quality trade-off. The [saved profanity direction test](docs/author_profanity_direction_test_closeout.md)
completed 224 calibration responses: baseline and all six strengths had 0/32
lexical-swearing responses and 0/10 semantic profanity. Intervention mechanics
passed, but no strength qualified for held-out testing. A separate
[three-method transfer](docs/profanity_method_transfer_closeout.md) subsequently
confirmed GCAD 5 on 32 held-out questions: PF 31.13/100, lexical swearing 14/32,
coherence 82.69 versus baseline 85.34; baseline/random PF 0. Fixed/SAPS did not
qualify. GCAD is retained in the [profanity workflow](docs/profanity_controller_workflow.md).
No joint or14B work is released.

**Latest user clarification:** reliable individual control is the goal. Prefer
GCAD if effective, with trait-specific method/settings fallback accepted.
Cross-effects describe correlations and coupling; strong coupling is reported,
not a rejection or mandatory disentanglement target. Joint control is optional.
The [2026-11-16 sequence](docs/political_refusal_research_plan.md) is now:
new 7B political-information refusal/GCAD study and same-model method comparisons;
then a fixed-workflow Qwen3-4B control with all outcomes reported; GitHub and
Hugging Face releases; then Qwen3.8-27B and Gemma 4. No model-size causal claim
or expected 4B failure is assumed. Dataset/source review precedes the new study.
The political-information protocol and licensed fixture are now registered.
Grading calibration passed all 16 authored examples in four first-valid batches.
The original network-isolated attempt produced no labels and is preserved;
one registered recovery keeps its failed start charged to the same budget.
The 96-answer Qwen2.5-7B baseline/reference screen is complete: sensitive-topic
explicit refusal was 0/32, while reference coverage was 51.6% (prompted 62.5%).
The frozen rule selected the available coverage branch, but no steering ran.
See [the measured status](docs/political_refusal_headroom_status.md). Claude
recommends a user scope decision before the substantially larger omission
study: continue coverage research, finish this refusal screen and proceed
to the 4B controls, or try another published benign refusal pool. This is not
a GCAD success or failure.
The [sarcasm transfer](docs/sarcasm_method_transfer_closeout.md) closed with a
development null: GCAD induced sarcasm but the tested strengths lost too much
coherence, and SAPS also had no qualifying setting. The separately reviewed
[lower-strength study](docs/sarcasm_gcad_low_strength_closeout.md) is complete:
GCAD2 passed held-out confirmation (SARC63.78 versus baseline0.06/random0;
coherence89.59→70.72, a narrow1.125-point margin). It also increased
impoliteness1.63→38.75; coupling remains descriptive. The retained
[workflow](docs/sarcasm_controller_workflow.md) uses the same bank and criteria.
This bounded tuning is closed; no further sweep is released.
The [refusal headroom screen](docs/refusal_headroom_closeout.md) closed at
3/100 published benign development prompts, below the predeclared threshold
of five. The secondary prefix count was five, exposing classification
sensitivity. No GCAD refusal study or automatic dataset/model fallback is
released. No joint grid is released.

## Current status and execution priorities (2026-11-14)

- Latest user-selected sequence, reviewed with Claude: return to Qwen3-4B for
  isolated profanity, rudeness and sarcasm steering, in that order. Each trait
  gets construction, extraction, bounded development, locked confirmation,
  save/reload and reporting. Score all three traits even in isolated tests.
  Consolidate literature-based isolated workflows first. Joint control remains
  experimental and deferred for later discussion. Finish this pipeline on 4B before
  further 14B work; retain the completed 14B baseline as historical evidence.
- Trait-rating calibration: complete/pass after a separately registered
  infrastructure completion. Both blinded passes agreed with all 48 intended
  labels for each trait; kappa was 0.9922/0.9851/0.9685. All 12 completion
  batches passed first try, with 13 cumulative invocations including the preserved
  initial sandbox DNS failure. All 1,100 protected bindings passed. See
  `docs/persona_rating_calibration_completion_execution.md`. This supports only
  clear-case repeatability, not human validity or subtle generated styles.
  No vector or target-model generation ran in calibration.
- Isolated profanity on 4B: closed with a development null. All 32 pairs passed
  eligibility; six settings at blocks 17/23 produced 0/16 raw/usable profanity.
  Hooks changed 6–14/16 token sequences, with content fulfillment 13–14/16 versus
  baseline 15/16. Confirmation, reload generation and retention were skipped.
  See `docs/profanity_steering_execution.md`; all 22 judge batches passed first
  try. The separately reviewed rudeness/sarcasm implementation passed 16 CPU
  checks with both corpora frozen before outputs.
- Isolated rudeness on 4B: closed with a development null. Eligibility retained
  31/32 pairs; all six settings produced 0/16 raw/usable target outputs, changing
  7–14/16 token sequences. Content fulfillment was 14–16/16 versus baseline
  15/16, with no incoherence/gross failures. All 22 judge batches passed first
  try. See `docs/isolated_persona_traits_execution.md`.
- Isolated sarcasm on 4B: also closed with a development null. All 32 pairs
  qualified; all six settings produced 0/16 raw/usable target outputs, changing
  6–12/16 token sequences. Content fulfillment was 13–15/16 versus baseline
  15/16, with no incoherence/gross failures. All 22 judge batches passed first
  try. No isolated control qualified in those historical pilots; the individual
  4B control milestone remains open. Joint steering is optional. The user then selected
  `docs/persona_literature_workflow.md`: author methods/data first, isolated
  rudeness (PV), sarcasm (SAPS), profanity (explicit PV new-trait adaptation).
  Source-tested implementations are complete. Rudeness and profanity full PV
  grids closed with development nulls. SAPS continuation completed a held-out
  paraphrase reference pass (27/100 reaching author sarcasm ≥3 versus 0 baseline/random), with
  reduced meaning preservation; compact capability checks passed. See `docs/persona_reference_execution.md`. The
  stronger-magnitude proposal was not run. Joint control is experimental,
  deferred for later discussion; no joint or further 14B run started.
- Qwen3-4B refusal study: closed after the final published-only replay. All 250
  token/input/output/layout/flag comparisons matched, with 681 protected bindings
  unchanged and no new grading. See `docs/qwen3_4b_closeout.md` and
  `results/xstest_replay_01.json`. The partial-effect/quality-tradeoff conclusion
  remains; no validated useful refusal setting or additional 4B run is claimed.
  This closes that refusal study, not the new 4B persona experiments.
- The coordinating Codex selected Qwen3-14B BF16 on one A6000 with Claude review
  under the user's direction to progress to a larger model. Official assets are
  pinned, downloaded and hash-verified; native tokenization matches the 250 4B
  inputs. The registered baseline is complete: safe candidate refusals 16/250,
  unsafe partial/full refusals 160/200, and all 57 grading batches accepted on
  their first attempts. The 48-item quality screen scored 10/16 general MMLU,
  9/16 physics MMLU, 8/8 strict GSM8K and 7/8 prompt-strict IFEval. See
  `docs/qwen3_14b_baseline_execution.md` and `results/qwen3_14b_baseline_01.json`.
  No intervention ran; reliable steering and size-related stability remain
  unestablished. The next study needs its own reviewed protocol.
- Refusal follow-up: the published XSTest-safe baseline adaptation is complete.
  All 250 responses were graded in 32 first-valid calls: 233 compliant, seven
  partially refusing and ten fully refusing. The 17/250 (6.8%) candidate refusals
  pass the practical headroom flag, not a legitimacy/repeatability or intervention
  test. There were 154 cap hits at the fixed 256-token budget. See
  `docs/xstest_baseline_execution.md`; source choices and Qwen3 compatibility
  limits are in `docs/refusal_reproducibility_review.md`. No steering sweep or
  automatic follow-up is active.
- A separately registered contextual Claude review of 19 stored XSTest items is
  complete: 16 review-supported candidate unnecessary refusals, one questionable
  primary refusal label and two supported compliant labels on lexical-only flags.
  Original grades remain unchanged; see `docs/xstest_stored_output_audit.md`.
  The user selected published prompts only for the next replication; the custom
  paraphrase proposal is not selected. The subsequent registered exact replay is
  complete as recorded above; no further 4B generation is scheduled. The refusal
  experiments are complete, with no validated reliable steering configuration.
- New reusable workflow: `docs/model_workflow.md`; Qwen3-4B writing protocol:
  `docs/physics_prl_protocol.md`. These supersede the initial smoke workflow for
  the physics-writing exercise, while preserving the historical results.
- Original quantization and QLoRA learning milestone: complete. The earlier GSM8K
  experiment already covered instruction tuning; the physics experiment used
  paper-text continued causal language modeling.
- The approved reporting, tone-feasibility and positive-refusal pilot sequence
  is complete. Discuss management planning with Claude first; the dedicated
  Codex worker implements code.
- Reporting cleanup is published and hash-verified in all four Hugging Face
  repositories; only cards and new compact summaries changed. See
  `docs/benchmark_summary.md` and `releases/README.md` for scores and exact
  documentation revisions.
- Tone contrast-construction feasibility is complete: upper bounds 5, 31 and 46
  did not reach the registered 50-pair requirement, so no experiment ratings,
  vectors or final tests ran. See `docs/tone_steering_execution.md`. Reusing a
  smaller existing set is optional exploratory backlog, not an automatic retry.
- Positive refusal pilot: completed as registered with a null at layer 17.
  Baseline benign refusals were 1/30; strengths 1/2 gave 2/30 each; strength 4
  degenerated on all 30. No setting passed, so final generation was skipped.
  Claude accepted the result. See `docs/refusal_execution.md` and
  `results/refusal_01.json`. No safety or model-improvement claim is supported.
- The user-selected layer-23 follow-up completed with no qualifying setting.
  Benign refusals rose from reused baseline 1/30 to 4/30, 7/30 and 24/30 at
  strengths 0.25/0.5/1, but strength 1 had six degenerate outputs and weaker
  settings missed the refusal threshold. This is an exploratory dose-related
  signal with a quality tradeoff, not a validated usable setting. Final
  generation was skipped and the refusal pilot is closed under its rule.
  See `docs/refusal_late_execution.md` and `results/refusal_02.json`.
  That positive-intervention pilot is closed; the new reduction study is
  specified below. NF4 transfer and new training remain deferred.
- Historical refusal-reduction and activation-steering study assessed willingness
  and answer quality separately, with persona deferred. The user
  authorized proceeding if Claude declined; it declined the core and limited
  measurement review. This design is not Claude-approved. The coordinating
  Codex owns scientific acceptance; the dedicated worker implements/tests.
- Refusal-reduction pilot 03: complete. Baseline and two fixed interventions
  were compared on development; subtraction at block 23 was selected and
  locked before final. Final valid answers on the refusal-triggering class
  rose from 1/30 to 7/30, but benign usefulness fell from 29/42 to 23/42,
  with two safe and one unsafe mechanical degeneration flags. Final quality
  criteria failed despite the tiny capability screen meeting its tolerances.
  This is a tradeoff result, not a validated usable configuration. See
  `docs/refusal_reduction_execution.md`, `results/refusal_03.json`, and
  `results/refusal_reduction_summary_03.json`. No reselection or further
  sweep is active; original positive results and model weights are unchanged.
- Run-03 stored-output diagnosis: complete with Claude advisory review.
  Nineteen unique pairs were inspected retrospectively, with no new target
  generation and no change to original ratings. For nine benign losses the
  reviewer coded seven as main-answer-present before cutoff, one before the
  main answer, and one unclear. Apparent errors, repetition/drift and mixed
  causes remain; a cap increase alone is not a sufficient proposed fix.
  See `docs/refusal_output_diagnosis.md` and the two-method explanation in
  `docs/activation_steering_methods.md`. All 315 original input hashes stayed
  unchanged. The review is unblinded qualitative evidence, not a new benchmark.
- Benign-only Phase A survey: complete through a separately registered grading
  completion that preserves the original incomplete study and valid 20 labels.
  All 36 requests (24 legitimate sensitive and 12 physics) finished with EOS,
  zero cap/empty/mechanical-degeneration flags, in 28.81 seconds including load.
  First-batch grading covered 20 items; the remaining batch returned 15/16 on
  both allowed attempts and stopped. The user authorized completion of only the
  remaining 16, split into two batches of eight; both passed first try.
  See `docs/benign_helpfulness_execution.md` and `results/benign_helpfulness_01.json`.
  Combined coverage is 36/36; the unchanged decision rule gives
  `insufficient_headroom`, with 0/24 sensitive refusals and no coordinator
  disagreement. Descriptive AI quality counts are 101/108 anchors and 11/36
  apparent incorrect/unsupported flags. See
  `docs/benign_helpfulness_completion_execution.md` and
  `results/benign_helpfulness_completed_01.json`. No candidate, held-out
  generation, activation vector or harmful-compliance optimization ran within
  that baseline survey.
  An evaluator CLI auto-update was handled by restoring
  the audited installed version with a documented process-local pin. The user
  selected finishing the grades and then moving to benign tone/persona work.
  That prompt-only pilot is now complete as recorded below. Prior gentler activation
  suggestions remain unregistered hypotheses, not an active run.
- Persona P1: completed as a registered development null, after Claude design,
  fixture and result review. Both matched system conditions generated 24/24 EOS
  with no cap/empty/degeneration flags; three grading batches passed first try.
  The fixed persona won 15/24 style comparisons (16 needed) and incurred two
  gross quality losses (one allowed), despite three gains and a net acceptable
  count change of 17 to 18. All calibration/experience/invalidity gates passed.
  Style wins were concentrated in physics and practical writing, with both
  losses on pushback. This is a directional prompt-level signal, not an accepted
  configuration or activation-control result. Held-out stayed unused, with no
  prompt revision, vector, training or automatic follow-up. See
  `docs/persona_prompt_execution.md` and `results/persona_prompt_01.json`.
- The user explicitly authorized GPU 0 coexistence if memory and utilization
  allow; leave existing jobs untouched and keep each comparison on one physical
  device. All registered steering stages finished and released project GPU memory.
- The JSON extraction SFT proposal is optional backlog, not a missing required
  step. See `docs/next_sft_pilot.md`; no new training run is registered.
- Routine default: one verified base, one fresh adapter configuration and seed
  42; a small development screen followed by one fixed final paired comparison.
  Reuse compatible baseline results. Rank/LR grids, extra seeds, BF16 controls
  and full quantization sweeps are optional investigations.
- Physics progress: 100 licensed PRL-associated papers frozen into 70/10/20 splits;
  NF4 base published privately and reload-verified. Four optimization runs and
  their development screens completed. The fixed rule selected all-linear rank
  128, LR 5e-5, epoch 1. Both matched controls completed and passed reload
  verification; the fixed final evaluation and private physics-adapter publication
  are complete. See `docs/physics_prl_execution.md` and
  `results/physics_rank_selection.json`. Release verification is in
  `results/physics_huggingface_publication.json`. Writing-quality assessment is deferred.
- Latest writing scope: keep this a lightweight exercise in learning the rough
  language and structure of condensed-matter theory PRLs. Writing-specific
  fine-tuning remains exploratory. Further training and writing ratings can wait;
  future Claude/Codex ratings will be recorded as AI judgments, not human ratings.
- Deferred physics objective: learn the language and structure of a balanced corpus of
  condensed-matter theory PRLs, focusing evaluation on titles, abstracts and
  introductions. No publication-acceptance claim is intended.
- Quantization choices: original BF16; independent GGUF Q8_0/Q4_K_M inference
  branches; saved NF4/double-quantized base before QLoRA. One BF16 LoRA control.
- Standalone NF4 serialization verified: 1,659 state tensors, probe logits and
  generated IDs match exactly; `results/nf4_checkpoint.json`.
- GitHub repository: created; this working directory is the project repository.
- Hugging Face organization: `RenormalizationTech` (confirmed by user).
- Hugging Face artifacts published and verified: public base GGUFs and private experimental adapter.
- Published revisions and scoped load checks: `results/huggingface_publication.json`.
- Bounded QLoRA completed: better answer format, lower math accuracy; see release cards.
- Publication visibility confirmed by user: public GGUFs; private experimental adapter.
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

### Initial work order (completed)

1. Verify the environment, GPUs, available disk, and upstream model/tool support.
2. Select one small model and pin its revision; establish minimal fixed evaluation.
3. Run baseline inference and record measurements.
4. Convert and quantize to Q8_0 and Q4_K_M; reload and compare.
5. Prepare a small licensed dataset and run QLoRA; reload and evaluate the adapter.
6. Create artifact repositories and model cards under the existing Hugging Face
   organization, then upload selected validated outputs.

Model cards live as `README.md` files in the artifact repositories. The exact
organization ID, artifact names, and visibility must be resolved before upload.

### Routine work order for an optional future SFT task

1. Select the task dataset and JSON schema; define labels, source-disjoint splits,
   primary metrics, retention checks and the runtime budget.
2. Pin the existing model/NF4 dependency and inspect formatted batches and loss
   masks. Record a development base reference or verify a compatible cached reference.
3. Register one task-appropriate configuration and seed 42 before optimization;
   record the full LR schedule and maximum exposure.
4. Train one fresh adapter and nominate one checkpoint by the registered
   development criterion. Screen that nominee once, retain the base if it fails,
   and run the fixed final base-versus-selected-adapter comparison once.
5. Verify reload, report the outcome and measured costs, and release selected
   experimental artifacts under the existing private publication scope.

The reusable workflow documents the larger physics calibration as completed
history. Further physics writing ratings or another model are separate follow-ups.

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

Before QLoRA training, inspect a decoded batch to verify loss targets, padding
and end-of-sequence handling. Instruction SFT masks prompt tokens; paper-text
causal training includes all within-paper prediction targets. Run a tiny training smoke test,
then train on a small licensed dataset with separate validation and test data.
Save and reload the adapter, and report held-out domain performance and general
retention without assuming improvement. Save run metadata with each result.

Multi-GPU training, TRL, DeepSpeed, hosted tracking, a web interface, additional
quantization formats, and custom training loops can wait until a concrete need
arises. Merging and requantizing the tuned model is a follow-up exercise.

### Historical GSM8K training experiment (completed)

The 1,024-example, 64-step GSM8K experiment and its fixed checkpoint-selection
protocol completed before the physics-writing exercise. The selected attention
LoRA rank-8 adapter improved answer formatting but reduced held-out math accuracy.
The private release preserves that negative result; see `results/qlora_real_01.json`
and `docs/execution.md`. It is a distinct objective and dataset from the current
paper-text causal training experiment.

The current workflow is `docs/model_workflow.md`; its exact frozen physics design
is `docs/physics_prl_protocol.md`, with the documented contingency addendum.
Completed development choices stay fixed during the repeat, BF16 control and
final evaluations. The final report separates paper-text likelihood, general
retention and pending human writing judgments: `results/physics_prl_final.json`.

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
workflow. The primary goal is a working train/save/reload/evaluate/release
exercise, not a proven improvement in physics writing. This is separate from
persona prompting and activation steering; neither requires the physics adapter.

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
-   select a checkpoint by the registered development rule; report regressions;
-   publish the selected adapter to Hugging Face;
-   optionally merge/export a standalone model;
-   optionally generate GGUF variants.

### Evaluation

Measure both:

1.  **Domain evaluation (improvement is not assumed)**
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

## Stage 5 - Profanity, Rudeness and Sarcasm Directions

The immediate user-selected milestone is isolated activation control of
profanity, rudeness and sarcasm on Qwen3-4B; any joint-control stage is optional. Finish the 4B pipeline before further 14B work. The previously substituted mild-tone roadmap
is superseded. Physics LoRA remains a separate fine-tuning demonstration.

Define and extract each trait using matched contrasts that distinguish it from
the other two. Swearing can be friendly; rudeness can be nonprofane; sarcasm
can be playful. Prompted contrasts are construction/reference data, not the
final activation-control mechanism. Do not assume every trait is one-dimensional.

Complete and report each individual direction first: construction, extraction,
bounded development, locked confirmation and vector save/reload verification.
Report cross-effects descriptively. Pairwise/all-three mixtures are optional
future studies and are not required for completion.
No successful refusal
configuration is required before investigating these distinct traits.
See [the project map](docs/project_tracks.md) for the development path.

------------------------------------------------------------------------

## Stage 6 - Descriptive Cross-Effects and Optional Geometry

Study relationships among individual controls to report coupling and score
correlations. The CPU GCAD profanity/impolite alignment record is retained;
further geometry requires a separate research question. No disentanglement or
joint-control result is required.

For directions (v_i) and (v_j), calculate:

\[ `\cos`{=tex}`\theta`{=tex}\_{ij} = `\frac{v_i\cdot v_j}`{=tex}
{\|v_i\|\|v_j\|}. \]

Questions include:

-   Are profanity and rudeness directions correlated?
-   Can sarcasm be controlled independently of the other two traits?
-   Do directions change across layers?
-   Are the representations genuinely one-dimensional?
-   Can multiple human-defined traits be described by a smaller latent
    behavioral space?

Construct a matrix of contrastive activation differences and use
PCA/SVD:

\[ `\Delta `{=tex}H = U`\Sigma `{=tex}V\^`\top`{=tex}. \]

Estimate the effective dimensionality of behavioral representations.

Strong correlations are a reportable result. No automatic orthogonalization or
tuning to remove coupling is required. Any later disentanglement experiment
would need a separate research objective; retain the original directions.

------------------------------------------------------------------------

## Stage 7 - Optional Simultaneous Persona Controls

This is optional research after the isolated 4B milestone; it is not required
for the primary individual-control goal. At least two validated individual
directions are needed for pairwise tests; all three are needed for the full
mixture. Poor joint behavior is an acceptable result, with no automatic retuning
or escalation to 14B. Complete the isolated 4B pipeline before model scaling; optional joint work
is not a prerequisite.

The target is three adjustable activation contributions:

\[
h_l' = h_l
+ \alpha_p v_{\text{profanity}}
+ \alpha_r v_{\text{rudeness}}
+ \alpha_s v_{\text{sarcasm}}.
\]

An illustrative, unimplemented interface is
`style_controls={"profanity": 0.0, "rudeness": 0.0, "sarcasm": 0.0}`.
Zero means no added contribution from that vector, not absence of the trait.
Layer, token scope, normalization and usable strength ranges need a protocol.
For multiple blocks, specify a direction at each intervention site.

Test each direction alone, pairwise combinations, then all-three mixtures.
Score all three traits in every condition and measure cross-trait interference,
content retention and degeneration. Additive activations do not guarantee
independent or linear output effects. Test negative strengths if included;
do not assume they reliably produce the opposite behavior.

This is the user's central persona research goal, with exact experimental
settings still unregistered. It requires no new physics LoRA training.

------------------------------------------------------------------------

## Stage 8 - Behavioral Robustness Under Quantization (optional)

After establishing a useful persona intervention, a separate study could ask
whether it transfers across model precision. Compare matched unadapted baselines
and technically comparable activation measurements, then assess causal effects
and quality. Torch hooks do not automatically transfer to a GGUF backend.

This is a deferred research question, not an automatic Q5/Q6 precision sweep or
a prerequisite for the completed quantization learning milestone.

------------------------------------------------------------------------

## Stage 9 - Interaction with Physics QLoRA

Deferred optional research only: study whether domain adaptation changes
behavioral geometry. This is not part of the physics LoRA demonstration or the
current persona program, and does not authorize new training.

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
-   Are the selected persona directions similarly separable?
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
-   [x] Publish public GGUFs and verify remote hashes/downloaded Q4 load

## M2 - First QLoRA exercise (physics remains a follow-up)

-   [x] Prepare first licensed dataset: GSM8K; physics specialization deferred
-   [x] Establish held-out smoke evaluation (16 validation/test rows each)
-   [x] Run two-step QLoRA smoke and verify fresh-base adapter reload
-   [x] Run 1,024-example/64-step experiment; selected step64 by validation loss
-   [x] Evaluate domain accuracy: observed 121/128 to 101/128 math regression
-   [x] Run small retention probes at 8/64 output tokens; no general equivalence claim
-   [x] Publish selected adapter privately as experimental; verify downloaded replay

## M3 - Refusal Research (completed with quality tradeoffs)

-   [x] Construct contrastive data and collect activations
-   [x] Extract candidate directions and test registered interventions
-   [x] Measure causal effects and response-quality side effects
-   No validated usable configuration passed the final quality criteria.
    See `docs/refusal_reduction_execution.md`; no further sweep is active.

## M4 - Isolated Persona Directions (fixed pilots closed; control unvalidated)

-   [x] Define the three trait scales and complete the bounded rating calibration
-   [x] Register and execute isolated profanity steering; development null
-   [x] Review the fixed isolated rudeness/sarcasm study design with Claude
-   [x] Complete isolated rudeness: development null, documented with fixed gates
-   [x] Complete the separately registered sarcasm run: development null
-   [x] Close each bounded axis with target, content and cross-trait results; passing
    control is not assumed
-   [ ] Establish or close/report each isolated control with its best supported method
-   Geometry analysis is optional and requires a separate research question.

## M5 - Optional Simultaneous Persona Controls

-   [ ] Calibrate each of the three directions individually
-   [ ] Test pairwise combinations and all-three mixtures
-   [ ] Measure cross-trait interference and response quality
-   [ ] Lock and evaluate a joint configuration; no API is yet implemented

## M6 - Quantization x Behavior

-   [ ] Compare behavioral representations across precision levels
-   [ ] Compare steering effectiveness across precision levels
-   [ ] Measure effective dimensionality
-   [ ] Analyze which traits are most robust

## M7 - Domain Adaptation x Behavior (deferred optional research)

-   [ ] Compare base and physics-adapted behavioral geometry
-   [ ] Test steering-vector transfer
-   [ ] Measure QLoRA-induced behavioral changes

## M8 - Cross-Architecture

-   [ ] Repeat selected experiments on Qwen3.8-27B
-   [ ] Compare conventional Transformer and hybrid architecture
-   [ ] Document architecture-dependent findings

------------------------------------------------------------------------

# 12. Original Setup Checklist (historical)

These describe the initial setup, not the current next steps. For current scope
and proposed future persona work, see `docs/project_tracks.md`.

Original sequence:

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
