# Project tracks and future persona steering

Scope clarification from the user, 2026-11-14; discussed with Claude before
updating the roadmap. This is the current project map. Historical protocols
and results continue to describe the experiments as registered.

## Portfolio objective and next steps (2026-11-16)

The user's high-level goal is a CV/job-search project demonstrating LLM research
ability across quantization, LoRA/QLoRA, refusal steering and trait steering.
Keep the work proportionate to that goal; successful production controllers,
novel methods and positive outcomes on every experiment are not required.
ARA is explicitly deferred. Existing studies cover the four-topic portfolio;
the user subsequently selected the bounded 27B extension below. Broad method
expansion and sweeps are not scheduled.
This clarification supersedes older pending-expansion language in the roadmap.

Next steps use existing evidence:

1. Condense the README into a portfolio overview with one results table covering
   the four areas and links to the underlying records and public artifacts.
2. Select a small set of saved baseline/intervention examples and make one
   compact figure summarizing measured effects and quality tradeoffs.
3. Document the shortest reproduction path for each area using existing scripts,
   configs and environments; check commands and links without rerunning studies.
4. Draft evidence-based CV bullets and a short interview narrative distinguishing
   implemented methods, measured findings and limitations.

Preserve frozen protocols, results, acceptance decisions and artifact hashes.
New work should serve a concrete portfolio question rather than extend the
research by default.

### Latest selected scope: publication, documentation, then 27B

The user authorized publication of the completed 7B refusal demonstration,
followed by GitHub and HF documentation updates. Preserve the original null
decisions and distinguish the released case-level demonstration from a reliable
controller. No 7B retuning or new evaluation is part of publication.
The release is now [public](https://huggingface.co/RenormalizationTech/qwen2.5-7b-refusal-steering-experiment)
with [anonymous pinned-file and CPU-load verification](../results/refusal_7b_huggingface_publication_01.json).
The release index and portfolio documentation receipt supersede older unpublished
status language below.

The selected larger-model candidate is **Qwen/Qwen3.8-27B**, with three bounded
exercises: quantization, sarcasm steering and QLoRA. Sarcasm replaces refusal as
the 27B behavioral target. Reuse existing datasets, methods and evaluation
structure where compatible; extract a new 27B bank rather than reuse 7B vectors.

- Quantization: use the official checkpoint in Transformers NF4 as the primary
  working representation. Verify save/reload and measure size, inference memory,
  throughput and a fixed small quality subset. A matched BF16 comparison and
  GGUF export are optional; without that comparison, do not attribute task-score
  changes to quantization. No required precision sweep.
- Sarcasm: adapt the existing GCAD workflow to the hybrid architecture, with
  zero-intervention checks, a small strength comparison, prompting/random
  references and one locked evaluation. Report coherence and impoliteness along
  with sarcasm. A null is an acceptable outcome; no automatic retuning follows.
- QLoRA: one short adapter run on the existing licensed physics corpus, one
  configuration and a fixed selection/evaluation rule. Check save/reload and
  domain-loss/general-task tradeoffs. No new sarcasm training corpus is needed.

Metadata inspection only: upstream revision
`37561816e66b5b9153b1b64c10968c18c1e6f4a0`, Apache-2.0,
`Qwen3_5ForConditionalGeneration`, 64 text blocks with hybrid attention,
55,563,006,776 safetensors bytes. Two 48 GB RTX A6000s are present. BF16 weights
exceed one card's capacity; a two-GPU inference path would be needed for an
optional BF16 reference rather than the primary 4-bit workflow.
An approximately 18 GB download is a quantized representation rather than this
BF16 checkpoint. Plan quantized inference and QLoRA separately from the BF16
reference; the 55.6 GB figure is not their weight-memory requirement.
Only text inputs are in scope. The official checkpoint contains vision
components, which are not the research target; keep them unused/frozen and
scope hooks/adapters to language modules. Ollama's approximately 18 GB Q4_K_M
artifact is a different backend/format from Transformers NF4; it is not the
QLoRA training base. Quantization on load may still require downloading the
full official source checkpoint before reducing its in-memory representation.
NF4 training memory and kernel/Transformers compatibility still require a smoke
check, not an assumption that 4B/7B code transfers unchanged. Use an isolated
environment; preserve existing environments and results.

Current publication/documentation is complete. The user then selected the
official checkpoint in Transformers NF4, text-only. Its source files were
downloaded and SHA-verified. An isolated environment and a bounded two-prompt
functional smoke completed, with 496 Linear4bit modules, no missing language
weights and no vision modules loaded. Stored NF4 reload in a separate process
reproduced both generated token sequences exactly. Saved weights: 17.65 GB;
peak PyTorch allocated memory: 18.02 GB on one A6000, not total VRAM.

The strict cached/full-prefix allclose diagnostic failed at unchanged
rtol 0.02 / atol 0.05 (max absolute logit difference 0.125). Finite logits and
matching one-step argmax do not establish cache correctness/equivalence.
The first numerical failure and the follow-up functional result are retained;
the latter is explicitly `complete_with_cache_difference`. Later compared arms
must use the same cached generation path and batch size. No threshold was
loosened or optimized kernels installed to make this diagnostic pass.
See [the execution record](../results/qwen38_27b_nf4_execution_01.json) and
[functional result](../results/qwen38_27b_nf4_smoke_02.json). The local NF4
artifact is not uploaded. No broad quality benchmark, sarcasm extraction or
QLoRA training has run. Register their exact data splits, software, resource
cap and evaluation settings before execution.
This latest scope takes precedence over older larger-model/pending-evasion
language below; historical experiments remain unchanged.

**Latest references:** see the [consolidated four-behavior record](steering_reference.md).
The subsequent [minimal 4B-to-7B refusal transfer](refusal_7b_transfer_execution.md)
is complete. Both fixed-workflow decisions were development nulls; a separate
selected benign diagnostic at raw coefficient -0.25 demonstrated two useful
rescues among three baseline refusals on 13 texts, with no new refusals and one
repetitive output. Four dose-seen cases are separate. The experimental setting
is a case-level reference, with broader reliability unvalidated. The user
accepted the completed experimental demonstration; no further run is scheduled.
Original results below are preserved.
GCAD profanity and the separate lower-strength GCAD sarcasm study on
Qwen2.5-7B passed held-out confirmation and are retained. Sarcasm has a narrow
coherence margin and measurable impoliteness coupling. Author-model GCAD
impoliteness and Mistral fixed/adaptive sarcasm also have positive references
with quality limits. Refusal reduction remains a partial effect that
failed final quality. The record spans three checkpoints; cross-model vectors
cannot be combined. The three Qwen7 GCAD style banks share coordinates, but
their joint use has now been tested in one fixed12-arm/24-question study.
**Phase1 is complete:0/5 combined candidates qualified, all failing a coherence
point-estimate rule.** The [public guide](https://huggingface.co/RenormalizationTech/qwen2.5-7b-gcad-steering-controls/blob/e1e56e6384214be873984f5fb9421ff6c61b9e85/joint_controls/GUIDE.md)
and [publication receipt](../results/steering_publication_joint_04.json) preserve
all outcomes. Friendly-PF scores and the separate post-hoc literal lexical
proxy differ: lower friendly score does not mean less swearing. Strong
cross-effects are descriptive, not independent knobs. The
[GCAD discussion](gcad_and_combined_control.md) records the evidence limits.
The Qwen7 benign refusal screen closed at3/100 semantic refusals versus five
prefix flags; no GCAD refusal controller was tested in that screen.
**Phase2 baseline is complete:** the [DECCP screen](deccp_headroom_closeout.md)
recorded English1 refusal+7 grader-classified evasions/36 (22.2%), controls0/11; it failed the25%
and12-item conditions despite a22.2pp gap. Chinese0 refusals+5 evasions/11 is
descriptive without Chinese controls. No fit/intervention was run and no public refusal controller was produced;
25 held-out rows remain protected/ungenerated. The
refusal-steered artifact remains incomplete pending a user decision about
changing target to evasion. No automatic extension; larger-model execution requires user discussion/decision.

**4B track status:** the 4B refusal and fixed persona-transfer studies are closed;
no4B controller qualified for release. The [three-trait rating calibration passed](persona_rating_calibration_completion_execution.md)
on clear authored cases. The [profanity pilot closed with a development null](profanity_steering_execution.md);
[rudeness and sarcasm also closed with development nulls](isolated_persona_traits_execution.md).
No isolated setting qualified in the tested range. The user selected the
[literature-first isolated workflow](persona_literature_workflow.md), with author
data/code before custom extensions. Source-tested implementations are complete;
full rudeness/profanity PV grids closed with development nulls. SAPS sarcasm
completed a held-out paraphrase reference pass, with a meaning-preservation cost
and small capability checks passing. See the [reference execution record](persona_reference_execution.md). The separate7B joint study is complete with no qualifying combined candidate.
Quality-preserving4B control remains unvalidated; no retuning is active.
The completed 14B baseline remains a historical measurement. Older closeout or
transition language in frozen protocols is superseded by this current roadmap;
those records remain unchanged to preserve their experiment hashes.

## Separate learning objectives

| Track | Mechanism | Purpose and status |
| --- | --- | --- |
| Quantization | Change the numerical representation of a pinned model | Demonstrate compression, serialization, reload and evaluation; initial milestone complete. |
| Physics LoRA/QLoRA | Train adapter parameters on physics-paper text; keep the base frozen | Demonstrate fine-tuning on domain data; completed experimental adapter. Writing-quality improvement remains unestablished. |
| Refusal activation steering | Intervene on hidden activations during inference | Learn behavioral representation/control methods; completed experiments showed an effect with quality tradeoffs, without a validated usable setting. |
| Persona prompting | Change system instructions at inference | Define a desired tone and provide a reference/control for persona research. No adapter training. P1 completed without passing its acceptance rule. |
| Persona activation steering | Intervene on activations to influence written tone/style | Custom 4B pilots closed with nulls; Qwen7 GCAD profanity/impoliteness/sarcasm and Mistral sarcasm references are established with their quality limits. Joint12-arm study complete:0/5 combined candidates qualified; coupling remains descriptive, and larger models await discussion. |

The physics adapter is a demonstration of the ability to fine-tune, not a
persona-steering task. Its physics-paper corpus, training loss, adapter rank
and learning rate belong to that training experiment. The practical learning
milestone is a working train/save/reload/evaluate/release workflow; it does not
require a better physics researcher or a proven PRL-quality writing assistant.

The behavioral program proceeds from refusal-steering research to persona/tone
research. A completed research stage can have a negative or tradeoff outcome.
Persona prompting is a preliminary reference experiment within the latter
program; it is neither fine-tuning nor an activation intervention. Both the
physics-training and behavioral tracks can start independently from the pinned
base model. Physics LoRA is not an input to the current persona experiments.

```mermaid
flowchart TD
    Base["Pinned open-source base model"] --> Quant["Quantization learning milestone"]
    Quant --> Physics["Saved NF4 base + physics LoRA training demonstration"]
    Base --> Refusal["Refusal activation-steering research: completed tradeoff"]
    Refusal --> Prompt["Persona prompt reference/control: P1 completed"]
    Prompt --> Persona["Isolated GCAD/SAPS references; joint study complete with quality failures"]
```

This diagram organizes the program rather than the exact chronology. The
consolidated references identify the accepted model-specific artifacts and
their limits.

## What the completed studies establish

- Earlier warm/neutral tone construction: three attempts stopped before judging
  or vector extraction, with mechanical upper bounds of 5, 31 and 46 eligible
  pairs against the registered 50-pair threshold. Length matching was a major
  constraint; the first warm prompt more than doubled typical response length.
  These attempts preceded the refusal studies. See the
  [tone execution record](tone_steering_execution.md).
- Physics adaptation: a saved/reloaded experimental adapter and documented
  evaluation. See the [physics execution record](physics_prl_execution.md).
- Refusal steering: single-block subtraction and all-block projection removal
  were tested. The selected intervention reduced refusal but failed final
  response-quality criteria. See the [methods guide](activation_steering_methods.md)
  and [execution record](refusal_reduction_execution.md).
- Persona P1: two system-prompt conditions on the unadapted BF16 base. The
  persona won 15/24 style comparisons but missed the registered style and gross
  quality-loss thresholds. No persona vector was extracted or applied. See the
  [P1 record](persona_prompt_execution.md).

The self-contained physics/math questions in P1 were content-retention probes.
They did not train a model, load the physics adapter, use the PRL training corpus
as persona data, or assess improvements in PRL writing. A task's subject matter
does not determine whether the method is fine-tuning, prompting or steering.

## Current user-selected goal: individual controls

The user clarified the objective after the GCAD discussion: establish reliable
individual controls for refusal reduction, profanity, rudeness and sarcasm.
Prefer GCAD if it works well across traits. Otherwise retain and tune the
best supported intervention for each trait. Different methods are an acceptable
final outcome. Simultaneous control is not the ultimate goal or a condition
for completing the individual-control learning pipeline.

| Behavior | Intended effect | Interpretation |
| --- | --- | --- |
| Refusal reduction | Fewer refusals under the registered task, with answer-quality and boundary measurements | Current reference showed a partial effect with failed quality. |
| Profanity | Amount/intensity of swearing | It can co-occur with rudeness; the retained study's original friendly-swearing rubric remains unchanged. |
| Rudeness/impoliteness | Abrasive, dismissive or impolite delivery | The author reference measures a broad construct; task/rubric differences stay explicit. |
| Sarcasm | Context-recognizable verbal irony | It may correlate with rudeness or profanity; those correlations do not invalidate a sarcasm effect. |

The previously substituted mild-tone roadmap remains superseded. Physics LoRA
remains a separate demonstration; a future persona LoRA would be a separate
training study.

## Per-trait workflow

1. Follow an established method/source first. Specify the target construct,
   checkpoint, extraction data, intervention sites and bounded tuning budget.
2. Extract and verify that trait's model-specific representation. Prompted
   references establish what behavior the model can express under instruction.
3. Tune the individual intervention against its own target and quality rules.
   Freeze a selected setting before held-out confirmation. Cross-trait effects
   are descriptive, not an off-axis rejection gate.
4. Retain a confirmed method/settings profile. If GCAD fails, preserve an
   established alternative method or document the bounded null. Do not force
   every trait to use the same mechanism or coefficient.
5. Save/reload/check and report both effects and limitations. The immediate
   DECCP7B baseline headroom stage is now complete with a failed gate, after
   the completed isolated sarcasm and joint studies. No refusal bank, steering
   or publication followed; a target change needs a new user decision/plan.

The fixed4B and7B joint studies are complete; later Qwen3.8-27B/Gemma4 work
requires user discussion, not an automatic milestone transition. Frozen historical protocols
and results remain unchanged.

## Descriptive cross-effects and optional joint work

Score relevant traits on individual-intervention outputs to describe coupling
or correlations. Use comparable questions/decoding where feasible, and label
model/task differences. A cross-effect matrix records how one active control
changes the other trait ratings; score correlations are a separate descriptive
summary. Neither is evidence of independent causal dimensions by itself.

If correlations are strong, report them. Do not automatically orthogonalize,
search for cancellation, or reject an otherwise qualifying individual control
because it also changes another trait. Existing study-specific constructs and
quality gates remain as registered; the clarification does not revise old scores.

The completed pair/triple study tested12 fixed arms without refit/selection.
Its failures do not erase model/task-specific individual references. Any further
mixture research requires a new protocol; none is currently released. Vectors from different model
coordinate systems cannot be added directly even if individual methods work.

A future persona LoRA would be another training experiment and would need its
own dataset and protocol. It is not the current persona-steering method or a
reason to restart the completed physics adapter exercise.

**Historical 4B pilot status:** profanity, rudeness and sarcasm closed with development nulls.
The [rudeness/sarcasm design](isolated_persona_traits_protocol.md), fixtures and
code are accepted; 16 focused CPU checks passed. Both studies had separate
pre-call registrations and preserved their stopping rules. See the
[execution record](isolated_persona_traits_execution.md). Confirmation, selected
vector generation replay and retention were skipped under the nulls. Joint
control is unexecuted and needs its own protocol after supported isolated
settings exist; this roadmap is not an execution registration.
