# Qwen3.8 27B evaluation and release plan

Updated 2026-11-19. The selected work is an evaluated NF4 base, one additional
**Q4_K_M GGUF**, and **two separate QLoRA adapters**: QASPER paper question
answering with evidence, and PRL-associated physics paper body to grounded
abstract. Both adapters start independently from the same unadapted NF4 base.
They are not mixed or trained sequentially onto each other.


**Qwen3.8 closeout:** all five planned packages are public: the NF4 and Q4_K_M
bases, independent QASPER/PRL adapters, and the separately registered
[experimental tunable sarcasm controls](qwen38_sarcasm_release.md). The latter
passed exact isolated GPU conformance after anonymous controls download; its
base was re-hashed from the earlier pinned anonymous-download cache. This
publication does not upgrade either sarcasm study's null or the PRL quality
failure. No additional model, retuning or judge run is scheduled.

The owner selected these formats and tasks, including the grounded-abstract
objective. This is the implementation plan, not an immutable run registration.
Exact data revisions, input hashes, resolved settings and source snapshots must
be registered before model experiments. The completed 4B/7B results and the
[27B NF4 smoke](../results/qwen38_27b_nf4_execution_01.json) retain their conclusions.
The owner resumed the separate 27B sarcasm study on 2026-11-18; see its
[execution record](qwen38_sarcasm_execution.md) and the
[roadmap](project_tracks.md). This completed quantization/QLoRA plan is unchanged;
new sarcasm runs use their own registration and budgets.

Implementation status: the registered NF4, Q4_K_M and local BF16-GGUF screens
are complete. Evaluated NF4 and Q4_K_M packages are public with remote file
hashes verified. Anonymous full-file downloads and GPU reload checks passed;
the packages are indexed in the [release catalog](../releases/catalog.json).
QASPER development testing selected the fixed schema-example prompt.
The [fixed quantization screen](../results/qwen38_27b_quantization_screen_01.json)
retains all three representations, exact denominators and interpretation limits.
The two-pass PRL support audit retained **57/70 training and 9/10 development
papers**, without changing prompts or targets. The single-configuration PRL run
completed 24 optimizer steps in 2.26 GPU-hours; step 24 had the lowest development
loss. QASPER completed 61 steps in 5.11 GPU-hours, selecting step 61 by development
loss. Both PRL final generation arms and all 80 blinded review calls completed.
The [PRL science report](../results/qwen38_prl_final_science_01.json) fails the
registered gates: adapter preference 0.175, 95% paper-bootstrap interval
[0.075, 0.2875]. PRL passed the fixed retention gates and is
[public as experimental](../releases/Qwen3.8-27B-tuneLLM-PRL-Abstract-QLoRA/README.md),
with anonymous full-file download and GPU tensor-equality reload verification.
QASPER passed all five fixed task gates on the 309-question primary subset:
strict Answer F1 is **0.690** versus **0.457** for base and **0.567** for the
development-selected schema prompt. The adapter's gain over the prompt is 0.123,
95% paired paper-bootstrap interval [0.083, 0.165]; lenient parsing also passes
with gain 0.106 [0.068, 0.147]. Evidence-strict F1 rises from 0.699 to 0.759,
and the adapter has zero invalid JSON outputs over all 403 questions. The
[task report](../results/qwen38_qasper_final_science_01.json) preserves every
subgroup and the original baseline cap failure/completion. QASPER also passed
the fixed retention gates: general MC 109/128 versus base 110/128, numeric GSM8K
119/128 versus 114/128, and strict IFEval 26/32 versus 23/32. These are registered
point limits, not a statistical noninferiority claim. Its
[public adapter](../releases/Qwen3.8-27B-tuneLLM-QASPER-QLoRA/README.md) passed
anonymous full-file download and GPU tensor-equality reload checks. All four
selected packages are published; the recommendation is limited to this custom
text-only task subset, not the full official QASPER benchmark.
Development loss selects checkpoints and does not establish task-quality gains.
The [preflight record](../results/qwen38_science_preflight_01.json)
retains the measurements and limits. For long PRL bodies, the registered
input policy now keeps fixed head and tail excerpts with whole math/environment
blocks and an omission marker; it uses no reference abstract for selection.
The QASPER evidence gate uses the frozen evidence-strict subset to avoid credit
from an empty-evidence reference. These refinements precede task optimization
and final-test generation.

The [resource registration](../results/protocols/qwen38-resource-plan-01.json)
reserves all remaining model phases within the original study caps. Shared base
retention is counted once, half in each adapter's evaluation pool. Each retention
arm is capped at two hours; its subordinate ledger adds no budget. The
[PRL review protocol](../results/protocols/qwen38-prl-evaluation-01.json) allows
20 development and 80 final tools-off AI calls, without automatic retries.
Development compares base and verification prompts before any adapter output;
final judgments require two valid blinded passes on every paper for a recommended
label. Support labels and pairwise preference share a response and are correlated.
These are AI judgments, not human factual validation.

The 10,000-second QASPER baseline phase stopped after 395/403 questions. Its
failed receipt and outputs are preserved. A separate, prospectively registered
[completion amendment](../results/protocols/qwen38-qasper-baseline-completion-amendment-01.json)
allowed 900 seconds for the eight questions not yet evaluated, within the original
12-hour QASPER evaluation pool. They completed in 244.55 seconds; the original
395 predictions were retained byte-for-byte. Full 403-question coverage is
distinguished from adherence to the
original uninterrupted phase cap; quality gates and the 403-question denominator
remain unchanged.

Measured GSM8K generation length (354 tokens on average in the native NF4
screen) exposed an overly optimistic batch-one retention timing forecast.
Before any retention execution, a separate synthetic preflight assessed
batch-four generation with matched groups across all three retention arms.
Science generation remains batch one; data, gates and study GPU-hour pools
remain fixed. Superseded CPU-only preparations are preserved. The
[prospective amendment](../results/protocols/qwen38-retention-batch-amendment-01.json)
records two of four invented probes differing between batch one and four;
no token-equivalence claim follows. All retention arms bind the same batch-four
groups and padding. The shared baseline and both adapter retention runs are
complete; both adapters passed the fixed retention gates. This does not override
the failed PRL abstract-quality gates.

PRL development generation is complete. Claude's five-hour usage limit interrupted
the judging phase after seven valid first-pass judgments; no paper had two complete
passes. The [registered fallback](../results/protocols/qwen38-prl-development-fallback-01.json)
therefore selects the base prompt and aliases the second baseline to it, without
retrying development judgments. A stronger-prompt advantage is not established.
Final review completed after the automatic quota reset with all 80 calls valid;
the final quality gates stayed unchanged. The development-only
[transfer probe](../results/protocols/qwen38-transfer-01.json)
uses 33 eligible QASPER questions and nine audited PRL papers. Only cross-task
cells describe transfer; same-task cells overlap checkpoint-selection data and
are explicitly labeled selection-set NLL.

## 1 Evaluate NF4 and add Q4_K_M

Use the official `Qwen/Qwen3.8-27B` revision
`37561816e66b5b9153b1b64c10968c18c1e6f4a0`. Preserve the saved NF4 model at
`artifacts/qwen38-27b-nf4-01/`. Convert the original higher-precision source to
GGUF and quantize to Q4_K_M; do not convert the NF4 model into the base GGUF.
No Q6_K, Q8_0, AWQ or other additional quantization is included.

First check the pinned llama.cpp build for this hybrid architecture, conversion,
chat template, special tokens and inference. Record vision and MTP inclusion or
exclusion explicitly. The release is text-only; do not imply image/video support.
Use standard Q4_K_M without an importance matrix for this bounded export.

Compare unadapted NF4 and Q4_K_M on the same registered science and general-task
subsets. Match rendered instructions, thinking mode and output limits, and record
tokenization differences. Use greedy generation, batch size one and the same
cached path within each backend. The existing strict cached/full-prefix failure
is retained; do not relax its tolerance or infer equivalence from token replay.

A BF16 Transformers reference is a priority subject to measured memory and device
availability. NF4 versus BF16 within Transformers can measure the representation
effect under the recorded implementation. A BF16 GGUF reference, if feasible,
can do the same within llama.cpp. NF4 versus Q4_K_M compares complete workflows;
backend differences prevent attributing all changes to quantization. Without a
matched reference, report absolute task results and workflow differences only.

Report model size, loading/reload behavior, prompt-processing and decoding speed,
time to first token, peak allocator and device memory with their measurement
methods, and task scores. Fix prompt/output lengths, warmup and repeats before
timing. Distinguish single-GPU results from any sharded BF16 run. Start with short
contexts, then measure only the registered context envelope; no broad long-context
or native multimodal claim follows from this screen.

**Deliverable:** pinned Q4_K_M export, baseline predictions and scoring, a compact
comparison table and a reproducible text-only runtime profile.

## 2 Train and evaluate two independent adapters

### Shared training and selection

Keep the 27B environment separate from historical environments. Inspect PEFT
targets against the actual text model: full-attention projections, supported
linear-attention projections and MLPs need an explicit inventory. Exclude vision,
MTP and unintended state parameters. Verify frozen base weights, trainable
parameter count, chat-template masking, padding and EOS before optimization.
Use assistant-answer loss, with paper bodies and user instructions masked.
The old physics causal-text chunker is not the new supervised task objective.
For QASPER this includes the complete serialized assistant JSON response
(answer, abstention flag, evidence IDs and EOS), not just the natural-language
answer field. PRL loss covers the abstract response and EOS. Record target-token
counts separately from prompt-token counts.

Run a small, separately recorded feasibility check on training/development data
to establish gradient correctness, kernel support, memory and step time. Select
the context cap from these mechanics and data coverage before viewing final
test outputs. A faster kernel path requires its own recorded checks; successful
short-prompt inference does not establish training feasibility.

Use one registered configuration per adapter, seed 42, microbatch one,
gradient accumulation, BF16 computation and gradient checkpointing as supported.
Fix rank, learning rate, token exposure, checkpoint schedule, maximum optimizer
steps and GPU-hour budget after preflight and before optimization. Do not run a
rank, learning-rate, seed or dataset-mixture sweep. Select the checkpoint by
minimum token-weighted development answer NLL; break ties toward the earlier
checkpoint. Apply quality/retention gates only to that checkpoint; do not search
other checkpoints after a gate failure.

For each task compare the same NF4 base with the fixed task prompt, a stronger
prompt-only baseline chosen using development data, and the adapter with the
fixed task prompt. Keep the instruction, available evidence, cache path, batch
size and decoding identical for each corresponding comparison. Register all
prompt templates before final testing. Thinking mode is off; no claim about
preserved thinking-mode performance follows without a separate evaluation.
Allow at most three candidate prompt-only templates, selected on development
Answer F1 for QASPER and the registered development review rubric for PRL.
One seed does not estimate seed-to-seed variance.

### QASPER paper questions and evidence

Use a pinned [QASPER dataset](https://huggingface.co/datasets/allenai/qasper)
revision, retain its CC-BY-4.0 attribution, and preserve its paper-level split
boundaries. QASPER concerns NLP research papers. Its answer and evidence labels
support grounded QA; they do not establish physics specialization. Public data
may have appeared in base-model training, so absolute scores are not an
uncontaminated generalization claim.

The output contains an answer, a flag for insufficient evidence, and evidence
paragraph IDs. Define a versioned JSON schema and deterministic parser. Preserve
yes/no, extractive, free-form and genuinely unanswerable annotations. Resolve
multiple training annotations deterministically; evaluation retains the
official scorer's handling of multiple valid references. Pin and test the
[upstream scoring semantics](https://github.com/allenai/qasper-led-baseline)
before adapting them to our output format.

Number source paragraphs before context selection. Use full documents where they
fit; otherwise use a deterministic question-dependent selector with no gold
answers or evidence as inputs. All compared arms receive the same context.
Do not label an originally answerable example unanswerable because selection
removed its evidence. Record context coverage and missing-evidence cases; choose
the training handling before freezing inputs. Do not silently drop such final
cases or count them as correct abstentions. Separately account for figure/table
evidence outside the text-only task. Any restricted subset must have its own
denominator and must not be called the full official benchmark.

Primary measures are answer F1 and evidence F1, with invalid JSON, abstention,
unanswerable accuracy, evidence availability, truncation and unsupported evidence
reported separately. Use paired comparisons and paper-clustered uncertainty.
Freeze bounded development/test subsets by paper IDs before generation, with
inclusion rules and counts for each answer type. No test-driven prompt tuning.
The primary subset contains text-answerable questions with at least one complete
valid reference-evidence set in context, plus originally unanswerable questions
whose full text fits. Report all sampled questions as a secondary denominator
and figure/table-dependent cases separately. Fix these memberships before model
generation and preserve every sampled record. False-unanswerable rate uses
answerable primary questions, not missing-evidence cases. Report strict JSON
scores alongside a frozen lenient answer parser to distinguish formatting gains
from improved answers; invalid strict outputs still fail the format measure.

### PRL physics bodies to grounded abstracts

Use arXiv source versions associated with PRL publications, retaining per-paper
CC-BY evidence, attribution, source hashes and Crossref DOI verification through
the existing corpus pipeline. The target is a faithful abstract of supplied
evidence, not journal acceptance, new physics or a verified physics solver.

Reuse eligible historical training/development papers after auditing target
extraction. Keep the historical 70/10/20 split immutable. Its old test papers
are exposed historical diagnostics; collect a separate, fixed fresh test set
with no paper/version overlap with any training or development source. Deduplicate
versions and near-identical papers, and disclose author overlap. Fresh selection
does not prove absence from pretraining.
Prefer arXiv v1 dates after the verified official base-model release date; count
eligible PRL-associated CC-BY sources before registration. If supply cannot meet
the target, record a smaller or older independent set and revise the claim and
power assessment before any model output. Do not relax source licensing or
quietly reuse the historical test set to fill the count.

Extract each reference abstract separately and remove the abstract environment,
equivalent macro forms and duplicated abstract passages from input. Remove
title-block metadata and any verbatim abstract in other source sections, while
retaining scientific body evidence, equations and required definitions. Audit
normalization and overlap before training. Choose body passages using a fixed
policy independent of the target; references must not guide final input
selection. If the available body omits the claim needed for an abstract, flag
the case and report coverage rather than reward invention.
Exclude training pairs with target claims absent from the selected body using a
fixed pre-training audit; record exclusions and their reasons. Keep test cases
and score omissions or unsupported content explicitly.

Compare generated abstracts with the supplied body for claim fidelity,
quantities/units, scope, limitations, concision and coverage. Register a
blinded paired review rubric and a separate judging session before generation;
identify AI review as AI review. Similarity metrics and abstract answer NLL are
secondary. Paper-text NLL is a separate continuity measure, not the selection
objective or proof of writing quality. Save failures and unsupported claims.
Randomize answer position, conceal model identities and use one overall paired
preference as the primary judgment; other rubric dimensions are secondary.
Disclose reviewer model and any advisory role rather than claim independent
scientific validation from an advisor's judgments.

### Retention and success criteria

For both adapters use the same fixed small GSM8K, MMLU and IFEval retention
subsets, plus the other science task as a transfer probe. Historical public
fixtures can measure regression, but do not become fresh held-out data again.
Select bounded sample counts and numeric acceptance thresholds before model
generation. Report raw counts, paired fixed/broken items, truncations and
uncertainty instead of treating tiny differences as established improvement.

The intended success criterion is improved task performance against the fixed
base, usefulness against the stronger prompt-only baseline, and acceptable
faithfulness and general-task retention under registered gates. Failure of those
gates remains a result. No larger corpus, rerun or retuning is automatic.
Reload each selected adapter in a fresh process and verify its identity and
behavior before staging a release.

**Deliverables:** separate QASPER and PRL adapters, data/split manifests,
provenance and licenses, selected-checkpoint records, saved baseline comparisons,
and explicit quality/retention decisions.

## 3 Publish evaluated Hugging Face packages

Plan the NF4 base, base Q4_K_M GGUF, QASPER adapter and PRL abstract adapter as
separate, clearly named artifacts under `RenormalizationTech`. Reuse the
[release storage rules](file_organization.md#releases-and-model-copies).
Each card records the exact base revision, text-only scope, task, data terms,
runtime/dependency pins, limits, evaluation denominators and known failures.
The adapters remain distinct from full weights and activation-steering banks.

Experimental or null adapters may be released with their failed gates visible;
they must not be described as improved models. After reload and evaluation,
an adapter that passes its gates can be merged into the original higher-precision
base and optionally exported to **Q4_K_M only**. Verify unmerged and merged
adapters on the same higher-precision base, then evaluate the merged Q4 export
as a new runtime path. Do not promise exact equivalence to NF4 plus adapter.

Stage complete local payloads, hashes, cards, licenses and reproduction commands
before upload. Verify the uploaded files and fresh download/reload before adding
new entries to the public release catalog and documentation. Keep failed and
historical artifacts intact. All four selected 27B releases are now public;
the [release index](../releases/README.md) records their verified revisions.

**Deliverable:** verified HF revisions and publication receipts, synchronized
local mirrors, and a concise comparison showing when each artifact is useful.

## Historical pre-registration readiness

The following observations describe the initial preparation, before the completed
runs summarized above. Local inspection on 2026-11-17 found the official source shards, saved NF4 and
historical 70/10/20 PRL corpus. Both A6000 GPUs were mostly free, workspace disk
had about 2.8 TB free, and system RAM had about 115 GiB available. These are
point-in-time observations, not reservations or proof of model-run feasibility.

The retained llama.cpp checkout is
`8ca5a8008b57aa39f2601546c0d07912045080a5`, with CPU/CUDA binaries and source
support for qwen35, text conversion, MTP exclusion and Q4_K_M. This specific 27B
conversion and runtime have not been tested. The 27B smoke environment lacks
PEFT and datasets; the old environment contains them but has incompatible
historical version pins. QASPER is not cached locally. The historical physics
trainer hardcodes legacy loading, GPU selection and causal-text chunk lengths;
reuse its provenance helpers, not its training assumptions.

The original pre-registration implementation sequence, now completed, was:

1. Prepare an isolated training/conversion environment and record a resolved
   package lock; preserve the existing smoke and historical environments.
2. Acquire pinned QASPER data and scorer source, inspect lengths and annotations,
   prepare answer/evidence pairs and leakage-audited PRL abstract pairs, and
   freeze manifests without model generation. Use synthetic fixtures to test
   masking, scoring, annotation handling and split/leakage checks.
3. Inventory adapter targets, verify text-only GGUF conversion and template
   parity, then run bounded feasibility checks before freezing the model-run
   registrations and their resource caps. Record any kernel or precision change.

These steps precede registered GPU baselines, training and publication. Exact
data counts, resource caps and numerical gates belong in per-study
registrations; this plan does not substitute for those frozen inputs.

### Bounded planning defaults

These defaults bound implementation and cost estimates. CPU coverage inspection
and a recorded runtime preflight may require an explicit revision before
registration. They cannot be changed after final outcomes to obtain a pass.

| Work | Data and checkpoint ceiling | Compute ceiling |
| --- | --- | --- |
| Base quantization screen | NF4 and Q4_K_M, with conditional local BF16 references; no new quantization formats | 6 GPU-hours, counted across devices |
| QASPER adapter | Up to 1,500 training questions, 150 development questions and a 400-question target final subset, selected by paper; one epoch, three fixed checkpoint slots | 12 GPU-hours for optimization/development checks, plus 12 for prompt-baseline selection, final evaluation and retention across all arms |
| PRL abstract adapter | Historical 70 training and 10 development papers before audit exclusions; target 40 independently selected fresh test papers; up to three epochs, one checkpoint per epoch | 6 GPU-hours for optimization/development checks, plus 6 for prompt-baseline selection, final evaluation and retention across all arms |

The final manifest must specify actual attainable counts and answer-type coverage,
not fabricate exact strata by splitting a paper. A shortfall in fresh PRL supply
or feasible QASPER context coverage requires a documented design revision before
test generation. The small PRL development set makes checkpoint selection noisy;
report that limitation without using the fresh test to choose checkpoints.
Count GPU-hours as device count times elapsed hours, including sharded runs.
Allow at most two additional GPU-hours for the recorded feasibility checks.
Choose an attainable prefix of the seeded, paper-grouped training order from
preflight step time before registering the actual exposure; do not secretly
shorten inputs or run until a preferred score appears. Register a separate
bounded AI-review packet and invocation count; GPU time does not authorize
unlimited external judging. Reaching a time cap produces an incomplete run, not
permission to extend it or silently treat a partial run as fully trained.

### Planned quality gates

Freeze the exact estimators, scorer versions and denominators in the registrations.
The following conservative defaults determine whether an adapter is recommended;
they are not guarantees of improvement or statistical equivalence tests.

- **QASPER:** at least five Answer-F1 points over each baseline, with the paired
  paper-bootstrap 95% interval excluding zero for both comparisons. Against the
  higher-Answer-F1 baseline, evidence F1 drops by at most two points,
  strict JSON failures are at most 2%, and false-unanswerable rate increases by
  at most five percentage points.
- **PRL:** overall paired preference of at least 60% against each baseline,
  counting ties as one half, and each paper-bootstrap 95% interval excludes
  50%. Unsupported claims per abstract do not increase against either baseline
  as a point estimate. At 40 papers the uncertainty condition is stringent;
  a 60% observed preference alone is insufficient. Report intervals and small
  sample limitations instead of loosening the gate after evaluation.
- **Retention:** relative to the unadapted NF4 base, at most five fewer correct
  answers on each fixed 128-question GSM8K and general-MMLU subset, and at most
  three fewer strict prompt passes on 32 IFEval items. These are point-estimate
  regression limits. Report physics-MMLU separately if included, with its own
  frozen denominator. Neither these limits nor a zero observed fidelity loss
  establish non-inferiority.

Use 10,000 paired paper-bootstrap draws with seed 20261117 for science-task
intervals. A gate failure prevents the recommended-adapter label and conditional
merged exports, even if another metric improves. Label such results experimental,
gain with costs, or null using the registered outcome definitions. Preserve all
metrics and uncertainty; this small single-seed study may miss modest benefits.
