# Project status and next steps

Updated 2026-11-18. This is the current roadmap. The
[earlier roadmap](project_tracks_history.md) preserves previous plans and status
statements; frozen protocols and results retain their original conclusions.

## Portfolio objective and next steps (2026-11-16)

The goal is a CV/job-search portfolio demonstrating practical LLM research in
**quantization, LoRA/QLoRA, refusal steering and trait steering**. Keep the scope
bounded: implemented methods, reproducible evidence and clear limitations matter
more than positive results on every experiment. ARA, broad sweeps and production
controllers are deferred.

## Completed work

| Track | Status and evidence |
| --- | --- |
| 4B quantization | BF16/Q8/Q4 GGUF and saved NF4 with reload checks, size/speed measurements and [fixed-subset evaluation](benchmark_summary.md). |
| Physics QLoRA | Published adapter; held-out paper NLL 1.975 → 1.734, numeric GSM8K 117/128 → 111/128 and strict-format correct 97/128 → 106/128. [Execution record](physics_prl_execution.md). Writing-quality improvement is unestablished. |
| Refusal steering | 4B and 7B interventions implemented and compared. Fixed-workflow decisions remain null under their original gates. A separate 13-case 7B diagnostic rescued two useful benign answers, with one repetitive output. The [experimental package is published](../releases/qwen2.5-7b-refusal-steering-experiment/README.md); no general benign-only controller is validated. |
| Trait steering | Individual GCAD/SAPS references show model-specific effects with quality costs. The joint 7B study qualified 0/5 combined candidates. [Consolidated evidence](steering_reference.md) and [joint-control discussion](gcad_and_combined_control.md). |
| Portfolio presentation | Public GitHub, nine HF cards, saved examples, a results figure, [reproduction guidance](reproduce.md) and the [live documentation site](https://renormalizationtech.github.io/tuneLLM/). |
| 27B initial NF4 smoke | Official text-only checkpoint loaded, saved and reloaded. Two short prompts replayed exact tokens. The strict one-step cached/full-prefix logit comparison failed; this remains a limitation. [Execution record](../results/qwen38_27b_nf4_execution_01.json). |

The 7B refusal package contains an inference-time direction and runtime, not new
base weights. Refusal reduction also occurs on unsafe contrasts; the selected
benign diagnostic is a narrow demonstration. No further 4B/7B retuning or
headroom/evasion extension is scheduled. The closed DECCP screen did not produce
a controller; it is separate from the later published refusal experiment.

## Selected next model work: Qwen3.8-27B

Use **Qwen/Qwen3.8-27B**, official revision
`37561816e66b5b9153b1b64c10968c18c1e6f4a0`, text-only through Transformers NF4.
The approximately 18 GB Ollama/GGUF download is a different representation and
backend; it is not the selected QLoRA base. The official BF16 source has about
55.6 GB of weights. Keep the [27B environment](reproduce.md#separate-qwen38-27b-nf4-environment)
separate from the existing 4B/7B environments.

The local NF4 smoke saved **17.65 GB** of weights and reached **18.02 GB peak
PyTorch allocated memory** on one A6000 for its short prompts. These are not
long-context or training memory estimates. The failed cache comparison had
maximum logit difference 0.125 at unchanged tolerances; matching argmax and token
replay do not establish cache equivalence. Use the same generation path and
batch size in later compared arms. A small fixed screen now covers 32 general
MMLU, 32 GSM8K and 35 QASPER development cases per base representation. NF4 and
Q4_K_M packages are public with remote hashes verified; anonymous download/reload
checks passed for their catalog entries. Independent QASPER and PRL optimization
and the fixed final science/retention evaluations are complete.

The owner selected this [three-stage evaluation and HF release plan](qwen38_release_plan.md)
on 2026-11-17:

1. **Evaluate the base and add one quantization:** screen saved NF4 and build
   **Q4_K_M GGUF** from the original higher-precision checkpoint. No Q6/Q8 or
   additional formats. Use a matched BF16 reference where resources permit;
   cross-backend differences are workflow comparisons, not isolated quantization
   effects.
2. **Train two independent QLoRA adapters:** **QASPER paper QA with evidence**
   and **PRL-associated physics paper body to grounded abstract**. Both start
   from the same unadapted NF4 base. Use one configuration per adapter, fixed
   development checkpoint selection, task/prompt baselines, general-task
   retention checks and fresh-process reload checks. Audit abstract leakage and
   paper-level splits; preserve the old physics split as historical evidence.
3. **Publish evaluated HF packages:** NF4, base Q4_K_M, and the two adapters,
   with pinned runtimes, data terms, saved comparisons and explicit limitations.
   Adapted Q4_K_M exports are conditional on quality gates and verified merging
   into the original higher-precision base. No new format is added by that step.

The bounded evaluation and release work is complete. The text-only **Q4_K_M conversion is complete**
(16.74 GB), with a local BF16 GGUF reference (53.81 GB). Both passed a two-prompt
runtime/tokenizer-parity smoke; no quality equivalence follows. Synthetic
rank-16 backward checks passed through 8,192 tokens with 256 supervised tokens.
The fixed three-representation quantization screen is complete; evaluated NF4
and Q4_K_M packages are public with anonymous download and GPU reload verification.
QASPER inputs and development prompt selection are frozen. The PRL
pipeline now retains 40 independently selected, source/license-verified test
papers; they are new to this project, not guaranteed absent from pretraining.
The two-pass PRL support audit retained 57 training and nine development papers.
Both adapters finished training. PRL failed the locked 40-paper science gates:
AI-review preference favored the unadapted base. Its higher reference overlap
did not establish better abstracts. QASPER passed its fixed text-only task gates:
primary Answer F1 is 0.690 versus 0.567 for the development-selected prompt baseline.
Matched QASPER retention also passed the fixed gates. See the
[preflight record](../results/qwen38_science_preflight_01.json).

The [27B PRL abstract adapter](../releases/Qwen3.8-27B-tuneLLM-PRL-Abstract-QLoRA/README.md)
is public as an experimental negative result, with complete matched retention and
anonymous-download/GPU-reload verification. The
[QASPER adapter](../releases/Qwen3.8-27B-tuneLLM-QASPER-QLoRA/README.md) is also
public with these checks; its recommendation is limited to the registered custom
text-only subset. The original baseline phase cap failure and first-time completion
of its eight remaining questions are preserved in the release evidence.
Exact splits, decoding, software, numeric
gates and resource caps are registered before model runs. A null result is
acceptable; it does not authorize a sweep.

**Sarcasm steering is deferred at the owner's request to reserve GPU capacity
for other projects.** It has not started; no new GPU study is scheduled.
Any future resumption needs a separate decision and registration. The existing
[portfolio summary](showcase.md#qwen38-27b-portfolio-summary) provides CV bullets
and interview wording using completed results.

The [27B closeout report](../results/qwen38_science_summary_01.json) binds all
four public revisions, final outcomes, limitations and settled GPU budgets.
The repository closeout publishes the source, compact evidence and documentation;
deployment is checked separately through GitHub Actions and the public site.
A [scoped cache cleanup](workspace_maintenance.md) is complete; scientific
evidence remains retained. Current work is CPU-only portfolio and publication
maintenance. No additional training, inference or judge calls are needed.

## Where to look

- [Results and saved examples](showcase.md): concise measured findings.
- [Reproduce](reproduce.md): current runnable setup and pinned downloads.
- [Release index](../releases/README.md): current HF revisions, card mirrors and local payloads.
- [Repository layout](../README.md#repository-layout): directory purposes and entry points.
- [Earlier roadmap](project_tracks_history.md) and [planning log](../plan.md): historical decisions.

Do not delete failed runs or rewrite registered acceptance decisions to tidy the
portfolio. Historical private/unpublished labels describe their original time;
current visibility and publication status are recorded in the release index.
