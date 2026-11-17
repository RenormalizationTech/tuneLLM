# Project status and next steps

Updated 2026-11-17. This is the current roadmap. The
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
| Portfolio presentation | Public GitHub, five HF cards, saved examples, a results figure, [reproduction guidance](reproduce.md) and the [live documentation site](https://renormalizationtech.github.io/tuneLLM/). |
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
batch size in later compared arms. No broad quality benchmark has run, and the
27B NF4 checkpoint is local, not published.

The selected extension remains three bounded exercises:

1. **Quantization:** finish a small quality/speed screen of the saved NF4 model.
   A matched BF16 reference and GGUF export are optional. Without a matched
   comparison, do not attribute score changes to quantization.
2. **Sarcasm steering:** adapt the existing GCAD workflow to the hybrid text
   architecture and extract a fresh 27B direction. Use zero-intervention checks,
   a small strength comparison, prompting/random controls and one locked
   evaluation. Report coherence and impoliteness alongside sarcasm.
3. **QLoRA:** one short run on the existing licensed physics corpus, with one
   configuration, fixed checkpoint selection and domain/general-task evaluation.
   Check adapter reload. Verify memory and language-module targeting first.

These three evaluations/experiments have **not started**. Register their exact
splits, decoding, software and resource caps before running them. A null result
is acceptable; it does not automatically authorize a new sweep. Separate
CV/interview wording can be drafted from the existing evidence when needed.

## Where to look

- [Results and saved examples](showcase.md): concise measured findings.
- [Reproduce](reproduce.md): current runnable setup and pinned downloads.
- [Release index](../releases/README.md): current HF revisions, card mirrors and local payloads.
- [Repository layout](../README.md#repository-layout): directory purposes and entry points.
- [Earlier roadmap](project_tracks_history.md) and [planning log](../plan.md): historical decisions.

Do not delete failed runs or rewrite registered acceptance decisions to tidy the
portfolio. Historical private/unpublished labels describe their original time;
current visibility and publication status are recorded in the release index.
