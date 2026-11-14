# Refusal research: literature and implementation audit

Reviewed 2026-11-14, with Claude consulted on the next baseline. This review
does not reopen historical studies or launch an intervention experiment.

## Findings

The original activation-steering paper has an official author implementation:
[Arditi et al.](https://arxiv.org/abs/2406.11717),
[andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction).
It is the primary reference for our method family, but published source does
not establish that the full pipeline runs unchanged on our current model.
The repository includes reference artifacts and pinned dependencies. Its
requirements specify Torch 2.3.0 and Transformers 4.44.2; its Qwen wrapper
accesses the original Qwen architecture (`model.transformer.h`), rather than
Qwen3's decoder layout. A Qwen3 implementation is an adaptation requiring
validation, not a direct reproduction of a supported reference configuration.
See the [requirements](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/requirements.txt)
and [Qwen wrapper](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/pipeline/model_utils/qwen_model.py).

The paper's extraction contrasts harmful and harmless instruction populations;
it does not require identical questions with refusal/answer continuations.
Our source-class contrast therefore follows that methodological family, but
our particular data/model/selection differ. Topic confounding is a limitation,
not an established explanation for our quality loss. An identical prefix under
the same deterministic model cannot have different pre-answer activations just
because different continuations are attached later.

## Reusable evaluation implementations

| Project | Available reference | Role in this project |
| --- | --- | --- |
| [Official XSTest](https://github.com/paul-rottger/xstest) / [paper](https://aclanthology.org/2024.naacl-long.301/) | Prompt CSV, model completions, classifier scripts and analysis | Small published benchmark for unnecessary refusal; first choice for baseline reproduction. |
| [Inspect Evals XSTest](https://github.com/UKGovernmentBEIS/inspect_evals/tree/a68dade7efb803ef50c8442d3f16264b293d1414/src/inspect_evals/xstest) | Task, dataset revision, prompt, scorer and metric implementation | Concrete reference settings and scoring semantics, independent of our historical harness. |
| [Official OR-Bench](https://github.com/justincui03/or-bench) / [paper](https://arxiv.org/abs/2405.20947) | Generation/evaluation code and [released dataset](https://huggingface.co/datasets/bench-llm/or-bench) | Possible separate diagnostic after a saturated XSTest result; not an activation-steering implementation. |

These resources provide standard prompts, labels, source code and methodological
references. They do not supply a universal gold-standard dataset of identical
questions paired with otherwise identical refusal/nonrefusal answers, or
portable Qwen3 activation vectors.

XSTest has 250 safe prompts and 200 unsafe contrasts. Its safe/unsafe contrasts
are different questions, and compliance labels do not establish answer accuracy.
The prompts are CC-BY-4.0; published model completions retain their providers'
original terms. OR-Bench's hard subset was selected for difficulty on other
models, and automated safety labels merit caution. A new diagnostic should use
fixed released data, not regenerate or select examples after seeing our scores.

## Reproducibility decision

First establish a baseline on the unmodified model using the published
XSTest-safe benchmark. No extraction, layer search, coefficient tuning or model
weight modification is part of this step. The existing author pipeline is a
literature/mechanics reference; its harmful-compliance optimization is not the
reproduction target.

Use a pinned evaluator definition and disclose any runtime or judge substitution.
In particular, Inspect's reference uses a helpful system message, temperature
zero, a 256-token cap and a GPT-4o judge. Using our local Hugging Face stack and
Claude evaluator is an **adapted benchmark reproduction**, not an exact replay
of the upstream environment or a replication of a published Qwen3 score.

Record source revisions/file hashes, native chat-template bytes, explicit Qwen3
nonthinking mode, model/runtime versions, generation completion flags and all
grading attempts. Separate full refusal, partial refusal and compliance from
empty (including whitespace-only), capped and degenerate outputs. Incomplete grading cannot become an improved
score by silently dropping items. Disclose overlap with previously used XSTest
prompts; this is a public descriptive benchmark, not fresh held-out evidence.

Candidate unnecessary refusals, if flagged, may justify a later legitimacy and
reproducibility diagnosis.
An absent or negligible effect is a valid stopping outcome. A saturated score
does not justify an automatic steering sweep or adaptive search for a failing
benchmark. Persona steering and the physics LoRA demonstration remain separate.

## Source revisions observed

Verified with `git ls-remote ... HEAD`; file hashes belong in the implementation
audit/registration before execution. Source inspection alone does not certify
installation, runtime compatibility or repeatability.

| Repository | Revision |
| --- | --- |
| andyrdt/refusal_direction | `9d852fae1a9121c78b29142de733cb1340770cc3` |
| paul-rottger/xstest | `d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d` |
| justincui03/or-bench | `4fad1f9e4d4ba368777585fa1f5b869b6d4442fa` |
| UKGovernmentBEIS/inspect_evals | `a68dade7efb803ef50c8442d3f16264b293d1414` |

The author and OR-Bench repositories declare Apache-2.0. Dataset and evaluation
licences are separate and must travel with the actual files used.
