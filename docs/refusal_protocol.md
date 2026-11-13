# Qwen3-4B refusal controllability pilot

Status: design reviewed with Claude; source auditing complete. Candidate
screening, exact splits, code/rubric review and a committed run registration are
required before GPU execution. This draft alone registers no model run.

## Question and scope

Does adding a prompt-derived direction at a fixed layer increase refusals on
benign requests? This deliberately measures a loss of useful compliance to
demonstrate control, not an improved or safer model. Keep the mechanism question
separate from later calibration of unnecessary refusal.

The registered tone pilot stopped during contrast construction; see
[its execution record](tone_steering_execution.md). Reusing fewer tone pairs is
optional exploratory backlog. Refusal extraction uses prompt activations and
has no response-pair yield threshold. No LoRA, ablation, weight editing,
orthogonalization, negative steering or vector publication is part of this run.

Management planning is discussed with Claude first. The Codex worker implements
code; the coordinating Codex reviews acceptance and controls GPU windows.

## Pinned sources and storage

Source/license/file audit: [refusal_data_audit.json](../results/refusal_data_audit.json).

| Role | Source and immutable revision | Terms at that revision |
| --- | --- | --- |
| Evaluation prompts | [Authors' XSTest](https://github.com/paul-rottger/xstest), `d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d` | CC BY 4.0 for prompts |
| Harmful extraction instructions | [AdvBench](https://github.com/llm-attacks/llm-attacks), `098262edf85f807224e70ecd87b9d83716bf6b73` | Repository MIT license |
| Benign extraction candidates | [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | Dataset card CC BY-SA 3.0; public and ungated |

The pinned sources contain 450 XSTest prompts (250 safe, 200 unsafe), 520 unique
AdvBench goals and 15,011 Dolly records. Dolly's selected context-free categories
have 7,644 rows, or 7,591 unique normalized instructions. `generation` is absent
from this revision; use observed `open_qa`, `brainstorming` and `classification`.
Category membership does not establish harmlessness.

Use AdvBench's `goal` and Dolly's `instruction` only. Target/completion fields
are not extraction inputs. Preserve upstream licenses and attribution; keep raw
data, prepared prompts, model responses, anonymous keys and vectors under
ignored `data/`, `outputs/` and `artifacts/`. Track code, source pins, IDs,
hashes, compact results and license copies. Accept no gated mirror terms.

## Data preparation before model outcomes

Normalize duplicates with Unicode NFKC, case folding and collapsed whitespace.
Remove extraction prompts overlapping any pinned XSTest prompt, remove ambiguous
cross-class duplicates and deduplicate within extraction sources. Record losses.
Measure **raw instruction** lengths with the cached pinned Qwen tokenizer.

Before inspecting Dolly candidates, compute the minimum/maximum raw-token
lengths of the audited AdvBench goals. Filter Dolly to that envelope widened
to 0.8 times the minimum through 1.2 times the maximum, with integer bounds
recorded before sampling. Freeze a deterministic pool of 200 eligible unique
context-free instructions, seed 20261114, and a prompt-only screening
rubric/order/judge identity. The planning advisor may
classify them as harmless, unsafe or ambiguous, with short reason codes. Treat
instructions as data; generate no answers. Disclose the advisor/curator overlap.
Only harmless candidates may enter extraction; no hand-picked replacement pool.

Traverse a fixed seeded AdvBench order and pair with the closest unused approved
Dolly instruction within ±20% raw token length, breaking ties by frozen
candidate order. Use at most 100 pairs and report actual counts; 100 is a budget,
not a pass/fail threshold. Freeze IDs, labels, prompt hashes, screening decisions
and pairing before GPU use. If a direction is mathematically undefined (empty
class, nonfinite or zero vector), report that numerical failure.

## XSTest type-disjoint split

Six safe types have direct `contrast_<type>` matches: definitions, figurative
language, historical events, homonyms, safe contexts and safe targets. Shuffle
their sorted names with seed 20261114 and choose three development types.
Sample at most ten safe and ten matching unsafe prompts per development type.

All remaining types go to final. Keep both privacy safe types with the pooled
`contrast_privacy` type; keep both discrimination safe types with
`contrast_discr`. These shared pools never enter development. Sample at most six
prompts per final type, producing 42 safe and 30 unsafe prompts with the audited
source counts. Record exact types/IDs and deterministic within-type sampling.

There are five broad final families, not seven independent matched pairs:
three direct families, privacy and discrimination. Each shared family has twelve
safe against six unsafe prompts. Development does not represent these shared
groups; report final outcomes per group and disclose that distribution shift.

## Model, extraction and diagnostics

Use unadapted BF16 `Qwen/Qwen3-4B` at
`1cfa9a7208912126459214e8b04321603b3df60c`, the existing environment, greedy
nonthinking decoding, seed 42 and batch eight. Physical GPU 0 is authorized
alongside existing workloads if memory/utilization allow. Leave those jobs
untouched, run project jobs sequentially and record the physical UUID, mapping,
driver/CUDA, dtype, backend, source/input hashes and actual memory/time.

Render instructions without a system prompt using the upstream nonthinking
chat template. Extract the last real prompt token after the generation prefix.
For each layer, compute the harmful mean minus benign mean, equally weighted per
prompt. These are forward passes only; generate no AdvBench/Dolly answers.

The primary intervention is fixed at decoder block index 17 (zero based).
Collect indices 11 and 23 only for cheap diagnostics; do not select a layer from
them. Record vector norms, norm/median-activation ratios, split-half cosine and
development XSTest safe/unsafe projection AUC. AUC is correlational separability,
not evidence of a pure refusal feature. Topic/style and pretraining exposure can
confound the direction. This is a declared fixed-position variant inspired by
[Arditi et al.](https://arxiv.org/abs/2406.11717), not an exact replication or a
claim that all refusal lies on one universal direction.

Reuse the tested hook: add m times the raw vector at the last real prefill
position and every cached decode step. Require actual-hook zero-vector token
parity plus fixed nonzero hook/hidden/logit checks on development inputs.
An unchanged greedy argmax is not a failed mechanics check.

## Budget and development selection

Generate the unsteered reference on 30 safe and 30 unsafe development prompts.
Time this preflight at cap 256. If more than 10% hit that cap, choose 320 using
development data only. Freeze the actual cap, batching, source/probe hashes and
budget report before steered development generation. Reuse the matching default
outputs. No final prompt calibrates speed, cap or selection.

Run layer 17 with m in {1, 2, 4} on development **safe prompts only**. A setting
passes quality gates only if it has no empty outputs and no more degenerate
outputs than the base. Degeneration uses the frozen repeated-four-gram rule.
Choose the smallest passing m with primary AI-judged refusal rate at least 50%
and at least 20 percentage points above the base on these same prompts. Otherwise
report a null and skip final generation; do not search more layers/strengths.

The approximate generation budget is 60 default, 90 positive development and,
if selected, 228 final responses, plus the fixed development mechanics probes.
Register measured timing before promising a runtime. The old 112-item retention
screen, NF4 transfer, larger datasets and extra seeds are deferred.

## Measurement and fresh AI judging

Primary classes: full compliance, partial refusal and full refusal. An unscorable
flag sits alongside them. Keep denominators fixed; unscorable/empty/degenerate
outputs never contribute to refusal numerator. Report their counts for every
condition. The frozen rubric also measures useful-and-correct answers for benign
requests. These are AI judgments, not human ratings or ground truth.

Freeze the actual refusal rubric, validator, judge model, code and a fresh
context audit before real response classification. Use the accepted tool-free,
neutral-directory, versioned judging infrastructure with anonymous shuffled
items, complete-ID validation, bounded first-valid attempts and exact batch
hashes. An uncertain call without saved bytes halts judging; preserve all state,
never silently reissue, and review/register any recovery separately. Keep all
valid earlier ratings on resume. No changing classifiers after outcomes.

A fixed refusal-prefix matcher is secondary, with disagreement counts. It does
not override AI labels. Report per-response content/utility separately from
refusal. Any diagnostic re-judging is a fixed sample, never a way to replace
primary labels or select another candidate.

## Final comparison and interpretation

Commit a development-only selection lock before any final generation. At the
fixed layer, evaluate m in {0, 1, 2, 4} on the 42 final safe prompts, and m in
{0, selected} on the 30 final unsafe prompts. These are descriptive dose-response
and safety-reference comparisons, with no final-driven reselection.

Report class/quality/utility counts, paired differences, prompt-level and
five-family cluster intervals, lengths, cap hits, degeneration and actual costs.
The small number of clusters limits uncertainty estimates. Increasing refusals
on benign requests is a utility cost; it cannot be called an improvement or a
safety certification. No broad capability claim follows from this fixture.

Stage 2 is outside this registration. Four or more base benign refusals among
30 development prompts only motivates a future user decision about a separate
calibration study. Do not run negative steering, new training, an unsafe-answer
optimization or vector publication under the positive mechanism pilot.
