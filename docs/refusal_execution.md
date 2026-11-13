# Qwen3-4B refusal mechanism pilot

The positive-only pilot adds a prompt-derived harmful-minus-benign direction
to layer 17 of the original BF16 Qwen3-4B. Increasing refusals on benign prompts
is a utility cost used to demonstrate control; it is not an improvement claim.
The fixed design and its stopping rules are in [the protocol](refusal_protocol.md).

## Preparation and registration

Public pinned authors' sources supplied AdvBench harmful instructions,
context-free Dolly benign instructions and type-disjoint XSTest evaluation
prompts. No AdvBench target completions were used. Claude semantically reviewed
a frozen 200-item Dolly pool, retaining all 200 as harmless; there was no
keyword blacklist. Raw sources, prompts, model responses, vectors and anonymous
judging keys remain in ignored local directories. Source licenses and compact
hash/ID records are tracked.

The prepared data contain 100 token-length-matched extraction pairs, 60
development prompts (30 safe/30 unsafe), and 72 final prompts (42 safe/30 unsafe).
Final families are definitions, figurative language, homonyms, privacy and
discrimination. Shared contrast pools stay in final.

The worker passed 27 refusal-specific CPU checks. Claude accepted the method
review. A fresh tools-off judge isolation audit passed, binding the exact
refusal adapter, validator, foundation and rubric. Requested/observed judge is
Claude `claude-opus-5-5`, effort requested medium; actual effort and dated
snapshot are not exposed. No experiment responses were used in the audit.
Commit `f6a0f87` froze the implementation and run/judge registrations before
GPU use; `6a8fa16` froze the measured budget before steered development.

## Extraction and mechanics

Extraction used 100 harmful and 100 benign prompts, capturing the last real
prompt token after the nonthinking generation prefix, with no system prompt.
It performed transformer forward passes only, without generating source
answers. Layer 17 direction norm was 14.495, split-half cosine 0.984, and
development projection AUC 0.637. The direction is stable but correlational: it
can encode source style and topic. Diagnostic layers 11/23 had AUC 0.639/0.974;
they did not change the fixed intervention layer.

Extraction took 7.67 seconds including loading, with 8.08 GB peak PyTorch
allocation. All GPU stages use the same physical RTX A6000 GPU 0, UUID
`GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b`, alongside existing jobs with the
user's permission. No existing job is stopped.

The real zero-vector/multiplier 1 hook exactly reproduced baseline token IDs.
The nonzero/multiplier 1 hook recorded prefill and decode calls, a maximum
hidden-state delta 1.296875 and last-prompt logit delta 7.5. These verify that the
intervention executes; they do not establish beneficial behavior.

The registered cap rule raised 256 to 320 because 40/60 initial responses hit
the cap. At 320, 27/30 safe and 9/30 unsafe baseline outputs still hit the limit.
Useful/correct ratings must be read beside truncation counts: incomplete long
answers and short refusals have unequal cutoff exposure. No further cap tuning
is allowed. The preflight projected 642.2 seconds remaining generation, excluding
loading, extraction and judging, if all final conditions run.

## Result: null at the registered layer

No positive strength met the rule requiring at least 50% benign refusals and
at least 20 percentage points above baseline, while passing quality gates.
The null decision was committed in `7e57435` before any final generation;
all 228 planned final responses were skipped. No final score is claimed.

| Safe development condition | Primary refusals / 30 | Useful and correct / 30 | Cap hits / 30 | Unscorable / 30 | Degenerate / 30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1 | 11 | 27 | 0 | 0 |
| Strength 1 | 2 | 11 | 29 | 0 | 0 |
| Strength 2 | 2 | 3 | 30 | 1 | 0 |
| Strength 4 | 0 | 0 | 30 | 30 | 30 |

Primary refusal counts partial plus full refusals on the fixed denominator;
invalid, empty, unscorable and degenerate responses cannot contribute. Here,
all valid refusals were classified as full refusals. The one-prompt increase
at strengths 1 and 2 is insufficient to demonstrate the registered control
effect. Strength 2 also reduced the indicative usefulness count, and strength
4 disrupted generation. This fixed experiment did not establish clean refusal
control; it does not rule out other positions, layers or directions.

On the separate 30 unsafe development baseline prompts, Claude classified
28 refusals and 2 compliant responses; 9 hit the token cap. No steered unsafe
comparison ran. Only 1/30 safe baseline refusals were observed, below the
predeclared threshold motivating a future negative-steering calibration study.

The secondary prefix matcher found 0/30 safe and 6/30 unsafe baseline refusals,
versus Claude's 1/30 and 28/30. Its 22 unsafe disagreements show poor coverage
of this model's refusal wording. It did not override AI classes or selection,
and was not retuned after seeing the outputs.

All 150 anonymous development items were classified in eight fresh-context
batches, each valid on its first attempt; no judgment was retried. Requested
and observed model remained `claude-opus-5-5`. These are AI rubric measurements,
not human ground truth, and the small development set is not a final benchmark.

## Development quality checks

All 90 steered safe-development responses completed. The baseline and strengths
1 and 2 had no empty or mechanically degenerate outputs. Strength 4 produced
mechanical degeneration on all 30 prompts and therefore fails the quality gate
regardless of the judge's refusal classes. Cap hits were 27/30 at baseline,
29/30 at strength 1, and 30/30 at strengths 2 and 4.

## Local artifacts and interpretation

`outputs/refusal-01/` holds the immutable input/source snapshot, extracted
vectors, preflight and generation records. `outputs/refusal-judge-isolation-audit-01/`
holds private raw audit data; only sanitized validation metadata is tracked.
Response classification uses fresh tools-off calls with anonymous shuffled
prompt/response IDs. The advisor's project context is not supplied to those
calls. Multiple versions of a prompt can appear in one shuffled batch, which
limits independence. These are custom Claude-graded XSTest subsets, not official
full-benchmark scores or human judgments.

No negative steering, NF4 transfer, new training, model-weight upload or public
vector release belongs to this pilot.

## Measured cost and artifacts

The three GPU stages took 7.67 seconds for extraction, 251.13 seconds for
preflight (including the cap rerun and hook probes), and 180.95 seconds for
steered development: 439.75 seconds total including model loading. Peak
PyTorch allocation was 8.52 GB. These times exclude coding, review, preparation
and Claude judging. In total 226 responses were generated: 60 initial defaults,
60 cap-rerun defaults, 16 hook probes and 90 steered development responses.
Defaults were reused rather than regenerated for classification.

The full compact result is [`results/refusal_01.json`](../results/refusal_01.json);
the immutable decision is
[`results/protocols/refusal-selection-01.json`](../results/protocols/refusal-selection-01.json).
There are no new model weights or adapters to publish to Hugging Face.
