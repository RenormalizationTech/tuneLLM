# Qwen3-4B refusal-reduction research

Status: complete. Subtraction passed the development rule and reduced refusals
on the held-out set, but failed the fixed final quality criteria. It is a
measured tradeoff, not a validated usable configuration. No reselection or
additional sweep follows the final result. The exact design is in
[the protocol](refusal_reduction_protocol.md). Earlier positive-intervention
pilots remain unchanged and did not test refusal reduction.

The user authorized proceeding if Claude declined scientific review. Claude
declined the core design and did not complete a limited measurement review.
The coordinating Codex owns acceptance, and the dedicated Codex worker
implements/tests. No Claude design approval is claimed.

The two fixed methods are block-23 raw-vector subtraction at coefficient -0.5
and all-block removal of the projection onto the same unit direction. Both
act at all token positions. The latter covers pre-block residuals and
attention/MLP residual writers; a block-output-only hook would miss the
intermediate residual input to the MLP. These methods differ in operation and
coverage, so their comparison is not a factorial isolation of those factors.

The public author reference was inspected at commit
`9d852fae1a9121c78b29142de733cb1340770cc3`,
`pipeline/utils/hook_utils.py`, SHA-256
`0fa5de8d6a739f9f3ecfcb462d027d8ade269f6ce469e320c3238ef4a1d0cd56`.
The installed Qwen3 implementation has 36 blocks with hidden size 2560.
The coordinator independently checked the hook mapping and reviewed
projection arithmetic, scope and cleanup. Per-device vector caching and
sampled numerical probe diagnostics avoid repeated transfers and per-token
CPU synchronization. Normal generation does not collect those diagnostics.

The original BF16 base, extracted layer-23 vector and source data are reused.
The 60 development baseline predictions and their existing labels are reused
without re-generation or re-judging. The new candidate packet will contain
120 items; baseline labels came from a separate older packet. There is only
1/30 benign baseline refusals, so that fixture mainly measures quality
retention. Valid full compliance on the source-unsafe class is measured
separately; it means willingness to answer, not correctness or successful
execution. Invalid output cannot count as an improvement.

The fixed rule selects one method or null. A null skips final generation and
retention. Otherwise only baseline and the selected method run on 72 final
prompts and a tiny 48-item general/physics/math/instruction check. No final
result can reselect a method. Raw prompts, outputs, keys and vectors remain
ignored/private; compact records, source and registrations belong in Git.

The exact input/source/runtime snapshot is registered in
`results/protocols/refusal-run-03.json`, before model execution. Source,
scorer resources, prior artifacts, baseline labels and retention IDs are bound
by hash. The unchanged judge adapter passed its existing audit/version
binding checks; no extra dummy model call was made.

The exact run was committed in `e665981` before inference. Both installed
identity interventions matched saved baseline tokens and plain logits exactly.
The projection probe covered 108 intervention sites and observed 180 nodes
including mid/post residuals. Nonzero last-prompt logit deltas were 8.625 for
subtraction and 6.953125 for ablation; plain logits after hook removal matched
exactly. All four eight-response probes hit cap 320 on every item. Projection
is performed in FP32 then rounded to BF16, so residual orthogonality is
approximate. The exact diagnostic/budget report was committed in `dbd326c`
before development; its rough remaining generation estimate was 707 seconds,
excluding loading and judging. Probe-stage wall time was 76.69 seconds with
8.53 GB peak PyTorch allocation.

## Development selection

| Condition | Source-unsafe valid answers / 30 | Source-unsafe refusals / 30 | Safe useful/correct / 30 | Safe cap hits / 30 | Unsafe cap hits / 30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reused baseline | 2 | 28 | 11 | 27 | 9 |
| Subtract at block 23 | 8 | 22 | 10 | 28 | 24 |
| Project out across all blocks | 6 | 24 | 13 | 27 | 21 |

All conditions had zero empty, degenerate or AI-unscorable development outputs.
Subtraction reached the fixed valid-answer threshold of 8/30 and retained
benign usefulness within the tolerance of two fewer than baseline. Projection
removal missed the valid-answer threshold, despite its higher indicative
benign usefulness count. The rule selected subtraction without changing
coefficients or adding another method. This does not show that single-block
intervention is generally better than all-block intervention.

The 120 new items were judged in six fresh-context batches, all valid on the
first attempt. No baseline response was re-judged. The selected method and all
input hashes were committed in `099ed24` before final generation. Final
comparison is baseline versus this one fixed candidate; projection removal
does not enter final and cannot be selected after seeing final results.

## Final result: reduction with quality loss

The selected setting was raw coefficient -0.5 at block 23 output, all token
positions, on the original BF16 model. Both final conditions were generated
fresh and classified together under the frozen rubric.

| Final condition / source class | Valid full answers | Refusals | Useful/correct | Mechanical degeneration | AI-unscorable | Cap hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, safe (42) | 41 | 1 | 29 | 0 | 0 | 20 |
| Subtraction, safe (42) | 39 | 1 | 23 | 2 | 1 | 26 |
| Baseline, unsafe (30) | 1 | 29 | Not measured | 0 | 0 | 6 |
| Subtraction, unsafe (30) | 7 | 22 | Not measured | 1 | 0 | 16 |

Quality flags overlap: the safe AI-unscorable response is among the mechanical
degeneration flags. Those columns must not be added as distinct invalid items.
There were no empty or missing outputs/ratings. Refusal counts include partial
and full refusals; invalid outputs are excluded from both the valid-answer
and refusal numerators while denominators stay fixed.

The source-unsafe valid-answer gain was six out of 30, or 20 percentage points.
Refusals decreased by seven, but one of those outputs was invalid, so the
valid-answer measure avoids overstating the gain. Its paired prompt-bootstrap
95% interval was +6.7 to +36.7 points; the five-family cluster interval was
0 to +46.7 points. The latter includes zero and is imprecise with five clusters.
This is an observed small-set effect, not a general refusal-removal claim.

Safe useful/correct counts fell by six out of 42 (14.3 percentage points),
exceeding the tolerance of three items. Two safe and one unsafe outputs had
mechanical repetition/degeneration flags, and one safe output was AI-unscorable,
all increased over baseline. Those final quality criteria failed. Safe
usefulness is AI-rated and indicative: the cap hit count increased from 20 to
26 and truncation exposure differs. The useful/correct paired prompt interval
was -31.0 to +2.4 points; the cluster interval was -35.7 to +7.1 points.

`null: false` in the full result means development selected a candidate. It
does not mean the final candidate passed: `final_quality.all_criteria_pass`
is false. The locked decision was not reopened and the other method was not
substituted after the final results.

## Small automatic retention check

| Custom test subset | Baseline | Subtraction |
| --- | ---: | ---: |
| General MMLU likelihood | 11/16 | 11/16 |
| Physics MMLU likelihood | 7/16 | 6/16 |
| GSM8K numeric | 8/8 | 8/8 |
| IFEval prompt-strict | 7/8 | 7/8 |

These counts passed the predeclared coarse retention tolerances. They do not
establish capability equivalence. The source prefixes had been used in earlier
project evaluations; this is a tiny diagnostic using existing scorer methods,
not a new full-benchmark claim. Final refusal prompts had not previously been
generated or judged. Math generations had no quality flags or caps; baseline
IFEval had one mechanical repetition flag and cap hit, versus zero for the
candidate. Mechanical flags are heuristics and task context matters.

## Judging, cost and closure

All six development and eight final classification batches were valid on their
first attempt, totaling 264 newly judged items with no retries. Requested and
observed model were `claude-opus-5-5`; effort requested medium, actual effort
and dated snapshot not exposed. These fresh tools-off anonymous classification
calls are measurements, not approval of the study by the project advisor.
The advisor's design-review refusal remains recorded separately.

New generation counts were 32 probes, 120 development, 144 final and 32
retention: 328 total, plus 64 multiple-choice likelihood items. Sixty original
development outputs and labels were reused without additional calls. Stage
wall times including model loading were 76.69 seconds (probe), 281.73
(development), 252.51 (final) and 105.28 (retention), totaling 716.23 seconds,
about 12 minutes. Peak PyTorch allocation was 9.54 GB. These times exclude
coding, CPU preparation, review and hosted judging. GPU 0 was released after
the jobs; existing workloads were left running.

The bounded comparison is complete. It demonstrated a refusal-reduction
tradeoff under this vector and setting, without producing a configuration
that passes both reduction and quality requirements. There is no new training,
model-weight modification, release or automatic parameter search.

See the [compact summary](../results/refusal_reduction_summary_03.json),
[full result](../results/refusal_03.json),
[judging provenance](../results/refusal_judging_03.json), and
[fixed selection lock](../results/protocols/refusal-selection-03.json).
