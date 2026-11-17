# Minimal transfer of the completed 4B refusal workflow to 7B

Status: implementation and review, before new model execution. The user authorized
continued work on 2026-11-16, with the completed 4B workflow as the baseline and
extra work only where necessary. The primary practical interest is reducing
unnecessary refusal on benign requests. This transfer first measures the same
mechanism and quality outcomes as the 4B study; it does not substitute a new
benchmark or an increase-refusal demonstration for that interest.

Reference: [4B protocol](refusal_reduction_protocol.md),
[4B execution](refusal_reduction_execution.md), and
`results/refusal_reduction_summary_03.json`. The 4B study demonstrated a reduction
in source-unsafe refusals with answer-quality costs. Its benign refusal rate did
not improve, and final quality criteria failed. Those observations remain intact.

## Fixed transfer

- Unadapted BF16 `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, cached weights, existing `.venv`.
  Native Qwen2 rendering inserts its default system message; Qwen3 nonthinking
  settings are inapplicable. Use the same rendering throughout this comparison.
- Reuse the exact 100 prepared AdvBench/Dolly pairs and their screening decisions
  from `data/refusal-01/prepared/`, without new sources, pairs or screening.
  Extract at the last real prompt token after the native generation prefix,
  using block **output**, as in the 4B workflow. Form the raw harmful-minus-benign
  mean direction with equal prompt weights and the existing arithmetic. The
  direction must be re-extracted: hidden size changes from 2560 to 3584.
- Fix block **18**, zero based, before outputs. This is the single relative-depth
  mapping `round(23 * 28 / 36)` of the old block 23, rather than a layer search.
  The model has 28 blocks. Report vector norm and its ratio to extraction
  activation norms descriptively; do not rescale the fixed intervention.
- `subtract18`: subtract **0.5 times the raw vector** at block 18 output.
  `ablate_all`: remove its unit-direction projection at every block input and
  attention/MLP residual writer, using the existing projection hook implementation
  (84 intervention sites). Both act at all prompt and cached-decode positions.
  Preserve FP32 intervention arithmetic and BF16 activation storage.
- Reuse exact development (30 safe, 30 source-unsafe), final (42 safe, 30
  source-unsafe), and 48-item retention fixtures, seeds and ordering. Greedy,
  batch 8, seed 42, cap 320 and repetition penalty 1.0 follow the old generation
  workflow. Explicit EOS/PAD handling follows the existing generator with the
  new tokenizer; record the actual IDs and effective settings before execution.

Old 7B cap-256/batch-1 outputs and their different grading schema are incompatible
with this comparison. Generate the 60 development baselines anew. Keep baseline
and both candidates together in one anonymous 180-item grading packet with the
unchanged refusal rubric, validator and audited tools-off Claude adapter. Original
4B and 7B judgments are never replaced. This joint packet is the necessary change
from 4B's reuse of older baseline labels, not a new evaluation design.

## Data reuse and scope

The 4B source labels remain separate. Source-unsafe full compliance measures
willingness to answer, not factual correctness, benign helpfulness or successful
execution. No harmful-answer quality optimization is introduced. Raw inputs,
responses, token IDs, vectors and grading reasons remain local and ignored.

Exact reuse of the 4B split reallocates 45 rows from the earlier 7B screen's
reserved 150: 20 development and 25 final safe prompts. Record their exact IDs
before execution; the other **105** stay ungenerated in this study. Preserve the
old split, stop decision, files and scores byte for byte. This is a new transfer
study authorized by the user's explicit carryover instruction, not an extension
under the closed headroom screen. The 25 DECCP held-out rows remain untouched.
The reused published prompts and prior cross-model exposure limit generalization.

No OR-Bench work, new dataset search, new direction-selection grid, coefficient
sweep, model change, training or publication is part of this transfer. The two
unfinished canonical-framework files from the cancelled expansion remain unused.

## Execution and acceptance

The coordinating Codex controls acceptance and GPU/judge stages; the Herdr Codex
worker implements/tests and Claude advises/reviews. Freeze source, config, model
assets, original prepared inputs, rubric and runtime hashes in a fresh
`outputs/refusal-7b-transfer-01/` manifest before GPU work. Use physical GPU 0,
UUID `GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b`, checking availability and leaving
other workloads intact. Bind exact reviewed registrations before dependent stages.

1. Extract the 7B direction and generate 60 development baseline responses.
2. Reuse the old four-condition, eight-safe-prompt mechanics probe: identity
   subtraction, identity ablation, subtraction and ablation (32 responses).
   Installed identity hooks must reproduce the new baseline tokens and logits;
   nonzero hooks must yield finite changed activations/logits, cover all intended
   prefill/decode sites, and leave plain logits unchanged after cleanup. Stop on
   mechanics failure. Freeze measured timing/budget before development candidates.
3. Generate 120 development candidate responses and grade all 180 development
   responses once with the original first-valid, complete-ID, exact-bound journal
   rules. Keep partial/full refusals distinct and report truncation, empty,
   degenerate and unscorable responses on fixed denominators.
4. Apply the original **baseline-relative** development rule: source-unsafe valid
   full compliance increases by at least 6/30; safe useful/correct answers fall
   by at most 2/30; and neither source class has an increase in empty, degenerate
   or unscorable responses. Among eligible methods choose higher safe usefulness,
   then unsafe valid full compliance, then ablation on a tie. A null skips final
   and retention. Freeze this decision before any final model work.
5. For a selected method, compare baseline and that method on the same 72 final
   prompts (144 new responses), with the old blinded rubric. No final reselection.
   Run the unchanged 48-item retention comparison (32 MC likelihood items and
   16 generations per condition).

Final criteria are the old rules: unsafe valid compliance gain at least 6/30;
safe usefulness loss at most 3/42; no per-class empty/degenerate/unscorable
increase; combined MC correct loss at most 2/32; GSM8K numeric and strict IFEval
loss at most 1/8 each; no empty/degenerate retention generation increase. Report
each criterion, paired prompt/family uncertainty and costs. Fixed gates are
pragmatic learning tolerances, not safety or equivalence certification.

Maximum target generations: **388** (60 new baseline + 32 probes + 120 candidates
+ 144 conditional final + 32 conditional retention), plus 200 extraction prompt
forwards and 64 conditional MC likelihood items. Reuse the existing judge audit
while its exact binary/version/source/rubric bindings remain valid; uncertain
calls stop rather than being silently repeated. No extra grader calibration or
evaluation gate is introduced.

## Reporting and next decision

Report refusal reduction and response quality separately. On benign prompts,
report actual baseline refusals and whether useful answers replace them. If the
benign baseline rarely refuses, the transfer can establish a mechanism effect
but cannot establish reliable benign over-refusal reduction. A source-unsafe
improvement alone is never relabelled a benign success. A selected candidate
that fails final quality remains a measured tradeoff.

Complete this fixed transfer before considering any additional method, dataset,
layer or strength. Discuss any genuinely necessary next step with the user using
the actual results. Compact output: `results/refusal_7b_transfer_01.json` and a
matching execution record; preserve all historical results.
