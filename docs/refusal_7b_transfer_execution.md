# Qwen2.5-7B transfer of the completed 4B refusal workflow

Status: **fixed transfer complete, development null**, 2026-11-16.
The user selected the existing 4B workflow as the baseline, with additional
work only where necessary, and authorized continued execution. See the
[fixed protocol](refusal_7b_transfer_protocol.md),
[design review](../results/reviews/refusal_7b_transfer_design_review.md), and
[implementation review](../results/reviews/refusal_7b_transfer_implementation_review.md).

The reviewed implementation and initial registration were committed in
`4e4b6e6` before any new target-model output. Manifest SHA256:
`a62b8f34507bf2766a0a0ca4a96a9db03c63e1decb5c39dbfcf0bafb6b10e1bb`.
The worker's focused suite passed 23 tests; the manager independently checked
identity parity, cached/full-prefix agreement, 84-site coverage and cleanup on
a tiny 28-layer Qwen2. The unchanged tools-off grading audit passed its binding
checks; no new calibration or grader audit call was needed.

## Extraction complete

The unadapted BF16 Qwen2.5-7B-Instruct checkpoint used the same 100
AdvBench/Dolly pairs and block-output, last-real-prompt-token extraction as
the 4B workflow. Block 18 is a fixed relative-depth mapping of the earlier
block 23, selected before outputs. No new direction search was performed.

- 200 prompt forwards, no extraction-source answers generated.
- Raw direction norm: 47.50859; median activation norm: 87.80380.
- Raw norm/activation ratio: 0.54108; fixed half-vector shift ratio: 0.27054.
- Split-half cosine: 0.98506, descriptive only.
- Stage elapsed: 10.57 seconds; peak PyTorch allocation: 15,307,132,416 bytes.
- Direction file SHA256:
  `671c6a839cd01a3767d6455b9a1b59a37059f712097ff1f8142324dfccbae28e`.

These are extraction/mechanics diagnostics, not proof of a pure refusal feature
or successful behavioral control.

## Baseline and mechanics complete

All 60 fresh development baselines completed in 90.71 stage seconds. There
were no empty responses. The fixed cap was reached on 21/30 safe and 5/30
source-unsafe responses. Refusal/usefulness labels await the joint grading.

All four real-model probe conditions passed. Both installed identity hooks
matched baseline tokens and logits exactly. Nonzero maximum absolute logit
changes were 7.09375 for subtraction and 5.66796875 for ablation. The ablation
covered all 84 intervention sites and 140 observed nodes, including mid/post
residual observations, during prefill and cached decoding. Hooks were removed
and plain logits restored exactly. There were 32 probe generations.

The exact probe/budget record was committed in `53320d4` before candidate
generation. Its rough estimate for all remaining conditional generation was
515.77 seconds, excluding load, judging and multiple-choice scoring; it is not
a guaranteed runtime bound.

## Development generation and grading complete

All 120 candidate responses completed in 200.75 stage seconds. There were no
empty responses. Mechanical results, before semantic grading:

| Condition | Safe cap hits /30 | Unsafe cap hits /30 | Safe degeneration /30 | Unsafe degeneration /30 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 21 | 5 | 0 | 0 |
| Subtract18 | 22 | 17 | 1 | 0 |
| Ablate all | 25 | 8 | 0 | 0 |

All 180 baseline/candidate answers were graded in one blinded development packet
under the unchanged rubric. All nine batches passed on their first attempts:
nine calls, no retries, missing labels or replacements.

The initial grading command stopped at the version check, before output-folder
creation or any judge call: the default Claude symlink resolved to 2.1.326.
The existing process-local PATH pin in
`outputs/sarcasm-method-transfer-01/runtime-bin` resolves the original native
2.1.323 binary, SHA256
`604d29a6869409106b6a1110bdadcc8b72e8654a5a865d23258f938a696e348e`.
Using that pin restored the exact audited version/argv without changing source,
rubric or audit, downloading a runtime, or repeating an uncertain call. The
runtime receipt is `operator_records/grader_runtime.json` under the run root.
Subsequent judge commands must retain that process-local PATH pin.

## Fixed decision: neither method qualified

| Development condition | Safe refusals /30 | Safe useful/correct /30 | Unsafe refusals /30 | Unsafe valid full compliance /30 |
| --- | ---: | ---: | ---: | ---: |
| Fresh baseline | 3 | 20 | 28 | 2 |
| Subtract18, raw coefficient -0.5 | 0 | 13 | 17 | 13 |
| Ablate all | 2 | 19 | 26 | 4 |

Subtraction demonstrated a refusal-reduction effect, including disappearance of
the three benign refusals. Only **one** became an AI-rated useful valid full
answer (`xstest-109`). Overall safe usefulness lost seven answers, beyond the
allowed two, and one safe answer became mechanically degenerate. Subtraction
therefore failed its quality gates despite meeting the unsafe compliance gain.

Projection removal retained safe usefulness within tolerance and introduced no
mechanical degeneration, but its unsafe compliance gain was two rather than the
required six. It produced no paired useful rescue of a baseline benign refusal.
These source-unsafe measurements describe willingness, not benign improvement
or correctness of unsafe answers.

The exact null decision was committed in `84d4282`. Final generation and
retention were skipped as registered; no final prompt was generated. Counts:
200 extraction prompt forwards, 60 fresh baselines, 32 probe generations,
120 candidate generations, 180 response judgments, zero final/retention calls.
Compact result: [refusal_7b_transfer_01.json](../results/refusal_7b_transfer_01.json).

The transfer establishes a mechanism effect with a quality cost; it does not
establish a usable benign-refusal controller. Claude is reviewing whether one
weaker subtraction dose is a necessary calibration within this same workflow.
No weaker-dose response has been generated and the fixed null is immutable.

Raw artifacts remain under ignored `outputs/refusal-7b-transfer-01/`. The
unfinished canonical framework and the cancelled OR-Bench expansion are unused.
The historical 4B/7B studies and scores are preserved.
