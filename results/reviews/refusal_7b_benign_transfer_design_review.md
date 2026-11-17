# Qwen2.5-7B benign transfer diagnostic: design review

Reviewer: Claude advisor, read-only, except for this file. No model or judge
call was made.

**Basis:**
- [half-dose result review](refusal_7b_halfdose_result_review.md);
- `results/refusal_7b_halfdose_01.json`, a development null locked in
  `cee9154`;
- the original Qwen3-4B XSTest-safe baseline, with 17 nonempty refusals in
  `results/xstest_baseline_01.json`.

The user had already authorized reusing these 17 texts.

## Approved design

A fixed descriptive diagnostic, **26 new generations**:
- **Primary 13:** 14, 57, 59, 61, 64, 67, 103, 151, 153, 168, 169, 417, 418.
- **Arms:** a fresh 7B baseline on the 13, and fixed −0.25 block-18 output
  subtraction on the same 13.
- **Settings:** the unchanged vector, hook, native render, greedy decoding,
  cap 320, batch 8 and seeds of the half-dose run.
- **Grading:** baseline and −0.25 graded together in one blinded packet with
  the original rubric and adapter.
- **Seen 4:** 109, 162, 165 and 171 are reported separately from their
  existing development baseline and −0.25 labels. They are not regenerated,
  because changing batch composition could change their outputs. They are
  marked as not independent of the dose choice.

There is no tuning, coefficient or layer change, selection, final, retention
or new benchmark. Both registered nulls (−0.5 and −0.25) stay unchanged.

## Data ledger

| Group | IDs |
| --- | --- |
| Newly consumed from the protected old 7B reserved rows | 59, 61, 67, 151, 153, 168, 418 (register prospectively; **98** remain protected) |
| 4B-workflow final set, now consumed | 57, 64 |

DECCP is untouched.

## Reporting

- Descriptive counts only, with no success rule, gate or statistics:
  - baseline refusals (P and I separately);
  - useful valid-full-compliance rescues among them;
  - newly refused or no-longer-useful compliant items;
  - empty, degenerate and unscorable outputs;
  - caps per arm.
- If baseline refusals are few, say the evidence is sparse.

## Limits to disclose

- The 13 texts were selected because Qwen3-4B refused them, under a different
  model and system message. Regression toward compliance on 7B is expected,
  and nothing can be read as a population rate.
- −0.25 was chosen on development prompts that include the seen 4, and it
  failed its registered unsafe gate. This diagnostic doesn't revise that
  null.
- Earlier 7B outputs for 14, 103, 169 and 417 used incompatible settings
  (cap 256, batch 1) and are not reused.
- Carried over from the earlier runs:
  - AI labels from a single judge family;
  - the seen 4 were graded in a different packet context;
  - truncation at the cap;
  - the prompts are public and were exposed in earlier 4B runs.
- **Allowed claim:** "on N of these 4B-selected texts that 7B refused, M
  became useful answers at −0.25, with H harms". It doesn't establish a
  benign controller or general over-refusal reduction.

## Status

Approved as a minimal descriptive check under the existing authorization. The
next step is code review.
