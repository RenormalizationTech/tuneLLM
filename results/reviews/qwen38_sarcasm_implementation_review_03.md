# Qwen3.8-27B sarcasm GCAD: implementation review 03 (R0 source signoff)

Advisor review, 2026-11-18, 15:50 EST. It covers current settled source only;
no file changed between 15:49:17 and this review. The review was read-only,
plus CPU tests and one local tokenizer round-trip. No model, judge or GPU
call was made.

## Signoff

**Signed off for source R0 registration, and for the synthetic smoke after
R0.** This applies only to the hashes below; any change to these files voids
the signoff. C6 completion is required before final registration (see below).

| File | sha256 |
| --- | --- |
| `scripts/qwen38_sarcasm.py` | `ed2b942dcdfcfb09c00531f89756b103d48d54657129197bc5b374ce09429904` |
| `scripts/qwen38_sarcasm_judge.py` | `483c711d015eb43ee61463e82286a51f71c53f80c5b3afb65ab1e4b36cfb0a4a` |
| `scripts/qwen38_gcad_port.py` | `3d76d8147ffbe6f802b285d643b4363e09349f0dd14a5f6a2924545b5966abe3` |
| `tests/test_qwen38_gcad_port.py` | `704bd3cdb5f7be56844af341d0ccd646d990dac43e2e124ead00a01f9e03d57c` |
| `tests/test_qwen38_sarcasm.py` | `9ff69b6998d7d0b9bc998287faabec21942b36718ba72aa737fc59e23193c806` |
| `configs/qwen38_sarcasm.json` | `0844af59d0a48cf968d494c96a1d6f5a6c42a164bc4d51ff2c59d21b9e7deecb` |

The plan (`docs/qwen38_sarcasm_plan.md`) was last amended in this pass to
disclose C5; R0 records its hash.

CPU tests:
`CUDA_VISIBLE_DEVICES= outputs/qwen38-science-01/env/bin/python -m pytest -q -m research tests/test_qwen38_gcad_port.py tests/test_qwen38_sarcasm.py`
gives **15 passed**.

## Criteria met

- **C4, raw trajectory.** `boundary()` uses the saved prompt IDs; the prompt
  is re-tokenized only to check equality and the system prefix. It then
  appends the saved `generated_ids` and removes one terminal EOS from the
  model's configured EOS list. No EOS is appended, and generated text is never
  re-tokenized. Special-token positions are excluded from the mask. An empty
  mask raises. The decode of the raw IDs must equal the judged `output`. The
  test covers all of these, including a special token inside the response and
  each failure path.
- **Synthetic trajectories.** Response IDs come from the fixed response text,
  are frozen in `inputs.json` at prepare and are hash-guarded. I checked with
  the local tokenizer that the 17 IDs decode exactly back to the text, so the
  decode-equality guard will not trip.
- **m1.** `frozen_rendering()` asserts that the prompt text and IDs equal
  `rendered.json`, which now includes the mechanics probes (`C01`, `C02`).
  Tested.
- **C1–C3 and counters.** Verified earlier and unchanged in substance.
- **C5.** Disclosed prospectively in plan §8, as PM allowed: rubric bytes and
  context are preserved, and the envelope and schema differ.

## Open before final registration (C6, not blocking R0 or smoke)

`report()` exists and applies the frozen `gates(baseline, selected, random)`;
for a single method this is equivalent to `final_decision`. Three items are
still missing:

1. **The H14-excluded descriptive figures.**
2. **Impoliteness descriptives.** The arm means are needed, plus the
   within-arm Pearson correlation between SARC and impoliteness for the
   selected arm, and the pooled correlation with the confounding caveat.
3. **A primary-complete path.** `report()` needs `judgments/final/scores.json`,
   which is written only after impoliteness completes. If the impoliteness
   stream stops fatally, the primary decision therefore cannot be reported.
   Write per-metric primary files, which already exist as
   `judgments/final/{trait,coherence}.json`. Then let `report()` emit
   `primary_complete_descriptive_incomplete` when the trait and coherence
   files are complete and impoliteness is not.

These must be implemented, CPU-tested and hashed into a registration before
final registration, as plan §6.3 requires.

## Freeze

After R0, this file, the design review, reviews 01–02 and the plan are frozen.
Later findings go in new review files.
