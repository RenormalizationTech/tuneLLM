# Qwen3.8-27B sarcasm GCAD: review 08 (elicitation, judging, eligibility)

Advisor review, 2026-11-19, of these files in `outputs/qwen38-sarcasm-gcad-01`:
- `elicitation/`;
- `judgments/elicitation/`;
- `eligibility.json`;
- `counters/`, `gpu-sessions/` and `approvals/`.

The review was read-only, using local JSON and hashes with CPU Python only. No
model, judge or GPU call was made. No development or final data exists, and
none was inspected. Frozen files were not edited.

## Verdict

**Eligibility verified and accepted: 90/100 pairs, `extraction_null = false`.**
I recomputed it independently from `scores.json` and `inputs.json` with the
frozen §6.1 rule. The result is the same 90 pairs, in the same order. The
journal provenance is complete. Nothing blocks extraction, which is already
running.

## Generation provenance

- **Rows.** 200 rows, in the same order as frozen `inputs.json['elicitation']`.
  Every frozen field (id, pair, side, question_id, condition, instruction,
  user, system) equals the frozen value.
- **Receipts.** All 200 row receipts match their row's sha256. There are no
  failure files.
- **Tokens.** 51,058 generated tokens. The generation reservations total
  51,392 (200 × 256 for elicitation plus 192 for the smoke), which is below
  the 92,544 cap. Counters are elicitation 200 and synthetic smoke 6.
- **Completion.** `generated_ids == output_ids` for every row. No reply is
  empty. **197 of 200 hit the 256-token cap**, and 3 ended on EOS.

## Judge provenance

- **Plan.** 25 trait batches and 25 coherence batches cover the same 200
  (question, condition) items. The opaque IDs are disjoint between the two
  metrics.
- **Attempts.** There were 50 starts, all first attempts (no `-1` retries),
  each with a matching receipt. Total study judge starts are 50, against a
  limit of 260.
- **Model.** `modelUsage` is only `claude-opus-5-5` in all 50 calls. The
  snapshot hash is unchanged (`343c744b…`). `invoke()` refuses any identity
  change before or after a call.
- **Scores.** `scores.json` covers exactly the 200 IDs with
  `{trait, coherence}`. It agrees item for item with `trait.json` and
  `coherence.json`. There are 0 trait refusals, and every coherence value is
  an integer from 0 to 100.
- **Approvals.** Approvals exist for each executed phase: synthetic smoke,
  judge audit, elicitation, judge-elicitation and extraction.
- **Binding.** `eligibility.json` records `scores_sha256`, which equals the
  current `scores.json`.

## Eligibility, recomputed

The rule is: pos SARC ≥ 50, neg SARC < 50, both non-refusal, and both
coherence ≥ 50.

- **Eligible:** 90 pairs.
- **Excluded:** 10 pairs. In 9, the positive reply scored SARC < 50. In 1, one
  side had coherence < 50. No pair was excluded because its negative side
  scored SARC ≥ 50.
- **Coverage:**
  - all 20 extraction questions are represented;
  - by instruction pair, i0 17, i1 20, i2 18, i3 16 and i4 19.
- **Separation:**
  - positive SARC mean 66.3 (range 15–92), positive coherence mean 76.8
    (minimum 58);
  - negative SARC mean 0.2 (range 0–5), negative coherence mean 82.6
    (minimum 40).
- **Expected teacher-forced passes.** 90 pairs × 2 = 180, within the
  300-pass cap. This matches the PM-reported extraction size.

## Disclosure

**Elicitation replies are almost all truncated.** 197 of 200 replies hit the
registered 256-token cap. Both the judge and the bank therefore see capped
responses: eligibility is judged on truncated text, and Δ is pooled over the
truncated response tokens. This is consistent with the registered design.
The raw saved IDs are teacher-forced exactly as generated (C4), so judge and
bank see identical text. It is a property of the bank and must be stated in
the closeout. It is not a defect, and it does not justify any change.

## Runtime update (measured, for PM planning)

- **Measured speed.** The elicitation GPU session took 9,185.6 s for 51,058
  tokens. That is about **5.6 tok/s of process wall time, including the model
  load**. This is below my planning range of 8–15 tok/s.
- **Charged so far.** Smoke 70.8 s plus elicitation 9,185.6 s = 9,256 s,
  with extraction still running.
- **Remaining generation at worst.** Mechanics needs 192 tokens, development
  128 × 160 = 20,480 and final 20,480, about 41.2k tokens in all. At 5.6 tok/s
  that is about 7,350 s, plus model loads.
- **Margin.** Unless extraction is unexpectedly long, the projected total
  stays well inside the 28,800 s stop, at roughly 17–19k s. If extraction
  exceeds about 2.5 h, the PM should recheck the margin before approving
  development.
