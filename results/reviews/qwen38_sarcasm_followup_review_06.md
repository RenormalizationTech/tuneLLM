# Qwen3.8-27B sarcasm follow-up: review 06 (Part B, judging, closeout)

Advisor review, 2026-11-19. CPU only, read-only.
- No model or judge call was made by the advisor.
- No frozen file was edited.
- The scores were re-derived from the saved raw judge replies with an
  independent parser. They were not taken from `judge/scores.json`.

## Verdict

**The registered outcome `exploratory_null_through_b16` is confirmed.**

- **Canonical result.** `results/qwen38_sarcasm_followup_01.json` (sha256
  `61362b38…81ad`) equals the `closed.json` binding.
- **Old outcome.** `old_outcome_unchanged: development_null`.
- **Errors.** `execution_error` is null.

## Generation integrity: verified (224 Part B rows)

**Receipts and bindings.**
- Every row receipt matches the sha256 of its row and start files.
- Every binding is `{manifest 87d1d0e6…, stage generation, key}`.
- There are no failure files.
- `generation/result.json` rows equal the per-row values.

**Inputs.** Every frozen `inputs.json` generation field matches exactly:
- id, question, arm and prompt;
- `input_ids` and `user_turn_ids`;
- `system_length`, which is 45 for prompted and none otherwise.

**Coverage.**
- Each steered row shows `{prefill 1, decode n−1, errors 0}` at all 6 sites.
- The telemetry token counts equal the prompt length and n−1.
- Every telemetry value is finite.
- Baseline and prompted rows carry no audit and no telemetry.

**Length.**
- 35,840 generated tokens.
- All 224 rows hit the 160-token cap, and none ended with EOS. This holds for
  every arm, baseline included.

**Text checks (frozen `text_checks`).**
- mostly-not-letters: 0 rows in every arm;
- empty: 0 rows;
- minimum distinct-bigram ratio: 0.888, against the invalid threshold of 0.6;
- invalid: 0 in every arm.

## Telemetry (token-weighted over 32 rows, actual ‖steered − original‖ / native per token)

| Arm | Prefill, range over sites | Decode, range over sites | Max single token | Decode gate |
| --- | --- | --- | --- | --- |
| b2 | 0.23–0.33 | 0.13–0.22 | 0.84 | 0.42–0.48 |
| b4 | 0.50–0.66 | 0.33–0.51 | 1.67 | 0.48–0.59 |
| b8 | 1.11–1.38 | 0.97–1.37 | 2.96 | 0.62–0.80 |
| b16 | 1.79–2.64 | 1.62–2.41 | 5.54 | 0.68–0.87 |
| random16 | 1.86–2.61 | 0.88–1.61 | 6.59 | 0.36–0.50 |

**Agreement.** The row-level sums agree within these bounds:
- requested vs actual: relative difference ≤ 0.002;
- algebraic vs requested: ≤ 0.001.

**Prefill gate at site 23.** It is 0.673 in every steered arm, as expected,
since nothing upstream of the first steered site differs between arms.

**Decode gate under b16 vs random16.** During decoding the gate stays high
under b16 and is much lower under random16. In effect, steering along the
bank direction raises downstream demand for itself, and the random direction
does not. **random16 is therefore matched to b16 in setting, coefficient
rule and ‖Δ‖, but not in realised decode shift.** Its decode actual ratio is
about 0.6–0.8× that of b16. It is still at least as large as b8's at most
sites, so the random control still applies a large perturbation.

## Judging provenance: verified

**Batches.**
- 84 batches: 28 each for trait, coherence and impoliteness.
- Exactly one start per batch, with no second attempts. There are 85 judge
  starts in total, counting the audit, against a cap of 169.
- Every receipt and binding matches. Every return code is 0.

**Model.** `modelUsage` is only `claude-opus-5-5` in all 84 replies.

**Parsing.**
- 17 replies (4 trait, 13 coherence) used a single surrounding ```json
  fence. This is accepted by the frozen historical rule in
  `tone_judge.result_json`.
- Every reply contains exactly the batch's opaque IDs, with integer scores
  from 0 to 100. There were no trait refusals.

**Independent parse.** The scores mapped through `plans.json` equal
`judge/scores.json` exactly.

## Independent recompute (frozen gate definitions)

| Arm | SARC (conservative) | Answers with SARC ≥ 50 | Coherence | Coherence < 50 | Coherent and sarcastic | Impoliteness |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.00 | 0 | 84.28 | 0 | 0 | 1.9 |
| prompted | 55.56 | 25 | 76.00 | 0 | 25 | 52.1 |
| b2 | 0.09 | 0 | 84.16 | 0 | 0 | 2.2 |
| b4 | 4.97 | 0 | 83.44 | 0 | 0 | 6.8 |
| b8 | 88.50 | 32 | 58.94 | 6 | 26 | 70.7 |
| b16 | 95.16 | 32 | 34.59 | 30 | 2 | 81.5 |
| random16 | 0.00 | 0 | 84.28 | 0 | 0 | 2.1 |

The random16 coherence mean equals the baseline mean by coincidence. The
per-item scores differ, and no random16 output is token-identical to its
baseline.

| Gate vs baseline | SARC ≥ base + 20 | ≥ 8 answers with SARC ≥ 50 | Coherence ≥ 50 | Coherence ≥ base − 20 (64.28) | Invalid ≤ 3 | All |
| --- | --- | --- | --- | --- | --- | --- |
| b2 | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ |
| b4 | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ |
| b8 | ✓ | ✓ | ✓ | **✗ (58.94)** | ✓ | ✗ |
| b16 | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ |

**Outcome.** No strength passes, so the outcome is
`exploratory_null_through_b16`. Every gate, metric, descriptive and label
matches the canonical result.

**Other checks.**
- **Prompted validity.** It is satisfied: prompted reaches +55.6 over
  baseline.
- **b16 direction check.** The margin of b16 over random16 is +95.2, with a
  question-bootstrap 95% CI of [94.7, 95.6] (my descriptive check). The ≥ 20
  margin holds. This is the only dose-matched control. It is
  setting-matched, not realised-shift-matched, as noted above.

**Paired descriptive differences in SARC** (question bootstrap, 10,000
resamples):

| Contrast | Difference | 95% CI |
| --- | --- | --- |
| b4 − baseline | +5.0 | [2.3, 8.2] |
| b8 − baseline | +88.5 | [85.6, 90.8] |
| prompted − baseline | +55.6 | [50.0, 60.6] |

**Part A consistency.** The canonical `part_A` is identical to my review 05
scratch run of the frozen `part_a()`.

## Ledger

| Item | Used | Limit |
| --- | --- | --- |
| Generations | 232 (8 mechanics + 224) | 232 |
| Reserved tokens | 36,096 | 36,096 |
| Teacher-forced passes | 194 | 194 |
| Logit forwards | 5 | 5 |
| Judge starts | 84 + 1 audit | 168 + 1 |
| GPU session | 6,348 s (1.76 h) | 4 h |
| Judge session | 491 s | — |
| Launcher wall | 6,843 s (1.90 h) | 12 h |
| Activation storage | 1.56 GB | 2 GiB |

Every item is within its cap.

## Interpretation (exploratory; within registered limits)

1. **The registered result is a null through b16, and it is a coherence
   failure, not an absence of trait movement.** The response is a sharp step
   between b4 and b8:
   - SARC is about 5 at b4 and 88.5 at b8.
   - At b8, all 32 answers score SARC ≥ 50 and 26 are coherent-sarcastic,
     compared with 25 for prompted.
   - b8 fails only the relative-coherence gate, by 5.3 points (58.94 vs
     64.28).
   - At b16 the outputs are sarcastic but mostly incoherent: 30 of 32 have
     coherence < 50.

   No strength between 4 and 8 was registered, and none may be tried
   post hoc under this study.
2. **Direction.** At b16, the bank direction produces sarcasm, and an
   equal-setting random direction produces none (+95 margin). This matches
   Part A's positive alignment, but the caveats are strong:
   - there is a single random seed set;
   - the comparison holds at b16 only;
   - it is setting-matched, not realised-shift-matched;
   - b16 itself fails coherence.

   Lower-strength passes are not direction-tested, and b8 has no random
   control.
3. **Relation to the old study.** On fresh questions, b2 again gives
   essentially zero sarcasm (0.09), as in the old dev32 b2 (0.19). The old
   `development_null` remains unchanged. It is consistent with these data
   that the old b ∈ {1, 2} lay below the strengths where an effect appears.
   As registered, however, **dose is not concluded to be the cause of the
   old null**. These are different questions, a different run and an
   exploratory design.
4. **Caveats for the PM's write-up.**
   - Every answer was truncated at 160 tokens.
   - Coherence is judged, and the judge may partly penalise sarcastic
     register. Prompted also drops 8 points.
   - Impoliteness rises together with sarcasm, from 70.7 at b8 to 81.5 at
     b16. Within-arm correlations are 0.31–0.44. The pooled r = 0.96 is
     confounded by arm.
   - The steered sarcasm is therefore not shown to be separable from
     rudeness.
   - There is no independent final set, n = 32, and the results are NF4 only.
   - No validated-controller or specificity claim is made.

## Wording the closeout may use

> On 32 fresh questions, the frozen GCAD bank produced no sarcasm at b ≤ 4,
> strong judged sarcasm at b = 8 with a coherence drop that failed the
> registered relative-coherence gate (58.9 vs ≥ 64.3), and sarcastic but
> largely incoherent output at b = 16. The registered outcome is
> exploratory_null_through_b16. At b = 16 a setting-matched random direction
> produced no sarcasm. The old development_null is unchanged, and dose is not
> concluded to be its cause.

This review closes the advisor's checks for follow-up 01.
