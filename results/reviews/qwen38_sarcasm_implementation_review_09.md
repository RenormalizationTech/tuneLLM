# Qwen3.8-27B sarcasm GCAD: review 09 (real bank and real-bank mechanics)

Advisor review, 2026-11-19, covering:
- `extraction/` (180 per-sample tensor files and their row records);
- `extraction/bank.safetensors`;
- `mechanics/`.

All checks were CPU-only, reading saved tensors and JSON. No model, judge or
GPU call was made. No development data exists. Frozen files were not edited.
The source is unchanged.

## Verdict

**Bank: verified. Real-bank mechanics: PASS against the registered criteria
1–4. Signed off for the registered development stage**: b ∈ {1,2}, baseline
and prompted, 128 replies. The diagnostic addendum (review 07) must be
hash-bound in the development registration, as the PM plans.

## Bank verification

**Binding.**
- `bank.safetensors` has sha256 `4fd9f86b…45de`, which equals both
  `extraction/result.json` and the mechanics approval dependency.
- The extraction approval binds `eligibility.json` (`03b9452d…6d25`), which
  equals the current file.

**Inputs.**
- There are exactly 180 samples: 90 eligible pairs, pos and neg. Every row
  receipt and per-sample tensor sha256 matches.
- Teacher-forced counters: extraction 56 as of review 08; 180 at completion.
  Extraction is also within the 300-pass cap.

**Raw-trajectory boundaries (C4).** For all 180 samples:
- the teacher-forced IDs equal the saved elicitation `input_ids` plus
  `generated_ids`, with one terminal EOS removed;
- `prompt_length` equals the saved prompt length;
- the response mask covers every generated token (175–256), so no special
  tokens appeared in the responses;
- there are 2 prefix checks per sample;
- system spans are 42–55 tokens.

**Counts.** `key_count` equals `system_length`, and `query_count` equals
`response_tokens`, at every site for every sample. There are 0 mismatches.

**Arithmetic, recomputed independently in float64.** I did not use
`port.bank`. I used:
- Δ = the mean over pairs of (pos − neg);
- K_pos = Σ key_sum / Σ key_count, over positives;
- q̄ the same way;
- D̄ = mean over heads of q̄_h·K_pos[h // 6] / √256.

The relative difference from the saved bank is at most 6.4e-8 for Δ and
3.5e-7 for K_pos. The absolute difference in D̄ is at most 1.9e-7. This is
FP32 rounding only.

**Shape, dtype, finiteness.** There are 18 tensors. Δ is (5120), K_pos is
(4, 256) and D̄ is a scalar, all float32 and all finite.

**Values (descriptive).**

| Site | ‖Δ‖ | D̄ |
| --- | --- | --- |
| 23 | 0.847 | 2.71 |
| 27 | 1.358 | 2.00 |
| 31 | 0.900 | 3.02 |
| 35 | 1.228 | 3.16 |
| 39 | 0.892 | 3.58 |
| 43 | 1.019 | 4.05 |

**Per-sample decomposition diagnostics (1,080 site-samples, not gated in
extraction).** They are consistent with the synthetic gate:
- gated error 0.0063–0.0108;
- omitted-gate error 3.23–8.30, at least 390× the gated error;
- split error 0.0031–0.0047.

## Real-bank mechanics (criteria 1–4, fresh process)

- **Binding.** `mechanics/result.json` equals `rows/phase.json` and its
  receipt (`0d5bfe32…`). The approval binds the bank hash above. The recorded
  `bank_norms` equal the bank's per-site ‖Δ‖.
- **1. Zero parity.** On my CPU reload, ordinary and b=0 give identical
  generated IDs on both probes, and `torch.equal` holds on the logits.
- **2. Cleanup.** `torch.equal(ordinary, post-cleanup)` holds. The runner's
  hook-signature assertions passed: the phase completed, and no failure
  record exists.
- **3. Nonzero and finite.**
  - The maximum |Δlogit| at b=1 is 1.25, which I recomputed. The final-norm
    hidden state differs, which is asserted.
  - All logits are finite, and all per-site ‖Δ‖ are greater than 0.
  - At b=1 the tokens diverge from ordinary on C01 at token 18. C02's 32
    tokens are unchanged. That is not required: the criterion is the logit
    change, and the C01 logits differ.
- **4. Coverage.** Every b=0 and b=1 call at the 6 sites (23–43) records
  `{prefill:1, decode:31, errors:0}`. The ordinary calls have no audit, as
  expected.
- **Budget.** 6 generations and 4 logit forwards. The session took 51.1 s of
  wall time, and peak allocated memory was 18.0 GB.
- **Information only.** The ordinary logits here are bit-identical to the
  ordinary logits from the synthetic smoke, which ran in a different process.
  Cross-process parity was never required.

## Notes for interpretation (no action)

- **The real Δ is smaller than the synthetic one.** The real ‖Δ‖ is
  0.85–1.36, against 1.23–4.94 for the synthetic Δ. Measured against the
  synthetic per-site attention-output norms, about 88–156, that is roughly
  0.6–1.5% before the gate, with a factor of at most 2b. The b=1 probe
  effect is correspondingly modest: max |Δlogit| 1.25, and one of the two
  probes is unchanged at 32 tokens. Strengths {1,2} are registered and must
  not change. A development null at these strengths is a valid, reportable
  outcome. Per review 07, no realized-energy value is inferred.
- **Ledger.** Completed GPU sessions total 2.62 h (smoke, elicitation,
  extraction and mechanics), within the 8 h stop. Development needs at most
  20,480 generated tokens, about 1 h at the measured 5.6 tok/s, plus the
  model load.

## Signoff

The bank and real-bank mechanics are accepted. Development may be approved,
bound to:
- the mechanics result sha256;
- the R0 manifest;
- the C6 registration;
- the review 07 addendum hash.
