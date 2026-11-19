# Qwen3.8-27B sarcasm follow-up: review 04 (registration and mechanics)

Advisor review, 2026-11-19, of `outputs/qwen38-sarcasm-followup-01`, read-only.
No model or judge call was made by the advisor. Frozen files were not edited.

## Registration: verified

- **Registration file.** `results/protocols/qwen38-sarcasm-followup-01.json`
  has sha256 `3b10bec5…c936`. The manifest has `87d1d0e6…8bf3`.
- **Bound files.** The registration binds the 8 files and hashes signed in
  review 03. The manifest records the same values.
- **Re-hashing.** All 241 bound source and external files, and all 36 frozen
  copies inside the run, re-hash to their manifest values.
- **Bank copy.** It equals the old bank (`4fd9f86b…`).
- **Other bindings.** The registration also cites:
  - the GPU0 UUID, the launcher hash, the algorithm string and the limits;
  - reviews 01–03, which match the current files;
  - the old outcome, `development_null`, from `00b155e6…`.
- **Judge audit: passed.** The snapshot copy in the run has the right hash
  and version. The argument list matches the historical isolated flags. The
  answer was exactly `{"project":"UNKNOWN"}`, and `modelUsage` is only
  `claude-opus-5-5`.

## Mechanics: PASS against plan §5

- **Binding.** `mechanics/result.json` equals the journal row value. The
  receipt matches, and the binding is the registered manifest. There are no
  failure records.
- **1. Zero parity.** On both probes, the b=0 tokens equal the ordinary
  tokens. The logit `torch.equal` assertion passed, since the phase
  completed.
- **2. Recorder equals frozen.** At b=2 the tokens are identical on both
  probes, and the logit `torch.equal` assertion passed.
- **Non-zero.** The b=2 tokens diverge from ordinary at token 6 (C01) and
  token 20 (C02). The ordinary ≠ b2 logit assertion passed.
- **3. Cleanup.** The post-cleanup logits are `torch.equal` to ordinary, and
  the hook signatures are restored.
- **4. Coverage.** Every b0, frozen b2 and recording b2 call shows
  `{prefill:1, decode:31, errors:0}` at the 6 sites (23–43).
- **5. Recording consistency.**
  - b=0 gives a coefficient, requested norm and actual norm of exactly 0 at
    every site and phase.
  - At b=2, the coefficient lies in [0.014, 3.97], inside [0, 4].
  - The manual recompute of requested and actual norms from the retained
    tensors passed at rtol 1e-5.
  - Requested, actual and algebraic agree to about 3 decimal places in the
    summaries. They are not forced equal.
- **6. Capture neutrality.** The C01_P_S logits with full capture equal a
  plain forward. Decomposition at the 6 sites:
  - gated error 0.0071–0.0102 (limit 0.02);
  - omitted-gate error 3.67–6.96, at least 428× the gated error;
  - split error 0.0036–0.0047 (limit 0.01).
- **7. Bank.** The bank is finite, and its hash is bound. At load, the
  random16 tensors' per-site ‖Δ‖ match the bank's and their K_pos and D̄ are
  identical.
- **Counters at this point.** Generation: mechanics 8. Teacher-forced:
  mechanics 2, plus Part A in progress. Logit: 5. Judge: audit 1. All are
  within the caps.

## Descriptive telemetry, mechanics probes only (not a result)

| Site, phase (C01 / C02) | Native per-token norm | Actual shift / native, mean (max) at b=2 |
| --- | --- | --- |
| 23 prefill | 9.4 / 9.0 | 0.34 (0.65) / 0.37 (0.61) |
| 23 decode | 9.2 / 8.7 | 0.26 (0.52) / 0.28 (0.72) |
| 43 prefill | 13.2 / 12.9 | 0.23 (0.57) / 0.25 (0.56) |
| 43 decode | 15.2 / 16.5 | 0.24 (0.34) / 0.14 (0.34) |

These are two short probes at cap 32. Two observations follow from them:
1. **The norm-unit correction is confirmed.** The native per-token norms here
   are about 9–17. The old whole-sequence Frobenius values were 88–156 over
   89 tokens, which gives about 9.3–16.5 after dividing by √89. They are
   consistent.
2. **At b=2 the actual applied shift is not small per token.** It is about
   14–37% of the native attention-output norm on average, reaching about 72%
   on some tokens. This contradicts the earlier, now-withdrawn "small Δ"
   reading. It is still not a causal account of the old null. The old study
   used b ∈ {1, 2} with the same bank. Part A's §A2 and Part B's full
   telemetry are the registered measurements. No conclusion is drawn here.

## Next

Part A is running; 38/192 teacher-forced passes are reserved so far. I will
review Part A, Part B, judging and the closeout when they complete.
