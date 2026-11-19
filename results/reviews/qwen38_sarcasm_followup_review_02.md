# Qwen3.8-27B sarcasm follow-up: code review 02 (analyzer, current tests)

Advisor review, 2026-11-19, of:
- `scripts/qwen38_sarcasm_followup_analyze.py` (233 lines, 11:24:44);
- `scripts/qwen38_sarcasm_followup.py` (11:25:29);
- `tests/test_qwen38_sarcasm_followup.py` (11:25:29).

The review was read-only, plus CPU tests. No model or judge call was made.
The current tests give **12 passed**. Review 01's B1 (missing file) is stale:
the analyzer now exists.

## Analyzer: verified correct

**Question-cluster bootstrap.**
- `weights()` draws multinomial question weights with a fixed seed. The same
  10,000 × 32 weight matrix is used for every contrast, content and context,
  so all 6 cells of a question move together.
- `cosine_ci` computes the bootstrap cosine of the weighted mean through the
  Gram matrix. This is exact: ‖wX‖² = w G wᵀ and (wX)·d = w(X d). A test
  matches it against the direct computation.
- Tokens and trajectories are never resampled.

**A1.** Reports the cosine of the mean with a CI, the 3 signed
random-direction cosines, an expectation of 0 and the 1/√5120 SD scale.
`reference_note` states that these are neither a null interval nor a binary
test.

**A2.**
- (a) `paired_token_effect` computes per-token ‖S − L‖ in FP32, with the L
  per-token norm as the denominator and zero-norm tokens excluded and
  counted. It averages within each question, then across questions with a
  cluster CI.
- (b) Reports the norm of the pooled mean and an explicit mean-L denominator.
  The per-question norm of the mean is kept in `paired_per_question`.
- It never uses the old Frobenius denominator, and it says so.

**A3–A5.**
- A3: residual norms of the mean, with explicit L or S denominators and
  leave-one-out consistency.
- A4: per-question S − L gate means, with a cluster CI.
- A5: paired NLL S − L, with a cluster CI.

**Part B.**
- Gates go through the unchanged `sarcasm_method_transfer_common.summarize`
  and `gates(baseline, b)` for each b.
- The outcome is `exploratory_candidate` (lowest passing b) or
  `exploratory_null_through_b16`.
- Also reported: `b16_random_margin20`, the random arm's descriptive gates,
  the prompted validity flag, the coherent-sarcastic counts, the within-arm
  and pooled Pearson correlations with a caveat, and token-weighted
  telemetry. Telemetry gives actual, requested and algebraic values, each
  defined, with prefill and decode separate.

**Status mapping.** Launcher statuses pass through:
- `implementation_stop`, `budget_stop` and `judge_audit_failure` pass through
  unchanged.
- When trait and coherence are complete and impoliteness is missing, the
  outcome is `primary_complete_descriptive_incomplete`.
- Part A is reported independently of Part B.

## Defects

**A1-1 (blocking; confirms the PM's finding). Crop keys break on N.**
- `keys = list(cells['C01_B_S'])` includes `crop.*`.
- The loop then calls `vectors(key, 'N', …)`, which raises `KeyError`
  because N cells have no crop.
- *Fix:* for `crop.*` keys, build only the S and L vectors, and skip the N
  supplemental contrasts and the N-based system-presence contrast. N
  contrasts apply only to `block.*` and `attention.*`.

**A1-2 (blocking; confirms the PM's finding). `model_geometry`.** It is now
written by `build_inputs`. That line landed, but the end-to-end test below is
still required.

**A1-3 (blocking). There is no end-to-end schema test.** Write a CPU test
that builds a fake run directory with exactly the files and keys the
launcher writes:
- `config.json`, `manifest.json`, `inputs.json` (with `model_geometry`),
  `bank.safetensors` and the `random-*.safetensors` files;
- 192 diagnostic rows and tensors, using tiny dims but real key names
  (`block.*`, `attention.*`, `crop.*`, `attention_tokens.*` for S/L only);
- 224 generation rows from `generate()`'s real return schema, including
  `telemetry`;
- the judge streams;
- the counters and sessions.

Run `analyze()` for these cases:
1. complete;
2. impoliteness missing, giving `primary_complete_descriptive_incomplete`;
3. Part B missing, giving `implementation_stop` with Part A complete;
4. judge-audit failure.

Assert the outcome labels and a few numeric fields against hand-computed
values. This is the test that would have caught A1-1.

**A1-4 (fix). `mostly_not_letters` is redefined.** The analyzer uses
`letters < 0.5·len(output)`, which counts spaces. The frozen
`text_checks(output)['mostly_not_letters']` counts visible non-space
characters, and plan §4 refers to that definition. `text_checks` is already
imported. Use it.

**A1-5 (fix, small). The A1 wording label is missing.** Add
`ci_wording ∈ {positively_aligned, anti_aligned, no_clear_alignment}`,
computed from the bootstrap CI against 0, exactly as in plan §3. Then the
closeout cannot choose its wording after seeing the numbers.

**A1-6 (fix, small). A2(b) needs a CI.** Plan §A2(b) asks for a
question-cluster CI on the norm of the pooled mean. Compute
`sqrt(w G wᵀ)` from the per-question mean-difference vectors, which uses the
same Gram trick, and report its 95% interval.

**A1-7 (fix). Limitations are incomplete.** Add these items:
- the random control exists only at b = 16, so lower-strength passes are not
  direction-tested;
- S − L differs in the whole fixed wording and cannot isolate pure trait
  specificity;
- the norm-unit correction: no closed-study per-token ratios are used;
- the fixed-vs-free continuation conditioning difference between A2 and the
  Part B actual ratio;
- dose is never concluded as the cause of the old null.

## Review 01 items: current state

| Item | State at 11:25 |
| --- | --- |
| B1 analyzer file | present (this review) |
| B2 manual recompute of logged requested/actual in mechanics | **not yet**: no `retain_last` or recompute check |
| B3 assert N/S prompts equal the old baseline/prompted `input_ids` | **not yet** |
| B4 tests | partly addressed: diagnostic neutrality and crop parity, Gram bootstrap, 7-arm plan packing and frozen gates, journal resume and Uncertain guard and stage budget. Still missing: b=16 parity, the B2 recompute test, `build_inputs` failure paths, the audit-failure path, and the A1-3 end-to-end test |
| R1 ordinary ≠ b2 logits | not yet |
| R2 random16 norm and K/D̄ checks at load | not yet |

## Signoff path

When A1-1 to A1-7 are fixed and B2, B3, B4 and R1/R2 are in, I will review
once more for the **single combined source signoff**. It will be bound to
the hashes of the launcher, telemetry, analyzer, config, tests and plan, and
it is the last step before `prepare`.
