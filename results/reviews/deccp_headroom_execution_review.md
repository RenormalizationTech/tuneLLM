# Final execution review: DECCP headroom baseline

Reviewer: Claude advisor, read-only. I made no model, GPU or native calls, ran
no tests and edited no scientific file.

**Read:**

- `docs/deccp_headroom_protocol.md`;
- `scripts/deccp_headroom_runbook.md`;
- the current `scripts/deccp_headroom.py` and `scripts/deccp_headroom_common.py`;
- `configs/deccp_headroom.json`;
- `tests/test_deccp_headroom.py`.

This follows my implementation review draft.

## Verdict

**Approved for fresh preparation, registration and execution. I found no
blocker.**

**Conditions:**

1. Root's final full run of `tests/test_deccp_headroom.py` passes.
2. All new files are committed before `prepare`, so `registered()` can verify
   them. At present they are untracked: config, protocol, runbook, fixture,
   LICENSE, both reviews, runner, common module and tests.

## Earlier items: resolved

- **R1, empty answers.** The rubric and protocol now call
  empty → `responsive_incomplete` a mechanical bookkeeping exception, and say
  it is not evidence of a substantive response.
  - `decision()` adds empty-output counts for English targets, Chinese targets
    and English controls.
  - It sets `empty_incomplete_bookkeeping_not_substantive: true`.
  - The gate denominators and labels are unchanged.
  - `report()` recomputes the decision and requires `decision.json` to match.
- **R2, item-specific example.** The phrase written around the "claimed blanket
  ban" development item is gone. The general premise-correction rule remains.
- **H1, generation config check.** A check is present. As written, though, it's
  tautological: `effective` is the defaults updated with the same overrides it
  is then compared against. That's not a blocker. The explicit kwargs passed to
  `generate` are what fix greedy decoding, and the effective config is saved for
  every row.

## Execution order (confirmed against code and runbook)

1. **Commit everything.** Run CPU tests only, with no preparation and no calls.
2. **`prepare` (fresh root only).** It compiles all 132 decisions, the 83
   included rows, the 58 development rows, the 25 held-out rows that are never
   generated, and the 20 anchors. It also freezes the renderings and all plans,
   and hashes the source, external files and model files.
3. **Root commits `results/protocols/deccp-headroom-01.json`** containing the
   manifest hash. Then run `verify`.
4. **Native calibration, before the GPU stage.**
   - `packet`, `judge --phase calibration --execute-native`, then `calibrate`.
   - 3 batches (8/8/4).
   - If calibration fails, the run stops permanently with `calibration_null`,
     and the GPU stage never runs.
5. **Root commits the calibration lock** (manifest, `calibration.json` and
   accepted hashes). `baseline()` requires it.
6. **`baseline --execute-gpu`.** It needs the registration and the calibration
   lock.
   - It refuses to run if the folder already exists, and never reruns or
     resumes.
   - It checks model hashes at entry and exit and writes an integrity receipt.
7. **`judge --phase primary`** needs the calibration lock and the baseline
   integrity receipt (8 batches). Then **`decide`**.
8. **`judge --phase duplicates`** needs the primary accepted file (1 batch of 6).
   Then **`report`** writes `results/deccp_headroom_01.json`.

## Budgets (confirmed)

- **Generations: 58.** `one()` writes a start record before each call and
  refuses a 59th. There's no command for held-out, fit or intervention.
- **First-valid batches: 12** (3 + 8 + 1). `prepare` asserts this.
- **Native starts: at most 24, across all streams.** This is checked before
  each new attempt. Each batch gets at most two structural attempts; any other
  failure is a permanent stop.
- **Plan seeds:**
  - calibration 107011, primary 107012, duplicates 107013;
  - the six duplicates are preselected with 107011 before any output;
  - aliases are separate per stream.

## Protocol and report wording (checked)

- **Gate (English only).** Denominators are fixed at 36 targets and 11 controls,
  with exact integer arithmetic.
  - The 12-response threshold is binding (33.3%).
  - Each control moves the control rate by 9.09 points.
  - The tests cover 11 vs 12 targets and 2 vs 3 controls at t = 12.
- **Wording.** A failure reads "did not meet prespecified headroom criteria",
  with no zero or "no measurable" claim. A pass only supports a separately
  registered plan.
- **Chinese.** It is descriptive only, with no Chinese controls. The repetition
  check is N/A and a nonempty answer is not treated as proof of quality. Empty
  answers and cap hits are still counted.
- **Controls.** They are genre comparisons, not causal matching. The selection
  by Qwen2, the Tiananmen dominance and the AI-audited EN/ZH mapping are all
  disclosed.
- **Scope.** There is no factual, coverage or official-framing scoring. The
  public output is counts only, with no raw text, tokens or grading reasons.
- **Earlier studies.** Reuse is disclosed. Exact reuse of earlier development
  or held-out items is rejected, and earlier studies are untouched.
- **No broader work.** The protocol introduces no safety-removal, harmful
  boundary, larger-model or publication work.
