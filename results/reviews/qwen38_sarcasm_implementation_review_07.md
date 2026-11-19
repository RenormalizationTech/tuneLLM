# Qwen3.8-27B sarcasm GCAD: review 07 (unmeasured descriptives, addendum)

Advisor review, 2026-11-18, in answer to the PM's prospective reporting check.
The review was read-only and covered source only: the frozen plan §2/§6.4/§7,
`scripts/qwen38_gcad_port.py`, `scripts/qwen38_sarcasm.py` and
`scripts/qwen38_sarcasm_analyze.py`. No study outputs, scores, model or judge
calls were involved. Nothing frozen was edited.

## Finding: both descriptives are absent from the frozen design's data

**Realized shift norms (plan §6.4 and §7): not recorded.**
- `Steerer.shift` computes the per-token coefficient
  `coef = 2b·σ(1.5(d − D̄))` and adds `coef·Δ`. It discards `coef`. The only
  audit fields are `{prefill, decode, errors}`.
- `generate()` saves only that audit dictionary. The mechanics phases save
  `max_logit_change`, the raw `bank_norms` (‖Δ‖ per site) and the logits.
- No stage writes per-token coefficients, demands or shift magnitudes. The
  analyzer does not compute them either.
- The realized shift energy can therefore not be reconstructed from saved
  data. It depends on the per-token decode-time queries, which are not
  stored.

**Cache/no-cache agreement (plan §2 and §7): no new measurement.**
- Every arm and every mechanics forward uses `use_cache=True`.
- The mechanics phases contain no steered full-prefix (no-cache) run.
- The `cache_disclosure` field in the synthetic smoke result is the config
  string quoting the *historical, unsteered* NF4 smoke (max 0.125). It is not
  a measurement made in this study.

**Correction to review 04.** Review 04 listed "cache disclosure" under
report-only items without saying it was the historical figure. Design review
F4 listed cache/no-cache agreement as report-only for the synthetic phase, and
that was never measured. This does not affect any F4 gate criterion or the
signoff, but it should be read as stated here.

## What remains reported (unchanged)

- Raw ‖Δ‖ per site for the real bank, from `extraction/bank.safetensors` and
  the mechanics `bank_norms`.
- Random-control raw ‖Δ‖ equal to the selected bank's per site. This holds by
  construction and can be verified from the stored random tensors and hashes.
- Mechanics `max_logit_change` for b=1 on the probes.
- The historical NF4 cached-vs-full-prefix failure (max 0.125), together with
  the registered mitigation: every compared arm uses the identical cached
  `generate` path at batch 1.

## Proposed protocol addendum (PM to register before development)

Suggested file: `results/protocols/qwen38-sarcasm-descriptive-addendum-01.json`.
Bind it with the R0 manifest hash and the C6 registration hash. Suggested
content:

> **Descriptive omissions, registered before any development generation.**
> Two plan §7 descriptives were not instrumented in the frozen R0 source and
> are declared **unmeasured** for this study:
> 1. **Realized shift norms / realized shift energy.** The frozen `Steerer`
>    does not record the per-token gate coefficients. No realized shift norm
>    or energy is reported for the selected or random arms. No claim of
>    energy matching between selected and random is made. Raw per-site ‖Δ‖,
>    and its exact equality between selected and random, remain reported as
>    tensor properties only. **No realized-energy value is inferred from raw
>    norms.** If a bound is shown, it is only the algebraic identity
>    0 < per-token shift ≤ 2b‖Δ‖, labelled as a bound and not as an estimate.
> 2. **New NF4 steered cache/no-cache comparison.** It is not run. No new
>    calls are added. The reported cache evidence is the historical unsteered
>    NF4 smoke (max logit difference 0.125) and the registered mitigation
>    (identical cached `generate`, batch 1, for every compared arm).
>
> Gates, strengths {1,2}, fixtures, arms, decoding, count limits, judge
> budgets and the C6 analyzer are unchanged. This omission is a transparent
> descriptive limitation. It is reported separately from the primary result
> and cannot change any gate, selection or outcome.

## Assessment

- **Scientific impact on the primary result: none.** Neither quantity enters
  a gate, selection or C6 decision.
- **Interpretive cost.** The selected-vs-random comparison controls the raw
  per-site direction norm, not the realized injected energy. This is because
  the demand gate σ(1.5(d − D̄)) depends on the real K_pos, which both arms
  share, and on the queries, which differ once the trajectories diverge. The
  closeout should state this when it interprets the random+20 gate.
- **Timing.** Register the addendum before the development approval, so the
  omission is declared before any steered evaluation data exists.
  Elicitation is unsteered and is unaffected.
- **No change to frozen source is advised.** Instrumenting the steerer now
  would require a fresh R0 and new mechanics. That cost is not justified for
  two descriptives.
