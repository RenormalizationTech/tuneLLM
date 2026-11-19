# Correction: activation-norm units in the 27B sarcasm interpretation

Recorded by the project manager on 2026-11-19 during preparation of the
owner-authorized activation/dose follow-up. This is a code-reading correction;
no new model or judge call was made. Original sources, reviews, results and
acceptance decisions remain frozen.

In `scripts/qwen38_gcad_port.py`, `extract()` records
`attention_output_norm = native_output.float().norm()`. That is the Frobenius
norm over the full batch/sequence/hidden tensor. It is the appropriate
denominator for the similarly aggregated reconstruction-error diagnostics.
It is **not** a mean per-token attention-output norm.

The bank's delta norm, 0.85–1.36, is the norm of a single hidden-dimension
vector applied at each position. Comparing it directly with the synthetic
whole-sequence norm (about 88–156) does not yield a per-token relative shift.
The synthetic 1.0–3.2% interpretation in review04, the 0.6–1.5% scaling
interpretation in review09, and the related small-dose interpretation repeated
in review10 therefore lack a consistent normalization.
It must not be used as evidence that the original strengths were insufficient.

The bank arithmetic, reconstruction relative errors, exact parity, measured
logit change, observed text changes and development-null decision are unaffected.
No realized intervention energy was recorded in that study, as its reporting
addendum already states.

The follow-up compares matched per-token native norms and the actual FP32
difference between the model-dtype steered and native outputs. It also reports
the norm of a pooled mean separately. An insufficient dose remains a hypothesis
to test, alongside a direction/site mismatch; neither cause has been established.
