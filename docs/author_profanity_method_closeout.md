# Profanity paper method: extraction stopped before steering

The reconstructed-input study closed as `extraction_null`. Its best registered
out-of-fold projection AUROC was0.13194, below the0.85 direction-validation
threshold. No steering generation, calibration or behavioral scoring occurred.
This outcome does not establish that profanity steering fails on Qwen2.5-7B.

| Registered diagnostic at the selected hidden index20 (block19) | AUROC |
| --- | ---: |
| Primary stratified row-fold, pooled raw projection | 0.13194 |
| Paired-question fold sensitivity | 0.60417 |
| Label-shuffled control | 0.45833 |

Across all17 searched layers, primary AUROC ranged0.06944–0.13194. The highest
paired-question sensitivity was0.72222 at hidden index9; this diagnostic was
never authorized to select a setting or override the primary gate. There is
no validated direction and no selected steering coefficient in this run.

The coordinator independently recomputed every primary layer AUROC from the
saved feature tensors, checked the positive/negative row order and verified the
raw full-data direction equals the mean paired difference exactly. Claude
independently audited the source, features, folds and score arithmetic; no
implementation error was identified.

## Why the validation failed

The12 reconstructed pairs mostly differ by one mild swear-word edit. Their
large shared content dominates their small stylistic difference. Eight pairs
are split between training and validation in the registered stratified row
folds. A held-out response's near-identical opposite-label partner then enters
the training class mean. Claude's saved-feature audit found reversed ordering
for seven of these eight pairs, while all four co-held-out pairs were ordered
correctly. Pooling raw projections from different folds also permits offsets
between folds. This is a design interaction between the reconstructed inputs
and the literal paper validation procedure.

A further pair-centered calculation by Claude separated the saved responses,
but it was post hoc and changes the evaluation. It neither rescues the
registered gate nor demonstrates steering. The manuscript's original triples
and exact code remain unavailable; their differences from ours cannot be
established. Do not infer that their reported success is contradicted.

The artifact review had accepted unchanged facts/helpfulness and disclosed
some additional emphasis/negative affect, a mild-swearing construct and the
shared generator/secondary-grader family. See the
[artifact review](../results/reviews/author_profanity_method_artifacts.json).
No artifacts were rewritten after seeing target features.

## Preserved result and next decision

Compute was24 teacher-forced assistant responses, zero target generations,
zero local judgments, zero logit probes and one first-valid native artifact
request, without a retry. All raw features, encodings, artifacts and journals
remain in ignored `outputs/author-profanity-method-01/`. The
[aggregate result](../results/author_profanity_method_01.json),
[protocol](author_profanity_method_protocol.md) and frozen sources are retained.

A new dataset or validation policy requires a separately reviewed study and
new registration. No automatic rescue, coefficient sweep, changed folds or
unvalidated steering is released. Original-Mistral sarcasm work continues
independently under its already reviewed plan.
