# Single profanity answer revision: extraction null

The approved revision closed as extraction_null before any steering or
calibration. Shorter, more clearly profane responses improved the signal,
but did not meet the registered row-level primary gate. The same accepted
data achieved perfect separation in the predeclared paired-question
sensitivity at the selected site. These are different validation designs;
only the registered primary governed this run.

| Diagnostic at hidden index7 / decoder block6 | AUROC |
| --- | ---: |
| Primary stratified row-fold projection | 0.36111 |
| Paired-question fold sensitivity | 1.00000 |
| Label-shuffled row-fold control | 0.37500 |

The0.85 primary threshold failed. No mechanics, target generation, coefficient
selection, local judging or held-out behavioral test occurred. Do not describe
this as a measured failure of profanity steering.

## What changed and what stayed fixed

The sole native artifact request generated24 short present/absent answers on
the same12 user requests. All32 calibration and32 report questions were copied
byte-identically. Root and Claude accepted the new answers unchanged before
teacher forcing, with intensity/affect and occasional awkward phrasing noted.
All model, pooling, raw-vector, folds, seeds, layer band and gate choices stayed
unchanged. First-study data, code and results remain intact.

Paired-question AUROC reached1.0 at layers7–10; it was at least0.85 at16 of17
searched layers. The primary still split opposite-label, closely worded
partners across training/testing. This motivates a validation discussion,
not a retroactive pass or a new steering claim. The same raw direction at
layer7 would be selected by either primary or paired sensitivity.

The paper specifies stratified five-fold projection AUROC without the grouping
unit; its main text says to score on pairs not used for fitting. Author code
is unavailable. Our row-level split was a reconstruction choice, not a known
literal author implementation. Earlier wording calling row folds paper-literal
was overconfident and has been corrected in the mutable closeout documentation.
The frozen protocols and registered nulls remain unchanged.

## Evidence and continuation boundary

Compute:24 teacher-forced responses, zero target/local-judge generations,
zero logit forwards and one artifact start, without a retry. Raw features,
encodings and journals remain in ignored
`outputs/author-profanity-revision-01/`. See the
[aggregate result](../results/author_profanity_revision_01.json),
[artifact review](../results/reviews/author_profanity_revision_artifacts.json)
and [frozen revision protocol](author_profanity_revision_protocol.md).
Claude independently checked the saved calculations and grouping mechanism;
the direction-validation result is accepted as registered.

No third dataset or automatic rescue is released. A
[proposed grouped-validation continuation](author_profanity_grouped_continuation_proposal.md)
would keep the exact accepted data and direction and test actual generated
behavior, with the already observed grouped gate acknowledged as a formality.
It requires the user's choice and a new registration. Both row-level nulls
stay preserved. Original-Mistral sarcasm continues independently.
