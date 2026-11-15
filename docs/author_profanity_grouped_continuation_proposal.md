# Proposed profanity continuation: hold out whole question pairs

The user subsequently authorized applying the direction and finishing the test.
Execution is governed by [the direction-test protocol](author_profanity_direction_test_protocol.md).
This proposal is retained to explain the validation change. Both prior
row-validation extraction_null results remain unchanged. No third dataset and
no newly generated contrastive answers are proposed.

The approved short-answer revision reached primary row-level AUROC0.36111
and the already registered paired-question sensitivity reached1.0 at hidden
index7. The manuscript specifies stratified five-fold projection scoring but
not its grouping unit; Step1 says to score on pairs not fit on. Author code
remains unavailable. Row grouping was our reconstruction choice. Do not blame
the paper for a verified implementation error or call its grouping known.

A new child study would hold both responses to each question in one fold,
using the existing frozen paired-fold assignment. Keep the accepted revision
triples and all32 calibration/32 report questions byte-identical. Reuse the
saved feature tensors, model/runtime identity and raw full-data direction;
no new teacher forcing is necessary. Use the same band7–23, earliest tie,
AUROC>=0.85 and raw steering math. A grouped label-shuffle control remains
report-only. No pair centering, sign flip or coefficient expansion.

The grouped gate's result is ALREADY KNOWN:1.0 at hidden index7/block6, the
same site selected by the row primary, with the exact same raw direction.
This continuation gains no new extraction evidence or selection leverage.
It proceeds from an exploratory validation clarification; the actual new
observations would be generated behavior in calibration and untouched held-out
questions. Never retroactively relabel either closed study as successful.

Before calls, freeze a new run/configuration/source/protocol acceptance and
hash-bound reuse registration. A CPU reuse stage must verify parent features,
vector and input bytes and reproduce the grouped calculation. Then perform
real mechanics, local self-judge validity, calibration/selection lock and the
original five-arm held-out report with three matched-norm random controls.
Original Qwen7/BF16, seeds, all-position hook, coefficients1/2/3/4/6/8, cap48,
self-judge definitions/parser/fill rules and repetition metric are unchanged.
No automatic rescue follows a mechanics/calibration null or failure.

Fresh maximum:394 target generations including10 mechanics probes,408 local
judge generations, two logit forwards, zero teacher-forced responses, zero MC.
There is no artifact request. Secondary Claude uses20 first-valid batches,
at most40 starts. Historical artifact/TF work is reported as reused, not fresh.

The interpretation must retain reconstructed-data, fixed-topic, affect/intensity,
missing-author-grouping and single-family secondary-evaluator limitations.
Any held-out effect is read from saved outputs and compared with baseline and
random controls; no outcome is promised. Joint steering and larger-model
transfer remain outside this study.
