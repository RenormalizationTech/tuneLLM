# Profanity: one paper-style dataset revision

The user selected one dataset revision after discussion with Claude. Preserve
all original `author-profanity-method-01` inputs, code and extraction-null
result. This is a new study, not a retry or relabelling of the old run.

Reuse all12 original shared user requests and the exact32 calibration/32
report questions by hash; none of the latter has been model-tested. Change
the contrastive-answer specification only:12 benign shared-user triples,
with short1–2 sentence present and absent answers. The present answer expresses
clear non-slur swearing throughout; the absent answer is plain. Preserve facts,
helpfulness and the shared user request. No swearing is directed at a person;
user prompts do not ask for swearing. Do not set a target swear-word count or
tune examples to saved activations. The paper's actual triples remain missing.

This follows the short, distinctly styled whole-response contrasts illustrated
by the paper more closely than the previous single-word minimal edits. It also
bundles stronger intensity and affect with swearing; the twelve topics remain
one fixed sample and row-fold partner leakage remains. Off-target toxicity must
be interpreted with that limitation. Passing direction validation is not
guaranteed. Do not claim an exact artifact or numerical reproduction.

Keep the existing documented-method protocol's model/revision, runtime,
assistant-content pooling, raw direction, seeds, folds, layer band, tie rules,
AUROC>=0.85 gate, missing-score handling, local self-judge, coefficients,
calibration selection, mechanics, held-out controls and secondary evaluation
unchanged. Calibration and report questions are copied unchanged from the first study.
The native artifact response contains only12 ordered IDs and present/absent
answers, with no user field: `{"triples":[{"id":"T01","present":"...","absent":"..."}, ...]}`.
Merge original requests by ID and original question sets deterministically,
retaining separate hashes for generated answers and reused inputs. No
generator paraphrase of frozen questions is accepted. Semantic review checks
short1–2 sentence form; no brittle sentence-count gate is added. No pair-centering,
changed folds, sign flip, relaxed threshold or expanded coefficient range is
permitted to pass the gate.

Use the same one-first-valid artifact process (at most two identical structural
attempts), root/Claude semantic review, exact data promotion and new committed
stage registrations. Each accepted candidate is used unchanged or closed;
no semantic editing/reroll. Use fresh explicit paths/config/source snapshots
for this study. Model weights and isolated environment may be reused by hash.
Keep all historical results unchanged.

The same hard maximum is394 target generations,408 local judge generations,
24 teacher-forced responses, two logit forwards, zero MC and42 native starts.
If this revision also closes at extraction, close this paper-method profanity
route on this model without another automatic dataset revision. Any further
method change requires a new discussion. There is no success expectation or
response-driven expansion. Sarcasm remains an independent reviewed study.

Implementation and exact new path bindings still require review before live
calls. This document records the approved scientific revision, not a stage
release.
