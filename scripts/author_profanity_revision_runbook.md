# One controlled paper-style profanity dataset revision

New paths only. Original `outputs/author-profanity-method-01` remains
extraction_null, with original users/questions/source/results unchanged.
This is the ONE user-approved revision, not a retry/relabel or method repair.
Worker implements/tests only; root owns all reviews, preparation, registrations
and target/native releases. No worker calls, installs or downloads.

## Exactly what changes

`author_profanity_revision.py` is a fresh copy of the frozen original driver so
its fixed path guards remain intact. Only profile paths, the answer-only data
factory/merge, parent-input bindings and revision provenance differ. The sole
additional operational difference is the artifact one-attempt restriction in
`native`/`read_native`; secondary grading keeps its original two attempts. CPU AST
checks enforce identical extraction/mechanics/generation/local-judge/native
journal/calibration/selection-lock/heldout method bodies. Pure algorithm
functions are imported by identity from the frozen original common module.

Same model/revision, environment, raw vector, assistant-content masks/EOT
exclusion, FP32 pooling, folds8401, control8402, band7..23, tie rule, >=.85 gate,
coefficient set, cap/batch/seed, all-position hook, missing-value rules, bigram
quirk, selected/random controls and secondary readout. No pair centering, sign
flip, threshold/seed adjustment or response-driven expansion.

Change only contrastive answers: twelve short1–2 sentence present/absent pairs,
with clear non-slur swearing throughout the present response, plain absent
response, same facts/helpfulness and no swearing directed at a person. No
target swear-word count. Form/intensity/semantic appropriateness are inspected
by root/Claude, not a brittle automatic sentence-count gate. Stronger affect
is bundled with profanity and limits off-target toxicity interpretation.
The generation request itself neutrally asks to keep content, warmth and
helpfulness the same apart from swearing. Intensity/affect confounding is a
review/report limitation, never an instruction to generate negativity.

## Immutable reuse and generator contract

The new request contains only the twelve exact old `{id,user}` fields, the
paper's two definitions, fixed lexicon and answer specification. It sends no
old answers, activations, AUCs, calibration or report questions. Return ONLY:

```json
{"triples":[{"id":"T01","present":"...","absent":"..."}]}
```

Exactly T01..T12 in order, no user or question fields. First-valid native data
stay answer-only. Merge joins users by exact ID and copies both32-question
sets unchanged. Separate raw answer, reused-user UTF8, source question-file,
preview, promoted dataset and native receipt hashes are retained. Child
calibration/report JSON bytes must equal the original files exactly. Original
accepted data registration/promotion hashes, closed status and unchanged
original source are verified; old feature/vector tensors are never read.
There must be no original mechanics/validity/calibration/report model phase.

## Commands and locks (root only)

Fixed run `outputs/author-profanity-revision-01`; public result
`results/author_profanity_revision_01.json`; new promoted directory
`evaluation/author_profanity_revision`. Use the same isolated GCAD Python3.11
for CPU preparation/preview and GPU stages, original `.venv` for native/CPU
reporting. Full method/lock details remain in the unchanged original
`scripts/author_profanity_method_runbook.md`; substitute revision paths below.

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_revision prepare
.venv/bin/python -B -m scripts.author_profanity_revision verify
```

Root first commits source acceptance, new protocol/design review and initial
`results/protocols/author-profanity-revision-01.json` with manifest_sha256.
Existing pinned native2.1.323 PATH/audit/model/flags remain unchanged.

Artifact generation has ONE attempt only. Any completed non-JSON, invalid-schema
or refusal response closes artifact_null immediately, preserving attempt0 and
its start/receipt. No artifact attempt1 is accepted or invoked, including on
resume. There is no automatic repair/reroll or retry registry. Identity/auth/
infrastructure/uncertain-start failures remain fatal rather than new attempts.
Secondary scoring alone retains two identical structural attempts per batch.

```sh
.venv/bin/python -B -m scripts.author_profanity_revision artifact --execute-native
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_revision preview
```

Root/Claude inspect the complete merged candidate, all renderings, literal
definitions and original self-judge envelope. Write the new semantic review
`results/reviews/author_profanity_revision_artifacts.json` binding accepted,
preview_sha256 and full datasets_sha256. Semantic failure =>artifact_null,
no hand edits or reroll. Otherwise root promotes/commits exact bytes:

```sh
.venv/bin/python -B -m scripts.author_profanity_revision promote
```

All stage registries have new prefix `author-profanity-revision`:

- initial: `...-01.json` (manifest_sha256)
- artifacts: `...-artifacts-01.json` (manifest_sha256, promotion_sha256,
  semantic_review_sha256, datasets_sha256)
- extraction: `...-extraction-01.json` (manifest_sha256, run_sha256,
  integrity_sha256, decodability_sha256, vectors_sha256)
- mechanics: `...-mechanics-01.json` (manifest_sha256, run_sha256,
  integrity_sha256, result_sha256, vectors_sha256)
- selection: `...-selection-01.json` (manifest_sha256, selection_sha256,
  calibration_run_sha256, local_judge_run_sha256, vectors_sha256)

Each GPU/native stage needs separate root release; no calls follow merely
from implementing the command. Ordered stage names/flags are unchanged:

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_revision extract --execute-gpu
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_revision mechanics --execute-gpu
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_revision validity --execute-gpu
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_revision calibration --execute-gpu
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_revision calibration_judge --execute-gpu
.venv/bin/python -B -m scripts.author_profanity_revision select
```

Only after passing selection is committed:

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_revision final --execute-gpu
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_revision final_judge --execute-gpu
.venv/bin/python -B -m scripts.author_profanity_revision secondary --execute-native
.venv/bin/python -B -m scripts.author_profanity_revision report
```

Artifact/extraction null closes this paper-method profanity route; no third
dataset, extra folds, rescue layer/coefficient or automatic new study. The
same calibration-null/fatal behavior, immutable first-valid records and
precondition/resource waits apply. Old null is never changed into a pass.

Maximum fresh394 target/408 local judge generations,24 TF,2 logit,0MC,
21 first-valid native batches; registered ceiling42 starts, effective maximum41
with the one-artifact-attempt restriction (1+20x2). Parent work is historical and not
claimed fresh. Public report adds aggregate reuse/answer-only revision
provenance and the intensity/affect/fixed-topic/row-fold limitations; no raw
requests, answers, reviewer reasons or token arrays.

Focused CPU checks (no live services/cache/bytecode):

```sh
.venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_author_profanity_revision.py
.venv/bin/python -B -m scripts.author_profanity_revision --help
```
