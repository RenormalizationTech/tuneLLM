# Existing-direction behavioral test

New child only: `outputs/author-profanity-direction-test-01`, configuration
`configs/author_profanity_direction_test.json`, protocol
`docs/author_profanity_direction_test_protocol.md`. Root owns preparation,
registrations and every live release. No worker preparation or live calls.

Both parent extraction_null results remain unchanged. Paired-question AUROC1.0
at hidden7/block6 was already known. CPU reuse reproduces it using the frozen
paired folds, verifies all raw means and copies the original tensor file byte
for byte. No new extraction evidence, teacher forcing, artifact request,
centering, normalization, sign flip or data revision. Grouped label shuffle
uses the same paired folds and is report-only.

## Preparation and releases

Use the isolated GCAD Python for prepare and GPU phases, original `.venv` for
verify, selection, secondary grading and report. Commands below are for root
after review; no installation or download is required.

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_direction_test prepare
.venv/bin/python -B -m scripts.author_profanity_direction_test verify
```

Initial registration `results/protocols/author-profanity-direction-test-01.json`
must contain exactly `manifest_sha256`, `parent_vectors_sha256` (same hash as
`reuse/vectors.pt`) and `grouped_reuse_sha256` (`reuse/grouped.json`). Commit
registration, sources, protocol and design acceptance before execution.

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_direction_test mechanics --execute-gpu
```

This performs ten cap16 probes and two explicit logit forwards with the same
frozen scientific mechanics body, changing only the direction authority to
the child's reused bank. All raw all-position hook settings are unchanged.
Root commits `results/protocols/author-profanity-direction-test-mechanics-01.json`
with exactly `manifest_sha256`, `run_sha256`, `integrity_sha256`,
`result_sha256`, `vectors_sha256` (all mechanics files except reused vectors).

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_direction_test validity --execute-gpu
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_direction_test calibration --execute-gpu
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_direction_test calibration_judge --execute-gpu
.venv/bin/python -B -m scripts.author_profanity_direction_test select
```

Selection is the unchanged distinct-bigram>=0.6, self-effect>=1 rule, closest
effect to2 then lower coefficient. Threshold equality is a deliberate boundary
choice. Lexicon counts never select. A calibration null closes a completed
behavioral test: report baseline and all six strengths, semantic scores,
lexical response counts/rates, repetition, caps and empties. No rescue follows.

If passing, root commits selection registration with exactly
`manifest_sha256`, `selection_sha256`, `calibration_run_sha256`,
`local_judge_run_sha256`, `vectors_sha256` before held-out generation:

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_direction_test final --execute-gpu
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_direction_test final_judge --execute-gpu
.venv/bin/python -B -m scripts.author_profanity_direction_test report
```

If secondary authentication is pending, this writes only private
`primary_report.json`, status `primary_complete_secondary_pending`, including
selected-minus-baseline and selected-minus-three-random semantic effects. No
public incomplete placeholder consumes the final result path. After login is
restored and root releases judging, prepend the registered native2.1.323 PATH
pin to this process only, retaining the same binary/model/flags/audit:

```sh
.venv/bin/python -B -m scripts.author_profanity_direction_test secondary --execute-native
.venv/bin/python -B -m scripts.author_profanity_direction_test report
```

`--resume` is only for an existing secondary stream's exact journal replay;
it never regrades valid batches or retries infrastructure/model/auth errors.
Twenty eight-item batches, maximum two identical structural attempts each,
forty starts total. Uncertain starts permanently stop. No artifact endpoint.

## Accounting and interpretation

Fresh maximum394 targets (10 probes+224 calibration+160 held-out),408 local
judgments (24 validity+224 calibration+160 held-out),2 logits,0 TF and0 MC.
Reused parent24 TF and artifact work are separate. Saved counters describe
completed work; interrupted starts remain separately journaled and may have
incurred additional work. Generation cannot resume/rerun an existing phase.
Resource waits start no work; fatal execution or integrity failures stop.

Current source, immutable inputs and model hashes are merged/deduplicated at
each live command entry/exit, not recursively expanded through old studies.
Only the new root is writable. GPU0 UUID/free-memory/utilization guards and
offline Qwen7 BF16/SDPA/batch8/greedy42/cap48 settings are unchanged. Local
24-behavior judges remain unsteered and cap1024. Reconstructed-data and
grouping uncertainty, fixed topics, affect/intensity confounds, self-judging,
mean filling and single secondary evaluator remain limitations. The fixed
11-word whole-word lexicon is a narrow context-dependent readout, not all
profanity or a new gate. No automatic new study, joint control or transfer.
