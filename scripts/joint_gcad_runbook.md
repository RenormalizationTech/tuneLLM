# Joint GCAD fixed study operator runbook

Read `docs/joint_gcad_protocol.md`, active config, exact fixture and finalized root
reviews first. Baseline has no hooks. Active traits share one native query;
their raw displacements sum in canonical trait-name order and enter attention
output once. No weights, banks or frozen runtime files are edited. Singles omit
zero banks. Hook device scalars are exported only after generation; every site
and prefill/decode call is counted, with finite/coefficient/norm diagnostics.

## CPU source verification

No installs or target forwards are needed. The immutable target environment has
no pytest; append the existing pure pytest path late:

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -c \
"import sys; sys.path.append('.venv/lib/python3.10/site-packages'); import pytest; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','tests/test_joint_gcad.py','tests/test_joint_gcad_study.py']))"
```

Tiny random CPU models and existing small banks only. No base-weight load,
GPU, native calls, network or run preparation is part of this test command.

## Freeze, prepare, initial release

Root finalizes/commits the protocol, design/fixture reviews, exact fixture,
config and sources before preparing in the GPU Python (CPU-only command):

```bash
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.joint_gcad_study prepare
```

Fresh root only: `outputs/joint-gcad-01`. Prepare freezes all rendered prompts,
anchors/rubrics, source/model metadata and all plans before outputs; no model
forward or native invocation. Root then commits
`results/protocols/joint-gcad-01.json` containing exactly
`{"manifest_sha256":"<SHA256 of run/manifest.json>"}`. Only after that:

```bash
.venv/bin/python -B -m scripts.joint_gcad_study verify
```

Source/bank/snapshot metadata are checked at command entry/exit; GPU stages also
check model weights at entry/exit in merged traversals, not per judge call.
Neither new input nor scientific-source edits are permitted after registration.

## Calibration, mechanics, main generation

Native commands require root release and host-network `require_escalated`.
Use process-local PATH prepend `outputs/sarcasm-method-transfer-01/runtime-bin`
(or the registered pin-equivalent native path). Verify the existing2.1.323 binary
hash with the frozen guard; never change argv, auth, audit or fallback models.
Logs use umask077 in the new run's operator_logs. No SDK or dummy calls.

Run these separately for metrics in fixed order **profanity, impolite, sarcasm,
coherence** (replace METRIC; no semantic retry):

```bash
.venv/bin/python -B -m scripts.joint_gcad_study packet --stream calibration --metric METRIC
.venv/bin/python -B -m scripts.joint_gcad_study judge --stream calibration --metric METRIC --execute-native
```

Then `calibrate` (same judge Python, no remote call). Each metric needs>=7/8
in-band plus high-low mean>=30. Failure writes calibration_null and closes.
Root commits `results/protocols/joint-gcad-calibration-01.json`:
manifest_sha256, calibration_sha256 (`calibration.json`), accepted_sha256 mapping
all four metric names to their `native/calibration/METRIC/accepted.json` hashes.

Root separately releases GPU mechanics:

```bash
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -u -m scripts.joint_gcad_study mechanics --execute-gpu
```

20 cap32 generations + ten logit forwards. Root inspects all gates and commits
`results/protocols/joint-gcad-mechanics-01.json` containing manifest_sha256,
run_sha256, integrity_sha256, result_sha256 for `mechanics/{run,integrity,result}.json`.
Root then releases the fixed288-response main stage:

```bash
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -u -m scripts.joint_gcad_study generate --execute-gpu
```

B1 cap160 greedy seed42, explicit do_sampleFalse/temp1/top_p1/top_k0/
repetition_penalty1.05, BF16/SDPA, native default system. Physical GPU0 only,
UUID check and>=24000MiB free/util<=10%; resource busy is WAIT before phase
creation. Each completed response is atomically retained under its phase’s `answers/`
directory before later calls; uncertain starts remain charged. Neither GPU
phase reruns/resumes. All non-WAIT errors stop permanently.

## Primary and independent duplicate grading

Root releases four primary streams in the fixed metric order, same packet/judge
commands with `--stream primary`.36 batches per metric. Only when ALL primary
streams are accepted run four `--stream duplicates` streams in the same order,
three batches per metric. They regrade24 preselected saved answers, not new
responses. Plans are packed before outputs, distinct questions per batch and
opaque aliases; no arm/settings/bands in payload. Duplicate journals cannot
replace immutable first-valid primary labels.

Exact native ceiling:4calibration+144primary+12duplicates=160 valid batches,
max320 starts. Two identical structural attempts at most; only malformed schema
permits retry. Auth/model/isolation/infrastructure/uncertain starts are fatal.
`--resume` only replays an existing nonterminal journal; no rerating after valid
attempt0 and no resume after uncertain/fatal errors. Starts charge before calls.

## Report and exploratory guide

```bash
.venv/bin/python -B -m scripts.joint_gcad_study report
```

Exclusive public output: `results/joint_gcad_01.json`; private progress remains
inside run while incomplete. Completed primary analysis remains visible if only
duplicate grading fails, with execution explicitly incomplete. Report every arm,
all three full pair/full triple/half triple guide assessments, matched single
references, refusal counts, caps/invalid/length, duplicate noise, factorial
mixed differences/additivity residuals and question-bootstrap intervals. No
selection/refit or new release follows automatically. Within-arm correlations
are descriptive; pooled and prompt-centred pooled correlations remain condition-
confounded. Coherence is not accuracy. All24 questions are practical advice and
AI-authored by the judge family; do not generalize to all tasks/independent traits.
After outcomes, root can write a measured user guide with failures/tradeoffs.
