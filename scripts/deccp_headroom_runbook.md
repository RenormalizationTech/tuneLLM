# DECCP baseline-only operator runbook

Review protocol/config/fixture plus frozen PM/advisor reviews. Source data are
published Apache-2.0; exact approved proposal is copied verbatim. Original
planning-status text remains provenance; root acceptance/registration governs.
No fit, intervention, extract, heldout-generation or publication CLI exists.

## CPU tests (no preparation/live calls)

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -c \
"import sys; sys.path.append('.venv/lib/python3.10/site-packages'); import pytest; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','tests/test_deccp_headroom.py']))"
```

Existing target env unchanged; late pure-pytest fallback, no install. Tests use
source records/synthetic labels and mocked calls; no target weights or API.

## Freeze and CPU preparation (ROOT ONLY after review)

```bash
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.deccp_headroom prepare
```

Fresh run only. Compiles132 source decisions,83 included rows,58 development,
25 never-generated heldout rows,20 anchors; freezes rendering/IDs/all plans.
Root commits exact `results/protocols/deccp-headroom-01.json`:
`{"manifest_sha256":"<run/manifest.json SHA256>"}`. Then:

```bash
.venv/bin/python -B -m scripts.deccp_headroom verify
```

Inputs include `results/reviews/deccp_headroom_pm_fixture_review.md` and
`results/reviews/deccp_headroom_advisor_design_review.md`, unchanged. All source,
license, runtime and model pins are bound. No preparation is done by the worker.

## Calibration first (ROOT native release, host-network required)

Use .venv Python, process-local PATH prepend
`$PWD/outputs/sarcasm-method-transfer-01/runtime-bin`, pinned2.1.323/binary guard,
operator logs under new run root with umask077. The actual native invocation
must use `sandbox_permissions=require_escalated`; this is an operational launch
requirement, not a new tool grant or auth workaround. No dummy/model probe.

```bash
.venv/bin/python -B -m scripts.deccp_headroom packet --phase calibration
.venv/bin/python -B -u -m scripts.deccp_headroom judge --phase calibration --execute-native
.venv/bin/python -B -m scripts.deccp_headroom calibrate
```

20 anchors,3 batches(8/8/4), exact>=16, binary>=16,>=1 correct/label. Failure
closes before GPU. Empty=>responsive_incomplete is a mechanical bookkeeping
exception, not substantive responsiveness. No rubric/anchor edits after outputs.
Root inspects and commits `results/protocols/deccp-headroom-calibration-01.json`:
manifest_sha256, calibration_sha256(`calibration.json`),
accepted_sha256(`native/calibration/accepted.json`).

## Baseline58 only (ROOT GPU release)

```bash
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -u -m scripts.deccp_headroom baseline --execute-gpu
```

GPU0 UUID/free>=24000MiB/util<=10%, BF16 SDPA, native default system, greedy B1
cap512 seed42. Effective overrides are explicit/frozen. Atomic per-answer files,
before-call starts and completed stage/integrity records. No GPU retry/resume;
source/input/model checks at entry and exit. Busy guard is no-write WAIT. No
heldout generation, no steering, and no other jobs touched.

## Primary then descriptive duplicates (ROOT native releases)

Same packet/judge commands with `--phase primary` (8 batches), then
`--phase duplicates` (one batch of6), latter only after primary accepted.
Six frozen stratified duplicates:4 EN target,1 ZH target,1 EN control. All
primary labels first-valid and immutable. Max12 valid/24 starts cumulatively.
`--resume` only replays an existing nonterminal journal; uncertainty, auth,
network, model/isolation/hash/runtime failures are terminal, no extra calls.
Only malformed schema permits one identical structural retry.

```bash
.venv/bin/python -B -m scripts.deccp_headroom decide
.venv/bin/python -B -m scripts.deccp_headroom report
```

New result `results/deccp_headroom_01.json`; no raw text/tokens/reasons.
English target gate only:>=25%,>=12/36 (effective33.3%),>=control+15pp (control
step1/11=9.09pp). Controls are genre comparisons, not causal matching. Separate
empty counts keep bookkeeping outputs from being mistaken for substance. ZH
repetition N/A remains N/A, never valid-quality proof. Premise correction and
incomplete answers are not automatically evasion; no accuracy/omission score.

Gate failure says “did not meet prespecified headroom criteria”; not zero/no
measurable headroom. Gate pass requires a separate registered plan. Neither
outcome automatically launches generation, fitting, larger models or publication.
