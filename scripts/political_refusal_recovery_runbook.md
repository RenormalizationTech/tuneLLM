# Political-information calibration/headroom: one network recovery

This is an execution-only recovery of `outputs/political-refusal-01`, which
remains permanently closed. Its single default-sandbox CLI call failed with
DNS EAI_AGAIN before any rating or Qwen output. That start is charged to the
same **32 initial / 424 whole-study** ceilings: at most 31 new initial starts.
The original 96 initial / 798 whole-study generation ceilings are unchanged.
No new scientific null, label or allowance is created by this recovery.

The scientific protocol and reviewed data are unchanged. Authority:
`results/reviews/political_refusal_network_recovery_review.md` and the committed
`results/protocols/political-refusal-parent-closure-01.json`. Every parent file
(including its log, fatal marker, source snapshots and native journals) is
hash-bound, plus its original registrations. Any changed or added parent file
blocks recovery. No parent lock/resume/receipt-repair routine is invoked.
The unused, zero-call snapshot archived at
`outputs/political-refusal-recovery-unregistered-preflight-01` stays untouched;
it is not a registration or a recovery attempt.
Its manifest SHA256 is bound in preparation and registration. Verification
checks its complete snapshot inventory and rejects any native/headroom stage;
the report discloses zero calls, zero scientific outputs and preserved inputs.

## Snapshot and source differences

`configs/political_refusal_recovery.json` differs from the parent only in five
operational locations and the explicit recovery metadata (review option A).
`inputs/parent_config.json` and `config.json` in
the recovery snapshot are **exact copies of the original scientific config**;
`execution_config.json` separately supplies the new run/result/registration
paths. All six `inputs/*.json` scientific artifacts are copied byte for byte:
data, plans, rendered prompts/IDs, both rubrics and model defaults. Preparation
recompiles source data and re-tokenizes/re-renders with the cached tokenizer
only to assert exact equality; it saves the copied original bytes, never a
replacement or reshuffle.
Aliases, membership, languages, all seven later branches and payload hashes
remain identical. Parent calibration-primary `payloads_sha256` is checked
against recovery payloads before any execution.

The new runner directly imports frozen `PRIMARY_RUBRIC`, `compile_fixture`,
`all_plans`, `render`, `bodies`, `parse`, `predictions` and `generate_headroom`.
Receipt replay, locks and aggregation retain the original method bodies where
new cumulative accounting must resolve locally. Native `f.invoke`,
`f.replay_or_call`, isolation, argv/model/version checks and receipts are
unchanged. No global monkeypatch, SDK, proxy, authentication change or dummy
call exists. No fit/intervention stage is exposed.

## Root-only preparation and registrations

Worker implementation makes no snapshot or live calls. After source review:

```bash
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.political_refusal_recovery prepare
.venv/bin/python -B -m scripts.political_refusal_recovery verify
```

Fresh root: `outputs/political-refusal-recovery-01`. Initial registration
`results/protocols/political-refusal-recovery-01.json` must equal
`registration_expected(Context())`, with exactly:

```json
{
  "manifest_sha256": "...",
  "parent_provenance_sha256": "...",
  "dns_preflight_sha256": "...",
  "unused_preflight_manifest_sha256": "...",
  "parent_charged_starts": 1,
  "initial_cumulative_maximum_starts": 32,
  "global_cumulative_maximum_starts": 424,
  "recovery_batch_maximum_structural_attempts": 2,
  "network_execution_policy": "require_escalated"
}
```

Commit the new source/config/runbook, closure record, both recovery reviews,
DNS preflight record and registration
before calls. The failed parent's API error is not a structural submission;
recovery batch 0 may use its normal two identical-payload structural attempts,
within the cumulative ceilings. First valid is frozen, never rerated. Existing
valid batches replay without charge. Uncertain starts, auth/network/runtime,
model/isolation/hash errors stop permanently; no second automatic recovery.

## Released calibration: host-network launch is mandatory

The operator **must request `sandbox_permissions="require_escalated"`** for the
real native command below. This is already authorized host-network execution,
not a requirement for another scientific/user decision. Do not first try the
same command in the default sandbox. The Python process cannot prove its outer
permission mode; it records the required launch policy without spoofing a
sandbox detector. Escalation relaxes the outer OS sandbox. Native argv,
environment policy, same-user auth/runtime, grader tools-off settings and
LLM tool permission policy remains unchanged; OS file permissions are not
claimed identical. No additional tool is granted to the grader. The basis for
argv/environment equality is the identical hash-pinned invocation source and
native binary. **Argv and environment were not recorded in either receipt;
no observed receipt-to-receipt diff of those fields is claimed.**

Immediately after recovery calibration/primary batch000 attempt0, compare its
actual `isolation` record and `canonical_stdin_sha256` exactly with the parent.
Any mismatch is fatal before parsing/retry. Receipt replay rechecks them. The
report records the measured comparison booleans and both raw-attempt hashes;
before completion it distinguishes not-attempted from a start with no completed
attempt. The unchanged foundation also rejects a wrong stdin hash before it
can save a completed attempt. The parent's isolation record includes removal
of `GIT_PAGER`; a different removed-variable list is a mismatch, not silently
normalized away.
Before the first recovery invocation, the runner also compares sorted launch
`GIT_*` names with the parent's removed-variable list. A mismatch returns
`Wait` before creating the native stream or any start/call/fatal marker. The
operator may relaunch with a matching environment; this consumes no budget.

`results/protocols/political-refusal-dns-preflight-01.json` records root's
read-only diagnostic: `api.anthropic.com`, UTC times, default resolution failure
and escalated resolution success, explicitly no model/grader call. Its SHA256
is bound in the manifest, initial registration and public recovery disclosure;
the record is not an authentication or API-availability probe.

With root's explicit stage release, use a fresh mode600 operator log and
umask077 (never overwrite an existing log):

```bash
umask 077
mkdir -p outputs/political-refusal-recovery-01/operator_logs
set -o noclobber
PATH="$PWD/outputs/sarcasm-method-transfer-01/runtime-bin:$PATH" .venv/bin/python -B -u -m scripts.political_refusal_recovery calibration --execute-native > outputs/political-refusal-recovery-01/operator_logs/calibration-01.log 2>&1
```

Four first-valid calibration batches maximum (two primary, two coherence).
Primary calibration failure stops before coherence. No GPU headroom until root
inspects passing calibration and commits its separate lock:
`results/protocols/political-refusal-recovery-calibration-01.json` with
`manifest_sha256`, `calibration_check_sha256`, `primary_accepted_sha256`,
`coherence_accepted_sha256` (same lock schema, new root only).

Later separately released stages:

```bash
# GPU host execution after committed calibration lock; no native calls here:
.venv/bin/python -B -u -m scripts.political_refusal_recovery headroom --execute-gpu
# Native judging also requires require_escalated and the identical PATH pin:
.venv/bin/python -B -u -m scripts.political_refusal_recovery judge --execute-native
.venv/bin/python -B -m scripts.political_refusal_recovery decide
.venv/bin/python -B -m scripts.political_refusal_recovery report
```

Headroom lock: `results/protocols/political-refusal-recovery-headroom-01.json`;
keys `manifest_sha256`, `headroom_choice_sha256`, `headroom_run_sha256`,
`headroom_integrity_sha256`, `primary_accepted_sha256`. A lock is not another
execution release. Output `results/political_refusal_recovery_headroom_01.json`
is aggregate only and discloses parent failure/charged start, recovery
registration and permission change. The frozen report behavior remains:
`incomplete` returns status without publishing an incomplete success result.
Original raw private records are never copied into public metadata.

## CPU checks

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_political_refusal_recovery.py
```

Mocks use the real first-valid journal/receipt functions, never an executable
or API. Checks cover cumulative pre-start budget, replay/uncertain/fatal
behavior, exact scientific copies/payloads, parent tampering/content-free
preconditions, no parent writes, safe public disclosure and direct reuse of
frozen scientific functions. No model load, GPU or snapshot preparation.
Additional checks cover actual parent-vs-recovery isolation/stdin equality,
both mismatch failures without retry, source/binary-only argv/env disclosure,
and DNS hash binding in the registration.
