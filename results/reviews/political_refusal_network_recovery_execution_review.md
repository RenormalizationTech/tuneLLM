# Political-refusal study: network recovery execution review

Reviewer: Claude advisor, read-only. I made no calls, ran no preparation and
ran no tests.

**Scope.** Only the items root asked for:

- `scripts/political_refusal_recovery.py`, read in full;
- `configs/political_refusal_recovery.json`, compared key by key with the
  parent config;
- the alias and identity test in `tests/test_political_refusal_recovery.py`
  and its test list;
- the disclosure section of the recovery runbook;
- `results/protocols/political-refusal-parent-closure-01.json` and the parent
  attempt and receipt fields.

All of this was checked against the accepted plan in
`political_refusal_network_recovery_review.md`, which I did not modify.

## Verdict

**Accepted once one small disclosure fix is made** (below). Nothing in the
science, data, plans, budget or stage locks needs changing.

## Checks that pass

### Only execution details differ (plan conditions 3a and 4)

- **The config diff** against the parent is exactly `run`, `result`, the
  three registration paths, and an added `recovery` block. That block holds
  the parent root, the parent config, the provenance record, this review,
  `parent_charged_starts: 1`, `require_escalated` and
  `maximum_recoveries: 1`.
- **`science_equal`** compares everything else key by key.
- **`validate_config`** runs the parent's frozen `validate_config` on the
  parent config and pins the recovery paths and provenance block exactly.
- **The test** rejects any change to the seed, cap, plan seeds, thresholds,
  systems or fixture paths.
- **Imports:** `PRIMARY_RUBRIC`, `all_plans`, `render`, `compile_fixture`,
  `bodies`, `parse`, `predictions`, `generate_headroom`, `stage` and `stop`
  are imported from the frozen parent module, and the test asserts
  `getattr(r, name) is getattr(r.parent, name)`. No old global is
  monkeypatched.
- **Copied functions are identical.** Some functions had to be copied because
  they call the module-level `native_count` or `binding`. Comparing their
  syntax trees with the parent's shows `binding`, `read_grades`,
  `calibration_result`, `headroom_scores`, `headroom_result`, `lock`,
  `judge_stream` and `main` are identical.
- **The only differences in the copies:**
  - `native_count` adds the parent's charge;
  - `run_journal` adds a redundant 424 check;
  - `report` adds the recovery disclosure;
  - `Context`, `prepare` and `validate_config` handle recovery paths and
    provenance.

### Exact snapshot, payload and config identity (plan condition 3)

- **Copied, not regenerated:** `prepare` copies the parent's `config.json`,
  `data.json`, `plans.json`, `rendered.json`, both rubrics and the model
  defaults. It also keeps `inputs/parent_config.json`.
- **Checked on every `verify`:** `verify_scientific_snapshots` compares each
  copied file with the parent's by hash, and checks that the calibration
  primary payload hash equals the parent's bound `payloads_sha256`, as
  recorded in the committed closure record.
- **Recompiled for parity:** `verify_recompiled_inputs` recompiles the
  fixture and re-renders every prompt and token-ID list with the cached
  tokenizer, and requires equality with the parent bytes.
- **Plans** are recomputed and compared on every `verify`.

### Parent untouched, and no labels exist (plan conditions 1 and 2)

`parent_facts` and `content_free_failure` require all of the following:

- exactly one start, at calibration/primary batch 000 attempt 0, with
  exactly the start, attempt and receipt files;
- no `accepted.json`, no `calibration/check.json` and no `headroom/`;
- the exact fatal marker;
- the parent binding equal to a recomputed manifest-plus-payload binding;
- a checked attempt with return code 1, `is_error: true`, empty `modelUsage`,
  the exact DNS error string, and no rating or grade keys.

`verify_parent` requires the full parent file inventory to match the
committed closure record and the parent registrations. All of these reads are
read-only, and nothing writes to the parent root.

### Cumulative budget (plan condition 6)

- `native_count` equals 1 plus the recovery's starts. It is checked before
  each new invocation (fewer than 32) and when grades are read (at most 32),
  so at most 31 new initial starts are possible.
- The registration records the 1-start parent charge, the cumulative 32/424
  ceilings and the 2 structural attempts per batch, so recovery batch 0 is
  not penalised.
- The generation ceilings (96 initial, 798 global) are unchanged.

### Later stages and locks use the recovery root (plan condition 8)

- `RUN`, `guard`, the result path and the calibration and headroom
  registrations all point to the recovery.
- `lock` reads the recovery registrations.
- `registered` also requires the committed closure record and the recovery
  review.
- On any fatal error, the stop marker is written only in the recovery root.

## Required fix: make the disclosure of unchanged argv and environment honest (plan condition 5)

**The problem.** `recovery_disclosure` hard-codes `native_argv_unchanged: True`
and `native_environment_unchanged: True`. The parent receipt and attempt
files don't record argv or the environment. They hold `isolation`,
`canonical_stdin_sha256`, the return code, stdout and stderr. So a
receipt-to-receipt diff of argv and environment, as the plan asked, isn't
possible, and those two fields currently state more than was measured.

**What to change:**

1. **Label how equality was established.** For example,
   `native_argv_env_basis: "identical hash-pinned invocation source and
   binary; argv/env not recorded in receipts"`.
2. **After the recovery's batch 0 attempt 0, compare it with the parent's
   attempt and fail closed if any of these differ:**
   - the `isolation` record must be equal;
   - `canonical_stdin_sha256` must be equal, which confirms the identical
     payload.

   Record both comparisons in the report.
3. **Record the public DNS check** in a committed record, either the closure
   record or the recovery registration: when it ran, which host or resolver
   it used, the outcome, and a statement that no model call was made. The
   plan requires this, and the closure record currently doesn't contain it.

`outer_os_sandbox_relaxed: true`, the runbook's statement that OS file
permissions are not claimed to be identical, and `verification_scope` are
already honest and stay as they are.

## Notes (no change required)

- The redundant `>= 424` check in `run_journal` is harmless.
- `prepare` runs `parent.Context().verify(models=True)`, which re-hashes the
  model weights. That costs time, but it is read-only and appropriate.
- The copied-function identity test covers five functions. The comparison I
  ran also showed `binding`, `read_grades` and `main` identical. Optionally,
  add them to the test so any drift fails.

## Acceptance

Once the disclosure fix is in, with a test showing that a mismatched
`isolation` record or stdin hash fails closed, this recovery runner is
accepted for registration. That covers calibration only and the later
headroom stages under the same rules. Root still owns the DNS record, the
commits and every live release, which must use the escalated launch policy
described in the runbook.

## Addendum: final required fixes checked

I checked only the disclosure fixes. I made no calls and ran no tests. The
worker reports 30 focused tests passing in 3.75 s.

### Verified

- **Unmeasured claims removed.** `recovery_disclosure` no longer asserts
  `native_argv_unchanged` or `native_environment_unchanged`. It now states
  how equality was established: "Identical hash-pinned invocation source and
  binary; argv/env not recorded in receipts. No observed receipt-to-receipt
  argv/env diff is claimed." It also includes the `tone_judge.py` source hash
  and the binary hash. `outer_os_sandbox_relaxed: true` stays.
- **Measured comparison of the first attempt.**
  - `first_attempt_comparison` compares the recovery's
    calibration/primary batch 0 attempt 0 with the parent's attempt on two
    fields: the `isolation` record and `canonical_stdin_sha256`. It records
    both attempt hashes.
  - In `run_journal` it runs immediately after the call, before parsing, and
    fails closed if either field differs. It also runs again on every replay
    in `read_grades`.
  - The report includes it non-fatally, with a status of `not_attempted` or
    `started_no_completed_attempt` when nothing has been compared yet.
  - The tests show that a mismatch in either field fails closed, both in the
    live loop and in the direct comparison. The stdin case is rejected even
    earlier by the unchanged foundation code.
- **DNS record bound.** `political-refusal-dns-preflight-01.json` is required
  in the config's recovery block. Its hash is in the manifest, the
  registration and the report, it is checked on every `verify`, and it must
  be committed.

### One pre-call guard to add before release (small; protects the single recovery)

**The risk.** The parent's `isolation` record includes
`removed_git_variable_names: ["GIT_PAGER"]`. `tone_judge.invoke` takes these
names from the launching process's `os.environ`. If the escalated host shell
has a different set of `GIT_*` variables, which is a harmless environment
difference, the new comparison fails only **after** the native call. That
charges a start, writes a fatal marker, and ends the only permitted recovery.

**Required, before the first recovery invocation:**

- Compare `sorted(k for k in os.environ if k.startswith("GIT_"))` in the
  launch process with the parent's `removed_git_variable_names`.
- If they differ, raise `Wait` with **no call and no start record**. The
  operator then relaunches with a matching environment. No change to frozen
  code or science is needed.
- Add one test showing that a mismatched `GIT_*` set makes no call.

The other isolation fields (`git_not_repository`,
`git_environment_clean`, the command and its return code) are fixed by the
unchanged temporary-directory logic. The post-call check remains as the
backstop.

### Final verdict

**Accepted for preparation and registration once the pre-call `GIT_*` guard
and its test are in.** Everything required by the plan review and by this
review is otherwise implemented:

- only execution details differ, with the scientific helpers imported
  unchanged;
- scientific snapshots and payloads are identical;
- the parent is proven closed with no labels or Qwen outputs;
- the budget is cumulative: 1 parent start plus new starts, against the
  32/424 ceilings;
- the host-sandbox disclosure is honest;
- the DNS record is bound;
- every later path and lock points to the recovery root.

The unused preflight archive
(`outputs/political-refusal-recovery-unregistered-preflight-01`) and the
parent's 32 files stay untouched and must be disclosed in the report. Root
still owns the commits, the registration and the escalated live release.

## Final addendum: last two conditions checked, ACCEPTED

I checked only the two remaining conditions. I made no calls, ran no
preparation and ran no tests. The worker reports 31 focused tests passing,
and root is testing independently.

1. **The pre-call `GIT_*` guard is in place.**
   - On the first recovery launch (`native_count(s) == 1`), `run_journal`
     compares the sorted `GIT_*` names in the launch environment with the
     parent's `removed_git_variable_names`. It does this before creating the
     stream directory, the binding, any start record or any call.
   - On a mismatch it raises `Wait`. `main` treats `Wait` as a no-call wait,
     so no fatal stop is written.
   - **The test** (`test_first_launch_git_name_mismatch_waits_before_any_start_or_call`)
     adds an extra `GIT_*` variable. It confirms that no call is made, that
     `native_count` stays at 1, and that neither a `native/` directory nor
     `stopped.json` is created.
   - The post-call comparison of `isolation` and stdin stays as the
     backstop. Root's escalated preflight returned `["GIT_PAGER"]`, which
     matches the parent.
2. **The unused preflight archive is disclosed.**
   - `unused_preflight_archive()` requires all of the following for
     `outputs/political-refusal-recovery-unregistered-preflight-01`:
     - no symlink;
     - no `native/` and no `headroom/`;
     - `no_model_or_judge_calls: true`;
     - a full file inventory equal to its own manifest.
   - The manifest hash is bound into the recovery manifest's external hashes,
     the registration and the report. It is re-checked on every `verify`.
   - The report states: `unused_never_registered_cpu_preflight`, 0 native
     starts, 0 model generations and 0 scientific outputs.

**FINAL VERDICT: ACCEPTED.** The network recovery runner meets every
condition of the accepted recovery plan and of this review. It is accepted
for preparation, registration and the calibration and headroom stages under
the escalated launch policy in the runbook. The science, data, plans, budget
(1 parent start, plus recovery starts, against the cumulative 32/424 ceilings)
and the recovery root locks are unchanged from the reviewed state. Root owns
the commits, the registration and each live release.
