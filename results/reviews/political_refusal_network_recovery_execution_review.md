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
