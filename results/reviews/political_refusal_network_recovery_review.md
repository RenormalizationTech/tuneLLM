# Political-refusal study: network recovery review

Reviewer: Claude advisor, read-only. I made no calls and no probes. This
reviews root's recovery plan for the calibration infrastructure failure. I
have not read the recovery code, which doesn't exist yet.

## Facts as reported

- **The parent run** is `outputs/political-refusal-01`.
- **Calls made:** one native start for calibration primary, batch 0,
  attempt 0, with a durable start record and receipt.
- **What the receipt shows:**
  - return code 1, `is_error: true`, empty `modelUsage`;
  - the result is "API Error: Can't reach API server", with DNS `EAI_AGAIN`.
- **Outputs:** no rating, label or model text exists. There are no Qwen
  outputs.
- **Status:** the fatal marker was written as designed.

## Verdict

**Accepted, with the conditions below.**

The failure happened before any judge content existed. A recovery that sends
the identical payload therefore can't select on, or re-roll, an observed
calibration outcome. Keeping the science frozen and charging the failed start
to the same ceilings keeps the study's provenance and budget intact.

## Conditions

### 1. The parent stays closed and unchanged

- Never write to, resume, retry, or remove the parent's stop marker.
- The recovery manifest records the hash of every parent file, including
  `stopped.json`, the start record, the receipt, `manifest.json` and the
  registrations.
- The recovery fails closed if any parent byte changes.

### 2. Check that the failure is content-free before any recovery call

The recovery's CPU checks must confirm all of the following:

- the receipt has `is_error: true` and an empty `modelUsage`;
- its result is the network error, with no `ratings` or other judge JSON in
  stdout or the result;
- there is exactly one parent native start, at calibration/primary batch 0
  attempt 0;
- there are no accepted files, no `calibration/check.json`, and no
  `headroom/` directory.

If any of these doesn't hold, there is no recovery. The case goes back for
review.

### 3. Identical science, checked by hash

These must match the parent manifest byte for byte:

- the inputs: `data.json`, `plans.json`, `rendered.json`, both rubrics and
  the model defaults;
- the config, under the rule in condition 3a below;
- every fixture, protocol and review hash bound by data acceptance.

**The calibration batch 0 payload** must be byte-identical to the parent's.
In particular, the recovery's `payloads_sha256` for calibration/primary must
equal the parent's `run_binding.json` value.

**No new prepare may change scientific bytes.** If the recovery re-prepares,
the resulting input hashes must equal the parent's.

The model, revision, seeds, plans, thresholds, methods, prompts and the
first-valid rules are unchanged.

### 3a. Config identity, with an execution-only allowlist

This resolves the conflict between conditions 3 and 4. Either implementation
is acceptable.

- **(A) Recovery gets its own config.**
  - Keep the parent's exact config bytes as `inputs/parent_config.json`, with
    a hash matching the parent manifest.
  - The recovery's own config may differ from the parent only in fields on an
    explicit allowlist, listed in the recovery source and tested:
    - run root;
    - result path;
    - config path;
    - registration paths;
    - implementation-stage or scope label;
    - recovery provenance pointers (the parent root, the parent manifest and
      receipt hashes, and the network-permission record).
  - With those fields removed from both configs, the two must be equal key by
    key.
  - The comparison must fail if any key is added, removed or changed outside
    the allowlist. That covers every scientific field: model, revision,
    snapshot, seeds, batch size, cap, systems, methods, preferences, residual
    rule, minimum pairs and coherence, random, bootstrap and plan seeds,
    thresholds, framing, budgets, and every fixture, protocol and review path
    and hash.
- **(B) The parent config stays unchanged and in force.** The recovery keeps
  the parent config bytes exactly and records its execution paths and
  provenance in a separate execution-metadata file. That file is bound by
  hash and contains only fields from the same allowlist.

**Under both options:** the budget fields stay identical, and cumulative
charging (condition 6) is implemented in recovery code, not by changing a
budget value. The recovery must record its real execution paths, so no
recorded path claims to be the parent root.

**Code reuse.** The recovery imports the frozen helpers and calls them
directly: `PRIMARY_RUBRIC`, `compile_fixture`, `all_plans`, `render`,
`bodies`/`parse` or their pure equivalents, and `political_refusal_common`.
It does not copy scientific logic or monkeypatch old module globals such as
`RUN`.

- **Where a frozen helper hard-codes the parent root:** wrap it with an
  explicit path argument, or reproduce only that I/O line.
- **The reviewed diff** (condition 4) must show that only path or budget I/O
  differs.
- **Each scientific output** of a frozen helper (plans, payload bodies,
  rendered prompts) is recomputed in the recovery and compared with the
  parent's frozen bytes before any call.

### 4. The recovery copy or wrapper differs only in execution details

- **Allowed differences:** the run-root and path constants, the
  registration paths, cumulative budget accounting, and the parent
  provenance check.
- **The diff:** a reviewed diff against the frozen
  `scripts/political_refusal.py` must show no other change.
- **Imports:** the recovery imports `political_refusal_common`, the rubric and
  plan functions unchanged and checked by hash. It does not reimplement them.
- **Frozen files stay unedited:** the frozen source, config and protocol are
  not changed.

### 5. Only the outer execution environment changes; native settings stay the same

- **Unchanged and verified against the parent's receipt:**
  - the pinned binary and version;
  - the model (`claude-opus-5-5`) and medium effort;
  - the native grader's argv and environment;
  - tools off and the native permission settings;
  - the neutral working directory;
  - the pinned runtime and the user's existing authentication;
  - the payload bytes.
- **No new tools are granted to the model.**
- **What changes, stated plainly:** the native process runs as an escalated
  host execution. The **outer OS sandbox is relaxed**, which is broader than
  a network-only toggle. That is what makes the API reachable.
- **This review does not claim that OS-level file or network permissions
  match the parent's.**
- **Record exactly this:**
  - `outer_os_sandbox_relaxed: true`;
  - `native_permission_settings_unchanged: true`, backed by an argv/env diff
    against the parent's receipt that shows no difference;
  - the outcome of the public DNS check.

  A vague field such as `tools_files_credentials_scope_change: false` must
  not be used.
- **Why this is acceptable:** the grader runs with tools off on a fixed
  payload, so the relaxed outer sandbox doesn't change what the model can
  see or do. Root's existing authorization for external grading covers this
  bounded API call.
- **If an automatic approval reviewer rejects the escalation:** treat it as
  an execution wait. No workaround or alternative route is tried, and the
  user is asked only then.
- **Not allowed:** retries, proxies or authentication workarounds.
- A public DNS check that makes no model call is fine.

### 6. Budget is cumulative, with no new allowance

- The parent's failed start counts against the initial 32-start ceiling and
  the global 424-start ceiling. `native_count` therefore includes the 1
  parent start, leaving at most 31 initial starts.
- The generation ceilings are unchanged and nothing has been used: 96 initial
  and 798 global.
- First-valid batches are unaffected (0 used).
- **Per-batch attempts:** the parent start produced no submission, so it was
  neither a structural attempt nor a valid one. Recovery batch 0 still gets
  at most the two identical structural attempts the protocol allows. This
  rule must be written into the recovery registration before any call, not
  decided afterwards.

### 7. Only one recovery

- Any further authentication, network, runtime or uncertain-start failure in
  the recovery is fatal, and the recovery's own stop marker is written.
- A second recovery needs a new reviewed registration. Every earlier start is
  charged to it.
- No automatic or looping retries.

### 8. Later stages use the recovery root

- The calibration registration and all later stages (headroom GPU, judging,
  decision, report, and any extension) use the recovery root.
- The parent is cited only as provenance.
- The headroom extension conditions in the design review apply unchanged.

### 9. Reporting

- The public result discloses the parent's infrastructure failure: one start
  charged, no labels, no outputs.
- It also discloses the recovery's registration, its hash and the network
  permission change, and states that no science was changed or re-rolled.
- Parent status: `incomplete_infrastructure_no_labels`.

## Not a reason to stop the study

An infrastructure failure before any label exists says nothing about
calibration, headroom or steering. The parent's closure is a provenance
event, not a scientific null.
