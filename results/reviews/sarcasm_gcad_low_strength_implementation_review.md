# Sarcasm GCAD lower-strength study: implementation review

Reviewer: Claude advisor, read-only. Scope: `scripts/sarcasm_gcad_low_strength.py`
(409 lines) and `configs/sarcasm_gcad_low_strength.json`, both read in full,
checked against [the protocol](../../docs/sarcasm_gcad_low_strength_protocol.md)
and [the design review](sarcasm_gcad_low_strength_design_review.md).

**Limits of this review.** I did not re-read the parent functions the child
delegates to: `controller`, `validate_banks`, `random_controls`,
`validate_controls`, `controls_metadata`, `rating_parse`, `journal`,
`summarize`, `choose` and `final_decision`. For those I rely on the parent
study's earlier review and the root's tests. I ran no tests and made no calls.

## Verdict

**Conditionally accepted.** Acceptance becomes final once the root has checked
the required fix and the two recommendations below in the frozen source.

## Checks that pass

- **The parent stays closed.** `parent_scope` requires the parent's status,
  selection and stop cause all to be the development null, and its held-out
  directories and selection registration to be absent. `verify()` re-checks
  this. The parent's manifest, report, selection, bank, journals and
  registrations are bound by hash. The child only reads the parent and writes
  nothing into it.
- **Reuse of the exact bank and 64 answers.** The bank is copied and checked
  against the parent's hash at every verify. The 64 reused baseline and
  prompted answers carry a hash per record and per output. `predictions()`
  requires the development baseline and prompted answers to equal them
  exactly, and every answer must match the parent's rendering bytes. All 160
  development items receive new grades, and
  `parent_grades_used_for_child_selection` is false.
- **Strengths and gates.** The config locks GCAD {1, 2, 3}, blocks 9–19, scale
  1.5, cap 160, seed 42 and the random seeds. Selection uses the parent's
  `choose`, which takes the lowest qualifying strength under the unchanged
  gates. The held-out decision uses `final_decision`, which includes the
  random-plus-20 clause.
- **Ceilings checked before every call.**
  - 230 generations, counted per kind across all phases before each start. The
    reused answers are never charged, so development charges only 96.
  - 2 logit forwards and 176 native starts.
  - Plans are frozen at prepare: 160 development items per metric (20
    batches), and 128 held-out items per metric (16 batches for each of 3
    metrics).
- **Mechanics.** Six cap-32 probes (ordinary, zero and coefficient 1 on two
  questions). Exact token equality and first-step logit equality with
  `torch.equal`. Nonzero changes in both the generation logits and the 2
  explicit logit forwards. Mechanics must be registered before development.
- **Random control.** The held-out random arm uses the selected setting with
  the parent's control contract, locked by hash in the selection registration
  before held-out.
- **The H14 exclusion is descriptive only.** Its figures sit beside the primary
  32-question results with `primary_decisions_unchanged`, and it is marked not
  applicable for a development null.
- **No changes to the parent's code.** The child calls parent functions with
  its own context and paths, imports them read-only, and uses fresh run,
  config and registration paths.

## Required fix (accepted by root; worker implementing)

**A failure in the descriptive impoliteness journal must not hide a completed
primary result.** As written, if held-out sarcasm and coherence are complete
and the impoliteness journal then stops fatally, `report()` sets the status to
`incomplete` and writes the primary decision only to `progress_report.json`.

Required behaviour, now also stated in the protocol:

- A fatal stop **during the held-out impoliteness journal**, after the primary
  decision is complete, publishes to `results/` with terminal status
  `primary_complete_descriptive_incomplete`. It keeps the primary decision,
  the H14 descriptive figures and the stop record.
- Any fatal stop before the primary decision is complete stays a private
  `incomplete` record.
- Source and hash guards are unchanged.

## Recommendations (accepted by root; worker implementing)

1. **Mechanics coverage must not pass vacuously.** `all(...)` over an empty
   `shift_audit` returns true. Assert exactly the 11 layer entries (blocks
   9–19), plus `hooks_removed`, before checking each entry.
2. **Enforce the grading order.** Held-out impoliteness must not start until
   both held-out sarcasm and coherence are accepted, so a descriptive failure
   can never block the primary streams.

## What final acceptance needs

The root confirms in the frozen source that:

- the publication rule above is implemented;
- the 11-layer coverage assertion is present;
- the order check is present;

and that the focused tests cover each of them. No other change is needed.

## Root final acceptance

Root read the completed runner/config/runbook and confirmed all three changes.
Coverage requires exactly blocks9–19 plus successful cleanup, including nonzero
prefill and decode changes. CLI prerequisites enforce primary grading before
the descriptive stream. A completed primary result is published after a
descriptive execution failure with its stop metadata; entry/post-integrity
failures remain incomplete and private. The corresponding focused tests pass.

Independent root verification: all15 focused CPU checks passed in8.47seconds,
including actual pinned GCAD hooks at1/2/3 on CPU fake attention modules,
real-bank random controls, exact parent reuse and the176-start replay boundary.
No target-model, GPU or native calls were made by these tests. Accepted for
source freezing and CPU preparation; root still registers each live release.
