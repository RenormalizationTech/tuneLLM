# Refusal pilot planning and data review

Date: 2026-11-13. Roles: coordinating Codex manages acceptance, Claude advises
on planning/science and curates source prompts, dedicated Codex implements code.

Claude first proposed a larger refusal study. The manager reduced it to a
learning exercise: fixed layer 17, three positive multipliers, small type-disjoint
XSTest subsets, BF16 only and approximately 378 generated responses if selection
passes. Claude approved this scope, with numerical stability, confound and
uncertainty disclosures. No improvement or broad safety claim is intended.

Reviewed decisions before model outcomes:

- Prompt-side harmful-minus-benign means avoid tone's response-pair yield gate.
  Up to 100 matched prompts per side is a budget, not a minimum sample gate.
- Raw instruction token lengths are matched within ±20%; benchmark and
  cross-class normalized duplicates are excluded.
- Three of the six directly matched XSTest types enter development. All shared
  privacy/discrimination pools remain final; the final set has five broader
  families, including two shared groups with unequal safe/unsafe counts.
- Primary AI refusal counts partial plus full refusals, with fixed denominators
  and unscorable/empty/degenerate outputs excluded from the numerator.
- Select the smallest quality-passing positive multiplier with refusal rate
  at least 50% and at least 20 percentage points above the base. Otherwise
  report null and skip final generation. No extra layers or strengths are tried.
- Projection AUC is correlational and cannot prove a pure refusal feature.
  Topic/style confounds, unknown pretraining exposure and five final clusters
  limit interpretation.
- Negative steering, NF4 transfer, the larger retention screen, new training
  and vector publication remain outside this pilot. Baseline benign refusals
  can motivate a future user decision about calibration.

## Source auditing and screening

The worker verified pinned public authors' sources, exact files, counts and
licenses in `results/refusal_data_audit.json`, with no gated terms or credentials.
The raw sources remain ignored. Preparation and source-audit CPU checks passed.

Before source screening, commit `7668545` froze the deterministic 200-item Dolly
pool, semantic rubric, reviewer identity/overlap, order and hashes. The manager
and advisor explicitly rejected a keyword blacklist: clearly benign intent
remains harmless even when wording sounds alarming. Ambiguity is about intent
or required context, not isolated words.

Claude reviewed all 200 instructions under that rubric and wrote the complete
hash-bound labels file. All were classified harmless; no candidate was replaced.
The advisor noted several imperfect-but-benign instructions and retained them.
This is AI source curation with protocol-advisor overlap, not human ground truth
or blinded model-response grading. Dated snapshot and reasoning effort were not
exposed. Exact metadata/labels are in `results/refusal_dolly_screen.json`.

Final preparation used those labels without manual exclusions and produced
100 extraction pairs, 60 development prompts (30 safe/30 unsafe) and 72 final
prompts (42 safe/30 unsafe). The development types are historical events, safe
contexts and safe targets. Final families are definitions, figurative language,
homonyms, privacy and discrimination. Source/input IDs and hashes are in
`results/protocols/refusal-prepared-01.json`.

No refusal-model generation, steering vector, development selection, final
response or experimental response rating exists at this review stage. Actual
code/rubric/audit and run-registration review must precede execution.

## Implementation acceptance and extraction review

Before model execution, the dedicated worker passed 27 refusal-specific CPU
checks and froze the runner, validator, judge adapter, config and rubric. Claude
accepted the implementation review and the attention-mask position-ID fix.
The new adapter passed one fresh dummy isolation audit with tools disabled,
a neutral temporary working directory and an independent Git-not-repository
check. Requested and observed judge model were `claude-opus-5-5`; exact dated
snapshot and actual effort were not exposed. Only sanitized audit metadata
is tracked in `results/refusal_judge_validation.json`. Commit `f6a0f87` froze
the code and exact run/judge registrations before GPU extraction.

Extraction then produced finite directions from 100 harmful and 100 benign
prompts, without generating answers. Claude reviewed the diagnostics: layer 17
split-half cosine 0.984 indicates a stable source-class mean difference, while
development projection AUC 0.637 gives only modest safe/unsafe separation.
Neither establishes causal refusal control. Diagnostic layer 23 had AUC 0.974,
but the intervention remains at the registered layer 17. Source register and
topic confounds remain. High-dose disruption must not count as refusal.

The initial cap 256 preflight hit the limit on 40/60 prompts and therefore
triggered the registered cap 320 rerun. Claude accepted this fixed rule, with no
further extension. At 320, 27/30 safe and 9/30 unsafe baseline responses remain
truncated. Refusal is classified from the available text, with unscorable
responses handled by the frozen rubric. Useful/correct ratings are indicative
only: cutoff can affect completion, and length changes can make its effect
differ across conditions. Report cap-hit counts beside usefulness results.

Actual zero-vector/multiplier1 generation matched baseline tokens exactly;
nonzero/multiplier1 had active prefill/decode hooks, hidden-state delta 1.296875
and maximum last-prompt logit delta 7.5. The remaining generation upper
projection was 642.2 seconds, excluding judging/loading/extraction, within the
reviewed small-pilot budget. The exact report is committed before development.

## Final acceptance

Claude reviewed the completed development aggregates and accepted the null
as a valid outcome of the registered pilot. Benign refusals were 1/30 at
baseline and 2/30 at strengths 1/2; strength 4 degenerated on all 30 prompts. No
strength met the fixed rule, and final generation was skipped under a committed
selection lock. Baseline unsafe refusals were 28/30; these are descriptive
development measurements, not a final benchmark. The small benign increase
does not establish clean control, and the design does not rule out other
refusal mechanisms.

The manager retains the topic/style confound as a possibility rather than a
proven explanation of the null. The secondary prefix matcher missed many
AI-classified refusals and was not retuned or used to change selection.

Claude recommends one separately registered late-layer follow-up if the user
wants to continue steering, motivated by the already-seen layer 23 diagnostic.
This would be exploratory follow-up with disclosed development reuse; its
strengths, layer and rules need a new registration. Closing steering or returning
to tone/emotion work are alternatives. No new job starts before the user's
scientific-priority choice. The layer 17 null is preserved.
