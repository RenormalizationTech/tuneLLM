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
