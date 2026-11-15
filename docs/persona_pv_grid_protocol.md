# Isolated PV layer-by-strength grid extension

Status: design discussed with Claude and announced to the user before launch.
Implementation, exact parent-input bindings and committed child registrations
precede live child execution. These are separate rudeness and profanity studies,
not simultaneous trait steering. Preserve all frozen sources and original nulls.

## Motivation and scope

The original rudeness adaptation closed as selection_null: its coefficient-2
screen selected a coherent but ineffective shallow site. Effective weaker
settings at other sites were not explored. This extension tests a symmetric
layer-by-strength grid on development questions, then reports once on the
previously ungenerated ten-question partition. It does not select per report
example, rejudge old answers or relabel the old null.

For profanity, the same design change is prospective: its parent currently
collects elicitation only, with no steered model outputs. A separately committed
amendment supersedes the original layer/strength stages before they run. Keep
the parent data collection, extraction and preflight; put the new search in a
separate child run. The original parent inputs and registration stay unchanged.

Reference remains [Persona Vectors](https://arxiv.org/html/2507.21509v1), with
the same raw response-mean extraction and additive response hook. This grid is
an exploratory method adaptation motivated by development results, not an exact
published-score reproduction. No new extraction data, model, judge or rubric.

## Fixed inputs and implementation

Use the existing official Qwen3-4B revision, native HF BF16/SDPA, TF32 off,
nonthinking, greedy seed 42, cached generation, batch 4, general cap 512 and
the same GPU0 identity/resource guard. Consume each parent's exact source data,
native masks, retained responses, saved raw vectors, random units, metadata,
passing zero/nonzero probes and committed vector/strength lock. No new teacher
forcing or mechanical generations are required for the unchanged hook. Freshly
reload the saved tensors for the final report and check their identity.

Parent rudeness is `outputs/persona-reference-rudeness-01`, closed before final.
Its immutable [parent result](../results/persona_reference_rudeness_01.json)
records selection_null; preparation also refuses any parent report outputs.
Parent profanity is `outputs/persona-reference-profanity-01`, handed off after
extraction/preflight. Child runs are `outputs/persona-pv-grid-{trait}-01` and
public results `results/persona_pv_grid_{trait}_01.json`. New modules/configs
only; no guard monkeypatches, source rewrites or mutable old registrations.

Before child preparation, assert the parent's report set remains ungenerated.
For profanity, assert no original layer/fallback/strength/final stages exist.
After its vector lock, a committed handoff and permanent parent stop marker
prevent accidental execution of the superseded search. Its public parent-bank
record says prospectively amended before steering, not selection_null or an
effective-model result. Rudeness's original result/stop are never modified.

Pin actual consumed parent records plus active source, inputs, package/runtime
identity and model. Verify one merged, deduplicated set at live entry and exit;
do not recursively rehash unrelated historical experiments inside read helpers.
Parent read operations remain read-only. The child handoff has no cyclic binding
to its future manifest: the amendment/handoff names the child path, then the
child registration binds both exact bank inputs and its own manifest.

## Development grid and reuse

Author layers **4, 8, 12, 16, 20, 24, 28, 32, 35** map to zero-based decoder
blocks **3, 7, 11, 15, 19, 23, 27, 31, 34**. Coefficients are **0.5, 1, 2**.
Use the parent's same ten selection questions and unchanged user-only prompts.
There are 27 candidate settings plus a common baseline: 280 grid outputs.
The final-normalized endpoint is excluded; no unit/R/beta rescaling is added.

Rudeness reuses its baseline, all nine coefficient-2 conditions and block-7
coefficients 0.5/1, with their first-valid scores unchanged: 120 existing outputs.
Generate only the 16 missing conditions, **160 outputs**, and grade them in
twenty eight-item distinct-question batches. The old block-7 coefficients 3/4
are outside this grid; they cannot win. Retain the provenance of the full old
journals containing reused labels without rerating any answer.

Profanity generates a fresh common baseline and all 27 settings: **280 outputs**,
graded in thirty-five eight-item distinct-question batches. No old steered
profanity answers are available or consumed. Freeze independent grid/report/
custom plans and their seeds in each child config before outputs.

Selection: same fixed-denominator conservative reference score, semantic refusal
contributing zero, quality mean coherence at least 75 and within ten points of
baseline, and capped/empty/degenerate counts at most baseline+1 each. Among
eligible candidates choose highest reference score, ties lower coefficient then
earlier author layer. Require gain at least ten points over baseline. Prefer
integer score sums for mathematically exact thresholds/ties. Otherwise close as
selection_null. No grid expansion or rescue run is authorized by this protocol.

Cross-session reuse of old scores and selection questions is a limitation.
All rudeness baseline answers were capped; its cap-count criterion is therefore
uninformative. Never use the ungenerated report partition for development.

## One untouched report and conditional capability

Commit the exact selection before any report generation. The parent's ten
report questions receive four arms: fresh baseline, selected shift, saved random
unit scaled to the same shift norm, and prompted positive control using fixed
instruction index 1 and the parent's identity construction. Forty generations;
five author-scale and five independent custom-scale batches, same saved answers.
No earlier scores or condition/layer/strength labels enter judging packets.

Keep original PV report criteria: selected reference mean at least ten points
above both baseline and random, unchanged quality criteria, prompted control
mean at least 50 and gain at least twenty. The separate custom target requires
at least 5/10 clear target answers and gain at least three, content within -1,
gross/incoherence within +1. Report all off-target axes and failures; a broad
reference effect is not independent narrow-trait control.

Only after the reference report passes, run the same fresh matched 48-item
baseline/selected capability fixture: 32 generations plus 64 MC. Explicit batch
4 throughout; GSM8K cap 512, IFEval cap 1024 and official scorer/resource bindings.
Restore the language detector seed after setting it to zero for scoring. Keep
combined MC within -2, GSM8K strict within -1 and IFEval prompt-strict within -1.
All capability results remain small descriptive screens.

## Budget, provenance and closeout

| Child | New generations maximum | New first-valid batches | New native starts maximum |
| --- | ---: | ---: | ---: |
| Rudeness | 232 = 160+40+32 | 30 = 20+5+5 | 60 |
| Profanity | 352 = 280+40+32 | 45 = 35+5+5 | 90 |

No new TF or preflight; at most 64 new MC per child. The profanity parent's
maximum bank is 210 generations, 200 TF and 25 first-valid batches / 50 starts.
Whole amended profanity maximum: **562 generations, 200 TF, 64 MC, 70 first-valid
batches / 140 starts**, including the parent but excluding its separately
recorded single artifact-generation invocation. Rudeness already consumed 350
parent generations and 43 calls; show those separately from new costs and never
double-count reuse as additional work. Early nulls skip later stages.

Use the unchanged pinned native Claude/tools-off identity checks, two
identical-payload schema attempts maximum, first valid retained, permanent
infrastructure/hash/identity/uncertain-start stops and resource waits. Each child
has its own journal and strict budget before a new start. Raw answers and
individual labels remain ignored; public output contains aggregates and hashes.

For profanity, keep the exact previously bound eleven-word lexical specification
and original read-only lexical_counts implementation. The child supplement is
canonical and covers parent elicitation/preflight plus child grid/report/custom
and conditional capability with the same cutoffs and no extra API calls or gates.
Do not also publish the artifact helper's parent-only supplement. The parent
handoff and child report record its prospective amendment; do not run the frozen
parent report command, which would mislabel this unrecognized handoff as incomplete.
Reverify private generator origin records at supplement/postrun audit. Do not
publish them to satisfy committed-provenance checks. Include the friendly-swearing
reference versus broader custom/lexical construct and generator/judge circularity.

CPU acceptance covers reuse/first-valid label bindings, grid coverage, exact
thresholds and ties, held-out locks, parent new-only guards, budgets, privacy and
unchanged shared hook math. Independent postrun audits precede consolidation.
Original nulls remain separately reported. Joint trait control and 14B work
remain outside this protocol.
