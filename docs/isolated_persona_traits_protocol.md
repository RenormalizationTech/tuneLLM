# Isolated rudeness and sarcasm on Qwen3-4B

Design reviewed with Claude and accepted by the coordinator, 2026-11-14.
These are two separately registered studies, executed
rudeness first, sarcasm second, after the bounded profanity pilot closes. The
user selected isolated controls before joint steering and the full 4B pipeline
before further 14B work. Physics LoRA is independent.

The [profanity protocol](profanity_steering_protocol.md) supplies the common
model, response-mean extraction, sites, units, mechanical checks, stage locks,
quality screen, resources and interpretation limits. The next-trait plan was
discussed with Claude before profanity behavior results were inspected; it
recommended shared tasks, fixed parameters and a second confirmation judge pass.
The concrete overrides below require review and committed registrations before
any live call. Design acceptance alone does not authorize live execution.

## Common model and fixed settings

Keep official Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`,
unadapted BF16, SDPA, nonthinking native chat, helpful system, seed 42, generation
batch 4 and cap 256. GPU 0 UUID and the 12,000 MiB/10% resource checks stay the
same; leave other jobs untouched. Use zero-based blocks 17 and 23, equal-pair
FP32 response-token mean differences, unit vectors and beta 0.05/0.10/0.20 times
the median neutral response-mean norm. Record per-token/actual BF16 norms too.
No parameter changes are chosen from the profanity results.
The stored-output [divergence diagnostic](../results/diagnostics/profanity_development_divergence_01.json)
showed 6–14/16 token sequences changed per setting. The grid therefore remains
fixed: the earlier null was not a complete lack of intervention on greedy
outputs. This does not prove that every other magnitude would fail the target.
Record token-identical counts, first-divergence positions and length changes
against baseline for each new condition, alongside the per-token norm fractions.

Each study retains the same vector serialization, actual zero-vector parity,
nonzero hook checks, committed extraction strengths, fixed six-setting
development, committed selection, exact four-prompt same-batch vector reload,
one norm-matched random direction and fresh matched capability comparison.
One new lean runner may serve both profiles; it must preserve all old sources,
registrations and results, without patching their guards or global variables.

## Data

For each trait author 32 neutral/target answer pairs in 16 topic/template
families, two each, new and disjoint from the other extraction corpora and
calibration. Reuse the existing 16 development and 24 confirmation tasks, with
their eight factual probes, **unchanged** from the profanity fixture. These are
shared historically used prompts, not newly unseen tests. Generate fresh matched
baselines and outputs for each trait. No baseline/judge cache substitution is
part of these first isolated studies.

Match facts and requested artifacts within each pair. Annotate the neutral and
target style spans by offsets and require the text outside those spans to be
identical. Preserve task/reference agreement. Both sides have zero hits under
the fixed profanity lexicon. Target spans must be actual phrases, with at least
four declared phrasing families, none above one third of retained pairs, at
least two natural phrasings per family and none repeated more than three times.
Require the same 0.85–1.15 token-length ratio.

For the 32 authored targets use at most 12 opening, at least 10 middle and at
least 10 closing placements of the annotated target span's first word. Use
zero-based whitespace-word index and exact integer thirds, with punctuation and
hyphenated forms remaining one whitespace word. Validate annotations and count
actual placements; report retention attrition by phrase, family and position.

Rudeness means clearly brusque or dismissive task-completing delivery toward the
fictional addressee or their request; merely expressing frustration with a task
or object is insufficient under the unchanged rating rubric. Sarcasm means contextual irony about
objects or situations, keeping reference facts intact. Do not replace these
traits with conciseness or warmth. Use no slurs, threats or real-person abuse;
retain the calibrated narrow gross-failure definition. Sarcasm/rudeness overlap
is an expected possibility and a valid failure, with no relaxed crosstalk gate.

Only the helpful system and `user` field enter model prompts. `context` is
evaluator metadata: rudeness uses a neutral fictional single-turn description,
without inventing repeated requests. All scenario facts needed for sarcasm
appear in `user`, so the model and evaluator share the basis for the irony.
The authored sarcasm uses generic ironic praise of malfunctioning objects, a
similar genre to calibration despite distinct topics. Neutral replacement spans
are sometimes unnatural meta-statements. These limit generalization; eligibility
and generated-output evaluation remain necessary.

## Eligibility and development

Blindly rate all 64 authored texts once on all three unchanged calibrated trait
scales plus content/coherence/quality fields. The positive side must score at
least 2 on its intended trait, the neutral at most 1 on all traits, and positive
off-target traits at most 1. Both sides must fulfill content, be coherent and
have no gross failure; nonfactual correctness is ignored. Require at least 24
eligible pairs, at least four retained declared style families, and no family
above one third of the retained pairs. Retain all passing pairs without
favorable subsampling. Authored position quotas and two-phrasings-per-family
requirements are not re-applied after eligibility: report phrase/family/position
attrition instead, including the two closely worded rudeness families. Claude
confirmed this clarification before registration, preserving the common
profanity-style retention rule. A failure closes that study.

For generation define **usable target presence** as target score at least 2,
content fulfilled, coherent, no gross failure, and no empty/mechanical
degeneration flag. Report raw presence, ordinal scores and score-1 ambiguity.
The two off-target axes are profanity/sarcasm for rudeness, and profanity/
rudeness for sarcasm. Always report lexical profanity as a separate diagnostic.

Development remains baseline plus six fixed settings on 16 prompts, 112 outputs.
An eligible setting must keep each off-target mean within +0.5, clear-presence
counts within +1, content fulfillment within -1, and incoherence/gross failures
within +1 of baseline. Choose the highest usable target count, then lower beta,
then block 17. Require at least 8/16 usable target outputs and a gain of at least
seven over baseline. Otherwise close with a development null and skip final.

## Two-pass confirmation

Generate baseline, locked steer and the one norm-matched random control on the
24 shared confirmation prompts: 72 outputs. Rate those same fixed outputs
twice, each in fresh isolated contexts with separately frozen constrained batch
membership. Never put two conditions of the same prompt in a batch; the
second pass cannot repair or replace the first. Keep stable opaque item aliases
across passes and prove the payload contents per alias are identical.

For primary counts, the target is present only when **both** passes score it at
least 2. Usability requires content/coherence in both passes and no gross flag
in either, plus the mechanical checks. For each off-target axis use the per-item
maximum score across passes, reporting individual-pass results too. Content and
factual correctness use agreement of both passes; incoherence/gross flags use
either-pass presence. Report presence disagreements and ordinal weighted kappa
per axis. These repeated judgments share a model family and are not human truth.

All final gates remain fixed:

- At least 12/24 usable targets and at least ten more than baseline.
- Random-control raw target presence in at most 2/24, using **either** pass at
  least 2 as the conservative raw-presence measure.
- Off-target mean increases at most 0.5 and clear counts at most +2; content
  fulfillment at most -2, incoherence/gross failures at most +2 versus baseline.
- At least 7/8 factual probes correct, no more than one below baseline.
- Fresh 48-item capability screen: combined MC within -2, GSM8K strict within
  -1 and IFEval prompt-strict within -1. Report fallback and mechanics separately.

Report rudeness-only, sarcasm-only and co-occurring clear-presence counts so
entanglement remains visible. No retuning or replacing confirmation labels.

## Budgets, blinding and closure

Maximum generation per trait stays **230**: 10 mechanical preflight, 112
development, four reload, 72 confirmation and 32 capability, plus 64 MC items
and at most 64 teacher-forced authored sequences. Development/eligibility nulls
skip later phases.

Judging is 8 eligibility, 14 development and 9 + 9 confirmation batches:
**40 first-valid batches, at most 80 invocations per trait**. Each batch has
eight distinct underlying tasks, at most two structural attempts, immutable
first-valid results and the unchanged pinned native tools-off judge. Seeds are
501/502/503 for the first three plans and 504 for the repeat confirmation plan.
Constrain pair sides/conditions exactly as in the profanity protocol. Bind
sources, masks, aliases, memberships, outputs and receipts; validate service
model identity and cumulative budget before every new start. Network-enabled
execution and GPU release belong to the coordinator after registration.

Preserve all earlier experiments. Stop on fatal infrastructure, identity, hash
or uncertain-start failures; no semantic retry or unregistered new study. Export
known aggregates only, with raw texts and individual judgments local/ignored.
Scientific review, code/fixture acceptance and actual tests must be described
accurately when an advisor response is unavailable.

Each axis closes independently, with a null/tradeoff a valid learning outcome.
Save both layer directions and metadata, while treating only a qualifying locked
setting as validated on this fixture. Joint control needs at least two supported
isolated directions and its own protocol; full three-way control needs all three.
Further 14B experiments remain deferred until this bounded 4B pipeline is closed.
The tested strength range is limited. The profanity grid's maximum was about
15% of mean per-token residual norm; each new trait reports its actual fractions.
A null within this range does not rule out other strengths or representations.
