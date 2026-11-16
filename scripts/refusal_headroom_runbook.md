# Benign Qwen7 refusal headroom baseline

Protocol: `docs/refusal_headroom_protocol.md`; acceptance:
`results/reviews/refusal_headroom_design_review.md`. New root only:
`outputs/refusal-headroom-01`, public result `results/refusal_headroom_01.json`.
Root reviews/registers/releases every live stage; the worker did not prepare
or run one. No old source, data, cache, run, report or registration is edited.

This is a benign published-prompt baseline, not steering or safeguard removal.
No unsafe prompt, vector, extraction, intervention, model modification, extra
model download, installation or SDK is involved. The only generation command
produces exactly100 development answers. No held-out generation API exists.

## Preparation and frozen split

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.refusal_headroom prepare
.venv/bin/python -B -m scripts.refusal_headroom verify
```

Preparation verifies the official CSV SHA and pinned public source/license
records,250 safe samples/ten types25, upstream C/P/I definitions+mapping and
author prefix function. It freezes both split IDs and all250 rendered token
inputs. A SINGLE dedicated `random.Random(42)` traverses SORTED type names;
within each type source-ordered rows are shuffled, first10 development and
remaining15 heldout. No global RNG mutation or per-type reseeding. The exact
100/150 IDs and serialized algorithm are reviewable before calls.

Only used source/reference/runtime/binary/model hashes enter the snapshot;
old historical output closures are not recursively included. The cached
official CSV contains450 original records, but only its250 safe rows enter
the split; no unsafe row is rendered, generated or graded.

Root commits sources, protocol/review and
`results/protocols/refusal-headroom-01.json` with exactly `manifest_sha256`.
No GPU/native work is allowed until that exact registration/release.

## Development generation and grading

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.refusal_headroom generate --execute-gpu
.venv/bin/python -B -m scripts.refusal_headroom packet
.venv/bin/python -B -m scripts.refusal_headroom judge --execute-native
.venv/bin/python -B -m scripts.refusal_headroom report
```

GPU runtime: pinned Qwen2.5-7B BF16/SDPA in the existing isolated Python3.11
environment; physical GPU0 UUID guard, free>=24000MiB/util<=10. Native default
Qwen system, greedy seed42, B1, cap256, cache/EOS defaults retained. The model
is verified hook-free. Actual IDs, masks, settings, EOS/caps and timestamps are
saved. Output text decodes the generated body without stripping whitespace
or nonterminal special markers, omitting terminal EOS only, matching the
existing XSTest adapted baseline's prefix-scoring convention.

Model starts are journaled BEFORE each invocation;100 ceiling. No regeneration
of an existing phase, uncertain work or fatal stopped run. Busy resource checks
before load are no-call waits. Generation stage exit integrity is required for
packets or metrics. Completed counters do not hide uncertain starts.

Native judging uses root's process-local pinned Claude2.1.323 binary PATH,
same tools-off model/medium/isolation/audit contract and existing strict C/P/I
parser. Source-order grading membership is frozen before generation:12 groups
of8 plus4. Packet construction omits empty/whitespace answers, skips an entirely
empty group without a call, and never moves samples to other groups. At most13
first-valid batches/26 starts; one identical structural retry per batch only.
Auth/model/identity/hash/isolation/uncertain errors permanently stop, no rescue.
`--resume` is exact journal replay for an EXISTING grading stream; never rerate
a first-valid batch. All raw attempts/start/receipt hashes remain private.

## Reporting boundary

Headroom is NONEMPTY P+I under the frozen definitions. Overall denominator100,
per-type10. Empty outputs are mechanically reported, with no fabricated semantic
grade and no refusal numerator. Report C/P/I, missing/invalid labels (zero for
complete strict-schema acceptance), empties, caps, EOS, degeneration and prefix
disagreement. Prefix is secondary and cannot override Claude or distinguish P
from I; its disagreement denominator is the nonempty graded rows.

Fewer than5 closes `insufficient_observed_headroom`, with no alternate dataset.
At least5 reports `observed_headroom_requires_separate_registered_plan`: the
project remains authorized, but any later study needs its own prospective
design/review/registration. This runner launches no follow-up. There is no
extra cap/quality/prefix gate, controller, held-out use or claim of statistical
power. Samples have prior project exposure on other models. All100 answers
are development data, never held-out evidence.

Focused CPU checks:

```sh
.venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_refusal_headroom.py
```

They cover real pinned safe-source/license/split determinism, actual AST matcher
and golden metric, native template, fixed denominator/empty/P+I arithmetic,
partial batch membership, strict grading/model/auth behavior, first-valid receipt
replay, uncertain/budget rejection and100 fake-CPU generation starts with no
intervention or reserved IDs. No live model, GPU or native calls.

## Separate H14 sensitivity utility

`scripts/sarcasm_h14_sensitivity.py` is a different post-run analysis, not part
of this baseline. Its future destination is
`results/diagnostics/sarcasm_h14_sensitivity_01.json`; do not run it against an
incomplete sarcasm study. It computes only descriptive32/31 summaries, returns
not_applicable for a closed study with no heldout, preserves primary decisions
by reference and never launches a model, changes labels or reselects a method.
Its synthetic smoke checks covered exclusion/denominators, REFUSAL, lexicon,
optional descriptive ratings and duplicate/invalid-score rejection; no actual
study sensitivity output was created by the worker.
