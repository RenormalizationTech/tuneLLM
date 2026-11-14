# Single infrastructure completion of calibration 01

Authority: `docs/persona_rating_calibration_completion_protocol.md`. Root owns
review, registration and network-enabled launch. Worker preparation/live calls
are not authorized. No old source, fixture, stop marker, report or output changes.
This wrapper has no GPU, target-model, generation or steering stages.

```bash
.venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_persona_rating_calibration_completion.py
.venv/bin/python -B -m scripts.persona_rating_calibration_completion --help
```

After acceptance, root alone prepares/verifies the new snapshot:

```bash
.venv/bin/python -B -m scripts.persona_rating_calibration_completion prepare
.venv/bin/python -B -m scripts.persona_rating_calibration_completion verify
```

Fresh output: `outputs/persona-rating-calibration-completion-01`. Preparation
requires exactly one parent start, one failed return1/is_error response with
EAI_AGAIN and empty modelUsage, no rating response and no accepted pass. It binds
the entire failed parent, its incomplete public report and historical protections.
Text/rubric/plans/lexicon are unchanged; all first-valid labels must be new because
the parent had none. The original fixture, payloads, gates, native2.1.323 binary,
requested Claude model, medium effort and tools-off environment stay unchanged.

`inputs/budget.json` is immutable and hash-bound through the new manifest and
grading binding: parent1, new maximum23, cumulative maximum24. The old binding's
maximum_total_calls24 is a legacy overall upper bound, not24 new calls. Before
every potential new invocation, verify the budget and append one exclusive bound
charge. Only then may the unchanged foundation create its start marker. At
exhaustion, neither charge nor start for another invocation is written. Valid
replays spend nothing. A charge interrupted before a durable start is retained
as conservatively spent and is fatal; it is never silently reissued.

After committed registration and a separate root release, use the host execution
permission with process-local pinned native PATH, preserving the existing argv:

```bash
umask 077
PATH=PINNED_BIN:"$PATH" .venv/bin/python -B -m scripts.persona_rating_calibration_completion judge --execute-judgments > outputs/persona-rating-calibration-completion-01-grading.log 2>&1
.venv/bin/python -B -m scripts.persona_rating_calibration_completion report
```

Do not run live judging in the network-restricted sandbox. No wrapper, native
environment, audit, payload or model changes are made to obtain completion.
Maximum two structural attempts per batch; first valid frozen. Exact journal
`--resume` can reuse completed receipts, but cannot continue a stopped run or
repeat uncertain starts. Infrastructure/auth/model/isolation/hash failures stop
for user input; no further completion is authorized. Budget exhaustion exports
incomplete measurement without another call. A first-pass measured axis failure
also stops without additional grading.

Public `results/persona_rating_calibration_completion_01.json` contains aggregate
agreements/kappa/lexical and per-cell counts, frozen batch compositions, hashes,
first-valid provenance and explicit parent/new/cumulative counts. Actual start
counts and conservatively charged slots are distinguished. Raw texts, individual
ratings, private command responses and credentials stay ignored. Parent hashes
are checked before and after calls/export. Passing permits separately reviewed
profanity preparation only; it authorizes no activation experiment.
