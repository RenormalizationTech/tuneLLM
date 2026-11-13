# Tone pilot runbook

Core is frozen for `outputs/tone-01`; the coordinator owns GPU windows and
registration commits. The worker does not run GPU stages. Do not change frozen
core, fixtures, rubric, protocol, or old outputs to accommodate later results.
All model reads are local/offline; there is no training, LoRA or model download.

**tone-01 feasibility stop:** root completed the registered 160 BF16 extraction
responses. Only five of 80 pairs can mechanically meet the unchanged length and
literal rules, before AI judgments. It cannot retain 50; do not judge, extract
vectors, or run development/final stages for tone-01. A separately approved and
registered follow-up is required; no eligibility relaxation is implemented here.

**Tone steering is now stopped under the registered B rule.** Tone-02A's
mechanical upper bound was 31/80, triggering the single permitted B rewrite.
Root completed B with 46/80 mechanically possible pairs: literal coverage 76,
length-window pass 49, uncapped/nonempty 80, 31 below 0.85 and none above 1.25;
median token ratio 0.880476. All 80 complied with the instructed word budget,
which remains diagnostic only. This cannot yield the required 50 pairs, so no
experiment judging, vectors, development/final runs or further construction is
authorized. Frozen tone-01/02/02B artifacts and criteria remain unchanged.

## Completed CPU checks

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_tone_common.py tests/test_tone_workflow.py tests/test_tone_judge.py
.venv/bin/python -m scripts.tone_steering validate-fixture
.venv/bin/python -m scripts.tone_steering --help
.venv/bin/python -m scripts.tone_judge --help
```

The optional cached-tokenizer CPU test never loads model weights. Random tiny
CPU Qwen3 tests verify actual cache/padding hooks, zero-vector parity and cleanup.
They are unit-test models, not experimental generations or extraction responses.

## Original registered stage order (reference; tone-01 stopped)

```bash
.venv/bin/python -m scripts.tone_steering prepare --run outputs/tone-01
```

The coordinator audits template separation and commits a registration containing
`manifest_sha256` and `semantic_fixture_audit_passed: true`, plus core source and
fixture/rubric/protocol hashes. Preparation copies inputs/code and streaming
model hashes into an ignored fresh run directory. It never runs a model. GPU
selection is frozen in the config (`0` or `1`); the device must have an explicitly
approved window. Every GPU stage checks the same UUID, records physical/logical
mapping, name, driver/CUDA, versions, timing and allocator memory. Existing jobs
must remain untouched. The same device is used for each comparison.

With an approved window, only the coordinator invokes GPU stages, for example:

```bash
.venv/bin/python -u -m scripts.tone_steering extraction-generate \
  --run outputs/tone-01 --registration results/protocols/tone-01.json --execute-gpu
```

Extraction generates neutral/warm responses on 80 fixed family-round-robin
scenarios. No responses are authored by the worker. Before judging, check the
mechanical upper bound; if fewer than 50 could pass, stop and report feasibility.
Do not judge tone-01. For an eligible separately registered run, the packet and
import interface is:

```bash
.venv/bin/python -m scripts.tone_steering packets --run RUN --packet-stage extraction
.venv/bin/python -m scripts.tone_steering import-ratings --run RUN \
  --packet-stage extraction --kind content --ratings CONTENT_SUBMISSION
.venv/bin/python -m scripts.tone_steering import-ratings --run RUN \
  --packet-stage extraction --kind warmth --ratings WARMTH_SUBMISSION
```

Give a fresh judge only frozen `rubric.json` and anonymous `*_packet.json`.
Never supply `keys.local.json`, paths, source code, condition names or result
summaries. The judging directory is mode 700 and keys mode 600. Packet manifests
bind source predictions and condition files; imports require exact complete IDs,
boolean coverage/task completion, claim counts, scales and judge disclosures.

Successful extraction requires exactly the first 50 eligible pairs, with all
families/attrition recorded. Eligibility never silently relaxes. The `extract`
GPU stage re-encodes both responses in identical neutral contexts, uses verified
response-only offsets, excludes template/EOS/padding and whitespace-only boundary
crossings, and saves raw mean-difference vectors for layers 11/17/23. Non-whitespace
crossings fail. All hooks detach on success or failure.

`preflight` times 30 development default responses at cap 256; more than 10% cap
hits selects 320. It checks a zero vector at multiplier one through real hooks,
then records layer17/m1 hook firing and hidden/logit delta on fixed development
rows. Unchanged tokens are permissible for that nonzero smoke. Copy successful
`preflight/preflight.json` to a tracked budget file and commit it exactly before
`development --budget BUDGET --registration REGISTRATION --execute-gpu`.

Development reuses final-cap preflight default outputs and generates nine
candidates; then create/import development packets and run `select`. Gates use
independent AI fact/event/task judgments, with literal disagreements separate,
plus length, truncation, empty-output and four-gram degeneration limits. Selection
uses only development warmth and the fixed multiplier/score/layer tie rules.

`lock-final --run RUN --output TRACKED_LOCK` creates an immutable final manifest.
Review and commit it before `final --lock TRACKED_LOCK --representation bf16`
and the matching `nf4` command (both require registration and `--execute-gpu`).
No final prompt content is read by selection. Final lock verification checks all
selection-recorded hashes, not just vectors. Saved NF4 forbids requantization and
uses the exact BF16 direction/settings. Zero selection reuses default outputs.

Create/import separate `final-bf16` and `final-nf4` packets with `--lock LOCK`,
then `summary --run RUN --lock LOCK --output NEW_SUMMARY`. Summaries report family
bootstrap intervals, content/task/literal diagnostics, descriptive gates and
costs. Final outcomes never cause reselection. No human or broad-capability claim.

## Independent Claude wrapper

The helper is outside the frozen core. Its hash and isolation audit must be
reviewed/registered before any real judgments. It sends at most 20 anonymous
items per fresh call from a neutral empty `/tmp` directory. Tools, MCP, commands,
plugins/customizations and session persistence are disabled. It uses argv, never
a shell; no resume/session/project file is passed. Requested judge is
`claude-opus-5-5`, requested effort `medium`, Claude Code `2.1.323`; actual
`modelUsage` identifiers must match, without silent fallback. Actual effort and
dated snapshot are marked unavailable when not exposed.

The authorized one-call isolation audit contains **no experiment examples**:

```bash
.venv/bin/python -m scripts.tone_judge audit \
  --output outputs/tone-judge-isolation-audit-01
```

This old-version call occurred once. The original audit failed JSON parsing
because its otherwise matching answer used one complete Markdown JSON fence.
The original `audit.json` and `attempt.json` remain intact. An offline-only
reparse exists separately under `outputs/tone-judge-isolation-audit-01-parsed`;
it makes no second call and records the original hashes and both implementation
hashes. Both records are historical: neither authorizes future judging after
the final wrapper's Git-isolation and resume changes.
The answer reports no project identity/files/project memory but does disclose
automatic environment/account metadata, so this is an isolation diagnostic,
not proof that all runtime metadata is absent.

The final wrapper removes every inherited `GIT_*` variable, alongside project
session variables. Before every inference call, in the same neutral temporary
cwd and sanitized environment, `git rev-parse --git-dir` must report that the
directory is not a repository (exit 128). Attempts record the command, return
code, clean-environment boolean and removed variable names only. If this check
fails, Claude is never invoked. A separately authorized single new dummy call
under the frozen final wrapper uses a fresh directory:

```bash
.venv/bin/python -m scripts.tone_judge audit \
  --output outputs/tone-judge-isolation-audit-02
```

This final-version call completed once and passed. Git returned 128 outside a
repository with all inherited `GIT_*` variables removed; the requested and
actual modelUsage identifier was `claude-opus-5-5`. Its local `audit.json` binds
the raw attempt, exact stdin and final wrapper SHA256
`63361396d908d219d2886a35d3bdb98d7a783e5b209d804e5aa791c45da7d4d5`.
The earlier failed and offline-reparsed audits remain untouched. Root/adviser
review is still required for any future use; the tone stop prohibits actual
experiment ratings regardless of this infrastructure result.

Future real judging requires root acceptance of that final audit. The helper
checks its current source hash, argv, requested/actual model IDs, original raw
attempt hash, exact stdin hash and Git proof; historical offline reparses fail
this guard. Raw audit/attempt files remain local under ignored outputs; do not
copy account identifiers or full answers into tracked summaries.

After separate approval, validate a packet offline; no judge call occurs:

```bash
.venv/bin/python -m scripts.tone_judge validate --packet PACKET --rubric RUBRIC \
  --kind content --output UNUSED_PATH
```

For a future approved eligible run only, use `judge` with the same arguments,
`--execute-judgments`, a fresh output directory, and
`--isolation-audit AUDIT_DIRECTORY/audit.json`. Import only its `submission.json`
through the frozen core. At most **two attempts total**, identical stdin, only
for malformed/incomplete outputs; use the first valid result. Model mismatch,
infrastructure/auth failures and errors never trigger fallback or retry. All
attempt files, hashes, exact batch IDs and canonical stdin SHA256s are retained.

For a future approved run, `--resume` requires the same output directory, raw
packet/rubric/audit file hashes, canonical packet/rubric/stdin hashes, wrapper,
CLI version and argv. An exclusive file lock prevents concurrent calls. The
helper replays and validates saved attempts and reuses the first valid response,
even if interrupted before batch/global summaries were written. Only pending
batches may call Claude; two total attempts remain the maximum, never reset by
resume. Complete results, receipts and previous interruptions are preserved.

Resume never retries saved auth/model errors. An attempt-start record without
a completed response means the remote outcome is unknown: stop and consult the
coordinator, retaining start/interruption files. Do not assume that rerunning
grants an extra attempt or permission to re-judge. Known malformed/incomplete
outputs alone may consume the one remaining retry. Final JSON artifacts are
written atomically without overwriting existing files.

Later, `swap-audit --kind warmth --primary-submission PRIMARY ...` judges a fixed
10% reversed-order sample in separate batches/directories. Its agreement and
A-preference shares are diagnostic only. Never import it over primary scores or
use it to select a different candidate. No actual ratings are authorized now.
