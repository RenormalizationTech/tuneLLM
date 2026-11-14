# Persona rating calibration 01 (CPU / judge only)

Authoritative design: `docs/persona_rating_calibration_protocol.md`. This stage
contains no target-model loading, GPU, generation, steering or extraction code.
Root owns design/fixture reviews, registration, execution release and narrative.
Do not prepare until fixture and code reviews are accepted. No new dependencies,
network downloads or isolation audit are needed.

```bash
.venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_persona_rating_calibration.py
.venv/bin/python -B -m scripts.persona_rating_calibration --help
```

New fixture: `evaluation/persona_calibration/texts.json`, 48 authored fictional
or impersonal texts, six in every profanity/rudeness/sarcasm binary cell. Author
intentions are not human ground truth. Context describes events/referents, not
the intended style. Mild unambiguous swearing counts as clear presence, not an
ambiguous rating. Nonrude sarcasm examples concern objects/situations; playful
person-directed teasing is not tested. No claim of that distinction is made.
`assess`, `class`, `hello`, `scrap` and `scrapbook` are innocent lexical traps.
The non-slur lexicon enumerates whole-word variants; no stemming/substrings.

After root review:

```bash
.venv/bin/python -B -m scripts.persona_rating_calibration prepare
.venv/bin/python -B -m scripts.persona_rating_calibration verify
```

Fresh private run: `outputs/persona-rating-calibration-01`. Snapshot binds source,
fixture/rubric/protocol/reviews, frozen pass orders (421/422), six batches of eight
per pass, local cell compositions, software metadata, historical evidence and the
existing audited native binary/wrapper. No author labels, cell counts, shuffle
seed, prior scores or pass identity enter the judge payload. Every batch is a
fresh isolated context. Raw labels, payloads, model usage IDs and attempts stay
local. The source schema rejects bool-as-int, missing/duplicate/extra IDs, extra
fields and out-of-range ratings.

After the exact registration is committed and root releases judgment execution,
prepend root's private pinned-native directory to PATH only for this process:

```bash
umask 077
PATH=PINNED_BIN:"$PATH" .venv/bin/python -B -m scripts.persona_rating_calibration judge --execute-judgments > outputs/persona-rating-calibration-01-grading.log 2>&1
.venv/bin/python -B -m scripts.persona_rating_calibration report
```

The audited CLI is native2.1.323, requested model `claude-opus-5-5`, medium effort,
all tools disabled via unchanged foundation argv/environment/isolation. Runtime
pin comes from the existing 14B config's judge runtime record. Default CLI version
may differ: a mismatch stops, it does not authorize replacement or a fresh audit.
No API calls occur during CPU checks/preparation. Live authorization is not
assumed from the existence of a credential file.

Maximum two structural attempts per batch, first valid frozen: 12 batches and at
most24 calls. Model identity absence/mismatch, unreadable service envelope, API,
auth, infrastructure, isolation, hash or uncertain-start failures are fatal.
No semantic retry, rephrasing or new label choice. `--resume` replays only the
exact existing journal, preserves valid results and remaining attempt budget,
and never continues a stopped run. Unexpected or post-first-valid attempts fail.

Presence is ordinal score>=2. Each pass/trait needs41/48 intended agreements;
both passes need zero gross failures. Lexical agreement needs46/48. Every trait
needs quadratic weighted kappa>=0.7 across aligned passes; undefined fails.
Fractions enforce the exact kappa boundary. A measured first-pass axis failure
stops before additional calls. The report distinguishes incomplete measurement
from a complete outcome and retains observed early failures. No result authorizes
an automatic profanity pilot or any steering call.

Public `results/persona_rating_calibration_01.json` contains only aggregate
agreement/gate/kappa/lexical counts, per-cell presence, frozen batch compositions,
hashes, bounded-call provenance and limitations. It contains no individual IDs,
raw contexts/texts or per-item ratings. Passing permits preparation of a separately
reviewed isolated profanity-on-4B protocol only. Rudeness, sarcasm, joint work and
14B transfer remain separate later stages.
