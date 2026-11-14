# Separate rudeness / sarcasm studies on 4B

Authority: `docs/isolated_persona_traits_protocol.md`, with the common mechanical
contract in the frozen profanity protocol. Root owns reviews, preparation,
registrations and every GPU/network release. Worker runs CPU checks only.
No source/guard/global in any old module is patched. Pure old generation,
safetensors, pooling, delta and cache-hook helpers are reused unchanged.

```bash
.venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_persona_traits.py
.venv/bin/python -B -m scripts.persona_traits --help
```

Both extraction corpora are frozen before outputs. Shared16 development and24
confirmation tasks (eight factual), plus rubric, are byte-identical copies from
the profanity study. They are historically used tasks, not unseen evidence.
One annotated neutral/target span differs; outside-span characters are identical.
Actual first-span-word positions use integer thirds. Both sides have zero fixed
lexical profanity. Retention uses all eligible pairs: minimum24, at least four
declared style families, none above one third. Authored position/phrase quotas
are not reapplied after eligibility; attrition is reported. Related style-family
names do not establish semantic independence (notably unwelcome/closed discussion).

Model rendering is **helpful system plus row.user only**. `context` is judge
metadata and never enters the model input. All answer facts are in user text;
rudeness context describes a single-turn fictional exchange, without invented
history. Sarcasm context describes observations, not a tone label.

Each profile has independent config/output/registration:

- `--trait rudeness`: `outputs/rudeness-steering-01`, `results/rudeness_steering_01.json`.
- `--trait sarcasm`: `outputs/sarcasm-steering-01`, `results/sarcasm_steering_01.json`.

Rudeness runs first. Sarcasm live registration checks require a closed rudeness
report; no command automatically launches the other profile. Use the following
commands with `--trait rudeness`; sarcasm is a separately approved later run.
Fatal/incomplete rudeness blocks live sarcasm and needs a separate management
decision/amendment. No automatic restart or bypass of that predecessor rule.

After fixture/code acceptance, root prepares and commits initial registration:

```bash
.venv/bin/python -B -m scripts.persona_traits prepare --trait rudeness
.venv/bin/python -B -m scripts.persona_traits verify --trait rudeness
.venv/bin/python -B -m scripts.persona_traits packet --trait rudeness --phase eligibility
```

Initial `results/protocols/rudeness-steering-01.json` binds `manifest_sha256`.
Source/protected hashes include both corpora/configs, shared tasks, all1206 prior
protections and the complete closed profanity run/report/diagnostic. No prior
baseline or grades are substituted. Root host/network release uses the unchanged
pinned native2.1.323 PATH and audited tools-off model/flags:

```bash
PATH=PINNED_BIN:"$PATH" .venv/bin/python -B -m scripts.persona_traits judge --trait rudeness --phase eligibility --execute-judgments
.venv/bin/python -B -m scripts.persona_traits eligibility --trait rudeness
```

An eligibility null stops. Otherwise separate GPU release:

```bash
.venv/bin/python -B -m scripts.persona_traits extract --trait rudeness --execute-gpu
.venv/bin/python -B -m scripts.persona_traits preflight --trait rudeness --execute-gpu
```

Same original4B BF16/SDPA/nonthinking/helpful/seed42, batch4/cap256, physicalGPU0
UUID guard12,000MiB/10%; leave jobs untouched. Teacher-forced response-only FP32
means, blocks17/23, R median neutral mean norm, unit beta .05/.10/.20, token-norm
diagnostics, actual BF16 norms, fixed CPU-seed random unit and exact safetensors
round-trip remain unchanged. Trait identity binds metadata; wrong-profile vectors
fail. Actual zero hooks must match tokens and nonzero probes must fire. No setting
or fallback comes from the previous null or these unscored probe texts.

Commit `results/protocols/rudeness-strength-01.json` with `strength_sha256`, then:

```bash
.venv/bin/python -B -m scripts.persona_traits dev --trait rudeness --execute-gpu
.venv/bin/python -B -m scripts.persona_traits packet --trait rudeness --phase development
PATH=PINNED_BIN:"$PATH" .venv/bin/python -B -m scripts.persona_traits judge --trait rudeness --phase development --execute-judgments
.venv/bin/python -B -m scripts.persona_traits select --trait rudeness
```

Usable target means judge target>=2 plus content/coherence, no gross or mechanical
failure. Other two axes are crosstalk; lexical profanity remains diagnostic. Null
stops. Otherwise `lock` saves exact source/rating/vector/input hashes, root commits
`results/protocols/rudeness-selection-01.json` with `lock_sha256`, then releases:

```bash
.venv/bin/python -B -m scripts.persona_traits lock --trait rudeness
.venv/bin/python -B -m scripts.persona_traits final --trait rudeness --execute-gpu
.venv/bin/python -B -m scripts.persona_traits packet --trait rudeness --phase confirmation
.venv/bin/python -B -m scripts.persona_traits packet --trait rudeness --phase confirmation_repeat
PATH=PINNED_BIN:"$PATH" .venv/bin/python -B -m scripts.persona_traits judge --trait rudeness --phase confirmation --execute-judgments
PATH=PINNED_BIN:"$PATH" .venv/bin/python -B -m scripts.persona_traits judge --trait rudeness --phase confirmation_repeat --execute-judgments
.venv/bin/python -B -m scripts.persona_traits retention --trait rudeness --execute-gpu
.venv/bin/python -B -m scripts.persona_traits report --trait rudeness
```

Final generation occurs ONCE: four same-batch reload probes, then72 baseline/
selected/random outputs. Repeat judging reads those same outputs, with stable
opaque aliases and identical per-item payloads under rearranged seed504 batches.
Seeds501/502/503/504 are frozen before calls. No same-task conditions share a batch.
Second pass cannot repair/replace first. Primary target is present in BOTH passes;
content/coherence/factual correctness AND, gross/incoherence OR, off-target scores
MAX. Random raw target uses EITHER pass. Export per-pass/consensus, entanglement
counts, disagreements and ordinal kappa; undefined kappa is reported, not gated.

Per trait:40 first-valid batches, max80 calls (two structural attempts each),
230 generations,64 MC items and max64 authored forwards. No duplicate generation
for repeat judging. First-valid, uncertain-start, identity/hash/fatal and exact
journal resume rules remain fixed. No semantic retry or payload reframing.
Native live calls require root's network-enabled execution, never the sandbox.
Generated token divergence/length diagnostics are private per item and public
aggregate only, alongside normalization and norm fractions. No diagnosis changes
the fixed grid. Fresh capability uses the same48-item fixture and strict math
gate, with language seed0 restored. Every old source/output stays unchanged.
