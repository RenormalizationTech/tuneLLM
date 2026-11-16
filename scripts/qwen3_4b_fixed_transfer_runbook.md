# Qwen3-4B: three isolated fixed GCAD transfers

One runner, one config, one Qwen3 port; no tuning, selection or new refusal
compute. Root owns review, preparation approval, commits, R1/R2, live releases
and publication. A command or registration is not permission for another stage.

Scientific authority: `docs/qwen3_4b_fixed_transfer_plan.md` and
`results/reviews/qwen3_4b_fixed_transfer_design_review.md`, plus the bound port
and implementation reviews. Run: `outputs/qwen3-4b-fixed-transfer-01`.
Aggregate: `results/qwen3_4b_fixed_transfer_01.json`.

## Frozen settings and scope

Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`, native
BF16, SDPA generation and separate eager extraction loads. Physical GPU0 UUID
`GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b`; free >=24000MiB, utilization <=10%.
Do not touch other workloads. No install or download. The registered GCAD
Python3.11 runtime is required for preparation and GPU stages.

The proportional block rule maps 7B blocks9..19 to 4B blocks12..25, inclusive.
Scale1.5, inverseFalse; b is PF5, impolite3.5, SARC2. These are fixed settings,
not matched effective doses or validated 4B strengths. All banks are freshly
fit on this checkpoint; no 7B response, CSV, score or tensor enters a bank.
The three original 7B outcomes are reference results, not predictions for 4B:
PF retained5; author impolite3.5 (no random arm there); SARC retained2.

All prompts explicitly use `enable_thinking=False`. Qwen3's user-only native
rendering has no default system. Elicitation uses the exact approved persona
systems. Greedy PF/SARC evaluation has no explicit system. The prompt includes
the empty closed think block; generated think tags are separately counted.
Decoding pins the 7B-resolved values: repetition penalty1.05, top-p.8, top-k20,
temperature1. Elicitation cap256 sampled B1; greedy evaluation cap160 B1;
impolite cap500 sampled in three matched20 sub-batches; probes cap32. Resolved
GenerationConfig values, actual tokens/masks/EOS/cap flags and audits are saved.
Both EOS151645 and151643 matter; tokenizer PAD is set to EOS151645, while the
generation config retains its saved PAD151643. No truncation repair/reroll.

The port preserves the author gate, update and pooling. Changes are native
head_dim128, per-head Q/K normalization, Qwen3 rotary helpers, fail-closed
geometry and applied/skipped counters. Delta/key are raw, never unit-normalized.
The Apache license and derivative NOTICE accompany the new module. All frozen
shared helpers and author source remain unchanged.

## Preparation and R1 (CPU only, after root accepts source)

```bash
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.qwen3_4b_fixed_transfer prepare
```

Preparation hashes local model files and source/data/reviews/runtime. It loads
only the cached tokenizer, computes the block rule, checks exact normalized
extraction/evaluation overlap (must be zero; never drops anything), and saves
all renderings/IDs, resolved decoding, full author question groups, and all
8 extraction-availability branches. It validates all grading-plan shapes and
opaque ID/group separation before creating the new root.

Root commits R1 `results/protocols/qwen3-4b-fixed-transfer-01.json`:

```json
{"manifest_sha256": "..."}
```

After committing R1 and its bound sources, verify before the first call:

```bash
.venv/bin/python -B -m scripts.qwen3_4b_fixed_transfer verify
```

No registered stage may run from an uncommitted source or registration. All
live commands verify sources/inputs; GPU stages verify the model files on
entry and exit. There is no snapshot overwrite or GPU resume. Missing ordinary
prerequisites return Wait without calls or stage/stop-marker writes. Hash,
budget, isolation, identity, uncertain-start and real call failures are fatal.
Report never writes a terminal marker and returns incomplete for pending grades.

## Elicitation, separate grading, eligibility, extraction

Run the following for each trait in fixed order `profanity`, `impolite`,
`sarcasm`, only with root's corresponding release. Host GPU execution must use
`require_escalated` through the operator tool. Example for the first trait:

```bash
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -u -m scripts.qwen3_4b_fixed_transfer elicitation --trait profanity --execute-gpu
```

Use umask077 and a fresh, non-overwritten private operator log. Native grading
**must** use network-enabled `sandbox_permissions=require_escalated`; the
process cannot detect outer permission mode. Process-local PATH pin:

```bash
PATH="$PWD/outputs/sarcasm-method-transfer-01/runtime-bin:$PATH" .venv/bin/python -B -u -m scripts.qwen3_4b_fixed_transfer judge --stream elicitation_profanity_trait --execute-native
PATH="$PWD/outputs/sarcasm-method-transfer-01/runtime-bin:$PATH" .venv/bin/python -B -u -m scripts.qwen3_4b_fixed_transfer judge --stream elicitation_profanity_coherence --execute-native
.venv/bin/python -B -m scripts.qwen3_4b_fixed_transfer eligibility --trait profanity
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -u -m scripts.qwen3_4b_fixed_transfer extraction --trait profanity --execute-gpu
```

Use the corresponding trait names for the other two streams/stages. Each trait
has200 newly elicited answers and25+25 separate trait/coherence batches. Mask:
positive trait>=50, negative<50, both coherence>=50, no trait REFUSAL; retain all
passing pairs. E<10 is extraction_null for that trait, with no bank/re-elicitation;
other traits continue. Eager extraction costs exactly3E source forwards,
charged durably before each call. Delta[36,2560], key[36,8,128], demand[36] are
saved as FP32 `bank.safetensors` with model/trait/input/port/tensor hashes.

Pooling uses the author's decoded-response re-tokenization and original
prompt_len. Per-item response-prefix mismatches are recorded, not repaired or
used to change pooling. Saved prompt IDs, system start/prefix and non-empty
response checks remain fatal. `encodings.json` registers the actual mismatch
count at R2. This boundary limitation is disclosed, not hidden or retuned.

Native runtime comes from `configs/qwen3_14b_baseline.json`: the same tools-off
Claude2.1.323 binary SHA604d29a6869409106b6a1110bdadcc8b72e8654a5a865d23258f938a696e348e,
requested `claude-opus-5-5`, medium effort, neutral cwd, Git isolation and no
session/project tools. Both requested/observed models, binary/version and audit
source are in the aggregate. Integer0..100 plus JSON/private reason is an
adaptation of the author evaluator. Semantic trait REFUSAL is null/conservative0,
not a schema retry; coherence is always numeric. At most2 identical structural
attempts, first valid frozen; no auth/network/model/isolation retries. `--resume`
means validated replay of accepted batches, never regrading uncertain starts.

## Mechanics, random controls and R2

After all three eligibility files and each available bank exist:

```bash
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -u -m scripts.qwen3_4b_fixed_transfer mechanics --execute-gpu
```

Anchor is the first available bank in PF/impolite/SARC order. No available bank
means no mechanics/evaluation load: publish all extraction_nulls via report.
Otherwise probes are2 ordinary,2 real b0,2 per available bank,2 anchor no-cache;
max12 generation calls,5 explicit logit forwards. Zero tokens/logits must match;
every available bank must change hidden/logit states, cover all active sites
and cached steps, and clean up. Any skipped hook—including zero/random—is a
pipeline error. Cache/no-cache token agreement is diagnostic only. A failed
mechanics result is port_or_mechanics_failure, not a method null.

Random controls use dedicated CPU generators, one raw-delta norm match per
layer, retaining keys/demands/gates. Seeds cycle by fixture index: PF9801..3,
SARC9811..3, impolite9821..3, with layer_seed=question_seed+1000*layer. No global
RNG is consumed. All bank/control tensors are safetensors; metadata saves
per-row/per-layer hashes, norms and seeds. There are no new torch.save/load
artifacts. Fresh banks and controls roundtrip exactly before use.

Root commits R2 `results/protocols/qwen3-4b-fixed-transfer-evaluation-01.json`
exactly as `evaluation_registration_expected(Context())`:

```json
{"manifest_sha256": "...", "files_sha256": {"...": "..."}, "available_traits": ["..."], "no_selection": true}
```

Its file map binds all three eligibility records, each available bank/metadata/
encodings/extraction integrity, mechanics run/result/integrity, and the realized
controls plus metadata. No evaluation until passing mechanics and committed R2.

## Fixed evaluation and grading (no selection)

```bash
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -u -m scripts.qwen3_4b_fixed_transfer evaluation --execute-gpu
```

Greedy evaluation generates one shared32 baseline plus selected/random32 for
each available PF/SARC bank. Impolite is baseline/impolite/random,60 each. All
three use the same sub-batches by conversation index mod3,20 each, seed42 reset
before each. Author full20x10 groups are sampled/shuffled before taking first
turns; no shortcut changes questions. The unchanged author generation function
runs through explicit nonthinking tokenizer and trace proxy. Duplicate literal
questions across source conversations remain disclosed; batching separates
conversation IDs/arms, not every duplicated literal string.

Grade active primary streams (new independent tools-off calls, PATH and host
network policy above):

```bash
.venv/bin/python -B -u -m scripts.qwen3_4b_fixed_transfer judge --stream profanity_trait --execute-native
.venv/bin/python -B -u -m scripts.qwen3_4b_fixed_transfer judge --stream sarcasm_trait --execute-native
.venv/bin/python -B -u -m scripts.qwen3_4b_fixed_transfer judge --stream greedy_coherence --execute-native
.venv/bin/python -B -u -m scripts.qwen3_4b_fixed_transfer judge --stream impolite_trait --execute-native
.venv/bin/python -B -u -m scripts.qwen3_4b_fixed_transfer judge --stream impolite_coherence --execute-native
```

Omit only streams absent from the pre-frozen active branch. There is no dynamic
filter/repack. SARC descriptive grading **must be last**, after ALL active
primary streams above are accepted:

```bash
.venv/bin/python -B -u -m scripts.qwen3_4b_fixed_transfer judge --stream sarcasm_impolite --execute-native
.venv/bin/python -B -m scripts.qwen3_4b_fixed_transfer report
```

Each ordinary plan is fully packed8, except impolite's last batch4. Maximum
252 first-valid batches/504 starts:150 elicitation,12 PF,12 SARC,12 descriptive
impolite,20 shared coherence,23+23 impolite. Global model ceilings952 generations,
900 teacher-forced forwards,5 explicit logits; no MC or new refusal calls.

PF and SARC use unchanged gates with `invalid` replaced by the UNION of old
empty/word-bigram<.6 and think tags, counted once. Frozen invalid and tag counts
are also reported separately. All14 site measurements are required for active
norms; baseline norm is explicitly0. Hook counters are excluded from norm sums.
Impolite's author numeric-mean +30 contrast and new random/coherence/invalid
qualification are separate. Author-only success is author_effect_only; only
both passes qualify a working controller. Gate keys and margins are exported.

SARC descriptive impoliteness and H14-excluded31 are non-gating. If only the
last descriptive stream fails after all primaries, primary results remain
publishable as primary_complete_descriptive_incomplete, with the fatal stop
and explicit failure; execution is never labelled complete. Other integrity/
fatal failures remain incomplete. No scientific controller is selected by a
report or by descriptive coupling. Passing banks alone are candidates for a
later separate 4B release; root owns publication.

The earlier refusal result is carried forward without new compute: subtraction
partial effect with failed quality rule, projection not selected, no retained
refusal controller. Old 4B PV nulls and all 7B records stay unchanged.

## CPU verification (no environment modification)

The GCAD env lacks pytest. Use pure pytest as a late import fallback, with
plugins disabled; target-installed torch/Transformers win. Do not install into
or mutate the frozen environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B - <<'PY'
import sys
sys.path.append('.venv/lib/python3.10/site-packages')
import pytest
raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider', 'tests/test_qwen3_gcad_port.py', 'tests/test_qwen3_4b_fixed_transfer.py']))
PY
```

Port receipt:15 checks passed independently in target runtime (root5.21s).
Runner checks cover all8 branches and all-null/one-bank/all-pass reports,
constructor/real tiny36-block hooks, 7B-shaped random parity, safetensors I/O,
retokenized-boundary diagnostics, invalid union, zero baseline norm, R2 tamper,
precondition waits/no writes, descriptive-last/failure status and actual first-
valid/retry/uncertain/budget journal behavior. No cached model weights are loaded.
The unrelated pkg_resources/sre_constants deprecation warning comes from the
late pytest fallback; no environment patch is applied.
