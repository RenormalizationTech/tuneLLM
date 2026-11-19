# Qwen3.8-27B sarcasm GCAD: review 10 (development null and study closeout)

Advisor review, 2026-11-19, of these files:
- in `outputs/qwen38-sarcasm-gcad-01`: `development/`,
  `judgments/development/`, `selection.json`, the `random-*.safetensors`
  files, `closeout.json`, `approvals/`, `counters/` and `gpu-sessions/`;
- `results/qwen38_sarcasm_01.json`;
- the study protocols in `results/protocols/`.

All checks were CPU-only, using JSON, hashes and saved tensors. No model, judge
or GPU call was made. Frozen files were not edited.

## Verdict

**Outcome: `development_null`. Independently confirmed. The study is closed
as a valid, prospectively registered negative result.**
- Neither registered GCAD strength (b = 1 or 2) passes the unchanged
  development gates.
- No final stage was authorized, and no final rows exist.
- The runner closeout and the authoritative C6 analyzer agree.
- No deterministic implementation failure occurred, so this is a scientific
  null under plan §9a, not an implementation stop.

## Provenance

**Registration order.**
- Development was registered in
  `qwen38-sarcasm-gcad-development-01.json` at 05:56:12 UTC. The first
  development row started at 06:02:09 UTC.
- The registration binds the following, each of which I checked against the
  current file:
  - R0 manifest `13254530…`;
  - development approval `d9b817b4…`;
  - mechanics `0d5bfe32…`;
  - bank `4fd9f86b…`;
  - C6 registration `6e849b72…`;
  - review 07 addendum `5e742afb…`;
  - review 09 `3f6161c4…`.
- The addendum was registered at 01:48 UTC, before any development data
  existed.

**Rows.** There are 128 development rows, in the same order as frozen
`inputs.json['development']`, and every frozen field equals the frozen value.
All 128 row receipts match.

**Judging.**
- There were 32 batches, all accepted on the first attempt, with no retries.
- `modelUsage` is only `claude-opus-5-5`.
- `scores.json` covers exactly the 128 IDs and agrees with `trait.json` and
  `coherence.json`. There are 0 refusals.
- The study has no failure records and no starts without a completed row.

## Independent recomputation

I recomputed every figure from the rows and `scores.json` directly, not with
the frozen helpers. Invalid uses the frozen `text_checks` definition.

| Arm | SARC (conservative) | n ≥ 50 | Coherence | Invalid | Caps / EOS | Mean tokens |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.000 | 0 | 85.719 | 0 | 32 / 0 | 160.0 |
| prompted | 69.219 | 29 | 80.344 | 0 | 32 / 0 | 160.0 |
| GCAD b=1 | 0.000 | 0 | 85.625 | 0 | 31 / 1 | 159.6 |
| GCAD b=2 | 0.188 | 0 | 84.906 | 0 | 32 / 0 | 160.0 |

**Gates for b=1 and b=2.** Both fail on two counts:
- SARC gain +20: b=1 gives +0.00 and b=2 gives +0.19, both fail;
- at least 8 answers with SARC ≥ 50: both have 0, both fail.

Both pass coherence ≥ 50, coherence ≥ baseline − 20 and invalid ≤ 3. Neither
arm passes all gates, so no strength qualifies, `selected = {}` and
`selection_null = true`. This equals `selection.json`, including every gate
boolean and every helper metric (lexicon hits are 0 in all arms).
`prompted_validity_concern` is false, because prompted 69.2 ≥ baseline + 20.

**Bindings.** `selection.json` binds the current development `scores.json`
(`3492af61…`) and the bank.

**Random controls.** They were created at selection as the runner always does,
but they were never used. Their hashes match. Per-site ‖Δ‖ equals the bank's
to within 2.4e-7. K_pos and D̄ are identical to the bank's.

## Closeout consistency (C6 registration rule)

- The runner `closeout.json` gives `development_null`, bound to selection
  `aa7ef7b2…`.
- `results/qwen38_sarcasm_01.json` (sha256 `00b155e6…`):
  - `outcome = development_null`;
  - `registration_verified = true`;
  - the analyzer, test and R0 hashes equal the C6 registration;
  - its `development` block is identical to `selection.json`.

  Under the registered rule, the outcome and gate booleans are equal, so there
  is no implementation stop.
- `before_first_final_start` is null, which is correct because no final run
  exists. No `final/` directory exists either. The PM's final-phase seal was
  therefore never needed.

## Budget ledger (from the analyzer, consistent with my counts)

| Item | Used | Registered maximum |
| --- | --- | --- |
| Generations | 340 (smoke 6, elicitation 200, mechanics 6, development 128) | 468 |
| Generated tokens | 71,908 completed (72,064 reserved) | 92,544 |
| Teacher-forced passes | 182 (2 + 180) | 302 |
| Logit forwards | 8 | 8 |
| GPU process wall time | 12,370 s = **3.44 h** | 8 h stop |
| Judge starts | 82 study + 1 audit, in 82 batch groups | 260 starts, 130 batches |

The GPU wall time breaks down as follows: smoke 71 s, elicitation 9,186 s,
extraction 110 s, mechanics 51 s and development 2,953 s. Development
generated about 6.9 tok/s including the load. The design estimated about 1.7–3.2
GPU-h of generation and 6–9 h end to end. Actual use was 3.44 GPU-h, because
elicitation ran slower than estimated.

## What the null does and does not show

- **The judge pipeline can detect sarcasm on these questions.** The prompted
  reference scores 69.2, with 29/32 answers at SARC ≥ 50, under the same
  judge, rubric and decoding. The null is therefore not an insensitive-judge
  artifact.
- **The steering was active, not a no-op.** All 32 answers at b=1, and all 32
  at b=2, differ token for token from baseline. The median first divergence is
  at token 18 for b=1 and token 10 for b=2. Mechanics parity, coverage and the
  nonzero effect were verified (reviews 04 and 09). The steering changed the
  text without making it sarcastic, and coherence stayed close to baseline,
  at −0.1 and −0.8.
- **What cannot be concluded.**
  - The data cannot distinguish "the direction is wrong for 27B" from "the
    dose is too small at the registered strengths". Realized shift energy is
    unmeasured (review 07 addendum). Raw ‖Δ‖ is 0.85–1.36, small compared with
    the attention outputs (review 09).
  - No higher strength may be tried or claimed in this study. Doing so would
    be post-hoc tuning, and the user and PM ruled it out.
  - The result is specific to this NF4 representation, the cached decoding
    path, the 6 full-attention sites, matched-pair extraction from capped
    elicitation replies (review 08: 197/200 hit the cap), and the
    `claude-opus-5-5` judge.
- **Fixture role.** The final fixtures were never used. Their reused
  cross-model status is moot for this study.

## Limitations the closeout must state

1. NF4 representation. The historical NF4 cached-vs-full-prefix difference was
   at most 0.125. All arms used the identical cached path at batch 1. No new
   steered cache comparison was made (addendum).
2. Realized shift norms and energy are unmeasured (addendum). Only raw per-site
   ‖Δ‖ is reported, with no energy claim.
3. The elicitation and bank use capped responses (197/200 at 256). All
   development answers were judged on text capped at 160 tokens: 127/128 hit
   the cap and 1 ended on EOS.
4. Judge payload envelope and schema differ from the historical 7B/4B format
   (C5). The rubric bytes are preserved.
5. The judge audit was a single negative isolation probe.
6. FP32 accumulation and raw-trajectory teacher forcing are registered port
   differences from the 4B/7B pipelines.
7. Elicitation, development and the fixture follow the existing sarcasm
   protocol. Known x18↔H14 topical overlap concerns only the final fixture,
   which was never run.

## Closure

The study is complete. Further GPU or judge calls, strength changes or
re-analysis are not warranted under this registration. Any follow-up, such as
a dose-response study at higher strengths or measured realized energy, would
need a new, separately registered study with fresh development data. It must
not be presented as a continuation of this result.
