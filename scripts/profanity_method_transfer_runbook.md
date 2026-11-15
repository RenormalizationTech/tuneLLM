# Three-method PF transfer: root execution runbook

New comparison only, governed by `docs/profanity_method_transfer_protocol.md`.
Run: `outputs/profanity-method-transfer-01`; public result:
`results/profanity_method_transfer_01.json`. Historical sources, data, nulls
and reports remain unchanged. The worker implemented and CPU-tested only;
root owns preparation, registrations and every live release.

Use the isolated GCAD Python3.11 environment for prepare/GPU stages, original
`.venv` for CPU selection/report and native grading. No installation, download,
new fixture, OpenAI SDK/key or original author pipeline execution is required.
Qwen2.5-7B is offline/pinned, BF16; physical GPU0/UUID only, >=24000MiB free,
utilization<=10. Evaluation is ordinary default-system replies, greedy seed42,
B1, cap160; elicitation is temperature1/cap256 with approved matched systems.

## CPU preparation and initial registration

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.profanity_method_transfer prepare
.venv/bin/python -B -m scripts.profanity_method_transfer verify
```

Preparation checks NFKC/case/whitespace overlap: all20 extraction questions
must remain, no overlap with the unchanged32 calibration+32 held-out questions.
It binds five instruction pairs/full approved systems, actual rendered prompts
and IDs, literal PF and GCAD coherence rubrics, model/runtime/tokenizer hashes,
pinned source files and all grading plans. No old response/vector is fitted.
The old impolite bank is bound only as a forbidden tensor-identity list.

Root commits sources/protocol/design acceptance and initial registration
`results/protocols/profanity-method-transfer-01.json` with exactly:
`manifest_sha256`.

## Fresh elicitation and separate ratings

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.profanity_method_transfer elicitation --execute-gpu
.venv/bin/python -B -m scripts.profanity_method_transfer judge --phase elicitation --metric trait --execute-native
.venv/bin/python -B -m scripts.profanity_method_transfer judge --phase elicitation --metric coherence --execute-native
.venv/bin/python -B -m scripts.profanity_method_transfer eligibility
```

Native commands require root's process-local pinned2.1.323 PATH, matching
binary/audit/model identity. Existing tools-off isolated foundation and disk
journals are unchanged. Trait and coherence are separate contexts. JSON+brief
reason is a declared numeric-only rubric adaptation. Trait REFUSAL is null,
counted as zero in conservative fixed-denominator means, with numeric-only
means/refusal counts separately reported. Coherence is always integer0..100.

At most two identical structural attempts per batch; first valid frozen.
Infrastructure/auth/model/isolation/hash errors and uncertain starts stop,
never trigger a retry. `--resume` is only exact replay of an existing grading
stream, never GPU regeneration. Plans have <=8 items, distinct question IDs
per batch, opaque aliases and no condition/system labels in payloads. Removed
GCAD/final branches filter the frozen full plan without new randomization.

Source eligibility: pos PF>=50, neg PF<50, coherence>=50 both. Fewer than10
pairs marks **GCAD only** unavailable; fixed/adaptive development still runs.

## Extraction, mechanics and release

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.profanity_method_transfer extraction --execute-gpu
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.profanity_method_transfer mechanics --execute-gpu
```

Projection fitting uses300 prompt forwards:100 plain,100 styled,100 separately
styled cyclic-next-instruction targets, raw post-block last-prompt states at
all28 blocks. It preserves sequential FP32 source fitting. No response pooling
or extraction AUROC gate blocks behavior generation.

GCAD uses AST-isolated exact pinned author delta/key-demand functions on the
eligible full systems+saved responses:2E delta plus E key/demand forwards.
Extraction loads SDPA for prompt fitting, synchronizes/unloads it, then uses a
separate BF16 eager load for actual attention weights. A bounded read-only
driver-resource settling wait precedes reload. Failure after completed work
is fatal/incomplete, not an alleged no-call resource wait. Both attention loads
and their resource guards are recorded. Source decoded-response retokenization
and system/response boundaries are saved explicitly.

Full mechanics uses16 cap32 generations: two genuinely hook-free ordinary,
two capture-only counterparts, and two each zero/nonzero fixed, adaptive and
GCAD. Without GCAD,12. Two extra logit forwards are bounded separately. Gates
check capture-only equality to hook-free tokens/logits, actual-zero parity,
nonzero hidden/logit effects, prefill/cache coverage and cleanup. Saved h0 is
block10's first-prefill last-real-token output. No extra uncounted h0 forward.

Root commits `results/protocols/profanity-method-transfer-mechanics-01.json`:

```text
manifest_sha256
extraction_run_sha256
extraction_integrity_sha256
banks_sha256
mechanics_run_sha256
mechanics_integrity_sha256
mechanics_result_sha256
```

## Development and selection

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.profanity_method_transfer development --execute-gpu
.venv/bin/python -B -m scripts.profanity_method_transfer judge --phase development --metric trait --execute-native
.venv/bin/python -B -m scripts.profanity_method_transfer judge --phase development --metric coherence --execute-native
.venv/bin/python -B -m scripts.profanity_method_transfer select
```

One baseline plus fixed/adaptive alpha1/2/4 at block10 and available GCAD
base3.5/5/7 at blocks9..19, scale1.5. All arms use the same32 items. Each
phase's own baseline captures and binds h0 without modifying generation.
Per method choose lowest strength satisfying PF>=base+20, lexical hits>=8
and>=basehits+8, coherence>=50 and>=base-20, invalid<=3. Invalid is empty OR
distinct-bigram<.6. No extra quality, factual, missingness or extraction gate.
Every strength's actual semantic/lexical/cap/repetition/changed-answer results
are reported even if all fail. No selection means completed development_null;
skip held-out work and run report, with no rescue.

Selection writes dedicated-CPU-generator controls, `controls.pt` and individual
`controls_metadata.json`. Projection random displacement norm is the selected
development **realized** after-minus-before L2 norm averaged over every hooked
position, prefill and decode. It is not full-prefix Frobenius norm. GCAD random
delta preserves each layer's raw norm with the same keys/demand/gate.
Question seeds cycle8603/8604/8605; GCAD layer seed=questionseed+1000*layer.
No global RNG draws. Store unit/delta/displacement hashes, seeds and norms.

Before held-out work, root commits selection registration:

```text
manifest_sha256
selection_sha256
development_run_sha256
trait_accepted_sha256
coherence_accepted_sha256
controls_sha256
controls_metadata_sha256
```

## Held-out confirmation and public report

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.profanity_method_transfer heldout --execute-gpu
.venv/bin/python -B -m scripts.profanity_method_transfer judge --phase heldout --metric trait --execute-native
.venv/bin/python -B -m scripts.profanity_method_transfer judge --phase heldout --metric coherence --execute-native
.venv/bin/python -B -m scripts.profanity_method_transfer report
```

One shared held-out baseline and one selected+one random arm per qualifying
method, generated once. Held-out baseline has its own h0; no development h0
reuse. Random controls are energy/site controls, not identical adaptive laws.
Final gates add PF>=ownrandom+20 to the development thresholds; no lexical
versus random requirement. Rank passers by PF gain over own random, then
coherence, then fixed>SAPS>GCAD; no test-driven strength reselection.

Full ceiling:760 generations (200+16+320+224),300 prompt forwards,300 GCAD TF,
2 extra logits,0 MC. Max186 first-valid native batches/372 starts. Branches
only reduce this. Before-call journals charge starts; completed counters do not
pretend interrupted work was free. No GPU-stage reruns. Entry/exit source/input/
model checks are merged/deduplicated once per live command, without historical
recursive closure. Raw prompts/answers/ratings/tensors remain private. Public
report exports aggregate findings and provenance; unfinished status is private.

These are reconstructed-input method transfers, not author numerical
reproductions. Coherence is not factual quality; lexical/repetition diagnostics
are limited. If all fail, retain the canonical literature workflow with its
positive effect unreproduced and missing-data limitation, not a working claim
or a new automatic rescue study.
