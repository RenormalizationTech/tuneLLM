# Isolated sarcasm transfer

Governed by `docs/sarcasm_method_transfer_protocol.md`. New root only:
`outputs/sarcasm-method-transfer-01`; public result:
`results/sarcasm_method_transfer_01.json`. Root owns preparation, reviews,
registrations and all live releases. Worker performed CPU implementation/tests
only. No old sources, data, runs, reports or guards are modified.

Qwen2.5-7B remains pinned, offline, native-HF BF16 in the isolated GCAD env.
Physical GPU0 UUID/free>=24000MiB/util<=10 guards remain. Ordinary replies:
greedy seed42, B1, cap160. Elicitation: temperature1, cap256. No new install,
download, SDK, credentials or dummy audit is needed. Native execution uses
root's process-local pinned2.1.323 PATH and unchanged audited tools-off flags,
model and medium effort. Runtime/model/isolation failure stops without retry.

## Initial snapshot and one artifact request

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.sarcasm_method_transfer prepare
.venv/bin/python -B -m scripts.sarcasm_method_transfer verify
```

Root commits initial registration `results/protocols/sarcasm-method-transfer-01.json`
with exactly `manifest_sha256`, plus the bound sources/protocol/design review.
Preparation freezes the request/role, actual tokenizer default system content,
known C/H questions, model/runtime fingerprints and FULLY PACKED plans for
every possible branch BEFORE artifact generation. No train question text is
invented locally. No Qwen weights are loaded during prepare.

```sh
.venv/bin/python -B -m scripts.sarcasm_method_transfer artifact --execute-native
.venv/bin/python -B -m scripts.sarcasm_method_transfer preview
```

Artifact JSON has role `artifact_generation`: five distinct pos/neg instruction
pairs,20 unique questions,0..100 rubric with exactly `{question}` and `{answer}`
slots. Anchors0/25/50/75/100 are specified in the frozen request. Neither the64
evaluation questions nor overlap list enters ANY request, including retries.
Local NFKC/case/whitespace overlap is a structural rejection; at most two
identical requests, first valid only. No semantic/content reroll or hand edits.

Root and Claude review the artifact and all full renderings in `preview.json`.
Root writes the configured semantic review, with exactly these required keys:

```text
accepted: true or false
preview_sha256: file SHA256 of preview.json
fixture_canonical_sha256: preview artifact_sha256 (canonical parsed fixture)
fit_inputs_canonical_sha256: preview fit_inputs_sha256 (canonical fit bundle)
```

These two canonical hashes differ from later serialized FILE hashes. Additional
private review metadata is allowed. Semantic rejection closes artifact_null;
no further artifact request is permitted.

```sh
.venv/bin/python -B -m scripts.sarcasm_method_transfer promote
```

Accepted parsed strings are retained unchanged at
`evaluation/sarcasm_method_transfer/extract.json`. Promotion saves the exact
fit bundle and data/rendering/basis mirrors. `promotion.json` binds their byte
hashes, first-valid receipt, preview and review; every later verification checks
the copies. This supplies the existing pure control helper's `inputs/data.json`
contract without any namespace/global patch.

Root commits the fixture/review and artifact registration with exactly:
`manifest_sha256`, `preview_sha256`, `semantic_review_sha256`, `fixture_sha256`
(actual serialized file hash), `promotion_sha256`.

## Elicitation, ratings and extraction

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.sarcasm_method_transfer elicitation --execute-gpu
.venv/bin/python -B -m scripts.sarcasm_method_transfer judge --phase elicitation --metric trait --execute-native
.venv/bin/python -B -m scripts.sarcasm_method_transfer judge --phase elicitation --metric coherence --execute-native
.venv/bin/python -B -m scripts.sarcasm_method_transfer eligibility
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.sarcasm_method_transfer extraction --execute-gpu
```

GCAD elicitation uses persona-only systems from `persona_reference_common.system`:
helpful sarcastic/helpful sincere. SAPS styled/target and prompted reference use
the actual frozen DEFAULT system content + ONE SPACE + positive full persona
system. Baseline/plain stays native default. This difference is intentional.
Prompted pair index0 is fixed. C/H prediction IDs remain C/H IDs; a separate
`*_prompted` render key selects the prompted input, never a different scoring ID.

Elicitation is200 responses. SARC and GCAD coherence ratings have separate
`rating` role payloads/journals, opaque IDs and distinct underlying questions
within every <=8-item batch. Trait REFUSAL is null/zero conservatively, separately
counted; coherence stays numeric. First valid only, one identical structural
retry maximum; all malformed attempts/receipts remain. `--resume` replays an
EXISTING stream, never regrades accepted batches or reruns GPU phases.

Eligible pairs: posSARC>=50, neg<50, coherence>=50 both. E<10 disables only
GCAD, never SAPS. No extra elicitation or extraction AUROC gate. SAPS uses300
raw last-prompt post-block captures: plain/style/cyclic-next-style target.
GCAD uses pinned AST-isolated exact author delta/key-demand functions,3E TF,
on full saved persona-system prompts + decoded responses. A separately
recorded eager model load follows SDPA prompt fitting; resource settling is
bounded/read-only and a failure after work is fatal/incomplete. No old loader
assertion is weakened. Banks are new SARC-labelled FP32/basis/hash-validated
objects; PF/impolite guards and data are not relabelled.

## Mechanics release, development and selection

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.sarcasm_method_transfer mechanics --execute-gpu
```

Ten cap32 generations: two hook-free ordinary, two SAPS-zero/nonzero each,
two GCAD-zero/nonzero each; six if GCAD is unavailable. Two bounded extra
logit forwards. Exact zero parity, nonzero hidden/logit effects, prefill/cache
coverage and cleanup are required. No fixed projection or h0 capture here.

Root commits mechanics registration with exactly:

```text
manifest_sha256
extraction_run_sha256
extraction_integrity_sha256
banks_sha256
mechanics_run_sha256
mechanics_integrity_sha256
mechanics_result_sha256
```

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.sarcasm_method_transfer development --execute-gpu
.venv/bin/python -B -m scripts.sarcasm_method_transfer judge --phase development --metric trait --execute-native
.venv/bin/python -B -m scripts.sarcasm_method_transfer judge --phase development --metric coherence --execute-native
.venv/bin/python -B -m scripts.sarcasm_method_transfer select
```

Full development:256 replies, baseline+prompted+SAPS1/2/4 block10+GCAD3.5/5/7
blocks9..19 scale1.5. GCAD-null branch160 replies. Per method choose lowest
qualifying strength: SARC>=base+20,>=8 answers scoring>=50, coherence>=50 and
>=base-20, invalid<=3 (empty OR distinct-bigram<.6). Prompted gain<20 is a
report-only validity concern. PF lexicon, impoliteness and correlation NEVER
gate selection. All strengths, caps/lengths, target-score distributions and
realized shifts are reported even for development_null.

Selection saves `controls.pt` and per-tensor `controls_metadata.json` using
dedicated CPU generators, seeds9603/4/5 by report row; GCAD layer seed adds
1000*block. SAPS constant-random norm is the selected development realized norm
weighted over ALL hooked positions, not Frobenius prefix norm. GCAD random
preserves raw layer delta norms and the same gate/key/demand/site. Report actual
energy differences; do not claim matched adaptive trajectories.

Root commits selection registration with exactly:

```text
manifest_sha256
selection_sha256
development_run_sha256
trait_accepted_sha256
coherence_accepted_sha256
controls_sha256
controls_metadata_sha256
```

## Confirmation, descriptive coupling and report

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.sarcasm_method_transfer heldout --execute-gpu
.venv/bin/python -B -m scripts.sarcasm_method_transfer judge --phase heldout --metric trait --execute-native
.venv/bin/python -B -m scripts.sarcasm_method_transfer judge --phase heldout --metric coherence --execute-native
.venv/bin/python -B -m scripts.sarcasm_method_transfer judge --phase heldout --metric impolite --execute-native
.venv/bin/python -B -m scripts.sarcasm_method_transfer report
```

If none selected, skip heldout and report development_null. Otherwise one
baseline+prompted+selected/random per method,128 or192 replies. The exact packed
SAPS-only/GCAD-only/both/none plan was frozen before artifact; no filtering or
post-output regrouping. Confirmation adds SARC>=ownrandom+20. Prefer GCAD if
passing, else SAPS, else null. No post-test strength reselection.

Primary confirmation is computed once target/coherence are accepted even if
the descriptive journal is pending: `primary_complete_descriptive_pending`.
Descriptive failure leaves confirmation intact with explicit execution-incomplete
status and permanent stop; it never retroactively fails a scientific gate.
The final descriptive impolite journal, narrow PF lexicon, within-condition
score correlations and separately labelled condition-confounded pooled
correlation do not determine retention. Undefined Pearson/prompt-gain fraction
is null, not a new rejection criterion. A complete report includes both primary
and descriptive evidence. Pending/incomplete reports remain private.

Maxima:658 generations,300 prompt forwards,300 GCAD TF,2 logits,0 MC;
187 first-valid native batches/374 starts INCLUDING artifact. Model/native
starts are durably charged BEFORE calls. Interrupted work cannot regenerate;
completed counters do not claim all uncertain work was free. Entry/exit hashes
are merged once per live command, not historical recursive closures.

This is reconstructed ordinary-reply transfer with generator/reviewer/rater
circularity and reused project evaluation questions, not an author-score or
generalization claim. Coherence is not accuracy; strong off-axis coupling is
a finding, not mandatory disentanglement or joint-control work. If unsuccessful,
the Mistral reference remains explicitly model/task-specific; no rescue sweep.

Focused CPU checks (no GPU/target/native calls):

```sh
.venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_sarcasm_method_transfer.py
```
