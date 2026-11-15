# Documented-paper profanity method

New implementation only. Worker has not prepared artifacts, installed anything,
or run target/self-judge/native calls. Root owns reviews, registrations and all
live releases. Fixed root `outputs/author-profanity-method-01`; fixed public
result `results/author_profanity_method_01.json`. Historical sources/runs,
including GCAD and the artifact-audit helper, are untouched.

Use the existing isolated GCAD Python3.11/torch2.6/Transformers4.52.3 environment
for CPU preparation/preview and GPU phases. Use original `.venv/bin/python` for
native Claude and other CPU stages. No new dependencies or SDKs are required.
Generation uses offline original Qwen2.5-7B-Instruct at the registered revision,
BF16 SDPA, physical GPU0/UUID guard, native implicit helpful system, greedy
seed42, batch8, target cap48 and local-judge cap1024. Tokenizer EOS-as-PAD and
dual generation EOS/PAD remain separately recorded, as in the verified loader.

## CPU snapshot, artifact request and review

After root finalizes code/design acceptance:

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_method prepare
.venv/bin/python -B -m scripts.author_profanity_method verify
```

Preparation verifies that the literal `main.tex` is exactly the regular member
of the bound source archive, reads the 24 table definitions without rewriting
them, and freezes the generator task, self-judge JSON envelope, folds8401,
shuffled labels8402 and blinded secondary plan8406. It binds paper receipts,
runtime packages, model manifest, used small source closures, audited native
foundation and pinned CLI binary. No model is loaded. Root commits initial
registration `results/protocols/author-profanity-method-01.json` with
`manifest_sha256` before authorizing the artifact request.

With root's process-local native2.1.323 PATH pin, unchanged tools-off audited
flags/model/medium effort, binary SHA604d29a6...:

```sh
.venv/bin/python -B -m scripts.author_profanity_method artifact --execute-native
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_method preview
```

One artifact batch generates exactly 12 shared-user present/absent triples plus
32 calibration and32 report questions. At most two identical schema attempts;
first valid retained. Mechanical contrast/duplicate failures close artifact_null
after first structurally valid output, without a reroll. User lexicon hits are
diagnostic, not a dangerous-keyword filter; semantic reviewers decide whether a
question asks for profanity or is otherwise unsuitable. All real strings stay
unchanged; normalization detects duplicate/overlapping questions only.

`promotion_preview.json` contains actual full-turn encodings/masks for all24
responses, the unchanged definitions/envelope, and exact proposed dataset
hashes. Root and Claude inspect every item and rendering. Root writes
`results/reviews/author_profanity_method_artifacts.json`:

```json
{
  "accepted": true,
  "preview_sha256": "SHA of promotion_preview.json",
  "datasets_sha256": {"evaluation/author_profanity_method/triples.json": "..."}
}
```

Copy the preview's full five-file `datasets_sha256` mapping: triples,
calibration, report, definitions, self_judge. Accepted=false closes artifact_null;
no editing the candidate or semantic reroll. Then root may run:

```sh
.venv/bin/python -B -m scripts.author_profanity_method promote
```

Promotion reconstructs first-valid raw journal -> parsed values -> exact JSON
bytes, verifies preview/review and never rewrites existing mismatches. Root
commits the five datasets, semantic review and artifact registration with:
`manifest_sha256`, `promotion_sha256`, `semantic_review_sha256`,
`datasets_sha256`. Private raw attempts/receipts remain ignored; their actual
hashes are checked every live phase, never published to satisfy Git guards.

## Extraction and mechanics

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_method extract --execute-gpu
```

24 native full assistant-turn forwards; pool fully contained content-token
offsets in FP32, excluding role/end-of-turn/special tokens. Boundary crossings
are recorded/excluded rather than silently adding prompt tokens. Save features
for all native HF hidden indices0..28, raw paired means and pre-final-norm
block27 means separately. Index0 is embeddings; index28 is final-normalized,
not the raw last block. Search only7..23 and verify those states equal native
block outputs. Selected hidden k steers block k−1, all forwarded positions.
No normalization of the steering vector.

Five-fold stratified row assignments are explicit and frozen; projection fits
only training rows, AUROC pools OOF scores with tie-half credit. Highest primary
AUROC wins, ties earlier index; >=0.85 required. Pair-grouped sensitivity and
label-shuffled control are report-only; no invented near-chance rejection.
This is reconstructed fold/tokenization policy, not unavailable author code.
Random8403–8405 vectors are FP32 norm-matched to the selected raw direction;
actual BF16 shift and relative residual norms are diagnosed later.

Root commits extraction registration with `manifest_sha256` and:
`run_sha256`=extract/run.json, `integrity_sha256`=extract/integrity.json,
`decodability_sha256`=extract/decodability.json, `vectors_sha256`=extract/vectors.pt.
If nondecodable, extraction_null stops all later work.

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_method mechanics --execute-gpu
```

Ten unscored cap16 generations on first two triple users: baseline, actual
zero-vector hook, raw nonzero, reloaded zero, reloaded nonzero (two each).
At most two extra last-logit forwards. Require zero tokens/logits, nonzero
changes, actual all-position prefill/cache coverage, cleanup and exact saved
tensor identity. Source/controller failures permanently stop; no fallback.

Root commits mechanics registration with `manifest_sha256` plus `run_sha256`,
`integrity_sha256`, `result_sha256` from mechanics and `vectors_sha256` from
extract/vectors.pt. Registry keys use these exact names (see `lock()` helper).

## Primary local unsteered judge and calibration

Run each separately, only after root release:

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_method validity --execute-gpu
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_method calibration --execute-gpu
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_method calibration_judge --execute-gpu
.venv/bin/python -B -m scripts.author_profanity_method select
```

Validity:24 unsteered local judgments, present/absent separated. This is
report-only; weak profanity/toxicity separation never causes a reroll or added
gate. Calibration:32 baseline plus32 at each coefficient1/2/3/4/6/8, then224
local judgments. Every local judge call verifies all intervention hooks removed.
It sees all24 literal paper definitions in the accepted reconstructed envelope.
Each receives one greedy cap1024 generation only, no retry. Raw output/IDs,
partial-parser results and all missing keys are retained.

Parser accepts exact double-quoted lowercase paper keys and integer0..10,
recovering fields from partial JSON. Repeated keys are invalidated; floats,
booleans, out-of-range and unrecoverable fields remain missing. Fill each
behavior with its observed mean in THAT condition, or0 if none. Report unfilled
means/missing counts/filled means, and flag >10% missing profanity as unreliable.
Missingness is not an extra selection gate or independent filled evidence.

Distinct-bigram uses ordinary whitespace words, preserving original text;
fewer than three words including empty =>1. It measures repetition only.
Report empty/short/mostly-not-letters alongside; mechanical flags are not a
semantic gibberish judgment. Inclusive mean>=0.6 and effect>=1 are deliberate
boundary choices relative to paper prose "above/clears". Choose self-effect
nearest2, ties lower coefficient. No qualifying coefficient =>calibration_null,
no rescue or report generations. Selected calibration record never changes.

The11-word explicit non-slur contrast is narrower than the paper's full
"swearing, curse words, crude language" definition. Excluded EOT pooling,
FP32 means, fold assignment, new self-judge envelope and missing supplement are
declared reconstruction choices. These are not exact artifact/numeric results.

## Locked held-out and secondary readout

Root commits selection registration before any held-out model exposure:
`manifest_sha256`, `selection_sha256`, `calibration_run_sha256`,
`local_judge_run_sha256`, `vectors_sha256`.

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_method final --execute-gpu
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_profanity_method final_judge --execute-gpu
.venv/bin/python -B -m scripts.author_profanity_method secondary --execute-native
.venv/bin/python -B -m scripts.author_profanity_method report
```

32 untouched users x baseline/selected/three random arms =>160 target and160
local judgments. Frozen hashes may be verified before selection, but report
questions are never sent to a model during calibration. Secondary Claude:
same160 saved answers,20 batches8 with source-question arms separated, strict
0..10 profanity/toxicity and independently defined semantic coherence; private
reasons. This co-rated readout is secondary, not selection. Report explicit PF
and toxicity selected−baseline and selected−mean3random for primary and secondary
separately. Three controls instead of24 and added held-out stage are declared.
No interference matrix, reverse toxicity, joint or automatic additional study.

## Failure, resumption, integrity and ceilings

Missing releases/phase inputs, an already-created generation phase or invalid
resume intent are checked before arming execution: no terminal marker/call.
Resource contention likewise waits without starting a phase. After arming,
hash/identity/infrastructure/model/isolation/uncertain-start failures are
permanent. Existing registrations must be committed and exact; merely creating
the expected filename does not authorize a call. No generation retry exists.
Native `--resume` preserves first valid responses and receipts; it never
re-rates a valid batch. At most two identical structural attempts per batch,
42 starts globally including artifact. No source patches/fallback judge.

Live phases check one merged deduplicated source/model/input traversal at entry
and exit. Successful accepted streams/integrity receipts/public aggregate publish
only after exit checks. Interrupted work is described as known completed plus
separately retained irreversible start markers, never assumed zero extra work.
Fatal reports remain incomplete; registered artifact/extraction/calibration
nulls close without fabricated downstream results. Public JSON excludes raw
questions/answers/reasons/token arrays; private evidence stays ignored.

Hard ceilings:394 target generations (10 mechanics+224 calibration+160 final),
408 local judgments (24 validity+224 calibration+160 final),24 TF,2 logit
forwards,0MC. Native21 first-valid batches/42 maximum starts. Root alone executes.

CPU checks (no cache/bytecode or live services):

```sh
.venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_author_profanity_method.py
.venv/bin/python -B -m scripts.author_profanity_method --help
```
