# Fixed selected benign diagnostic

Root owns `docs/refusal_7b_benign_transfer_protocol.md`, registration, commits,
GPU0 and judge execution. There are no selection/final/retention/probe stages.
Initial and halfdose nulls, source labels and all old artifacts stay immutable.

Use the EXISTING pinned grading PATH for preparation/verification and judging;
no shim or installation is needed:

```bash
export PATH="$PWD/outputs/sarcasm-method-transfer-01/runtime-bin:$PATH"
.venv/bin/python -B -m scripts.refusal_7b_benign_transfer prepare
.venv/bin/python -B -m scripts.refusal_7b_benign_transfer verify --models
.venv/bin/python -B -m scripts.refusal_7b_benign_transfer register --output results/protocols/refusal-7b-benign-transfer-run-01.json
```

That `claude` resolves to version2.1.323, SHA256
`604d29a6869409106b6a1110bdadcc8b72e8654a5a865d23258f938a696e348e`,
as recorded in `outputs/refusal-7b-transfer-01/operator_records/grader_runtime.json`.
The original audit/rubric/adapter/argv remain unchanged.

Prepare derives original17 from frozen4B nonempty primary-refusal records and
verifies all250 accepted grades. Original case67 remains P despite the later
questionable interpretation. Excluding dose-seen109/162/165/171 leaves exactly
13 new IDs, source order14/57/59/61/64/67/103/151/153/168/169/417/418.

Seven reserved IDs59/61/67/151/153/168/418 are prospectively consumed, leaving98
protected. IDs57/64 consume two previously unused final-set prompts: the earlier
final42 is no longer untouched. DECCP25 remains untouched. Source files, old
splits and old results are never rewritten. These are selected diagnostic
prompts, not confirmatory held-out benchmark evidence.

Native13 inputs are frozen before calls. Fresh baseline13 and candidate13 use
identical batches8/5, original settings (seed42,batch8,cap320,greedy,rep1,
tokenizerEOS151645/PAD151643) and native default-system template. Baseline gets
no vector; candidate gets half the preserved original block18 vector through
the old -0.5 output hook, effective -0.25raw at all positions. No new extraction,
baseline reuse across incompatible batch membership, or additional dose.

Root reviews/commits exact sources/protocol/run registration, then executes:

```bash
.venv/bin/python -B -u -m scripts.refusal_7b_benign_transfer compare --registration results/protocols/refusal-7b-benign-transfer-run-01.json --execute-gpu
.venv/bin/python -B -m scripts.refusal_7b_benign_transfer packets
export PATH="$PWD/outputs/sarcasm-method-transfer-01/runtime-bin:$PATH"
.venv/bin/python -B -m scripts.refusal_judge judge --packet outputs/refusal-7b-benign-transfer-01/comparison/judging/packet.json --rubric outputs/refusal-7b-benign-transfer-01/inputs/rubric.json --audit outputs/refusal-judge-isolation-audit-01 --registration results/protocols/refusal-judge-01.json --output outputs/refusal-7b-benign-transfer-judgments-01 --execute-judgments
.venv/bin/python -B -m scripts.refusal_7b_benign_transfer import-ratings --ratings outputs/refusal-7b-benign-transfer-judgments-01/submission.json
.venv/bin/python -B -m scripts.refusal_7b_benign_transfer summary
```

Comparison persists per-batch start/result journals. Existing stages cannot
restart, including interrupted stages; root reviews failures without deleting
history. Exactly26 new answers enter one anonymous26-item packet (two judge
batches20/6), fixed packet seed20261118. No four-seen responses enter grading.

Report `results/refusal_7b_benign_transfer_01.json` contains paired refusal,
usefulness, useful-valid-full rescue IDs, new-refusal IDs, usefulness-loss IDs,
invalid-union/component flags and cap counts. Harm categories may overlap and
must not be added as disjoint cases. The four previously seen paired outputs/
labels are reused only in a separate descriptive block, including xstest109;
there is no pooling with13. Original joint-baseline versus separate-candidate
grading contexts remain disclosed. No gates, rates, p-values or benign-success
Boolean, no further data/method/coefficient/tuning after this diagnostic.
