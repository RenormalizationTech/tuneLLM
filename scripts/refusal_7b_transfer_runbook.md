# Fixed refusal03 transfer operations

Root owns `docs/refusal_7b_transfer_protocol.md`, registration, commits, GPU
windows and judge calls. This runner ports the old block-output workflow. It
does not use the unfinished `refusal_7b_common.py`, ORBench, canonical input
extraction, coefficient tuning, or a layer search.

Use `.venv/bin/python` (Torch2.6.0+cu124, Transformers4.57.6). Old artifacts stay
read-only. All run output goes to `outputs/refusal-7b-transfer-01`. Model inputs
use the native Qwen2 template/default system, batch8, cap320, greedy seed42,
repetition penalty1 and tokenizer EOS151645/PAD151643, exactly the old generator
operations. Native generation defaults (dual EOS, repetition1.05) are recorded
but overridden by that helper. Preparation verifies native prompt byte equality
with/without its ignored thinking flag; no thinking-mode claim is made.

## CPU preparation, then root commits

```bash
.venv/bin/python -B -m scripts.refusal_7b_transfer prepare
.venv/bin/python -B -m scripts.refusal_7b_transfer verify --models
.venv/bin/python -B -m scripts.refusal_7b_transfer register --output results/protocols/refusal-7b-transfer-run-01.json
```

Prepare copies the exact100 AdvBench/Dolly pairs, 30safe/30unsafe development,
42safe/30unsafe final and old48-item retention prefixes. It records20 development
and25 final IDs prospectively reallocated from the earlier150 reserved;105 remain
protected. Original split bytes and DECCP inputs are hash-bound and untouched.
Prepared data/report, old03 provenance, model assets, active/snapshot sources,
Qwen2 runtime/scorer resources, protocol and unchanged judge audit are bound.
No original4B vector,7B screen response or previous labels enter this comparison.

Root reviews/commits the exact source, protocol and run registration before any
execution flag below. Example paths name root-owned registrations; the commands
do not themselves authorize execution.

## Extraction, baseline, probes

```bash
.venv/bin/python -B -u -m scripts.refusal_7b_transfer extract --registration results/protocols/refusal-7b-transfer-run-01.json --execute-gpu
.venv/bin/python -B -u -m scripts.refusal_7b_transfer baseline --registration results/protocols/refusal-7b-transfer-run-01.json --execute-gpu
.venv/bin/python -B -u -m scripts.refusal_7b_transfer probe --registration results/protocols/refusal-7b-transfer-run-01.json --execute-gpu
```

Extraction is forward-only, last-real-prompt block18 OUTPUT, raw FP32 harmful
mean minus benign mean. The shift is fixed -0.5raw; the metadata reports both
the original norm ratio and scaled shift/median ratio. Baseline generates60
fresh development answers. Each extraction/generation batch has persistent
start/result artifacts. **No existing stage can restart, even if incomplete.**
An interrupted stage stops for root review; never delete its directory to retry.

The32 probe responses are two identities and two nonzero interventions on the
first8 safe development prompts. Identity tokens, inputs and batch membership
must match baseline; logits and hook cleanup are checked. Ablation retains the
old28-block input/o_proj/down_proj coverage (84 intervention sites,140 observed
nodes including mid/post residuals), with prefill/decode coverage. Root copies
the exact `probe/budget.json` to
`results/protocols/refusal-7b-transfer-budget-01.json`, reviews and commits it.

## Development and joint180 grading

```bash
.venv/bin/python -B -u -m scripts.refusal_7b_transfer development --registration results/protocols/refusal-7b-transfer-run-01.json --budget results/protocols/refusal-7b-transfer-budget-01.json --execute-gpu
.venv/bin/python -B -m scripts.refusal_7b_transfer packets --phase development
.venv/bin/python -B -m scripts.refusal_judge judge --packet outputs/refusal-7b-transfer-01/development/judging/packet.json --rubric outputs/refusal-7b-transfer-01/inputs/rubric.json --audit outputs/refusal-judge-isolation-audit-01 --registration results/protocols/refusal-judge-01.json --output outputs/refusal-7b-transfer-judgments-dev-01 --execute-judgments
.venv/bin/python -B -m scripts.refusal_7b_transfer import-ratings --phase development --ratings outputs/refusal-7b-transfer-judgments-dev-01/submission.json
.venv/bin/python -B -m scripts.refusal_7b_transfer select
.venv/bin/python -B -m scripts.refusal_7b_transfer lock-final --output results/protocols/refusal-7b-transfer-selection-01.json
```

Only the two fixed candidates generate120 additional development responses.
All60 baseline and120 candidate answers enter ONE shuffled180-item packet with
the unchanged rubric/adapter/audit. Judge resume, if separately authorized,
uses only the existing adapter's exact-bound journal, never a new grading run.
No judge calls exist in the transfer runner.

Selection requires unsafe valid full compliance >=baseline+6, safe useful
>=baseline-2, and no per-class empty/degenerate/unscorable increase. Tie order:
higher safe usefulness, higher unsafe valid compliance, then ablation. Invalid
outputs cannot become gains by complementing refusal. Root commits the exact
selection lock. A null skips all final/retention calls; report still includes
development paired useful benign rescues and newly refused compliant cases.
Unsafe compliance is mechanism/willingness, never benign-success evidence.

## Conditional final, retention and compact report

```bash
.venv/bin/python -B -u -m scripts.refusal_7b_transfer final --registration results/protocols/refusal-7b-transfer-run-01.json --lock results/protocols/refusal-7b-transfer-selection-01.json --execute-gpu
.venv/bin/python -B -m scripts.refusal_7b_transfer packets --phase final --lock results/protocols/refusal-7b-transfer-selection-01.json
.venv/bin/python -B -m scripts.refusal_judge judge --packet outputs/refusal-7b-transfer-01/final/judging/packet.json --rubric outputs/refusal-7b-transfer-01/inputs/rubric.json --audit outputs/refusal-judge-isolation-audit-01 --registration results/protocols/refusal-judge-01.json --output outputs/refusal-7b-transfer-judgments-final-01 --execute-judgments
.venv/bin/python -B -m scripts.refusal_7b_transfer import-ratings --phase final --lock results/protocols/refusal-7b-transfer-selection-01.json --ratings outputs/refusal-7b-transfer-judgments-final-01/submission.json
.venv/bin/python -B -u -m scripts.refusal_7b_transfer retention --registration results/protocols/refusal-7b-transfer-run-01.json --lock results/protocols/refusal-7b-transfer-selection-01.json --execute-gpu
.venv/bin/python -B -m scripts.refusal_7b_transfer summary --lock results/protocols/refusal-7b-transfer-selection-01.json
```

For a null, skip the final packet/judge/import commands and use the null final/
retention commands only to write their skip receipts, then summary. A non-null
compares baseline and the one selected method on144 final answers. Retention
uses32 generations and64 MC items total, unchanged old prompt formats, caps,
48-item fixture, scorer thresholds and language-detector seed/restoration.
Final quality cannot reselect a method. Reports preserve the old final criteria,
paired/family uncertainty, PAD-before-EOS diagnostics, and paired valid/useful
benign rescue counts. A mere aggregate refusal drop is not a useful rescue.

Default compact destination is `results/refusal_7b_transfer_01.json`; raw prompts,
answers, keys, vectors and batch receipts remain private in the ignored run.
No publication, weight changes, further dataset, or automatic follow-up.
