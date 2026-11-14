# Isolated 4B profanity pilot (root execution only)

Authority: `docs/profanity_steering_protocol.md`. Worker authors code/fixtures and
runs CPU checks only. Root owns fixture/code approval, preparation, three exact
registrations, GPU/network releases and narrative. Old sources/outputs remain
read-only. No tuning, 14B use or joint-trait steering belongs here.

```bash
.venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_profanity_steering.py
.venv/bin/python -B -m scripts.profanity_steering --help
```

Fixtures: 32 authored neutral/profane pairs, 16 development tasks and 24
confirmation tasks (eight factual). Families/topics are disjoint by declared
taxonomy and require human/advisor semantic review; automated checks establish
exact input uniqueness. First lexical-hit positions and annotated phrases are
derived/validated from actual answer text. The calibrated trait definitions stay
byte-identical; content, coherence and reference-correctness booleans are added.
For factual:false, emit factual_correct:false and ignore it: content_ok still
requires reference agreement, including extraction. Only the eight factual:true
confirmation items enter the factual-correctness gate.

After approval, root prepares and commits `results/protocols/profanity-steering-01.json`
with the exact `manifest_sha256`. Preparation freezes every fixture, anonymous
constrained plan, tokenizer boundary, source/model hash and historical protection:

```bash
.venv/bin/python -B -m scripts.profanity_steering prepare
.venv/bin/python -B -m scripts.profanity_steering verify
.venv/bin/python -B -m scripts.profanity_steering packet --phase eligibility
```

Blind plans use seeds501/502/503, eight distinct task groups per batch, and hide
condition, block, beta and pair side. No same-task alternatives share a batch.
After a separate host/network release, process-local pinned native Claude2.1.323:

```bash
umask 077
PATH=PINNED_BIN:"$PATH" .venv/bin/python -B -m scripts.profanity_steering judge --phase eligibility --execute-judgments
.venv/bin/python -B -m scripts.profanity_steering eligibility
```

Every eligible pair is retained. Fewer than24, fewer than four lexical families
or any family above one third is an eligibility null: stop, report, no GPU.
Retained position/phrase counts are attrition diagnostics, not new retained-set
gates. Neither regeneration nor rebalance is authorized.

Root GPU release is separate, physical GPU0 UUID only, >=12,000 MiB free and
<=10% utilization. Busy guards wait without disturbing jobs; other live failures
stop. No partial phase is resumed or regenerated:

```bash
.venv/bin/python -B -m scripts.profanity_steering extract --execute-gpu
.venv/bin/python -B -m scripts.profanity_steering preflight --execute-gpu
```

Teacher-forcing uses the same helpful system for both authored answers. Exclude
template/prompt/EOS/pad using verified offsets. FP32 per-response means produce
raw differences and unit directions at zero-based blocks17/23. R is the median
neutral response-MEAN norm. Fixed deltas are beta*R*unit, beta .05/.10/.20. Token-
weighted per-token norms are diagnostics only. Random units use local CPU seeds
4242+layer and are serialized before development. Safetensors round-trip must be
exact. Actual GPU deltas are BF16, with scale metadata explicitly recorded.
The inherited hook adds only at last-real prefill and each cached decode position;
its actual zero-vector path must match tokens, and nonzero probes must fire and
change hidden values/logits. Development/final/reload/retention hooks record all
prefill/decode calls and sample hidden deltas once per cache phase. Immediate EOS
need not produce a decode call; the dedicated preflight remains strict. Probe
text is unscored. Native generated bodies omit
only terminal EOS, retaining other whitespace/markers.

Root commits `results/protocols/profanity-strength-01.json` containing
`strength_sha256` of `outputs/profanity-steering-01/strength.json`, then releases:

```bash
.venv/bin/python -B -m scripts.profanity_steering dev --execute-gpu
.venv/bin/python -B -m scripts.profanity_steering packet --phase development
PATH=PINNED_BIN:"$PATH" .venv/bin/python -B -m scripts.profanity_steering judge --phase development --execute-judgments
.venv/bin/python -B -m scripts.profanity_steering select
```

Development null stops confirmation/retention. Otherwise root generates the lock:

```bash
.venv/bin/python -B -m scripts.profanity_steering lock
```

Commit `results/protocols/profanity-selection-01.json` containing `lock_sha256`
of the private `final_lock.json`. Lock verification rechecks selection against
current development labels and binds all vector/input/prediction/packet hashes.
No final-driven reselection. After separate GPU and judging releases:

```bash
.venv/bin/python -B -m scripts.profanity_steering final --execute-gpu
.venv/bin/python -B -m scripts.profanity_steering packet --phase confirmation
PATH=PINNED_BIN:"$PATH" .venv/bin/python -B -m scripts.profanity_steering judge --phase confirmation --execute-judgments
.venv/bin/python -B -m scripts.profanity_steering retention --execute-gpu
.venv/bin/python -B -m scripts.profanity_steering report
```

Final first replays four selected development items with the same batch4/cap256
and freshly loaded saved vector; token IDs/layout must match exactly. Confirmation
then uses baseline/selected/norm-matched random on24 tasks. Capability is fresh
baseline versus selected: raw-prefix MC batch16, GSM8K/IFEval generation batch8
with existing512/1024 caps, language-detector seed0 restored afterward. Math gate
uses strict scoring; fallback is descriptive. The same single-block hook observes
MC prompt masks and changes last-real next-answer logits, with no CLI mutations.

At most31 first-valid batches (8/14/9), two structural attempts each,62 total calls.
Exact `--resume` reuses journals/first-valid results; uncertain, identity, auth,
infra, schema-exhausted or hash failures permanently stop. No prompt reframing,
third attempt, alternative judge or fresh infrastructure completion is authorized.
Model envelope/usage must validate before schema retries. No different conditions
of one prompt are judged together. Saved accepted labels are never replaced.

Maximum target generations230 = probes10 + dev112 + reload4 + confirmation72 +
capability32, plus64 capability MC and at most64 authored teacher-forced forwards.
Per-phase records include synchronized completion timing and allocator peaks;
failed phases disclose that saved counts can omit interrupted unsaved work.
Known native-CLI cost/duration fields are aggregated without exposing private
account/session fields. No prediction text, token arrays, task IDs or per-answer
ratings enter public `results/profanity_steering_01.json`. Nulls/tradeoffs are final
learning outcomes; no new setting or further run follows automatically.
