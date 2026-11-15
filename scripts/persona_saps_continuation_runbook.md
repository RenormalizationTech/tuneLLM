# Bounded SAPS continuation — parent failed gate remains failed

Worker implements/CPU-tests only. Root owns prepare/review/registration and every
GPU/native release. Frozen original SAPS/PV/grid/artifact files and old results
are untouched. No downloads/installations, new data, new judge or joint steering.

Config `configs/persona_saps_continuation.json`; command:
`.venv/bin/python -B -m scripts.persona_saps_continuation STAGE`.
Fixed new run `outputs/persona-saps-sarcasm-continuation-01`, aggregate
`results/persona_saps_sarcasm_continuation_01.json`. No alternate write paths.

The parent `outputs/persona-saps-sarcasm-01` is CLOSED elicitation_null. Its
extra pre-extraction gate failed only clarity3.46<3.5; style3.04/36of50>=3 and
grammar4.64 passed; entailment2.44 versus baseline4.60 is a known quality risk.
The author learns from prompt activations, not these generated responses. This
new registration intentionally omits our extra pre-extraction gate; it does NOT
claim the parent gate passed, rerun it or loosen any downstream criterion.

1. After code/protocol acceptance root runs `prepare`, then `verify`. Parent must
   have NO directions/preflight/validation/test/custom/retention or downstream
   grading artifacts, checked at preparation and every verify. Bind original
   stop/failed eligibility/result/registration,100 raw initial answers, first-valid
   five batch journals/starts/receipts/model identity, original50-input/four-block
   read-only first-prefill h0 capture and costs. No rerating or rewriting.
   Exact input BYTES are copied: all data/splits, seed702 test100/custom-first20,
   frozen1000 draws, templates/token IDs, rubric, retention, all plans/seeds and
   absent-method branches. No new sampling/choices. Commit
   `results/protocols/persona-saps-sarcasm-continuation-01.json` with manifest_sha256.
2. Separately released `directions --execute-gpu`: unchanged native HF BF16 SDPA,
   nonthinking seed42 B1.500 plain+500 styled+500 target prompt forwards,1500 total;
   all36 raw post-block states including block35 BEFORE final norm. Imported frozen
   capture/normalization/controller primitives; original1000 style draws unchanged.
   Save exact safetensors roundtrip and metadata binding child failed-bank record
   and original accepted grades. No false eligibility.json/pass record is created.
3. Released `preflight --execute-gpu`: ten cap32 generations (four ordinary with
   read-only capture, four ACTUAL zero hooks, two nonzero), plus two logit forwards.
   Exact capture-inclusive zero parity, real hidden/logit differences, cache scope
   and cleanup. Same original mechanics; fixed c is CPU FP32 dot, separate .item()
   operands, Python subtraction/alpha and model-dtype unit multiplication.
   Adaptive T remains0D and B1 correction broadcasts to full prefill and each
   cached token. Root commits
   `results/protocols/persona-saps-sarcasm-continuation-strength-01.json` containing
   strength_sha256 before validation steering.
4. Released `validation --execute-gpu`:600 outputs, same12 settings (blocks11/15/
   19/23; fixed1, SAPS1/1.4). Reuse parent baseline metrics/grades/h0 without any
   regeneration/rejudging. `packet --phase validation`, released
   `judge --phase validation --execute-judgments`, then `select`. Original floors
   grammar/clarity>=3.5, entailment>=3, mechanics<=base+3; choose per method then
   overall highest style, lower alpha/earlier block/fixed ties; gain>=0.5.
   No eligible method becomes absent; no eligible winner/gain closes selection_null.
5. Root commits
   `results/protocols/persona-saps-sarcasm-continuation-selection-01.json` with
   selection_sha256 BEFORE test. Released `test --execute-gpu`: up to500 outputs,
   untouched100 strings, baseline/best fixed/best SAPS/random/prompt-index0; absent
   method arm skipped. Fresh test-only h0 captured in read-only baseline first,
   never val h0. Saved scalar-controller random and per-position norm ratio flag
   0.8–1.25 are unchanged/report-only. Packet/judge phase `test` (up to25 batches).
   Independently packet/judge phase `custom`: same20 fixed source strings across
   saved arms, up to100 existing answers/13 batches, no generation/prior scores.
6. Only if original reference report criteria pass: released
   `retention --execute-gpu`, fresh matched48 baseline/winner, B1/32 generations/
   64 MC with fixed per-case h0, GSM512/IFEval1024, official scores and detector
   seed0 restored. Independent off-task transfer only. `report` publishes aggregates
   after closure, never raw prompts/answers/IDs or a premature exclusive result.

Fresh maxima1142 generations/1500 prompt forwards/2 logit forwards/64 MC;
68 first-valid batches/136 native starts. Parent100 generations/five first-valid
starts stay separate. Whole maximum1242 generations/73 first-valid batches and
141 starts with parent actually5, not a fictional10. Missing arms/nulls reduce
work. No initial-generation or initial-rating API exists in this child.

Same pinned native2.1.323 binary/model/audit/tools-off medium runtime. Actual
modelUsage, payload/start/receipt/first-valid bindings checked. Two identical
structural attempts max per batch; fatal infra/hash/identity/uncertain-start or
schema exhaustion stops permanently, no semantic repair. `--resume` only replays
an existing exact child phase; first valid is never rerated. Accepted records
publish after live entry/exit integrity. Resource waits do not start/stop phases.

One merged deduplicated model/source/input traversal per live command entry/exit;
no recursive unrelated-history closure or cross-command hash cache. Parent read
helpers remain read-only. Report resource accounting distinguishes fresh child
work from100 reused generations and five old grade calls. Failed unfinished work
may be additional to saved counters. Parent null/result hashes are retained.

Limits: observed low prompted quality makes later null plausible. Old/new grading
sessions, co-rating20-item contrast, single Claude versus three author judges,
strict sentence-alone sarcasm versus contextual custom definition, hostile/profane
inputs and incomplete transformation lineage remain disclosed. No human validity,
general assistant-persona control, test oracle or future sweep is implied.

CPU only:
`.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_persona_saps_continuation.py`
