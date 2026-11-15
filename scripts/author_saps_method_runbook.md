# Original-Mistral SAPS, source-native TransformerLens

Worker implementation/CPU tests only. Root owns preparation, reviews,
registrations and all live releases. No worker target-model, GPU or native
judge calls, installs or downloads. No edits to frozen PF/GCAD/SAPS studies.
Fixed root `outputs/author-saps-mistral-01`; result
`results/author_saps_mistral_01.json`.

## Source and runtime fidelity

Use source commit c56fb28ac0d14c8fd70035e035c5452e3c6c6ccc and actual
`steering.py` helpers: chat builder, capture, selector, complete learner, both
hook factories and output stripping. No native-HF port, HF/Qwen direction,
validation sweep or test oracle. All500 train and450 test rows stay in source
order; first two released validation rows are frozen unscored mechanics users.

GPU phases use the provisioned SAPS Python3.11/torch2.6.0+cu124/
TransformerLens2.17.0/Transformers4.57.6/NumPy1.26.4 environment. Native judging
uses the original `.venv` and audited Claude2.1.323/claude-opus-5-5/medium
tools-off foundation; no new SDK/API credential is needed.

The loader calls actual `HookedTransformer.from_pretrained` with the official
Mistral ID, exact revision, project cache and `local_files_only=True`, with
network/remote-code disabled. It injects a tokenizer loaded from that exact
local snapshot because TL's internal tokenizer path does not forward revision
or cache_dir. It does not supply `hf_model`, override centering/folding defaults,
or replace `torch_dtype=float16` with another precision mechanism. TL2.17's
backward-compatibility handling sets actual model dtype from that keyword.

CPU equivalence was checked in the SAPS environment: local default tokenizer
versus the inspected internal AutoTokenizer kwargs (add_bos_token=True,
use_fast=True, trust_remote_code=False, token=None, plus immutable local/offline
path). Both pass through actual TL set_tokenizer on tiny CPU models. BOS/EOS/
PAD IDs, padding side, add_bos_token and actual default TL.to_tokens match for
all3450 rendered prompts:500 train plain+2500 train style+450 test plain.
No7B weights or target-model forward is used by this check.

Python and torch seed42 are set before actual TL loading. The explicit tokenizer
load is checked to consume neither RNG; no reset occurs after model loading.
Actual1000 instruction draws are inferred from the unchanged helper's real
prompts/1500 forward calls, not reconstructed assuming loading consumed no RNG.
Post-load and post-learning RNG hashes are saved. Runtime records actual config,
parameter dtypes, normalization classes, processing-default arguments,
block12 parameter fingerprints and all warnings. Fresh-arm basis fingerprints
must match the direction load. Defaults such as doubled BOS are preserved.

## CPU prepare and initial registration

After final code/protocol/design acceptance only:

```sh
outputs/persona-author-replication-audit-01/envs/saps/bin/python -B -m scripts.author_saps_method prepare
.venv/bin/python -B -m scripts.author_saps_method verify
```

Preparation freezes source data/prompts/rendered strings, mechanics rows,
original Appendix C rubric and blind plans; binds author canonical blobs,
current model manifest, all installed TL Python source and runtime package
inventory. It loads no target weights. Only small used source dependencies and
current model files are bound, no unrelated historical raw-output closure.
Root commits `results/protocols/author-saps-mistral-01.json` containing exact
`manifest_sha256` plus source/protocol/design acceptance before live release.

## Directions and mechanics

```sh
outputs/persona-author-replication-audit-01/envs/saps/bin/python -B -m scripts.author_saps_method direction --execute-gpu
outputs/persona-author-replication-audit-01/envs/saps/bin/python -B -m scripts.author_saps_method mechanics --execute-gpu
```

Root releases each stage. Physical GPU0/UUID34646... only, free>=24000MiB and
utilization<=10%, leaving all other jobs untouched. Author learner executes
500 plain+500 random-style prompt forwards, then500 newly drawn style target
forwards. The proxy observes tokenization/calls only, without replacing author
globals/math or source cache returns. Save source-compatible units/targets and
metadata using exact FP32 CPU tensor roundtrip checks. Loading uses
weights_only=True; no arbitrary pickle execution. All32 native post-block
coordinates are retained. Layer12 is fixed prospectively.

Ten unscored cap16 generations, B1: two hook-free, two actual SAPS-alpha0, two
fixed-zero, two SAPS1, two fixed1. Fixed0/fixed1 each independently capture two
unsteered h0s, for a separate maximum FOUR mechanics projection forwards.
Two additional source-TL logit forwards (alpha0/alpha1) are bounded separately.
Require actual zero tokens, finite projections/hidden samples, nonzero changes,
prefill/cache coverage and complete cleanup. No source-controller patches or
rescue if any mechanics gate fails.

Direction metadata/roundtrip is verified before mechanics; root then commits
`results/protocols/author-saps-mistral-mechanics-01.json` with exact keys:

- `manifest_sha256`
- `direction_run_sha256`, `directions_sha256`, `direction_integrity_sha256`
- `mechanics_run_sha256`, `mechanics_result_sha256`, `mechanics_integrity_sha256`

Paths correspond to direction/{run.json,directions.pt,integrity.json} and
mechanics/{run.json,result.json,integrity.json}. No scored output before this lock.

## Four separate source-style loads, fixed order

```sh
outputs/persona-author-replication-audit-01/envs/saps/bin/python -B -m scripts.author_saps_method generate --arm saps0 --execute-gpu
outputs/persona-author-replication-audit-01/envs/saps/bin/python -B -m scripts.author_saps_method generate --arm fixed1 --execute-gpu
outputs/persona-author-replication-audit-01/envs/saps/bin/python -B -m scripts.author_saps_method generate --arm saps1 --execute-gpu
outputs/persona-author-replication-audit-01/envs/saps/bin/python -B -m scripts.author_saps_method generate --arm saps1p4 --execute-gpu
```

One new process/model load per arm,450 original test rows, cap60 greedy B1,
same plain instruction. Native TL cache, stop/EOS, default prepend-BOS and
string return are not overridden. Fixed1 makes450 additional source h0 prompt
forwards. Its c uses source FP32 CPU dot, separate item() operands and Python
subtraction/multiplication; the actual author fixed hook multiplies model-dtype
unit by Python c. SAPS receives the0D target and uses the actual source's
last-token projection with whole-prefix B1 broadcast, including cached decode.
Do not replace it with last-token-only assignment or normalize differently.

Each arm writes the source-style six-field JSONL plus private full trace.
Generation is called exactly with prompt string, max_new_tokens, do_sample=False,
verbose=False. A temporary, restored delegate observes the **actual final
tokenizer.decode input**; no return_type=tokens override or regenerated answer.
The actual TL2.17 embedding trajectory is also checked in full, including its
final sampled token embedded before EOS break. Both EOS and cap-final tokens
are retained, including double BOS input. No fallback to hypothetical loops.
Original full decoded string and actual author stripping stay separate.

Hook auditing delegates the original callable and returns its output unchanged.
It counts every call; first prefill/first decode numerical diagnostics are
PER POSITION. Full-prefix Frobenius L2 scales sqrt(S), squared energy scales S;
neither is mislabeled as a per-token norm. No source arithmetic selected by
diagnostics or output-based layer/alpha changes.

## Blinded native grading and report

```sh
.venv/bin/python -B -m scripts.author_saps_method packet
.venv/bin/python -B -m scripts.author_saps_method judge --execute-judgments
.venv/bin/python -B -m scripts.author_saps_method report
```

Root supplies the pinned native PATH only to the authorized process. Exact
1800 saved stripped answers in90 frozen batches20, no two arms of a sentence
together. Items label original_complaint/generated_sentence. Original complaint
is for entailment; sarcasm is sentence-alone. Exact original four0..5 rubrics
and hard constraints are preserved, including fragments<=2 and truncated or
non-terminated sentences<=3. The list/ID envelope and single Claude replacing
three judges are declared adaptations. No reasons/other score fields.

At most two identical structural attempts per batch; global max180 starts.
First valid retained. Schema repair never changes payload. Infrastructure,
identity/hash/isolation/uncertain-start errors permanently stop. Explicit
--resume can replay a complete saved first-valid response, including receipt
publication, never a response-less started call or a different payload. No
semantic reroll/fallback judge. Native receipts bind exact payload/start/bytes.

Report four distributions, paired sentence differences, sarcasm>=3 and>=4,
caps/EOS/empty/degeneration and caps with score>3 (diagnostic, no silent score
edits). The three fixed directional checks are descriptive, not selection or
release gates. Meanings/entailment remain next to style. No random arm,
capability benchmark, oracle, validation sweep or joint/larger-model follow-up.
Published Table1/arithmetic context has different/unknown input and evaluator
lineage; no exact numerical comparison is claimed by this report.

## Integrity, failure and budgets

Entry/exit use one merged deduplicated source/current-model traversal per live
command. Successful integrity/accepted/public artifacts publish only after exit
checks. Small stage reads check their own files, without recursive old model
hashing. Existing generation directories never rerun. Missing release/phase or
invalid resume waits before arming; busy resources wait before loading. After
arming, failures are permanent. Failed costs disclose known completed work and
separate irreversible starts rather than assuming interrupted work is free.

Totals:1810 generations (1800 main+10 mechanics),1950 extraction/main prompt
forwards (1500+450), separately bounded4 mechanics projection forwards,
2 logit forwards,0MC;90 first-valid native batches /180 maximum starts.

CPU checks run without bytecode/cache writes:

```sh
.venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_author_saps_method.py
.venv/bin/python -B -m scripts.author_saps_method --help
```

Pytest runs from existing original env; child CPU-only scripts use existing
SAPS env and tiny random TL models/local tokenizers. No pytest install or7B
model loading is needed. The worker does not execute prepare or live stages.
