# Gemma 4 31B: QASPER QLoRA study plan

Prospective design, written by the advisor on 2026-11-19 before any Gemma task
training, development generation or final-split generation. The owner authorized
**QASPER only**. There is no PRL adapter and no Hugging Face upload in this study.

## Question and claim scope

The question: does the **same QLoRA recipe on the same prepared QASPER data**
improve `gemma-4-31B-it` beyond Gemma's own unadapted base and its own
development-selected prompt, without failing the fixed retention limits?

- The primary claim is **within Gemma**: the adapter against Gemma baselines.
- Comparison with the Qwen3.8-27B result is descriptive and enters no gate.
- The 403-question final set has already been used in project evaluation, for
  Qwen. It is a **reused project evaluation set, not a fresh held-out test**.
  Gemma itself has not been tuned on it, and no Gemma output on it exists.
  Every report must carry this label.
- The existing limits carry over: public NLP papers may be in pretraining; there
  is no physics or general scientific-comprehension claim; scores are on a custom
  text-only subset, not the official QASPER benchmark; and retention gates are
  point-estimate limits, not noninferiority tests.

## Fixed inputs (reused unchanged)

- Model: `google/gemma-4-31B-it` at revision
  `842da3794eaa0b77d5f08bae87a17459d91ff475`, text decoder only, NF4 with double
  quantization and BF16 compute. This is the profile from the
  [preflight](gemma4_preflight.md).
- Environment: the isolated science environment from the preflight (Torch
  2.12.0+cu130, Transformers 5.23.0, PEFT 0.21.0, bitsandbytes 0.49.2, Accelerate
  1.12.0), with no upgrades.
- Prepared data: the Qwen prepared bytes, bound by hash. The prepared manifest is
  `89c5e45e…9464`; train input `bb9b1664…f897` and development input
  `7028881d…51fc4`.
  - Training uses the **482 selected train IDs** and the **101 development IDs**
    listed in `results/protocols/qwen38-qasper-02.json`, including its exclusions.
  - Prompt selection uses the **35 frozen IDs** from
    `results/protocols/qwen38-qasper-evaluation-01.json` (input `3bde8d40…`).
  - The final evaluation uses the 403 records (input `6dc33a31…`), with the
    frozen primary (309) and evidence-strict primary (267) memberships.
- The worker re-tokenizes these bytes for Gemma. Records are not re-selected,
  re-cut, expanded or re-ordered.

## Arms

| Arm | Weights | Prompt |
| --- | --- | --- |
| Base | Unadapted, with the LoRA wrapper attached and disabled | Base prompt |
| Prompt | Unadapted, with the LoRA wrapper attached and disabled | Development-selected prompt from base, verify and schema. If base is selected, this arm is an alias of Base, not an independent arm |
| Adapter | Selected checkpoint | Base prompt, the training prompt |

All arms share one loading path: identical FP32 preparation of the non-quantized
floating modules, the same wrapper and the same generation code. The only
differences are whether the adapter is enabled and which prompt text is used.

## Training configuration (one run, no sweep)

| Item | Value |
| --- | --- |
| Rank, alpha | 16, 32 |
| Target modules | All 410 decoder projections: q, k, v, o, gate, up and down. The global layers share K=V, so there are 50 `v_proj` modules. Vision and audio towers are excluded |
| Learning rate, schedule | 5e-5. Linear warmup over 4 steps, then linear decay to 0 at step 61 (Qwen `lr_factor`) |
| Optimizer | Torch AdamW, weight decay 0.01, maximum gradient norm 1.0 |
| Batch | Batch 1, gradient accumulation 8, equal-example mean loss per group. The final partial group is divided by its actual size |
| Steps | 1 epoch = 61 optimizer steps |
| Registered checkpoints | 21, 41 and 61. File hashes are recorded at save time |
| Selection | Minimum token-weighted development response NLL (answer plus `<turn|>`) on the 101 development IDs. The earliest step wins ties |
| Seed | 42. Example order is `random.Random(seed + epoch).shuffle` over the 482 IDs in Qwen input order, which reproduces the Qwen order |
| Maximum length | 8192, with no truncation of input or response |
| Memory profile | Gradient checkpointing (non-reentrant), SDPA, response-only logits |
| Precision | Outer `torch.autocast("cuda", dtype=torch.bfloat16)` around every forward: training loss, development NLL, generation and retention MC. Parameters stay FP32 after preparation, AdamW state is FP32, and norms run in FP32 wherever the model forces it. This matches Qwen. See below |
| Dropout | **0.05**. See the decision below |

### Dropout decision (fixed now)

**Use 0.05, matching Qwen.** The design is "same recipe, same data". The only
other differences are ones the model family forces: the template, the end token
and the module set. Dropout is not forced, so changing it would add an avoidable
confound to the descriptive cross-family comparison.

The preflight measured dropout 0. Dropout 0.05 adds per-module dropout
activations, which gradient checkpointing recomputes layer by layer. The added
peak should be well inside the 7.24 GB measured headroom. The longest real
training sequence is 7,091 tokens, against the 8,192-token synthetic probe.

Before step 1, the worker runs **one mechanics-only forward and backward pass**
on the longest training record with dropout 0.05, and discards it: no optimizer
step and no saved state. This is a memory check, not a quality probe.

- If it fits, train with 0.05.
- If it does not fit, fall back to dropout 0 through a registered amendment
  written before training, and disclose the change from Qwen.

No other reason may change dropout.

### Precision decision (fixed now)

**Use outer BF16 autocast, matching the frozen Qwen trainer and evaluator.**
After PEFT preparation the embedding and other floating modules are FP32. Without
autocast, activations, SDPA and the LoRA matmuls would run in FP32, which is a
different numeric recipe from Qwen and slower. This is a pre-output mechanics
choice, not quality tuning.

**Disclosure:** the [preflight](gemma4_preflight.md) ran *without* outer
autocast. Its memory and timing figures describe that profile, not this one.
The study profile is pinned by the startup mechanics below.

Before step 1, startup mechanics run under the actual study setup: BF16
autocast, FP32 prepared parameters, the real model, and training row 0 for the
short checks. The tolerance is `atol = rtol = 1e-4`, registered before the
check. Gates:

- **G1, the model's compact forward at native head shape.** Compare
  `model(..., logits_to_keep=positions).logits` with the full forward's
  `logits[:, positions]`, both upcast to FP32. Maximum absolute logit
  difference ≤ 1e-4, and cross-entropy difference ≤ 1e-4.
- **G2, loss and gradient parity on one shared logit tensor.** Take one full
  forward's logits as a detached FP32 leaf `L`. Compute the masked full
  cross-entropy over `L[:, :-1]` against `labels[:, 1:]`, and the
  response-loss arithmetic on `L[:, positions]` against
  `labels[:, positions + 1]`. Loss difference ≤ 1e-4; gradient with respect to
  `L` within 1e-4; and that gradient exactly zero on every unsupervised row.
  This tests the shift, mask, position and target logic without
  shape-dependent BF16 effects.
- **G3, the longest training row in the actual training setup.** Training
  mode, dropout 0.05, BF16 autocast. Finite loss; all 820 adapter gradients
  present and finite; peak memory recorded. The initial adapter, optimizer,
  scheduler, RNG and buffers are restored, gradients are cleared, and there
  are 0 optimizer updates.

Reported diagnostics, not gates:

- **D1:** native full-path versus compact-path LoRA gradients (maximum
  absolute and relative error, tolerance ratio).
- **D2:** the compact path run twice on the same input, with its maximum
  gradient difference.

D1 makes no claim that the native gradients are equivalent. A D1 mismatch with
D2 = 0 is *consistent with* BF16 rounding that depends on the head matmul's row
count; D2 alone does not prove that cause.

**Disclosure of the redesign.** An earlier pre-registration check (mechanics 02)
required elementwise native full-versus-compact LoRA gradient agreement at 1e-4.
It failed before any optimizer update or study data:

- full and compact cross-entropy were identical (9.066858291625977);
- all 820 gradients were finite;
- the maximum absolute gradient difference was 0.0078125, a tolerance ratio
  of 65.9.

That requirement was mis-specified for a BF16 backbone. It was replaced by
G1–G3 with the same tolerance, before the whole-study registration. The
failed receipt is preserved and bound in that registration.

After the startup checks, the initial RNG state is restored and gradients are
cleared, so the checks do not alter training. If a gate fails, the tolerance is
not widened; it is a mechanics failure handled by a registered amendment.

Numeric requirements under autocast:

- Cross-entropy and log-softmax run on FP32 logits. Upcast before the loss.
- Gemma's final logit soft-cap (30) is applied inside the model, on BF16
  logits. This is accepted as part of the native runtime and disclosed.
- Gradient checkpointing recomputes under the same autocast state (the
  non-reentrant default preserves it).
- Retention MC letter ties under BF16 logits follow the existing Qwen rule
  (`qwen38_retention.py`: argmax over A–D, first letter wins ties). Report the
  number of tied items per arm.

## Formatting and tokenization

- Messages are rendered with Gemma's native chat template, thinking off. The
  message roles and texts are the same as Qwen's for each prompt variant.
- Training sequence = the native generation prompt, including the empty thought
  block `<|channel>thought\n<channel|>`, then the answer JSON, then `<turn|>`
  (id 106).
  - The prompt and the empty thought block are masked.
  - The answer and `<turn|>` are supervised.
  - EOS (id 1) is never appended or trained.
- Generation uses exactly the same prompt prefix, greedy decoding with the cache,
  batch 1, a 512-token generation cap and an 8192 sequence cap.
  - Stop tokens are an explicit text-only override, **{1, 106}**, the same as
    the preflight. The native `generation_config` end set, [1, 106, 50], is
    recorded. Token 50 (`<|tool_response>`) is excluded because this study
    uses no tools.
  - The scored text is the new tokens before the first 1 or 106, decoded
    with special tokens kept (`skip_special_tokens=False`). No other
    stripping is done, so token 50 and any other control marker stay visible
    to the strict parser.
  - If a stray channel or thought marker is emitted, it is left in place and
    scored by the unchanged strict parser. In practice it is usually invalid
    JSON.
- **Coverage check before any output, tokenization only.** For every
  development, prompt-selection and final record under each of the three prompt
  variants, verify prompt tokens + 512 ≤ 8192.
  - Report the maximum and the count over the limit.
  - With Qwen the margin was about 940 tokens, and Gemma tokenizes about 2%
    longer, so zero overflows are expected.
  - If any record overflows, stop. Register a disclosed handling rule before
    any generation. Never truncate silently.
  - This check reads final-split prompts only to count tokens; it produces no
    outputs or scores.

## Prompt selection (development only)

- Run base, verify and schema on the 35 frozen IDs with the Gemma base model
  (adapter off).
- Choose the largest **strict primary Answer F1**. Ties go to base, then verify,
  then schema.
- Record the scores, the selected prompt text hash and that no final-split data
  was used.
- Selection is final once it is recorded. Qwen's choice (schema) is not carried
  over.

## Final evaluation and frozen gates

The gates are unchanged from Qwen (`qwen38-qasper-evaluation-01`). The strict
parser decides every gate; lenient results are diagnostic only.

1. Adapter strict primary Answer F1 (n = 309) gain **≥ 0.05 over each
   baseline**.
2. For each baseline, the 95% interval of the paired paper bootstrap excludes
   zero. The bootstrap uses 10,000 draws, seed 20261117, paper as the unit, and
   the percentile method. An aliased Prompt arm reuses the Base comparison.
3. Evidence F1 on the evidence-strict primary subset (n = 267): the adapter may
   be at most **0.02 below** the better-answer baseline. That baseline is chosen
   by strict primary Answer F1, and base wins ties.
4. Invalid JSON over all 403: **≤ 2%**.
5. False-unanswerable rate on all-answerable primary rows: the increase over the
   better baseline is **≤ 0.05**.
6. Retention, as defined in the next section.

A science pass requires gates 1–5. A recommendation requires gates 1–6.
Subgroups, lenient metrics, exact-evidence scores and any Qwen comparison are
reported as descriptive results only.

**Analysis freeze.** Before *any* final-split generation, PM registers the
following, recording hashes where content already exists:

- the analysis script and scorer, by source hash;
- these gates and the bootstrap settings, verbatim;
- a **prospective run mapping**: each arm's role, its planned output path, the
  source and input hashes that will produce it, and the alias rule. It holds
  only roles, paths and rules.

Prediction and result files do not exist yet, so no hash of them is recorded
or invented at freeze time. After generation, a separate **completion seal**
records the SHA256 of each produced prediction and result file. The seal
cites the frozen registration hash and checks every artifact against the
prospective mapping (role, path, source and input hashes). Any mismatch is a
deterministic-bug case under the policy below.

Reusing the Qwen analysis and scorer code with family-neutral parsing is
preferred. Final outputs are not scored or viewed before the freeze.

## Retention

- Use the same fixture items: 128 general MC, 128 GSM8K and 32 IFEval.
- Use the same scorers, the same generation caps (GSM8K 512, IFEval 1024) and
  greedy decoding.
- MC uses full-vocabulary log-probabilities of single answer letters. CPU tests
  check that each letter is a single token in Gemma's tokenizer.
- The MC prompt is the raw fixture text (no chat template), as for Qwen, with
  Gemma's BOS token (id 2) prepended. The Gemma tokenizer does not add BOS by
  itself, and Qwen has no BOS, so this is the Gemma-native form of the same
  raw prompt. Both arms use it.
- Arms: the matched base (wrapper disabled, same FP32 preparation) and the
  selected adapter.
- Gates on net loss in counts: MC ≤ 5, GSM8K ≤ 5 (fallback numeric, with strict
  also reported), IFEval strict ≤ 3.
- **Batch 1 throughout.** Batching would require a separately predefined
  mechanics amendment registered before any retention output.
- No noninferiority or broad-capability claim.

## Execution

- GPU 0 runs training, development NLL and checkpoint selection.
- GPU 1 runs everything else. These jobs need no adapter and can run during
  training:
  - prompt selection;
  - Base and Prompt final generation (after the analysis freeze);
  - base retention.
- After selection, the adapter final generation and adapter retention can be
  split across both GPUs.
- Every job runs in its own **persistent Herdr pane**, not as a tool subprocess.
  Each job writes its own receipt.
- There are no study-level resource caps. Each generation is bounded by its
  token cap.
- **GPU sharing with idle neighbours (operational, not a gate).** Other
  projects' processes may already hold a CUDA context on a study GPU; the
  study never stops or changes them. A job starts only if, at launch, the
  device's total used memory is ≤ 1 GiB and utilization is ≤ 5%, read from
  one NVML snapshot immediately before model loading. Larger or active
  occupancy is refused, and the job does not start. The memory rule is the
  main safeguard; a single utilization reading is a weak signal, which is
  disclosed. Each job records the full occupancy it found
  (processes, memory, utilization) in its receipt. The first case was a
  464 MiB idle context on GPU 0 (487 MiB total, 0% utilization).
  - Measured timings are therefore not exclusive-device timings and are not
    claimed as such.
  - The preflight headroom (7.24 GB free at the 8K steady-state step) covers
    an idle resident context of this size.
  - The watchdog also logs other processes' memory on study GPUs, so later
    neighbour growth is visible.

**Rough planning estimates.** These are scenarios, not bounds:

- Training: about 6–7 GPU-hours.
  - Real records (median 5.7K tokens) run between the measured 4K and 8K rates.
  - Add about 1–1.5 h for development NLL at three checkpoints.
- Final generation: 403 × 3 arms, plus 105 prompt-selection generations, at
  roughly 20–40 s each (about 5.2K-token prefill plus a short JSON answer).
  That is about 7–15 GPU-hours.
- Retention: 2 arms, about 2–4 GPU-hours. Most of it is GSM8K and IFEval
  decoding.
- Calendar: roughly 12–18 h when GPU 1 overlaps with training.
- For comparison, the Qwen QASPER ledger was 5.11 h of training and 10.56 h of
  evaluation.

**Watchdog, for hung jobs only.** A separate Herdr pane polls every 60 s:

- each job's heartbeat file (last completed step or item ID, timestamp);
- process state;
- `nvidia-smi` utilization and memory.

A job is flagged as **hung** when there has been no new completed unit for
longer than a stall threshold and GPU utilization is near zero, or the process
is gone or defunct. The thresholds:

| Job type | Stall threshold | Typical unit time |
| --- | --- | --- |
| Training | 30 min | One optimizer step ≈ 8 records ≈ 8–12 min |
| Generation | 15 min | One item ≤ about 3 min at the cap |
| Retention | 15 min | |

When a job is flagged:

1. The watchdog records a flag file with the GPU state and the log tail.
   It does not stop the job itself.
2. The PM inspects the flag, captures a stack dump if useful, and decides
   whether to stop the job.
3. A stopped job resumes under the recovery policy below.

The watchdog never stops a job that is still making progress. It never judges
output quality.

## Bug and recovery policy (prospective)

**Deterministic bugs** cover wrong masking, template or tokenization defects,
parser crashes, wrong IDs, hash mismatches and code exceptions.

1. Stop the affected job.
2. Preserve every output, log and receipt.
3. Fix the source.
4. Register an amendment *before generating more data*. It records:
   - a new run ID;
   - the old and new source hashes;
   - the reason;
   - which outputs are invalidated.

What follows depends on where the bug is:

- A bug confined to scoring may be re-scored under the amendment. All versions
  are reported.
- A bug affecting training or generation reruns the affected arm or run
  completely under the new registration. Superseded outputs are kept and
  labelled.

Never allowed: changes driven by output quality. That covers prompt edits,
hyperparameters, data or ID changes, checkpoint overrides, gate or parser
changes, extra steps or epochs, and dropping hard items.

**Memory failure on a real record.** Do not truncate or skip the record. Only
math-preserving mechanics changes are allowed, for example allocator settings
or chunked loss computation. Each must be registered and tested on CPU for
loss parity before resuming.
Any out-of-memory error stops the job, and the code refuses a plain resume. If
the watchdog's per-process log shows that another process's memory grew on that
GPU before the failure, the PM records that cause in the amendment, which may
then re-run the job with no source change. Either way, the cause and the
watchdog evidence are disclosed.

**Interruptions** (SIGTERM, reboot, a hung job flagged by the watchdog):

- **Training resumes only from an exact optimizer-step boundary.** Save resume
  state after every optimizer step and **keep every boundary, steps 0–61**
  (about 1.5 GB each, about 90 GB in total; disk has room). Each boundary is
  immutable and checksummed, which also preserves the full optimization
  history for audit. Resume state includes:
  - adapter weights;
  - AdamW moments;
  - scheduler state;
  - step and data-cursor (accumulation-group index);
  - Python, NumPy, Torch CPU and CUDA RNG states.

  A partially accumulated group is discarded and recomputed from its first
  record, so no sample contributes twice to any update. Resumed training is
  equivalent in data, order and state. It is not claimed bitwise identical,
  because GPU kernels are nondeterministic. Each resume is disclosed in the
  receipt.
- **Generation uses append-only per-item records** (ID, arm, prompt hash,
  output, token counts). A restart verifies the existing records and skips
  completed IDs. An item interrupted mid-generation is rerun from scratch.
  Completed items are never regenerated, discarded or duplicated. The precedent
  is the Qwen baseline-completion amendment.

**Ownership.**

- The worker writes the new Gemma code and its CPU tests only.
- The PM owns registration, configuration, final analysis and closeout.
- The advisor reviews the code before any registration.
- Frozen Qwen sources, reviews, protocols and results are never edited.

## Outputs and release gate

- The receipts include:
  - the training record with losses and selection;
  - the prompt-selection record;
  - the coverage check;
  - the final science report;
  - the retention report;
  - the combined final report, labelled "reused project evaluation set".
- A null or failed result counts as completion and is published as it is in
  the docs.
- **No Hugging Face upload** happens in this study. Any later release requires
  a passing quality result, saved-tensor validation in a fresh-process reload,
  and a separate owner decision.
