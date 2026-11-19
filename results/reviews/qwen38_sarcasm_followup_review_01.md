# Qwen3.8-27B sarcasm follow-up: code review 01

Advisor review, 2026-11-19. It covers the worker's files as of 11:21:
- `scripts/qwen38_sarcasm_followup.py` (launcher, 455 lines);
- `scripts/qwen38_sarcasm_followup_telemetry.py` (241 lines);
- `tests/test_qwen38_sarcasm_followup.py`;
- `configs/qwen38_sarcasm_followup.json`.

They are checked against `docs/qwen38_sarcasm_followup_plan.md` (sha256
`dd6b7b76…`). The review was read-only, plus CPU tests and `nvidia-smi -L`.
No model or judge call was made.

`outputs/qwen38-27b-nf4-01/env/bin/python -m pytest -m '' -q tests/test_qwen38_sarcasm_followup.py`
gives **7 passed**.

## Status: not ready for registration

The telemetry and launcher core are sound. B1–B4 block registration.

## Verified correct

**Recording steerer (`InstrumentedSteerer`).**
- It subclasses the frozen `Steerer` and never replaces its output.
- Hook order on each site's `self_attn` is observer-before, frozen shift,
  observer-after. PyTorch passes the frozen hook's returned output to later
  hooks, so the "after" observer sees the steered tensor.
- The "before" observer recomputes demand, gate and coefficient with the
  frozen expression and the same dtypes. `requested = coef·Δ.to(original)` is
  bit-for-bit the operand the frozen hook adds.
- `actual = ‖steered.float() − original.float()‖`, per token. The algebraic
  coef·‖Δ‖ and the bound 2b‖Δ‖ are kept separate. Prefill and decode are
  split, and decode steps are traced.
- Zero native norms are excluded from ratios and counted, not divided by an
  epsilon. Nonfinite telemetry raises.
- In tests, the instrumented steerer is bit-equal to the frozen one in logits
  and cached tokens at b ∈ {0, 2}. At b=0 the actual norm sums to exactly 0.
  A rounding unit test shows the actual norm can be 0 while the requested
  norm is above 0.

**Part A capture.**
- `ActivationCapture` stores FP32 response-token means of all 64 block outputs
  and the 6 attention outputs. With `retain_attention_tokens` it also keeps
  the per-token S/L attention outputs in native dtype (for §A2), and only for
  contexts that have a system span.
- The crop and decomposition use the frozen extraction arithmetic. The NLL is
  computed at response positions via `logits_to_keep = positions − 1`.
- `replay_ids` appends the saved raw IDs without re-tokenizing, checks that
  the decode equals the judged output, and excludes special tokens from the
  mask.
- `build_inputs` asserts that S and L have equal total and system lengths,
  that user-turn tokens are identical, that the continuation and mask are
  identical across contexts, and that the old row receipts and raw rows
  match.

**Launcher.**
- **Single approval.** One approval binds the manifest, the launcher hash,
  the algorithm string and the GPU UUID. Every bound source and input and
  every frozen copy is re-hashed at start.
- **GPU.** The configured UUID `GPU-34646ab9…` is GPU 0 (`nvidia-smi -L`).
  Preflight refuses if that GPU is busy and never falls back to another.
- **Rows and counters.** Rows are never retried; completed rows hash-replay.
  Counters match the registered counts: 232 generations (8 mechanics + 224),
  36,096 tokens, 194 teacher-forced passes, 5 logit forwards, and 169 judge
  starts (1 audit + 168).
- **Time limits.** A 4 h GPU limit and a 12 h wall limit are enforced by
  parent-process timeouts that kill the process group. Stage-entry budget
  checks run before Part A and Part B.
- **Stage order and stops.** The stages run as audit, GPU (mechanics, then
  Part A, then Part B), then judging. A failed audit still runs the GPU
  stages and skips judging, as §6 requires.
- **Mechanics.** It checks b=0 parity (tokens and logits), frozen b2 against
  recording b2 (tokens and logits), cleanup, coverage, capture neutrality
  (`torch.equal` logits with and without capture on C01_P_S), and the F4
  decomposition thresholds.
- **Judge.** The judge plan uses `make_plan`, which gives 224 / 8 = 28 batches
  per metric with 7 distinct questions' arms spread across batches and opaque
  IDs. Each call checks identity before and after. Streams run trait, then
  coherence, then impoliteness, sequentially, so impoliteness starts only
  after both primary streams are complete.

## Blocking (fix before registration)

**B1. The analyzer is missing.** `launch()` imports
`scripts.qwen38_sarcasm_followup_analyze`, and `SOURCES` lists it, so
`prepare()` cannot even compute the source closure. It must implement:
- A1–A5 with the **question-cluster bootstrap**: resample 32 question IDs and
  carry all 6 cells of each together, using seed 20261119 and 10,000
  resamples;
- signed cosines with the 3 random-direction reference values and the CI
  wording rule (§3);
- §A2 (a) per-token paired norms and relative norms with explicit
  denominators, and (b) the norm of the pooled mean;
- Part B gates through the unchanged frozen
  `sarcasm_method_transfer_common.summarize`/`gates`;
- the `exploratory_candidate` / `exploratory_null_through_b16` labels;
- `b16_direction_check`;
- the coherent-response checks;
- the impoliteness descriptives;
- telemetry summaries (actual, requested and algebraic, prefill and decode
  separate);
- the ledger;
- the terminal statuses: `implementation_stop`, `budget_stop`,
  `judge_audit_failure`, and, for a failure during impoliteness with complete
  trait and coherence, `primary_complete_descriptive_incomplete`.

**B2. Mechanics criterion 5 is not implemented.** The plan requires that on
one probe call, the logged requested and actual norms equal a **manual
recomputation** from the captured tensors, within FP32 relative 1e-5. The
launcher checks only the coefficient range and b=0 zero sums.
- *Fix:* give `InstrumentedSteerer` an opt-in `retain_last=True` that keeps,
  for the most recent call per site, the `original`, `requested` and steered
  tensors.
- In the record2 logit forward, recompute ‖requested‖ (model dtype,
  accumulated in FP32) and ‖steered − original‖ (FP32) per token, and compare
  them with the logged values.
- Add a CPU test of the same check on the tiny model.

**B3. Part A prompts are not asserted equal to the prompts the old content
was generated under.** The plan's replay depends on the N prompt being the
old baseline prompt and the S prompt being the old prompted prompt. My CPU
tokenizer check found they are equal today. The code re-renders them, though,
and does not assert it.
- *Fix:* in `build_inputs`, assert
  `rendered['N']['input_ids'] == lookup[q,'baseline']['input_ids']` and
  `rendered['S']['input_ids'] == lookup[q,'prompted']['input_ids']` for every
  question. Fail at prepare otherwise.

**B4. The CPU tests are far short of plan §9.** Add:
1. recorder vs frozen parity at **b = 16**; the plan lists {0, 2, 16};
2. the B2 manual-recompute check;
3. `diagnostic_forward`:
   - capture is bit-neutral, meaning the logits equal a plain forward;
   - the crop mean equals the frozen `port.extract` delta;
   - the block means equal a manual mean of `output_hidden_states`;
   - per-token attention retention happens only for S and L;
4. `build_inputs` failure paths: S/L length mismatch, the B3 assertion,
   continuation drift, and a changed old receipt;
5. plans: 28 batches per metric, 8 distinct questions per batch, each item
   once per metric;
6. launcher guards, with GPU and judge stubbed:
   - budget refusal before a stage;
   - resume without re-running completed rows;
   - an `Uncertain` raise on a started-but-unfinished row;
   - the audit-failure path running GPU and skipping judging;
   - the counter caps;
7. analyzer tests:
   - the question-cluster bootstrap on synthetic data, deterministic;
   - an API test showing that token-level resampling is impossible;
   - the gates and outcome labels;
   - the b16 direction check;
   - the primary-complete path.

## Recommended (cheap; include before registration)

- **R1. Non-zero check in mechanics.** Assert
  `not torch.equal(values[0], values[2])`, meaning ordinary ≠ frozen b2. If
  both the frozen and the recording hook silently became no-ops, the
  equality checks would still pass.
- **R2. Random arm integrity at load.** Assert that each site's ‖Δ_random‖
  equals ‖Δ_bank‖ within relative 1e-5, and that the random K_pos and D̄ are
  `torch.equal` to the bank's. The frozen hash already binds the file; this
  makes the "norm-matched" property explicit at runtime.
- **R3. Activation storage headroom.** The expected size is about 1.55 GB:
  1.26 GB of S/L per-token attention plus about 0.27 GB of block means. The
  cap is 2 GiB, which is adequate. Record the actual size in the Part A
  result.

## Notes

- Using `plan_seed + i` per metric gives three different but fixed plans.
  That is fine; the plan text will say "seed 20261120 + metric index".
- After the analyzer lands, I will re-review the launcher together with it,
  focusing on the status mapping and the analysis arithmetic.
