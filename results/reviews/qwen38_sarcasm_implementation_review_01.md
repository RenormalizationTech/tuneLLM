# Qwen3.8-27B sarcasm GCAD: implementation review 01

Advisor review, 2026-11-18. It covers three uncommitted files:
`scripts/qwen38_gcad_port.py`, `tests/test_qwen38_gcad_port.py` and
`configs/qwen38_sarcasm.json`. They are checked against the amended
`docs/qwen38_sarcasm_plan.md` and the [design review](qwen38_sarcasm_design_review_01.md).
No model, judge or GPU call was made.

## What I ran (CPU only)

`CUDA_VISIBLE_DEVICES= outputs/qwen38-science-01/env/bin/python -m pytest -q -m research tests/test_qwen38_gcad_port.py`
gave **4 passed**.

The default `-m portable` profile deselects this module. A plain `pytest` run
therefore reports "4 deselected", not a pass. The registration record should
cite the explicit command above.

The installed `modeling_qwen3_5.py` hashes the same in both 27B environments
(`1a475c45…0a68`). Both have Transformers 5.23.0, torch 2.12.0+cu130 and
bitsandbytes 0.49.2. That is enough to accept a run under the config's
`outputs/qwen38-27b-nf4-01/env` while the tests ran in the science
environment. Record both hashes at R0.

## What the tests establish

- **Q/gate de-interleave.** It is bitwise equal to the native chunking. A
  global-half split is shown to differ.
- **Partial RoPE.** The 12 non-rotated dimensions are unchanged. The port's
  softmax weights equal the eager attention weights.
- **Gated crop.** system + other, with the gate applied, reproduces the native
  `self_attn` output. The same sum without the gate fails.
  `extract()`'s Δ matches the manual crop.
- **Zero strength.** b=0 gives identical full-prefix logits and identical
  cached greedy tokens. Hooks are absent on the linear block and on the
  out-of-window full block. Decode calls are counted.
- **Cleanup.** b=1 changes the logits, and they are restored after exit.
  Hooks are cleaned up even when an exception occurs.
- **Bank arithmetic.** Raw pair Δ, token-weighted key, demand.

## Findings

**I1. Config budgets predate the amended plan (fix before R0).** The config
still has `maximum_generations 462`, `maximum_teacher_forced 300` and
`maximum_logit_forwards 4`. The plan needs 468, 302 (2 synthetic + 300 real)
and 8 (4 + 4). Also add a maximum of 92,544 generated tokens, and the fixed
synthetic-phase inputs:
- instruction pair index 0;
- extraction question index 0;
- the frozen response text bytes and their hash.

Give the synthetic phase its own counters, so that its 2 teacher-forced
forwards cannot consume the 300 allowed for the real bank.

**I2. Judge runtime snapshot fields are missing.** Add these fields:
- `judge_snapshot`: an absolute path inside the run directory;
- `judge_source`: `~/.local/share/claude/versions/2.1.327`;
- `judge_sha256`: `343c744baa14bfd38286a7cb90871051f04379956da6c5b803a9e66019a76ac9`;
- `judge_version`: `2.1.327 (Claude Code)`;
- a new audit directory.

The existing `qwen38_judge_runtime.py` resolves `shutil.which("claude")` and
pins 2.1.326. Leave it unchanged as historical. The new wrapper must invoke
the snapshot path only, and re-check its hash and version before and after
every call.

**I3. Steered cached decode is not tested against a steered recompute (add
test).** The zero-strength test cannot detect a wrong decode-time position or
query. The coefficient is 0 there, whatever the demand. Add a tiny fp32 test.
With b=1 on a non-uniform bank, run greedy cached `generate` for N tokens.
Then run greedy steps without the cache, recomputing the full prefix. Require
identical tokens, and closeness of the last-position logits at each step.
This is the real check that `position_embeddings` and q are picked up
correctly in cached decode.

**I4. The GQA mapping is invisible to the current tests (add test).** In the
bank test and the steering tensors, the key rows are uniform or random
without a reference. A mistaken `repeat` (tiling) instead of
`repeat_interleave` would still pass. Add a test with distinct key/value-head
keys, for example `key[g] = (g+1)·e`, and a known q. It should check the
demand in `bank()` and the coefficient applied inside `Steerer` against a
manual `repeat_kv` mapping (head h → kv h // groups).

**I5. Skipped-call accounting (design review F7, still open).** The `shift`
hook reads `self.state[i]` through `qk_gate`. If the capture state is missing,
it raises a bare `KeyError`. Convert this to an explicit error. Add an
`errors`/`skipped` counter that must be 0 in the mechanics records. Prefill
classification by `T > 1` is acceptable for the fixed prompts; assert that
prompt length > 1 at entry.

**I6. The SDPA mask path in `extract()` is untested (add test or R0
assertion).** The tests use eager attention. The real model loads SDPA, where
the native mask may be `None` or a boolean `[B,1,T,T]`. The code handles
both. Add a tiny test that loads the same weights with
`attn_implementation='sdpa'` and checks that `extract()` returns the same Δ
as the eager load, within fp32 tolerance. The synthetic phase's decomposition
check (≤ 0.02) then covers the NF4/BF16 case on the real model.

**I7. Config changes are consistent with the plan as amended.**
- Using the saved NF4 artifact `artifacts/qwen38-27b-nf4-01`, bound by its
  manifest, is accepted. The plan is updated to say so.
- The mechanics prompts come from `calibration.json`. The plan now fixes them
  as development questions 0 and 1, never graded or used for selection.
- The layers, strengths, seeds, caps, minimum of 10 pairs and the 28,800 s GPU
  stop all match the plan.
- `max_sequence_tokens 2048` is a guard, not a scientific setting.

**I8. Pre-R0 completeness.** The tree does not yet contain the run
orchestration, the elicitation/judge plans, boundary assertions (F5),
random-control generation, or gate code. These need review before
registration. The scientific gate code should reuse the frozen
`sarcasm_method_transfer_common` summarize/gates helpers, imported unchanged
and hash-recorded, rather than reimplementing them.

## Policy reminder (plan §9a)

A deterministic implementation bug is an implementation stop, not a
scientific null. Preserve the failed evidence, diagnose it, and have PM
register the new source hashes before any further data. Never make a change
motivated by sarcasm or coherence scores.

## Status

The port is **accepted for its arithmetic.** I1, I2, I3, I4 and I5 must be
fixed before R0. I6 is strongly recommended. I8 is pending worker delivery.
