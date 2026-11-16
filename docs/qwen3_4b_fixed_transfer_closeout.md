# Qwen3-4B fixed-setting GCAD transfer: closeout

**Status: complete. No 4B controller qualified.** Aggregate:
`results/qwen3_4b_fixed_transfer_01.json` (immutable). Protocol:
[plan](qwen3_4b_fixed_transfer_plan.md). Reviews: design, port,
implementation and [interpretation](../results/reviews/qwen3_4b_fixed_transfer_interpretation.md).

## What was run

- **Model and runtime:**
  - `Qwen/Qwen3-4B` revision `1cfa9a72…`, in native BF16;
  - thinking explicitly off;
  - Qwen3's native default rendering, which has no system message.
- **Decoding:** the values resolved on 7B (repetition penalty 1.05, top-p .8,
  top-k 20), passed explicitly.
- **GCAD settings carried over unchanged from the retained 7B controllers:**
  - PF b = 5, impolite b = 3.5 and SARC b = 2;
  - gate scale 1.5 and `inverse=False`;
  - the predeclared proportional window, blocks 12–25 (14 blocks; 7B used
    9–19 of 28).
- **Fresh 4B banks:**
  - elicitation and filtering of 200 replies per trait;
  - eligible pairs: **PF 45, impolite 63, SARC 63**;
  - 513 teacher-forced passes (3E);
  - FP32 safetensors.

  No 7B vector, response or score was reused. There was no grid, no
  selection and no rescue.
- **Evaluation:** held-out only, with baseline, GCAD and a random Δ for each
  trait.
  - The random Δ matches the bank's raw per-layer Δ norm. It is not matched
    for realized shift energy.
  - PF and SARC: 32 greedy questions each, sharing one baseline.
  - Impolite: the author's 60 first replies, which use 13 unique first
    questions.

## Execution notes

- **Port:**
  - `config.head_dim`;
  - Qwen3 `q_norm`/`k_norm` applied before RoPE;
  - Qwen3 imports;
  - fail-closed geometry and applied/skipped call counters.
  The Qwen2 path is bit-identical to the author code.
- **Precision:** the model and gate are in BF16, as on 7B. Bank files are
  stored in FP32.
  - As in the 7B code, each side's response-pooled Δ mean is taken in BF16,
    then cast to FP32 before the difference.
  - K_pos and D̄ are computed from FP32 copies.
- **Extraction inference context:** all three extraction stages ran inside
  a global `torch.no_grad` wrapper. This is an execution-only setting
  (`results/protocols/qwen3-4b-extraction-inference-context-01.json`). It
  prevented autograd graph retention from the norm parameters, and its CPU
  tests showed bit-identical keys, demand and delta.
- **Impolite batches:** the arms were generated as three identical
  sub-batches of 20, one per random seed group, each seeded with 42. This
  replaced the 7B batch of 64.
- **Budget used:** 952 generations, 513 teacher-forced passes and 5 logit
  forwards. 252 of 252 grading batches were valid on the first attempt, with
  0 retries. R1 and R2 stayed intact.
- **Mechanics passed for every bank:**
  - exact token and logit parity at zero;
  - nonzero hidden-state and logit effects;
  - full prefill and cached-decode coverage, with no skipped hook calls.

## Results (held-out)

| Trait | Arms | Trait score | Coherence | Other | Status, failed gates (margin) |
| --- | --- | --- | --- | --- | --- |
| PF, n = 32 | baseline → GCAD (random) | 0 → **37.34** (0) | 87.00 → **20.56** (85.03) | lexicon 0 → 28/32; invalid 0 → **6/32** | `heldout_failure`: coherence absolute (−29.44), coherence relative (−46.44), invalid (−3) |
| SARC, n = 32 | baseline → GCAD (random) | 0 → **56.13** (0) | 87.00 → **63.91** (87.06) | 22/32 ≥ 50; invalid 0; impoliteness 1.28 → 32.31 (descriptive) | `heldout_failure`: coherence relative only (−3.09) |
| Impolite, n = 60 | baseline → GCAD (random) | numeric 1.30 → **80.18** (1.40) | 77.32 → **52.02** (78.18) | invalid 0; 60/60 EOS; mean length 497 → 142 tokens | `author_effect_only`: the author rule (+30) **passed** (+48.88 over); the 4B qualification failed only on coherence relative (−5.30) |

**Reading the results:**

- **Every GCAD arm induced its trait strongly, and every random arm stayed at
  baseline.** So this is **not** a failure to steer.
- **No trait met its predeclared quality gates.**
  - PF degraded heavily: coherence 20.6 and 6 invalid answers.
  - SARC and impolite failed only the gate allowing a coherence loss of at
    most 20 points, by 3.1 and 5.3 points.
- **Impolite: the two rules are reported separately.**
  - The author-effect rule is the unchanged 7B rule, and it passed.
  - The coherence-relative clause belongs to the new 4B qualification,
    declared before any output. 7B had no such gate and no random arm, so
    this is not a retroactive 7B criterion. For reference, 7B's descriptive
    coherence loss was 6.8 points.
- The SARC result excluding H14 is essentially unchanged (55.52 and 64.19),
  and is descriptive only.

## Refusal (carried forward, no new compute)

Qwen3-4B conventional refusal is unchanged:

- **Raw subtraction, block 23, h − 0.5v:** source-unsafe refusals 29 → 22
  and valid answers 1 → 7, but benign usefulness fell 29/42 → 23/42. That is
  a partial effect with a **failed** quality rule.
- **All-block projection:** not selected, with 6/30 valid answers in
  development.

No refusal controller is retained.

## Conclusions and limits

- **No 4B bank is published, and no new 4B Hugging Face steering repository
  is created.** The existing 4B quantization and physics repositories are
  unaffected.
  All 4B outcomes are released as aggregates on GitHub and are linked from
  the 7B card's related-results section.
- **No rescue.** There is no strength reduction, window change, re-extraction
  or rerun. Lower or retuned 4B strengths would be a new, separately
  registered study.
- **Outcomes are model-specific.** Differences between checkpoints do not
  establish an effect of model size, and a 4B null does not show that the
  method fails on small models.
- **Settings are carried over, not matched strengths.** The 4B Δ norms
  differ, and the window has 14 blocks, so b is a fixed setting.
  - The realized mean shift norms were PF 6.09, SARC 2.10 and impolite 3.58,
    against 7B's PF 7.61 and SARC 4.04.
  - Which factor drove the quality loss was **not tested**. Candidates include
    strength, window size, the system message and the model.
- **Limits:**
  - one Claude judge family;
  - Claude-generated PF/SARC inputs;
  - an impolite bank built by our recipe, whereas 7B used the author's bank;
  - decoding pinned to the 7B values;
  - truncation at the cap (all greedy PF/SARC arms hit 160);
  - coherence is not factual accuracy;
  - n = 32/32/60.
