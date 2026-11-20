# Gemma 4 31B QASPER study: advisor source review 01

Advisor review, 2026-11-19. Read-only: the advisor made no model, GPU or judge
calls and edited no source. No Gemma task-training update or final-split output
existed when this review was written.

**Verdict: the source is accepted for registration (R0) and launch**, at the
hashes below. If any listed file changes, this signoff no longer applies to it.

## Signed source

| File | SHA256 |
| --- | --- |
| `scripts/gemma4_qasper_runtime.py` | `7a47b21c64733c8f41318df8019cd34d4bc3617c39dbeaf50522938efef7c247` |
| `scripts/gemma4_train_qasper.py` | `de1c50234cd5d51f9aeb9b7b783c3614f482b38ce0276b3887d8ee69fca5beb4` |
| `scripts/gemma4_eval_qasper.py` | `6838680092ea587de5612217c625d2127d37d52611db1cc4fa0d40647a658b76` |
| `scripts/gemma4_retention.py` | `69f4869920a5e8de8b570fa9ac27557669c18dde3f267a1ec6cb04e091ac64b8` |
| `scripts/gemma4_preflight.py` | `b8e8479b02af7a4e00209a4d14c677298135a3d46ebc01f6071f0983ecba2232` |
| `scripts/gemma4_qasper_analysis.py` | `f957f8a0de70b6a4624ea6c61f27c90e6ebc84a65290e7ab5be0596957337f2a` |
| `scripts/register_gemma4_qasper.py` | `aa151635de108f03fce9cd695048692e6f1e594aa0b608afb4ca76e199bd20cd` |
| `scripts/gemma4_qasper_pipeline.py` | `59616d92b26423dd73e730f7221592095fc18e7426cea2ef8003eba2ee77dee9` |
| `scripts/gemma4_qasper_watch.py` | `8afe04b8f6bc6c9a4bfd7680434dbf8d32e70119b7431dbc14aead5b5e48610b` |
| `scripts/gemma4_qasper_mechanics.py` | `b05a176c6a76838d2b2288e6b4c7c7fabbd5d451f770b84b1b7b625df0e24758` |
| `scripts/gemma4_refresh_qasper_preparation.py` | `2894a04e4a76927fca56a446f484ff58677407cba69ada62639cff48cd7172bc` |
| `tests/test_gemma4_qasper.py` | `ad5404b6e433f7428fe4bee2311bc8b7ef98180ffdc516a6c1c8b9a81f0cd561` |
| `tests/test_gemma4_qasper_analysis.py` | `f87dcccbd517d8b49fe17dae13bdb3582e7b9c2b04d1e93ea42b9a36b6f7c48a` |
| `docs/gemma4_qasper_plan.md` | `6d9d1a090d290cd339fd0ca85dc6488fdaaa50bd1e0c0f48af52740c316eac96` |

The frozen Qwen helpers are hash-checked by the analyzer at load time:

- `qwen38_science_score.py`
- `qwen38_qasper_analysis.py`
- `qwen38_retention.py`
- `qwen38_eval_science.py`
- `evaluation/gsm8k.py`

**CPU tests:** 83 passed. This is `tests/test_gemma4_qasper.py` plus
`tests/test_gemma4_qasper_analysis.py`, run in the isolated science environment
with `CUDA_VISIBLE_DEVICES=""` and `OMP_NUM_THREADS=1`, in 37 s.

## What was checked against the plan

**Training**
- **Data:**
  - Qwen's 482 training IDs, in Qwen's input order;
  - Qwen's 101 development IDs;
  - prepared bytes bound by hash;
  - the base prompt is empty, so the training and evaluation prompts match.
- **Optimization:**
  - `Random(42)` groups of 8, with the final group of 2 divided by its actual
    size;
  - Qwen's learning-rate factor;
  - rank 16, alpha 32, dropout 0.05, AdamW (weight decay 0.01, clip 1.0);
  - 61 steps, development NLL at steps 21, 41 and 61, earliest step wins ties.
- **Formatting and precision:**
  - the native empty-thought generation prefix is masked;
  - `<turn|>` (106) is supervised and EOS (1) is never trained;
  - outer BF16 autocast, with FP32 cross-entropy.

**Recovery and execution**
- **Checkpoints:** immutable, checksummed checkpoints at every optimizer
  boundary from step 0 to step 61. Each holds the adapter, optimizer,
  scheduler, step and all RNG states.
- **Interrupted work:** an interrupted accumulation group is rolled back.
  Interrupted generation rows are rerun, completed rows are never repeated.
- **Deterministic errors:** these refuse a plain resume.
- **GPU check:** the occupancy check runs before any attempt record is
  written.
- **Pipeline:** `--resume` and a whole-pipeline `flock`.

**Evaluation**
- **Weights per arm:** development loss, generation and MC all run in eval
  mode, so dropout is off. Baselines use `disable_adapter()`. The adapter arm
  reloads with exact tensor equality.
- **Generation:**
  - greedy decoding with the cache, batch 1;
  - stops on {1, 106}; the native end set [1, 106, 50] is recorded;
  - decoding keeps special tokens and strips nothing else;
  - the coverage check blocks any prompt plus 512 tokens over 8,192, and
    truncation is refused.

**Retention**
- **MC:**
  - the raw fixture prompt with Gemma BOS (id 2) prepended;
  - single-letter boundaries checked;
  - Qwen's argmax and first-wins tie rule.
- **Generation tasks:** GSM8K and IFEval use the chat template with caps of 512
  and 1024.
- **Scoring and gates:**
  - the official IFEval scorer runs under the venv path, with assets bound by
    hash;
  - fixed gates of 5, 5 and 3 net losses.

**Analysis and registration**
- **Gates:** the frozen Qwen gates, strict parser only.
- **Bootstrap:** paired by paper, 10,000 draws, seed 20261117.
- **Denominators:** 403, 309 and 267.
- **Prompt selection:** recomputed from the 35 development questions, with the
  base, verify, schema tie order. The prompt arm is an alias of base if base
  wins.
- **Seals:** every role is bound either through R0 job registrations or
  through the R0 seal. The completion seal is written after outputs exist,
  against the prospective mapping.
- **Outcome rules:** retention pending never yields a recommendation.

## Pre-registration mechanics

**Mechanics 02 (failed; preserved and bound in R0).**
- Receipt SHA256:
  `4f545bf9e1884cdfa9f41aabdd478875bddb0f337fb21ad91ef9e200e590a736`.
- Full and compact cross-entropy were identical, 9.066858291625977.
- The native gradient requirement failed: maximum absolute difference
  0.0078125, tolerance ratio 65.9.
- That requirement was mis-specified for a BF16 backbone. G1–G3 replaced it at
  the same tolerance before R0, as disclosed in the plan.

**Mechanics 03 (passed).**
- Receipt SHA256:
  `95e0ce1e995f4c97f355588a874b8c25fb942c42e961e6e92397c4cc1e3ca3ac`.
- Manifest SHA256:
  `eadc8ec5ddd14889a641b22b1f5dfc501d1c7c55e3e9ecaaafdb9db610a64b10`. The
  receipt's `manifest_sha256` matches it.
- Its `train.json` input SHA equals the final training preparation's
  `train.json` (`c3c7efb1cc2ee2cd…`).

| Check | Result |
| --- | --- |
| G1: maximum logit difference, cross-entropy difference | 0.0, 0.0 |
| G2: cross-entropy difference, maximum gradient error | 0.0, 0.0 |
| G2: unsupervised rows | 32, all with exactly zero gradient |
| G3: longest selected row | 7,044 tokens, dropout 0.05, all 820 gradients finite, state, RNG and buffers restored, 0 optimizer updates |
| G3: peak memory | 36.97 GB allocated, 39.31 GB reserved |
| Occupancy at launch | One idle neighbour context, 464 MiB (487 MiB total, 0% utilization), accepted under the disclosed 1 GiB / 5% policy |

Peak memory excludes the Adam moments, which are about 0.98 GB.

**Diagnostics (not gates).**
- **D1, native full versus compact LoRA gradients:** maximum absolute
  difference 0.0078125, tolerance ratio 65.9.
- **D2, two repeated compact runs:** difference 0.0.

The mismatch is consistent with BF16 rounding that depends on the head
matmul's row count. No claim of native gradient equivalence is made.

**Transient edit during mechanics 03 (disclosed by the PM).** After the probe
had imported its modules, the PM added an unused `validate_study` guard to the
analyzer and then reverted it byte for byte. The probe calls only `read`, `sha`
and `validate_startup` from the analyzer, and Python does not reload a module
that is already imported. All five manifest source hashes match the current
files, the analyzer's included (`f957f8a0…`), and the probe's own end-of-run
source recheck passed. **Assessment: no effect on the executed code or the
receipt.**

## Final CPU preparations (pre-R0)

Under `outputs/gemma4-qasper-study-01/`, none of these has attempts or a result:

| Role | `registration.json` SHA256 |
| --- | --- |
| coverage | `907c2f8b2f402bfe4b25742a194a6c0d999acf13d36379235c693655f9759348` |
| training | `3f780de5bcdad9b5954d83bb00d8f74636c14e80e9dc9357e85853ee31bc0f0c` |
| prompt-development | `66b75527fd1f6c98015f6a18027ae5e8f1b302d7a9afe60347cfb40335fdbe90` |
| retention-base | `d74319771730c062c8ae58ea44ed0f03f94ca16fb1e93b21afa19eef5de2ddb1` |

Advisor checks:

- Every training source-closure file matches the current repository file,
  apart from the registered empty package `__init__.py` exception.
- Prompt development points at the coverage registration listed above.
- Retention base uses the venv scorer path `.venv/bin/python` and
  `mc_bos_token_id` 2.

## Conditions and residual items

- **R0 binding:** R0 must bind the mechanics 02 and 03 receipts and this file.
  The refreshed CPU preparations must show:
  - encoded-input SHAs equal to the originals, which the refresh helper
    enforces;
  - coverage role identity, which the register script enforces;
  - the venv scorer path.
- **Manual recovery cases:**
  - A preparation that crashes partway remains a manual, CPU-only cleanup.
  - Any out-of-memory error needs an amendment that records its cause from the
    watchdog's per-process log.
- **Timings:** these include coexistence with recorded idle neighbour contexts
  and are not exclusive-device timings.
- **Output handling:** final outputs stay unread until the scripted closeout.
  No Hugging Face upload happens in this study.
