# Qwen3.8-27B sarcasm GCAD controls release: review 02 (source, CPU and payload signoff for R6)

Advisor review, 2026-11-19, of the fresh primary staging in
`outputs/qwen38-sarcasm-release-01`. It is read-only, plus CPU tests.
- No model, judge or GPU call was made.
- No source was edited.

The criteria come from release review 01.

## Signoff

**Signed off for the local R6 GPU mechanics run with the verifier below.**
- This review does **not** authorize publication.
- Publication additionally requires a passing R6 receipt reviewed in a new
  file, and then P5 after upload.
- Any change to a file listed below voids this signoff.

| Item | sha256 |
| --- | --- |
| `payload/release_manifest.json` (trust anchor) | `26f50f8a03c16420ce16507f317e5529c297cf81d071f6b28e32a15c1b330bd4` |
| `publication_manifest.json` | `7ee53368dd5d65e4a7c91421e026a62e6618f988e7ef7ecf17f83b448c4ab1a1` |
| `scripts/stage_qwen38_sarcasm_release.py` (builder) | `0c2a6aa16ec122e50d4bf54bd10e226588974053e880854c0b5673675b3265c0` |
| `runtime/loader.py` | `4597b202f4637d98d0503a4825e9e1313c6b8abe968b5417638a1c5471e9e93a` |
| `runtime/gcad.py`, which equals the frozen `scripts/qwen38_gcad_port.py` | `3d76d8147ffbe6f802b285d643b4363e09349f0dd14a5f6a2924545b5966abe3` |
| `runtime/__init__.py` | `52ef7e279a50a0e016ad31790ecf06e08dde054683c90f455b5ab2ebe798e82c` |
| `runtime/example.py` | `7e8a20a18111008093d05cf3ee545d0f8b71520e4d4b8a816b75c557b57cbaf7` |
| `scripts/verify_qwen38_sarcasm_release.py` | `0e9eeade5705a34429b7358009de043ad61fe8a9ff11bbac43638de838ab69bc` |
| `tests/test_qwen38_sarcasm_release.py` | `091a489759ee112dd712399021767dd30c306acd7288ba7871b3e7b4ebdcefe6` |
| `docs/qwen38_sarcasm_hf_card_draft.md` (release input, unchanged) | `33ad601d34e6d7787fa907bc325a8a8fef3c4eabd966fd53e8ca4c0c65f521a5` |

**CPU tests.**
`CUDA_VISIBLE_DEVICES= outputs/qwen38-27b-nf4-01/env/bin/python -m pytest -m '' -q tests/test_qwen38_sarcasm_release.py`
gives **40 passed**, run by the advisor.

## Payload: verified against release review 01 §3

**P1. Exact allowlist.** The payload holds 17 files and no symlinks:
- `bank.safetensors`;
- `runtime/{__init__,gcad,loader,example}.py`;
- `README.md` and `METHODS.md`;
- `evaluation.json` and `metrics.csv`;
- `LICENSE`, `BASE_LICENSE`, `qwen3_gcad_port_{LICENSE,NOTICE}` and
  `qwen38_gcad_port_NOTICE`;
- `requirements.txt` and `environment.lock.json`;
- `release_manifest.json`.

It contains no rows, raw judge replies, activations, random tensors or
fixtures. The observed inventory equals the manifest files plus the manifest
itself. Every byte count and sha256 matches. The publication manifest
matches every file, and its required commit message carries `26f50f8a…`.

**P2. Bindings.** The manifest binds:
- the bank `4fd9f86b…`, equal to the payload bank bytes;
- the base `RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4` at `586bb022…`,
  with the six runtime files that review 01 verified against the study run;
- the frozen port `3d76d814…` and the builder `0c2a6aa1…`;
- the canonical result `61362b38…`, the registration `3b10bec5…` and
  follow-up review 06 `134d0538…`, which is the current file;
- the card draft `33ad601d…`;
- sites `[23,27,31,35,39,43]`, scale 1.5, range [0, 16], default 8,
  extreme 16, and evaluated strengths 2/4/8/16;
- the exact environment: Python 3.13.3, torch 2.12.0+cu130, CUDA 13.0, the
  package pins, the absent optional kernels, and the native
  `modeling_qwen3_5.py` sha256.

The payload runtime files are byte-equal to the source files above.

**P3. Privacy.**
- My independent scan of every text file found none of the following:
  local paths, the username, email addresses, `outputs/`, session IDs, argv
  or cost fields.
- `evaluation.json` contains no question text or outputs.
- Its Part B metrics, gates, descriptives and limitations equal the
  canonical result exactly.

**P4. Card numbers.**
- The staged README table is generated from the canonical result.
- I checked every value: SARC, coherence, impoliteness, coherence below 50
  and the gate failures.
- The b8 row states "FAIL relative coherence: 58.94 < 64.28" and labels b8
  the owner-chosen experimental demonstration default.
- No draft or planned wording remains.

## Card content: meets release review 01 §1 and §4

- **Status line.** It states the registered null up front and says the
  package is not a validated controller.
- **Results table.** It shows the b8 gate failure and calls b2 "essentially
  no judged sarcasm".
- **Random16.** It is described as setting-matched, not realised-shift
  matched, with one seed set, at b16 only.
- **Impoliteness.** It is coupled to sarcasm, and sarcasm is not shown to be
  separable from rudeness.
- **Scope.** It covers:
  - the 160-token cap;
  - one English user turn with no system prompt;
  - greedy decoding with repetition penalty 1.05, batch 1 and native cache;
  - the 5-pair, 90-pair bank;
  - NF4 on one stack;
  - the AI judge and the coherence rubric caveat.
- **Untested values.** Interior values other than 2, 4 and 8 are marked
  untested.
- **Old null.** It is unchanged, with no dose-causation claim.
- **Safety and licences.** A safety warning is present, and Apache-2.0 plus
  the base licence and NOTICE files are included.

**Base licence bytes.** They equal the public NF4 record's `LICENSE`, as
checked by the builder. The card's line that only the exact stack "was
mechanically checked" becomes true only after R6 passes. Publication is
gated on R6 anyway.

## Runtime source: meets release review 01 §2

**R1. Arithmetic.** `runtime/gcad.py` is byte-identical to the frozen port,
so it uses identical arithmetic by construction.

**R2. CPU parity tests.**
- At b ∈ {0, 0.5, 2, 8, 16}, sequences and per-step logits are
  `torch.equal` to the frozen `Steerer` through cached generate.
- The hook signature is restored after normal exit and after an exception.

**R3. Refusals.**
- **Strength:** booleans, NaN, ±inf, negative values, values above 16 and
  non-numeric values.
- **Inputs:**
  - tensors are required, with batch size 1, at least 2 tokens and dtype
    long;
  - any attention-mask position that is not 1;
  - a token cap outside 1–160;
  - prompt plus cap above 2048.
- **Bank:** wrong key set, dtype, shape or non-finite values, and an
  inventory sha256 other than `4fd9…`.
- **Base:** the six file hashes, alternate weight files, the geometry
  (64 layers, hybrid pattern, 24 query heads, 4 KV heads, head width 256),
  NF4 double quantization, `Linear4bit` BF16 projections, and vision or
  MTP modules.
- **Environment:** package, Python, torch/CUDA build, optional-kernel
  presence and the native modelling source.
- **Device:** non-CUDA or offload devices.
- **Manifest and files:** a manifest that does not match the external
  sha256 anchor; unsafe paths; directory links; file links other than
  standard HF snapshot-to-blob links; extra or missing files.

**R4. Hook hygiene.**
- A per-model non-blocking lock refuses nested or concurrent use and any
  strength change during a call.
- b = 0 uses `nullcontext`, so no hooks are registered.
- The model must stay in eval mode.

**R5. Interface.** The CLI and API take a float strength in [0, 16],
default 8. The example prints the experimental warning. The card's example
calls `load_runtime` and `generate`, which are the same calls the verifier
exercises.

**R6. Verifier design.**
- It runs only in a fresh `python -I` process with no prior `runtime`
  import. It is offline, with `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`
  set and `CUDA_VISIBLE_DEVICES` set to the GPU0 UUID.
- It adds only the checked bundle to `sys.path`.
- It loads the frozen port by file and hash.
- **Two probes, b ∈ {0, 8, 16}:**
  - logits are `torch.equal` to the frozen `Steerer`, or to plain
    unhooked output at b = 0;
  - 32-token greedy suffixes are equal;
  - outputs are finite and no hook leaks;
  - cache coverage is `{prefill 1, decode n−1, errors 0}`;
  - cleanup parity holds for logits and tokens.
- **Saved-trajectory replay:** F01 baseline, b8 and b16 at cap 160 through
  the public `generate_ids`. Their bindings are checked:
  - the aggregate `2bf33ce5…`, which equals the canonical result's
    `input_sha256["generation/result.json"]`;
  - the study manifest `87d1d0e6…`;
  - the row receipts;
  - the user-only rendering equals the saved `input_ids`.
- The receipt records hashes only. It contains no text, tokens or paths, and
  it records only the type of any failure.
- `--compare-receipt` supports the post-download P5 equality check.

**Scope of the parity check.** Because `gcad.py` is byte-identical to the
frozen port, the frozen-vs-portable comparison mainly checks the loader
wiring. The informative R6 checks are:
- the b = 0 bypass and its equality to plain output;
- cleanup;
- the three saved-trajectory replays, which compare against rows produced
  by the follow-up's recording steerer.

The PM has chosen 3 replays plus 2 probes. That is sufficient for an
experimental mechanics release, and the optional extra b8 replay is waived.

## Residual notes (no action required)

- `metrics.csv` writes `reference` for random16's gate column. The card
  correctly calls random16 "descriptive only". This is cosmetic.
- `seed 42` has no effect under greedy decoding. It is stated accurately as
  a runtime setting.
- The strict pins (Python 3.13.3, torch +cu130, optional kernels absent)
  will refuse many user environments. The card states this. It is the
  correct choice for an exactness-pinned experimental release.

## Next

1. The PM runs the verifier locally on GPU0 against
   `payload/` (manifest `26f50f8a…`) and the pinned base snapshot.
2. Release review 03 reviews the R6 receipt.
3. After a passing R6, the PM may upload with the commit message
   `Release manifest SHA256: 26f50f8a…`.
4. Then the PM runs P5: an anonymous pinned download and a fresh-process
   `--compare-receipt` from the downloaded copy, reviewed in a new file.
