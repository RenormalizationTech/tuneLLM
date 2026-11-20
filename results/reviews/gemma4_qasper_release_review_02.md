# Gemma 4 31B QASPER release: source and staged-payload review (release review 02)

Advisor, 2026-11-20. Read-only and CPU only: no model, GPU, upload or quality
call was made, and no source was edited. This review checks the staged release
against `gemma4_qasper_release_review_01.md` (criteria `e1ddbd0d…`).

**Verdict: accepted for the root's local GPU mechanics** (reference phase, then
portable phase), then **upload of the 12 allowlisted files**, then **anonymous
pinned verification**.

If any hash below changes, this signoff no longer applies. Upload still
requires the local GPU receipts in section D.

## A. Signed hashes

**Active stage:** `outputs/gemma4-qasper-release-03/`. Stages 01 and 02 are
superseded and must not be verified or published.

| Item | SHA256 |
| --- | --- |
| `payload/release_manifest.json` (anchor) | `3665a184f92cbef94598919c1893fa3acefaa6bd3b98e1e2db1031688439dfbe` |
| `publication_manifest.json` (12 files, public) | `9ee5534335cf70534d7155321b5dce70d09cf6d6c918fbab9141bf240f45895b` |
| `payload/adapter_model.safetensors` | `5c539bb8e0744a3f0a4d95f98e8f1e05c1bb3ab46ed723a0e81a7643d6bc8eb7` |
| `payload/README.md` = `docs/gemma4_qasper_hf_card_draft.md` | `26ef6aa70814d95338b52cbf4b64db65d3ab50d7c963fa375af966b4a48bf542` |
| `scripts/gemma4_qasper_release_runtime/loader.py` (= payload copy) | `07b7a375cc79692593a7c5173e389926a453d17c159298c28d664c8a0ef33358` |
| `scripts/gemma4_qasper_release_runtime/example.py` (= payload copy) | `4086537b786a9ee5f3fc4887adf2af75e0e2ffb6342102c474493d0e20845c58` |
| `scripts/gemma4_qasper_release_runtime/__init__.py` (= payload copy) | `6f7346fdadd0da10242433483ba7341799102abc5d1d104e3a5087b6c5355570` |
| `scripts/verify_gemma4_qasper_release.py` | `c50871eab911b26c80ab5abbada60d74160a9a10c02fa070910e1441c6d17505` |
| `scripts/stage_gemma4_qasper_release.py` | `9268a6aede5c5ad8be8100ebd0bb00283af9767fce64cf55aa197445f69ddfa7` |
| `tests/test_gemma4_qasper_release.py` | `84a0498932116d5bbc698f345a4ee5f2331bf4f70d58c6e122e3f5cb242e18f0` |

**CPU tests:** 43 passed. The advisor re-ran them in
`outputs/qwen38-science-01/env` with `CUDA_VISIBLE_DEVICES=""`. The worker's
receipt is `cpu-tests-03.xml`, `5121ea4f…`.

## B. Criteria results

**Identity (review 01, section A)**
- **Base:** `google/gemma-4-31B-it` at revision `842da379…` everywhere.
- **Weights:** byte-identical to step 61.
- **Config:** a key diff against the saved config shows only
  `base_model_name_or_path` and `revision`. The saved `target_modules` are the
  7 suffix names.
- **Inventory:** 820 tensors, 122,429,440 parameters, and 410 targets equal to
  the registered geometry.
- **Payload files:**
  - The file set is exactly the 12-file allowlist.
  - Every manifest file hash and all 6 source hashes match.
  - `state.pt`, the PEFT README and the study receipts are absent.
- **Privacy:** searching the payload text files for local paths, the
  username, the email domain and `hf_` tokens finds nothing.

**Provenance**
- The manifest binds:
  - R0 `ba1983fd…`;
  - analysis amendment `29f41ff8…`;
  - amended analyzer `78506e39…`;
  - canonical result `a149de80…`;
  - final audit `06beb096…`;
  - training registration `3f780de5…` and result `c601e9e2…`;
  - completion seal `97aed854…`;
  - frozen runtime `7a47b21c…`.
- The verifier checks the training registration against the completion seal
  before compiling the preserved training source closure.
- The receipt states `working_tree_modules_imported: false` separately from
  the reference phase's hash-checked frozen snapshot.

**Loading and generation (review 01, section B)**
- **Loader matches the frozen study runtime:**
  - text-only `Gemma4ForCausalLM` with the `^model.language_model.` key mapping;
  - NF4 with double quantization and BF16 compute;
  - PEFT FP32 casts and the same LoRA wrapper;
  - fatal missing or unexpected keys;
  - an exact-equality adapter reload.
- **Inference mode matches the frozen study:**
  - eval mode, `inference_mode` and outer BF16 autocast;
  - greedy cached generation at batch 1;
  - stop tokens [1, 106], cap 512, context 8,192 with no truncation;
  - decoding keeps special tokens.
- **Prompt formatter:**
  - `format_question` reproduces the saved `user` text byte for byte on 399 of
    the 403 final rows. The advisor checked all 403 against `data.json`.
  - The other 4 rows (ordinals 353–356) contain a paragraph whose section is
    missing in the source data. The frozen preparation printed it as the
    literal string `None`. Passing `section="None"` reproduces those rows.
  - None of these rows is in the replay set.
  - The card quotes the template verbatim.
- **Generic PEFT loading:** the card warns that generic PEFT automatic loading
  does not reproduce the studied method.

**Download contract**
- **Base:** the card downloads it with `allow_patterns` for exactly the 8
  manifest files (`model-*.safetensors` matches only the two shards) into a
  fresh `local_dir`. If extra base files are present, the loader names them
  and points to the card's download.
- **Adapter:** downloaded at a specific commit, with no token.
- **Trust anchor:** the revision and manifest SHA come from the release commit
  message (`Release manifest SHA256: 3665a184…`) or from the catalog or guide
  after public verification.
- **No verification claim:** the card makes none. It explains why its own
  commit cannot appear in the card.
- **Pinned links:** the study evidence links are pinned to commits `15f0d1d4`
  and `9323a83c`.

**Verifier (review 01, sections C and D)**
- **Reference phase:** imports only the seal-bound frozen closure.
- **Portable phase:**
  - loads from the staged bundle only, offline;
  - logits and tokens must equal the reference exactly.
- **Saved replay:** 7 cells with exact tokens and exact decoded output:
  - the demo ID on base and adapter;
  - final ordinals 0–3 on the adapter;
  - ordinal 0 on the base.
- **Invented prompts:** 4 cells.
- **Cleanup:** adapter-flag and hook cleanup checks.
- **Receipt:** records the GPU UUID.
- **Download phase:**
  - fresh, empty `HF_HOME`, `HF_HUB_CACHE` and `HF_XET_CACHE` are enforced;
  - the model must be public at the pinned revision, with an exact allowlist;
  - all payload bytes are hash-checked before any downloaded Python is
    compiled.

**Card claims (review 01, section E)**
- format floor (all 403 base outputs fenced);
- lenient gain 0.3063, CI [0.2513, 0.3617], with median 35 → 3 words framed as
  concision, not comprehension;
- no evidence superiority, prompt arm an alias of base;
- retention within point limits, not non-inferiority;
- one configuration and seed, no Gemma-versus-Qwen ranking, timing caveat;
- Apache-2.0 from the pinned upstream metadata, the extracted text, and NOTICE
  attribution.

## C. Non-blocking notes

1. **Card ordering.** In "Local use", the sentence "Create `question.txt` …
   Run from the adapter directory:" comes before the formatter block, not next
   to the command. This is cosmetic only, so no restage.
2. **Mutable links.** The catalog and reproduction-guide links point to
   `main`. That is acceptable under the agreed non-circular contract. The root
   must make sure both resolve before upload, and add the Gemma entry only
   after public verification passes.
3. **Exact version pins.** The loader requires the exact studied package
   versions. This is deliberate and stated in the card. Other versions are
   refused, not silently run.

## D. Conditions on the GPU and upload sequence

1. **Commands:** run the root commands in
   `outputs/gemma4-qasper-release-pm-01/release-commands.json` exactly, one
   persistent process per phase, and run portable only after reference has
   exited.
2. **Before upload:** both local receipts must report:
   - `reference_mechanics_passed`, then `portable_mechanics_passed`;
   - manifest `3665a184…`;
   - every cell with exact saved-replay and portable-reference equality.
3. **On failure:** any mismatch stops the release. Record the first
   divergence, then escalate as a real decision. No tolerance may be widened
   and no payload changed.
4. **Upload:** only the 12 files in publication manifest `9ee55343…`, public,
   with commit message `Release manifest SHA256: 3665a184…`.
5. **Public check:** the anonymous download and portable phases must pass
   against the published commit. Only then do the receipt, catalog and guide
   publish the verified revision and manifest SHA.
6. **Advisor check:** the advisor checks the local receipts before upload and
   the public receipts after. Any finding goes in a new review file.
