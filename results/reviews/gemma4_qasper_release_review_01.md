# Gemma 4 31B QASPER release: advisor criteria (release review 01)

Advisor, 2026-11-20. This file sets the criteria for release before staging.
The advisor reviews against it for source and material defects only. It adds no
approval rounds.

- **Frozen and preserved:** R0, analysis amendment 01, all study outputs, the
  earlier reviews, and every Qwen, physics or other method file.
- **Delivery:** findings on the staged material go in a new file,
  `gemma4_qasper_release_review_02.md`.

**Bound result:**
- `results/gemma4_qasper_final_01.json` `a149de80…`
- final audit `results/reviews/gemma4_qasper_final_audit_review_02.md`
  `06beb096…`

## A. Artifact identity

1. **Base model:** `google/gemma-4-31B-it` (the exact upstream ID recorded in
   R0), revision `842da3794eaa0b77d5f08bae87a17459d91ff475`.
   - Staging, inference code, card and verifier all name this revision.
   - Nothing may resolve `main` or an unpinned name.
2. **Adapter:** the selected step-61 checkpoint
   (`training/checkpoints/step-00061`, development NLL 0.3469).
   `adapter_model.safetensors` must be byte-identical: SHA256
   `5c539bb8e0744a3f0a4d95f98e8f1e05c1bb3ab46ed723a0e81a7643d6bc8eb7`, 489,828,512
   bytes. It must not be re-saved, re-serialized or converted.
3. **Config normalization:** `adapter_config.json` (source `ed8a5428…`) may
   differ only in `base_model_name_or_path` (local path to the public ID) and
   `revision` (to `842da379…`).
   - A key-by-key diff must show no other change. That covers r 16, alpha 32,
     dropout 0.05, `target_modules` / module set (410 projections,
     50 `v_proj`), task type, bias and PEFT version fields.
4. **Excluded from upload:**
   - `state.pt` (optimizer and RNG state, 980 MB);
   - the PEFT-generated checkpoint `README.md` (replaced by the card);
   - any study receipt containing local paths.
   - If training state is published, that is a separate, stated decision.
5. **Privacy scan:** every staged and uploaded text file is free of absolute
   local paths (`/home/…`), usernames, email addresses and tokens. The manifest
   lists every file with its SHA256 and size.

## B. Loading protocol (must reproduce the study, not an approximation)

1. **Native Gemma 4 text-only path.**
   - The card and code use the same model class and module paths as the study.
   - Any key or prefix mapping is explicit and tested. After loading, the
     number of attached LoRA modules is 410, with 820 A/B tensors. There are
     zero missing and zero unexpected adapter keys, and the error is fatal:
     PEFT's warning-only behavior is not acceptable.
   - Every loaded LoRA tensor equals the safetensors tensor exactly.
2. **Quantization:** NF4 with double quantization and BF16 compute, then
   `prepare_model_for_kbit_training`-equivalent FP32 casts as in the study.
   Inference runs under outer BF16 autocast, with SDPA and batch 1.
   - The card states this is the tested configuration.
   - Full-precision BF16 base loading is untested and must be labelled so.
3. **Formatting and generation:**
   - the native chat template with the study's empty-thought generation prefix;
   - greedy decoding with the cache;
   - stop set {1, 106}, with 50 excluded;
   - a cap of 512 new tokens;
   - decoding that keeps special tokens;
   - the study's QASPER prompt template, the base template;
   - prompt plus 512 ≤ 8,192 tokens, with truncation refused.
4. **Versions:** record the `transformers`, PEFT, bitsandbytes, torch and CUDA
   versions the study used. A different version is allowed for users but must
   be labelled unverified.

## C. Offline reload parity, before upload (GPU, run by the PM)

1. **Isolation:** a fresh process with `HF_HUB_OFFLINE=1`, loading from the
   staged release folder only. No study objects are imported, and nothing is
   cached from the training run.
2. **Fixed rows, chosen before running and not by metric:**
   - the demo ID `1603.03876::47ffc9811b037613c9c4d1ec1e4f13c08396ed1c`;
   - the first 4 final rows by ordinal;
   - adapter arm for all 5, plus base arm (adapter disabled) for the demo ID
     and ordinal 0.
3. **Prompts:** the rendered prompt and `input_ids` must equal the saved rows
   byte for byte.
4. **Pass condition:** `generated_ids` exactly equal to the saved rows, with
   decoded text equal to the saved `output`.
   - A mismatch blocks upload. It is a real decision to escalate, with the
     first diverging position and the top-2 logit margin recorded.
   - Tolerances must not be widened after seeing the result.
5. **Results stay as recorded:** parity reproduces saved rows. Released
   results are never regenerated or rescored.

## D. Post-upload verification (anonymous, pinned)

1. **Clean environment:** no token (`HF_TOKEN` unset, implicit token
   disabled) and an empty fresh cache directory. Download at the uploaded
   commit SHA, not `main`.
2. **Manifest match:** the file set and every SHA256 equal the staged
   manifest. `adapter_config.json` resolves the public base ID and
   `842da379…`.
3. **License:** the repo is public and the license and model-card metadata are
   present. The base license is quoted from the pinned upstream card or
   LICENSE, verified, not asserted from memory. QASPER text in the card or demo
   carries CC BY 4.0 attribution.
4. **Optional GPU repeat:** one GPU parity row (the demo ID, adapter) from the
   downloaded files repeats C4. Recommended, not required, if C passed and the
   hashes match.

## E. Card, docs and demo content (non-negotiable claims)

- **Gate claim:** report "all registered task and retention gates passed",
  under the registered scope:
  - custom text-only QASPER on reused project IDs;
  - 403 / 309 / 267 records;
  - one configuration and one seed;
  - not an official or held-out benchmark;
  - the papers may be in pretraining.
- **Strict gate needs context:**
  - The strict base 0.0 is a **format floor**: all 403 base outputs are wrapped
    in ```` ```json ```` fences, which the strict parser rejects. It must not
    be presented as the base lacking the knowledge.
  - The lenient primary answer F1 diagnostic must sit beside the strict
    result: 0.3338 → 0.6401, gain 0.3063, paper-bootstrap CI [0.2513, 0.3617].
  - Strict: 0.0000 → 0.6401, CI [0.5884, 0.6922].
  - The lenient gain is largely answer style (median 35 words → 3).
- **No evidence superiority:** lenient evidence F1 is about the same (0.723 vs
  0.717 primary). No grounding or evidence-selection improvement claim.
- **Prompt arm:** it is an alias of base. All templates scored 0 strict
  because every output was fenced, so no claim about prompting can be made.
- **Retention:**
  - MC 108 → 105/128, GSM8K 126 → 126, IFEval 26 → 26;
  - within fixed point-estimate net-loss limits only;
  - no non-inferiority claim.
- **No ranking:** no broad Gemma-versus-Qwen claim. Qwen base strict was
  0.457, the baselines fail differently, and the families are not
  protocol-matched.
- **Output contract:** the adapter emits raw JSON `{answer, unanswerable,
  evidence}` with `p######` paragraph IDs. Users must supply paragraphs in the
  study's ID format.
- **Docs:**
  - `docs/gemma4_qasper_results.md` links forward to the release and usage.
  - The navigation replaces the feasibility tab with one Gemma tab, and
    `docs/gemma4_preflight.md` stays byte-unchanged and archived.
  - Qwen pages are untouched apart from a navigation link, if any.
- **Static demo:**
  - **Example:** the same Qwen QASPER ID, reused because it was fixed earlier,
    not picked by metric.
  - **Content:** the question, raw base output with the fences kept, the
    adapter output, and strict and lenient per-item scores from the frozen
    scorer, clearly labelled. The prompt arm is shown as an alias of base.
  - **Exact text:** the displayed outputs must equal the saved `output` fields
    exactly, checked by hash in a test.
  - **Limits:** no new inference and nothing in the browser. One example is
    not evidence.

## Advisor notes

- **Biggest technical risk:** silently not applying the adapter. Gemma 4's
  multimodal wrapper shifts module prefixes, and PEFT only warns on missing
  keys. Criteria B1 and C4 together close this. Card usage code must use
  the same loader the verifier tests.
- **Generation parity across processes:** under BF16 this is expected but not
  guaranteed. If C4 fails on a near-tie, that is an infrastructure fact to
  disclose and decide on, not something to repair by retuning.
- **The fence finding is useful to users:** the base model's default
  fenced-JSON behavior is worth one sentence in the card as practical guidance
  for base users. Lenient parsing recovers it.
- **Repo naming:** keep the name parallel to the Qwen release, for example
  `…/Gemma-4-31B-it-tuneLLM-QASPER-QLoRA`. The card must not imply
  endorsement by Google.
