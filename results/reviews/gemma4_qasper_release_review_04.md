# Gemma 4 31B QASPER release: local GPU mechanics signoff (release review 04)

Advisor, 2026-11-20. Read-only and CPU only: no model, GPU or upload call was
made, and no source or earlier review was edited. This review applies the
conditions of release reviews 02 and 03 to the local receipts.

**Verdict: the local reference and portable mechanics passed.** The 12-file
stage-04 payload is accepted for public upload.

## Receipts

| Receipt | SHA256 | Status |
| --- | --- | --- |
| `outputs/gemma4-qasper-release-04/local-reference-01.json` | `8c0e392e05ed3ec6bf3be57fea545ec7789a035b928b3fef0f54c23140164482` | `reference_mechanics_passed` |
| `local-reference-01.reference.safetensors` | `43bd823b069e7a25e162c2851d70380c6e4dba87778248229c2a9355742e11de` | CPU logits and tokens |
| `outputs/gemma4-qasper-release-04/local-portable-01.json` | `b3a99a12e784dcfafc4b900c3661acefbe5e4ecbf7df5f383a5aad521d796553` | `portable_mechanics_passed` |

**What both receipts share:**
- manifest `4e06c0de…`;
- verifier `c50871ea…`;
- base `google/gemma-4-31B-it` at `842da379…`, with base files hash-checked;
- adapter reload of 820 tensors with exact equality;
- `working_tree_modules_imported: false`;
- cleanup exact, no optimizer updates, no regrading;
- the same RTX A6000, UUID `34646ab9…`.

The portable receipt's `reference_receipt_sha256` equals `8c0e392e…`.

## Advisor's independent CPU checks

**Saved replay from the study rows.**
- For all 7 saved-replay cells, the advisor recomputed token hashes from the
  sealed `final-base` and `final-adapter` predictions. All match the reference
  receipt. The cells are:
  - the demo ID (ordinal 370) on base and adapter;
  - ordinals 0–3 on the adapter;
  - ordinal 0 on the base.
- The tokens in the reference tensor file equal the saved `generated_ids`.
- The logit hashes match, and all logits are finite.
- The decoded output was also checked exactly by the verifier.

**Portable equals reference.**
- The two receipts have the same 11 cells.
- Every logit hash and token hash is identical between them.
- Every cell reports `portable_reference_exact: true`.
- All 7 saved cells report exact token replay and exact decoded output.

**The adapter is active.**
- On both invented prompts, the base and adapter logits differ.
- Their 6–8 generated tokens are the same, which is expected for trivial
  prompts.

**The payload is unchanged after the GPU runs.**
- Still 12 files, with no `__pycache__` or `.pyc`.
- Every manifest file hash still matches.
- Manifest `4e06c0de…`, publication manifest `2b3289df…`.

## Conditions for the remaining steps

1. **Upload:** only the 12 files in publication manifest `2b3289df…`, to the
   public `RenormalizationTech/Gemma-4-31B-tuneLLM-QASPER-QLoRA`, with commit
   message `Release manifest SHA256: 4e06c0de…`. Record the published commit
   SHA.
2. **Anonymous download:** run the download phase with fresh empty caches
   against that commit. Then run the anonymous portable phase. Its cells must
   equal `local-portable-01.json`, and it must report
   `public_portable_mechanics_passed`.
3. **Publishing the pins:** only after the anonymous download and portable
   phases pass do the public verification receipt, catalog and guide publish
   the verified revision and manifest SHA. If either fails, report the failure
   publicly rather than leaving the pins pending.
4. **Advisor check:** the advisor checks the public receipts. Any finding goes
   in a new review file.
