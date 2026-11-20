# Gemma 4 31B QASPER release: public publication and anonymous verification (release review 05)

Advisor, 2026-11-20. Read-only and CPU only: no model, GPU or upload call was
made, and no source or earlier review was edited.

**Verdict: the public release is verified.**
- **Published:** `RenormalizationTech/Gemma-4-31B-tuneLLM-QASPER-QLoRA` at
  revision **`b3ca7a48aa6a95245c5778e2c2473d8129f58bbb`**, with release
  manifest **`4e06c0de10fd97b262bd032f34bc6481fa0d27fbd138b301befbc05e2cb8bdac`**.
- **Contents:** exactly the 12 reviewed stage-04 files.
- **Reproduces the study:** loaded anonymously, the adapter reproduces the
  local reference and the saved study generations exactly.
- **Pins:** the docs, catalog and final public receipt may now publish this
  revision and manifest SHA.

## Receipts

| Receipt | SHA256 | Status |
| --- | --- | --- |
| `results/gemma4_qasper_huggingface_publication_01.json` | `6d6dc24fa453e6a14e96228cc0cda8d4fefd788e967ff5b055680024e3ea44cb` | `published_remote_hashes_verified` |
| `outputs/gemma4-qasper-release-04/anonymous-download-01.json` | `6413d8dd04fad9207326abdd68aa097d41c61b18f452c22f0474aa9b8c82a117` | `anonymous_adapter_download_verified` |
| `outputs/gemma4-qasper-release-04/anonymous-portable-01.json` | `d64a6d46572c4b018a07191b4922ed45940752a34867e33ec7ac69086c79b879` | `public_portable_mechanics_passed` |

These follow the local receipts bound in review 04: reference `8c0e392e…`,
portable `b3a99a12…`.

## Checks

**Publication**
- **Repository:** public, at revision `b3ca7a48…`, with no credentials
  recorded.
- **Manifest:** `manifest_sha256` equals the stage-04 publication manifest
  `2b3289df…`.
- **Files:** the hashes and sizes of all 12 remote files equal that manifest.
  The only other remote file is `.gitattributes`.
- **Hub card metadata:**
  - license apache-2.0;
  - `base_model: google/gemma-4-31B-it`, relation adapter;
  - peft library and the expected tags.

**Anonymous download**
- **Pinned and clean:** at revision `b3ca7a48…` with manifest `4e06c0de…`,
  fresh caches and no token. The base was not downloaded.
- **Files:** the `files` map equals the publication manifest.
- **Inventory digest:** the advisor recomputed it with the catalog recipe and
  got `059efc4e…`, matching the receipt.
- **Bytes:** every downloaded file equals the manifest.
- **Extra local files:** only Hugging Face's `.cache/huggingface/` metadata
  and `.gitattributes`, which the loader's inventory check ignores.

**Public portable phase**
- **Identity:** public repo, revision `b3ca7a48…`, manifest `4e06c0de…`,
  verifier `c50871ea…`.
- **Base:** `842da379…`, with base files hash-checked. This reuses the pinned
  base cache; it is not a fresh base download, and the receipt says so.
- **Adapter and imports:**
  - adapter reload of 820 tensors with exact equality;
  - cleanup exact, no optimizer updates, no regrading;
  - no working-tree imports.
- **Cells:** its 11 cells are identical to `local-portable-01.json`, and every
  cell reports `portable_reference_exact`.
- **Saved replay:** all 7 saved cells replay exactly, both tokens and decoded
  output. The advisor had already recomputed these against the sealed study
  predictions in review 04.
- **Receipt bindings:**
  - `local_receipt_sha256` → `b3a99a12…`;
  - `download_receipt_sha256` → `6413d8dd…`;
  - `reference_receipt_sha256` → `8c0e392e…`.

## Scope and remaining items

- **What this verifies:** this is a functional and identity verification of
  the published artifact. It adds no new quality evidence. The efficacy claims
  remain those of the frozen study, as audited in final audit review 02.
- **Finalizer:** `scripts/finalize_gemma4_qasper_release.py` may now
  aggregate these receipts into
  `results/gemma4_qasper_public_verification_01.json` and the catalog entry.
  The advisor checked its schema against these real receipts and found no
  blocker. One optional hardening: check that the public receipt's
  `base_revision` equals `842da379…` before copying it.
- **Docs and catalog:** they should state the verified revision and manifest
  SHA above. They keep the card's scope:
  - the strict base score is a format floor;
  - the lenient gain is mostly concision;
  - no evidence or ranking claims;
  - retention is within point limits;
  - one configuration and seed.
- **Hugging Face card:** the published card cannot contain its own commit.
  Users take the pins from the catalog or guide, or from the commit message.
