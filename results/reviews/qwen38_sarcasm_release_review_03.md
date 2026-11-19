# Qwen3.8-27B sarcasm GCAD controls release: review 03 (HF Hub 2.4 cache fix; source and payload signoff for R6 retry 02)

Advisor review, 2026-11-19. It is read-only, plus CPU tests.
- No model, judge or GPU call was made.
- No source was edited.

Release review 02 is unchanged. Its binding is void because the loader,
builder and tests changed.

## Cause of R6 attempt 01

**Receipt.** `results/qwen38_sarcasm_release_local_smoke_01.json` records
`release_mechanics_failed` with `ValueError`, before model load.
- No forwards ran.
- No raw outputs were recorded.

**Cause.** The installed `huggingface_hub` 2.4.0 (`utils/_shared_blobs.py`)
uses a two-hop layout, which I read in this review:
- the snapshot file links to `<repo>/blobs/<etag>`;
- that entry is itself a relative link to
  `<cache>/blobs/<xet[:2]>/<xet>`;
- the store is marked by a regular `.huggingface-shared-blobs` file
  containing `1\n`.

The review 02 loader accepted only one hop, so it correctly failed closed.

## The fix: verified

`runtime/loader.py` `_cache_file_link` inspects each hop lexically before it
resolves anything.

**Directories.**
- The root is `snapshots/<40-hex>` under a `models|datasets|spaces--…`
  repository directory.
- The root, `snapshots`, the repository, the cache and the repository blobs
  directory are all real directories (checked with lstat).
- The existing check that refuses links in intermediate snapshot directories
  is kept.

**First hop.**
- It must land in `<repo>/blobs/` with a 40- or 64-hex name.
- If that entry is a regular file, it is accepted. This is the legacy layout,
  and it is unchanged.

**Second hop.** It must equal `<cache>/blobs/<h[:2]>/<h>`, with h 64-hex.
Each of the following must also hold:
- the store and prefix directories are real directories;
- the marker is a regular file and reads exactly `1\n`;
- the final entry is a regular, non-link file, so a third link is refused;
- the resolved path stays under `cache.resolve()`.

**Unchanged.**
- The size and content-sha256 check against the manifest still runs on the
  final bytes. That check is the integrity guarantee.
- The comment correctly notes that the Xet hash is not the content sha256.

**Tests.** The tests build the fixture with the library's own
`shared_blob_path`. They cover:
- the positive two-hop case, with content-hash success and tamper refusal;
- 10 refusal modes:
  - a wrong prefix;
  - a missing marker, a bad marker and a linked marker;
  - a third link;
  - a target outside the cache;
  - bad first and final names;
  - prefix and store directory links;
- the legacy one-hop case at both hash lengths;
- plain local files.

**Advisor run.**
`CUDA_VISIBLE_DEVICES= outputs/qwen38-27b-nf4-01/env/bin/python -m pytest -m '' -q tests/test_qwen38_sarcasm_release.py`
gives **53 passed**.

**README.** It now states that both the legacy one-hop layout and the
pinned Hub 2.4 two-hop shared-blob layout are supported and hash-verified,
and it lists the refusals. Nothing else in the card changed: the numeric
table is identical to the preparation-02 copy.

**Unchanged components.**
- `gcad.py` is still byte-identical to the frozen port `3d76d814…`.
- The verifier is unchanged (`0e9eeade…`).
- `__init__.py`, `example.py`, the card draft `33ad601d…`, the canonical
  result and the bank are unchanged.

## Payload: rechecked

- **Inventory.** 17 files, with no symlinks. The inventory equals the
  manifest plus the manifest itself.
- **Hashes.** Every byte count and sha256 matches. The publication manifest
  matches, and its commit message carries `cfaacd0e…`.
- **Runtime files.** They are byte-equal to the source files.
- **Bindings.**
  - builder `b908303f…`;
  - canonical result `61362b38…`;
  - review 06 `134d0538…`;
  - card draft `33ad601d…`;
  - bank `4fd9f86b…`;
  - base `586bb022…`.
- **Evaluation file.** The Part B metrics and gates in `evaluation.json`
  equal the canonical result.
- **Privacy.** The scan is clean.

## Signoff

**Signed off for local R6 retry 02 with the unchanged verifier.** This is
not publication approval. Any change to a file listed below voids it.

| Item | sha256 |
| --- | --- |
| `payload/release_manifest.json` (trust anchor) | `cfaacd0efa5a7b22a40fab5305217ee2c642b46ba230a2eb8259d5a28639e54b` |
| `publication_manifest.json` | `2c3efab74ceee3bb9f3b7fd11d2f2966e066088224376806947e0f52d2805d79` |
| `scripts/stage_qwen38_sarcasm_release.py` | `b908303f6a8b7b216f2f8f6d1c569482405223e66c3644576bb9438f31e94e01` |
| `runtime/loader.py` | `49cfb352004dd95688c0f1ec5d94a98f2c540c813f3e233eea51a8b8c194cb57` |
| `runtime/gcad.py`, which equals the frozen port | `3d76d8147ffbe6f802b285d643b4363e09349f0dd14a5f6a2924545b5966abe3` |
| `runtime/__init__.py` | `52ef7e279a50a0e016ad31790ecf06e08dde054683c90f455b5ab2ebe798e82c` |
| `runtime/example.py` | `7e8a20a18111008093d05cf3ee545d0f8b71520e4d4b8a816b75c557b57cbaf7` |
| `scripts/verify_qwen38_sarcasm_release.py` | `0e9eeade5705a34429b7358009de043ad61fe8a9ff11bbac43638de838ab69bc` |
| `tests/test_qwen38_sarcasm_release.py` | `7d95b0da5dd539c523c2d5fcf4dc0387960694b70b78fba97cf2e596128a241d` |
| `docs/qwen38_sarcasm_hf_card_draft.md` | `33ad601d34e6d7787fa907bc325a8a8fef3c4eabd966fd53e8ca4c0c65f521a5` |

Release review 02's content findings still apply, other than its hashes:
- the card, allowlist and bindings;
- the R3 and R4 refusals;
- the verifier design, with 2 probes and 3 F01 replays;
- the cosmetic `metrics.csv` random16 label.

## Next

1. The PM runs retry 02 on GPU0 against manifest `cfaacd0e…`. The attempt 01
   receipt is retained.
2. A new review covers the retry 02 receipt.
3. Publication may follow only after that receipt passes. Then P5 runs: an
   anonymous pinned download and a fresh-process `--compare-receipt`.
