# Qwen3.8-27B sarcasm GCAD controls release: review 05 (P5 public download and reload)

Advisor review, 2026-11-19. It is read-only.
- No model, judge or GPU call was made.
- No source was edited.

## Verdict

**P5 passes.** The public release at
`RenormalizationTech/qwen38-27b-sarcasm-gcad-controls` revision
`877e4b51e654e7410f792858e93fee6727d48646` is byte-consistent with the
reviewed payload. It reproduces the local release mechanics exactly.

## Publication record

The record is `results/qwen38_sarcasm_huggingface_publication_01.json`,
sha256 `5b68c827…6996`.
- Its status is `published_remote_hashes_verified`, with `private: false`
  and `credentials_recorded: false`.
- The 17 remote files' byte counts and sha256 values equal
  `publication_manifest.json` (`2c3efab7…`) exactly, as checked by the
  advisor.
- The bank uses the Hub LFS sha256 `4fd9f86b…`.
- The only additional remote file is the Hub-managed `.gitattributes`,
  which the loader excludes by rule.
- The card metadata says experimental, with Apache-2.0 and the pinned NF4
  base model.

## Download record

The record is `outputs/qwen38-sarcasm-public-01/download.json`, sha256
`93693585…a86f`.
- It records anonymous access, verified hashes for every file, and a commit
  message that carries the manifest digest `cfaacd0e…`.

**Controls repository.**
- It was downloaded into the verification cache at 14:50 today, revision
  `877e4b51…`.
- Its 17 payload files equal the publication manifest. `.gitattributes` is
  recorded separately.

**Base model.**
- All 17 files at `586bb022…` equal the public NF4 verification record.
  This includes the six runtime files.
- **Wording point:** the base snapshot in that cache was created on
  2026-11-17 by the earlier NF4 public verification. Today's step served the
  pinned revision from that cache, and did not transfer 18 GB again. That is
  why the record shows about 27 s.
- The bytes are hash-verified, and the earlier record documents its own
  anonymous full download.
- Docs should say: "base files hash-verified from the pinned-revision cache
  (anonymous full download recorded 2026-11-17)". They should not imply a
  fresh base download today.
- `model.safetensors` resolves through the Hub 2.4 two-hop shared store.
  This is the real layout the review 03 fix supports.

## Public reload receipt

The receipt is `results/qwen38_sarcasm_release_public_smoke_01.json`,
sha256 `e0605f35…b4f6`.
- Its status is `passed_release_mechanics`, with `python_isolated: true`,
  offline mode and GPU0 (an RTX A6000).
- It records `local_receipt_sha256: c9173dbe…`, which equals the local
  retry 02 receipt file, and `public_local_mechanics_equal: true`.

**Independent comparison by the advisor.** These fields are identical to the
local receipt:
- the release manifest and the frozen port;
- the canonical result and the aggregate;
- the verifier;
- the GPU UUID;
- all 6 probe records, including the logit hashes;
- all 3 saved-trajectory replays (F01 baseline, b8 and b16 at 160 tokens);
- both cleanup flags.

A privacy scan is clean, and no raw outputs were recorded.

**Scope.** The receipt contains no paths, by design, so it cannot itself
name the on-disk copy it ran from. Its guarantee is byte identity through
the manifest anchor. Together with the download record, that is the P5
requirement.

## Release chain: complete

| Step | Evidence |
| --- | --- |
| Criteria | review 01 |
| Source and payload | review 02 (superseded hashes); review 03 (Hub 2.4 fix, `cfaacd0e…`) |
| Attempt 01 (pre-load failure, preserved) | `635f07ee…` |
| Local mechanics | review 04, receipt `c9173dbe…` |
| Publication | `877e4b51…`, all remote hashes match |
| P5 | this review, receipt `e0605f35…` |

## Remaining conditions for docs and catalog

- **Outcomes.** Keep `development_null` and `exploratory_null_through_b16`
  exactly.
- **Labelling.** Call the package an experimental demonstration, not a
  validated controller. b8 failed relative coherence.
- **Citation.** Cite revision `877e4b51…` and the manifest `cfaacd0e…`.
- **Links.** Link release reviews 01–05.
- **Base download wording.** Use the wording above.
- **Scope of verification.** These checks prove byte consistency and
  mechanical reproduction on the studied stack. They do not prove publisher
  identity or quality on other hardware.

This closes the advisor's release review for the Qwen3.8 sarcasm GCAD
controls.
