# Gemma 4 31B QASPER release: stage 04 card-only review (release review 03)

Advisor, 2026-11-20. Read-only and CPU only: no model, GPU or upload call was
made, and no source or earlier review was edited.

This review supersedes review 02 (`ba1f4c7f…`) **only for the manifest,
publication-manifest and card hashes**. Every other finding and condition in
review 02 carries over unchanged.

**Verdict: stage 04 is accepted** for the root's local GPU mechanics
(reference phase, then portable phase), then upload of the 12 allowlisted
files, then anonymous pinned verification.

**Stages 01–03 are superseded** and must not be verified or published.

## Signed hashes (`outputs/gemma4-qasper-release-04/`)

| Item | SHA256 |
| --- | --- |
| `payload/release_manifest.json` (anchor) | `4e06c0de10fd97b262bd032f34bc6481fa0d27fbd138b301befbc05e2cb8bdac` |
| `publication_manifest.json` (12 files, public) | `2b3289df8e46d091063e191bd8f41908ad6b32022b6f046d8324613c9d063570` |
| `payload/README.md` = `docs/gemma4_qasper_hf_card_draft.md` | `fb64e4cfdf35e5e04685aeb550b76373ed1a006ff9983491ab28a310737be258` |

**Unchanged from review 02, re-hashed:**

| File | SHA256 prefix |
| --- | --- |
| weights | `5c539bb8…` |
| loader | `07b7a375…` |
| `example.py` | `4086537b…` |
| `__init__.py` | `6f7346fd…` |
| verifier | `c50871ea…` |
| staging script | `9268a6ae…` |
| test file | `84a04989…` |

## Checks

- **Only the card changed.** Byte comparison of stage 04 against stage 03:
  - 10 of the 12 payload files are identical;
  - only `README.md` and `release_manifest.json` differ;
  - within the manifest, only the `files.README.md` entry changed;
  - all manifest file hashes and source hashes match.
- **The card diff is limited to the usage workflow:**
  - `import sys; sys.dont_write_bytecode = True` before the `runtime` import;
  - `Path('../question.txt').write_text(user, encoding='utf-8')`;
  - input files, including `../question.json`, are kept outside the bundle.

  No claim, number, link or download allowlist changed.
- **The card's code was tested independently.** The root's receipt
  `root-card-usability-04.json` reports status passed. The test ran the card's
  exact Python formatter block (code block SHA `98bd1eda…`) without `-B`, in a
  scratch copy of the stage-04 files. It found:
  - no bytecode written;
  - the question file written outside the bundle;
  - `validate_bundle` passing with the new anchor;
  - zero model calls.
- **Upload manifest:** the publication manifest lists 12 public files, with
  commit message `Release manifest SHA256: 4e06c0de…`.
- **Commands:** `release-commands.json` refers only to stage 04 and
  `4e06c0de…`. It has no stage-03 path and no `3665a184…`.
- **Privacy:** searching the stage-04 payload text for local paths, the
  username, the email domain and tokens finds nothing.

## Conditions (from review 02, section D, with the new hashes)

1. **Before upload:** the local reference and portable receipts must pass with
   manifest `4e06c0de…`, and every cell must show exact saved-replay and
   portable-reference equality. Any mismatch stops the release and is
   escalated, with no widened tolerances and no payload change.
2. **Upload:** only the 12 files in publication manifest `2b3289df…`, public,
   with commit message `Release manifest SHA256: 4e06c0de…`.
3. **Public check:** the anonymous download and portable phases must pass
   against the published commit before the catalog and guide publish the
   verified pins.
4. **Advisor check:** the advisor checks the local receipts before upload and
   the public receipts after. Any finding goes in a new review file.
