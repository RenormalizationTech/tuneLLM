# Public release 03 review (7B steering bank; diff to `stage_steering_release.py`)

Reviewer: Claude advisor, read-only. I made no model, GPU, native or HF calls
and ran no tests.

**What I read:**

- the working-tree diff to `scripts/stage_steering_release.py` (+412 lines)
  and the new public tests;
- the dedup rule for `.gitattributes` in `scripts/publish_release.py`.

I checked the diff against the accepted public scope: 7B only, GitHub kept
private, a self-contained artifact.

## Verdict

**The structure is sound. Two wording defects must be fixed before building
03** (P1, P2). There is one condition for publishing (P3). Everything else
meets the accepted scope.

## Verified

- **Metadata:**
  - `private: false`;
  - `status: public_release`;
  - `repo_id` is exactly `RenormalizationTech/qwen2.5-7b-gcad-steering-controls`;
  - the builder hash.
- **No overwrite:** both 03 and the publication manifest must be new.
- **Source binding:** `resolve_source_ref` resolves an existing **tag** to a
  40-character commit. It requires the tagged bytes of the builder, the
  tests, the reload receipt, the 4B aggregate and closeout and the two refusal
  documents to equal the working copy. The tag and commit are labelled
  private provenance, and no GitHub link to the project repository is
  allowed: the link check rejects `github.com/RenormalizationTech/tuneLLM`.
- **Inference core:** every bank, `profile.json`, `provenance.json`,
  `runtime/*`, LICENSE and NOTICE file is copied from the reviewed 02, with
  manifest `05aea48f…`. Each is required byte-identical to 02 on every
  verify.
  - The `EXAMPLE` string literal was reformatted. Byte identity is still
    enforced, because `validated_parent()` runs `preflight(DEST)`, which
    compares `runtime_files()` with 02's bytes. The full old test suite must
    confirm this (pending).
- **Validation:**
  - `reload_proof` requires the 02 receipt (3 traits, 6 forwards, 0 answers,
    exact logits) and that its runtime hash basis equals 02.
  - It is stated as CPU byte identity, with no 03 GPU rerun claimed.
  - The 01 false-positive history is carried forward.
- **Exactly three derived public files are added:**
  - **The 4B JSON derivative** reduces the two project paths in
    `native_identity` to basenames and keeps all numbers and decisions. It
    records the hash of the immutable original (`004c2de…`), and asserts
    `complete` with zero passing traits.
  - **The closeout derivative** replaces backticked project paths with
    basename plus private hash, and redirects relative links.
  - **`METHODS.md`.**
- **Public preflight:**
  - an exact allowlist plus those three files;
  - the project-path scan (`configs|outputs|scripts|docs|results/`, which
    correctly ignores `related_results/`) over all JSON and Markdown;
  - the existing secret, path, pickle and raw-field scans;
  - Markdown links limited to payload files or four allowlisted public
    URLs;
  - draft or "research use only" wording rejected.
- **Licence:** Apache-2.0 with the NOTICE verbatim, plus "no additional use
  restriction".
- **Accuracy:**
  - The card's numbers match the retained profiles: PF, SARC with random
    coherence 85.69 and 87.72, and impolite 1.62 → 77.37 and 82.68 → 75.9.
  - The 4B summary matches the aggregate.
  - The Persona Vectors commit `b8e0f04…` matches the project's source
    review.
- **Publication manifest:** written outside the payload, and it includes
  `release_manifest.json`. That is consistent with the publisher's
  `.gitattributes`-only exception, so there is no hash cycle.
- **Tests:**
  - the inference core's byte identity and the strict CPU loader against
    the public bundle;
  - rejection of each tampered field: visibility, status, hash, secret, an
    extra file, a core change;
  - the tag guard.

## Required

**P1. The public closeout derivative makes a false claim.** The copied
closeout says "All 4B outcomes are released as aggregates on **GitHub**"
(source line 108). The project repository is private.

- In `public_closeout_derivative()`, replace that clause with "released as
  aggregates in this bundle (`related_results/`)".
- The redirected "[interpretation]" and "[plan]" links now point to
  `METHODS.md`, which misreads as the interpretation itself. Label them
  "(private review; not included)" or drop the link.
- **Test:** the public closeout contains no "on GitHub" release claim.

**P2. "Trusted" overstates what the manifest hash shows.** The card says to use
the hash H from the commit message "as the trusted bank manifest hash".

- **What H proves:** H gives integrity and consistency for a pinned
  revision. Authenticity rests only on the `RenormalizationTech` HF account.
- **Fix:** say "expected manifest hash", plus one line: "This checks that the
  files at the pinned revision are internally consistent. It does not
  independently prove who published them."

**P3 (operator condition, at publish).** The card depends on the HF commit
message reading exactly `Release manifest SHA256: <sha256 of 03 release_manifest.json>`.
The builder doesn't set it.

- **Before upload:** root passes exactly that string to the publisher's
  `--message`.
- **After upload:** check the remote revision's commit message and file
  hashes against the publication manifest. Record both in the publication
  receipt.

## Optional

- A one-line test that the 4B public derivative contains no string matching
  `PROJECT_PATH`. Preflight covers this already, so a test is only
  belt-and-braces.

**Status:** after P1 and P2, and the full test suite passing (26 old plus the
new ones), the worker may build 03. Root then tags it, runs `verify-public`,
and publishes with P3.
