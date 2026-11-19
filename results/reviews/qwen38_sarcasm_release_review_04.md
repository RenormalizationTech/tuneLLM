# Qwen3.8-27B sarcasm GCAD controls release: review 04 (R6 receipt; clearance for public HF upload)

Advisor review, 2026-11-19. It is read-only.
- No model, judge or GPU call was made.
- No source was edited.

## Receipt: verified

The receipt is `results/qwen38_sarcasm_release_local_smoke_02.json`, sha256
`c9173dbe…25e9`. Its status is `passed_release_mechanics`, with a
mechanics-only scope and `raw_outputs_recorded: false`. A privacy scan of
the receipt is clean.

**Bindings.**
- The receipt binds:
  - the release manifest `cfaacd0e…`, which matches the current payload
    file;
  - the frozen port `3d76d814…`;
  - the canonical result `61362b38…`;
  - the saved aggregate `2bf33ce5…`;
  - the verifier `0e9eeade…`, the file signed in review 03;
  - GPU0 `GPU-34646ab9…`, an RTX A6000;
  - `python_isolated: true` and offline mode.
- Every file signed in review 03 still has its signed hash:
  - the release and publication manifests;
  - the builder, loader, `gcad.py`, `__init__.py` and `example.py`;
  - the verifier, the tests and the card draft.

**Probes** (2 prompts × b ∈ {0, 8, 16}, cap 32).
- All 6 entries show equal logits and equal tokens, finite values and a full
  32-token suffix.
- At b = 0 the comparison is with plain unhooked output. At b = 8 and 16 it
  is with the frozen `Steerer`.
- The b = 0, 8 and 16 logit and suffix hashes differ from each other, so
  steering is active and not a no-op.
- `cleanup_equal` and `cleanup_tokens_equal` are both true.

**Saved-trajectory replays** (cap 160, through the public `generate_ids`).
- F01 baseline, b8 and b16 are all token-equal.
- For each row I independently checked two things against
  `outputs/qwen38-sarcasm-followup-01/generation/rows`:
  - the recorded `saved_row_sha256` equals the saved row file's sha256;
  - the recorded `suffix_sha256` equals the canonical-JSON digest of the
    saved `generated_ids`, which have 160 tokens.
- The released runtime therefore reproduces the study's exact b8 and b16
  trajectories on the studied stack.

**Attempt 01.** The attempt 01 failure is preserved: a pre-load
`ValueError`, receipt `635f07ee…`, with 0 model calls. The protocol records
its cause.

**Protocol.** `results/protocols/qwen38-sarcasm-release-02.json`
(`0a2ddd4d…`) binds:
- the same manifests and sources;
- review 03 (`59030cd0…`, the current file);
- the base `586bb022…` and the GPU UUID;
- 53 CPU tests.

It keeps both scientific outcomes unchanged and has no quality or judge
calls. It also states the publication condition and the P5 plan.

## Clearance

**Cleared for public HF upload** of exactly the 17-file payload in
`outputs/qwen38-sarcasm-release-01/payload` to
`RenormalizationTech/qwen38-27b-sarcasm-gcad-controls`, with these
conditions:
1. Upload exactly the files in `publication_manifest.json` (`2c3efab7…`).
   Add, regenerate or edit nothing. Any Hub-managed `.gitattributes` is
   excluded by the loader rule.
2. The commit message must contain
   `Release manifest SHA256: cfaacd0efa5a7b22a40fab5305217ee2c642b46ba230a2eb8259d5a28639e54b`.
3. **P5, reviewed in a new file.** Every step below must pass:
   - an anonymous `snapshot_download` of the controls repo at the published
     commit, and of the base at `586bb022…`;
   - every remote and downloaded file's bytes and sha256 equal the
     publication manifest;
   - the commit message hash equals `sha256(release_manifest.json)`;
   - a fresh `python -I` offline verifier run from the downloaded copy, with
     `--compare-receipt results/qwen38_sarcasm_release_local_smoke_02.json`,
     gives `public_local_mechanics_equal: true`.

   If any step fails, the repository's status must say so. The card must not
   claim download verification before P5 passes.
4. Catalog and GitHub docs must keep `development_null` and
   `exploratory_null_through_b16` exactly. They must describe the package as
   an experimental demonstration, not a validated controller. They must link
   reviews 01–04 and the P5 review.

This clearance is a byte-consistency and mechanics clearance. It is not a
scientific upgrade of either outcome.
