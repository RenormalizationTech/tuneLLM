# Steering release staging review (`scripts/stage_steering_release.py`)

Reviewer: Claude advisor, read-only. I made no GPU, native or HF calls, ran
no tests and edited no files.

**What I checked:**

- the current staging script (369 lines) and the test list;
- `.gitignore`: the staging root is ignored through `/outputs/`;
- the registered vector metadata for the impolite bank.

I compared these against my approved minimal release advice and
`steering_release_scope_review.md`.

## Verdict

**Acceptable as the local draft bundle, after two small fixes (S1, S2).** It
meets the minimal scope. It touches no 4B-bound file: the script is new,
imports only `author_gcad_impolite_common` and `save_nf4.tensor_hash`, and
modifies neither.

## Verified

**Source integrity before deserialization.**

- **Bound sources:**
  - each source profile, registration and aggregate result must be the
    committed `HEAD` bytes;
  - every `path`/`sha256` reference in the PF and SARC profiles is
    re-hashed.
- **Bank files:** each is hash-checked **before** `torch.load(...,
  weights_only=True)`. The original per-tensor hashes are then checked
  against the retained profile's `tensor_sha256`, or, for impolite, against
  the registered `vector_metadata`.
- **After conversion:** all sources are rechecked, and a source change fails
  closed.

**Raw tensors are preserved.**

- `delta` [28, 3584], `key` [28, 4, 128] and `demand` [28], as finite FP32.
  These are all 28 layers, unnormalized.
- The registered impolite metadata has these exact shapes. The author's
  residual vector [29, 3584] is correctly excluded.
- The safetensors round-trip requires `torch.equal`, and both the source and
  the converted tensor hashes are recorded.

**Runtime.**

- `runtime/gcad.py` is a byte copy of the reviewed port. Preflight rejects
  any change, and the port's Qwen2 path is test-proven.
- **The loader checks:**
  - the pinned bundle-manifest hash;
  - the per-file hashes;
  - base ID and revision, with the snapshot basename equal to the revision;
  - Qwen2 geometry (28 / 3584 / 28 / 4), BF16 and SDPA;
  - the snapshot metadata hashes, the actual vocabulary and the chat
    template;
  - tensor names, shapes, dtype, finiteness and hashes;
  - `validate_geometry`.
- No pickle is loaded and there is no network or remote code. The real-7B
  coefficient-0 self-test is deferred, as agreed.

**Scientific numbers and limits.** These match the retained profiles and
`docs/steering_reference.md`:

| Trait | Design | Trait | Coherence | Other |
| --- | --- | --- | --- | --- |
| PF | n = 32 | 0 → 31.125 | 85.34 → 82.69 | 14/32 lexicon hits |
| SARC | n = 32 | 0.06 → 63.78 | 89.59 → 70.72, a 1.125 margin | 29/32 scores ≥ 50; impoliteness 38.75 |
| Impolite | 60 first replies on 13 questions | 1.62 → 77.37 | 82.68 → 75.90 | no random arm |

- Each controller keeps its own design and n.
- The coupling, the single judge family, Claude-generated inputs and
  non-comparable strengths are all disclosed.
- The 4B results are pending. There is no refusal bank.

**Licence and provenance.**

- The author's LICENSE and NOTICE are copied verbatim, and the NOTICE credits
  Persona Vectors.
- `provenance.json` records:
  - the author repository and revision;
  - the source manifest, registration and result hashes;
  - the input licences;
  - the hash of the port source.

**Preflight.**

- An exact allowlist, with no symlinks.
- Pickle and other suffixes are rejected, because the allowlist contains only
  `.safetensors`, `.json`, `.md`, `.py`, `LICENSE` and `NOTICE`.
- Text is scanned for paths and secrets (`/home/`, `/tmp/`, `hf_…`, `sk-…`,
  `Bearer`).
- JSON keys that could carry answers, judge reasons or token arrays are
  forbidden.
- Every manifest hash is rechecked, and `private: true` with
  `status: local_draft_not_uploaded` is required.

## Required (small)

**S1. The run instruction in the README is wrong.**
`python -m runtime.example` only works with the **bundle root** (the folder
that contains `runtime/`) as the working directory, and `example.py` itself
resolves `root = parents[1]` to that folder. The README says "from the
bundle parent" (line 341). Change it to "from the bundle root".

**S2. Verify the model manifest before using it.** `stage()` reads
`outputs/persona-author-replication-audit-01/model_manifest.json` (line 283)
and takes the snapshot metadata hashes from it **before** any profile
reference check. It is verified only later and indirectly, through
`refs(profile)`.

- **Fix:** check it first against the PF profile's
  `model.manifest.sha256` (or call `load_source_profile` first), and
  only then use it.
- **Test:** a tampered manifest is rejected before any write.

## Notes (optional, no blocker)

- **Runtime requirements.** List them in the README: `torch` 2.6.0,
  `transformers` 4.52.3 (≥ 4.51 is needed, because the port imports the
  `qwen3` module), `safetensors` and `tqdm`.
- **PF and SARC bank attribution.** These banks are project-produced
  artifacts, released under the bundle's Apache-2.0 licence. Add one README
  line saying so. Leave the author's NOTICE verbatim.
- **The loader's EOS-as-PAD check.** The loader requires EOS as PAD and left
  padding, which matches the example. Mention it, so that callers who load
  their own tokenizer aren't surprised.

**Status:** after S1 and S2 and passing CPU tests, this is accepted as the
local draft bundle. The real-7B reload and coefficient-0 parity check, the
4B rows and any upload all remain deferred.
