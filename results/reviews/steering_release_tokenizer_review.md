# Steering release: tokenizer guard fix review (staging 02)

Reviewer: Claude advisor, read-only. I made no GPU, model, native or HF calls
and ran no tests.

**Read:**

- `git diff HEAD` of `scripts/stage_steering_release.py` and
  `tests/test_stage_steering_release.py`;
- the failure log `outputs/steering-release-validation-01/profanity.log`;
- the staging 02 tree and its manifest.

I compared the staging 01 and 02 bank files and the runtime port byte for
byte. This review doesn't modify the earlier staging review or any 4B-bound
file.

## Diagnosis: confirmed, a guard false positive

- **What failed:** the 01 loader compared the live tokenizer's
  `backend_tokenizer.to_str()` with the raw on-disk `tokenizer.json`. The
  log shows `validate_base` raising "actual tokenizer backend differs" after
  the shards loaded and **before any forward pass**.
- **Why:** these are two serializations of the same tokenizer. The on-disk
  merges are `"a b"` strings, while the installed `tokenizers` writes them as
  pairs, among other format differences. There was no tokenizer, model or
  bank problem.
- **Consequences:** no result is affected, and 01 is correctly kept as the
  record of the failed check.

## The fix: acceptable

**`validate_tokenizer`:**

- **Hashes first.** It requires the complete set of six snapshot metadata
  files and checks each against its pinned hash **before** loading the
  reference.
- **Source path.** The tokenizer must come from that snapshot.
- **A fresh local reference:** `AutoTokenizer.from_pretrained(snapshot,
  local_files_only=True, trust_remote_code=False)`, with the same
  EOS-as-PAD and left padding.
- **Comparisons:**
  - token IDs and padding side;
  - vocabulary and chat template;
  - six backend sections from each side's own `to_str()`: `model`,
    `normalizer`, `pre_tokenizer`, `post_processor`, `decoder` and
    `added_tokens`.
- **Mutable state is excluded.** The allowlist compares only those six
  sections, so the backend's `padding` and `truncation` state is
  excluded, as I asked.
- **No custom normalization code.** The serialization formats match
  automatically, because both sides come from the same library version.

**Everything else is unchanged:**

- base ID, revision and snapshot name;
- geometry, BF16 and SDPA;
- the pinned bundle manifest;
- the bank and tensor hashes;
- `validate_geometry`.

**Staging 02:**

- The three `bank.safetensors` files are **byte-identical** to 01.
- `runtime/gcad.py` is a byte copy of `scripts/qwen3_gcad_port.py`.
- 01 and the validation log are preserved.

## Tests

The new test uses the **real cached 7B tokenizer** files and checks that
those files are unchanged afterwards.

- **Accepted:** the unmodified tokenizer.
- **Rejected:**
  - a swapped merge pair;
  - a replaced normalizer.

**This is adequate in principle.** Three small cases are still needed to cover
the points in my advice. Each is one parametrized case, on CPU:

1. **Padding-state acceptance.** Call `tokenizer(["a", "bb"],
   padding=True)` before `validate_tokenizer`, and require it to pass.
   This guards against the next false positive.
2. **Tamper with an added or special token,** such as changing one
   `added_tokens` content, and **tamper with the chat template.** Both must
   be rejected.
3. **A wrong pad token, or right padding,** must be rejected.

## Required before the public stage

**Missing: the record of the 01 failure.** The 02 `release_manifest.json` has
no record of it. Add a field such as:

```
"prior_validation": {"staging_01": "real-7B load guard false positive (tokenizer serialization compare); 0 forwards; superseded by staging_02; 01 tree and log preserved", "log_sha256": "<hash of profanity.log>", "staging_01_manifest_sha256": "<hash>"}
```

Mention it in one README line as well.

**`real_7b_reload_validation`** should stay "deferred" until the coefficient-0
rerun. That rerun happens after the 4B evaluation generation has freed GPU 0,
while judging continues, as planned.

## Verdict

**Accepted**, once the three test cases and the 01-failure record in the 02
manifest are added. The worker can make these before the public stage. Then
run the real 7B coefficient-0 reload once the GPU is free. There is no
scientific change and no user decision is needed.

## Addendum: conditions resolved

I checked these read-only:

- `outputs/steering-release-staging-02-receipt.json`;
- the 02 manifest (sha256 `05aea48f…853e`, which I recomputed and which
  matches);
- the 02 README;
- the test diff.

**Results:**

- **The diagnosis is now specific.** Only `model.merges` and
  `model.ignore_merges` differ, and both are serialization or default-field
  differences:
  - **merges:** 151,387 pairs, as space-delimited strings on disk and
    two-element arrays in memory, with the canonical values and order
    exactly equal;
  - **`ignore_merges`:** absent on disk, and `false` when loaded.

  All other backend keys are equal. This confirms a false positive in the
  guard.
- **The tests cover all requested cases.** The parametrized real-cached-
  tokenizer test now covers:
  - **accepted:** the unmodified tokenizer, and the tokenizer after a
    `padding=True` call;
  - **rejected:** a merge, the normalizer, an added or special token, the
    chat template, a wrong pad token, and right padding.

  The cache files are unchanged afterwards, and no weights are loaded. The
  worker reports 26 tests passing, and root's independent run is still in
  progress.
- **The 01 failure is now on record.**
  - **In the 02 manifest:** `prior_validation` records the false positive,
    `model_forwards: 0`, `zero_selftest_completed: false`, the differing
    keys and the repair.
  - **Hashes:** I recomputed the `log_sha256` (`16c2da80…`) and the 01
    manifest hash (`256318b7…`), and both match the files.
  - **README:** line 54 states "0 forwards".
  - **Receipt:** the initial 02 manifest bytes are preserved, as the receipt
    shows.
- **Nothing else changed.** All three banks are byte-identical to 01, and
  `runtime/gcad.py` is still a byte copy of the port.
  `real_7b_reload_validation` correctly stays "deferred".

**Verdict: all conditions are resolved. Staging 02 is accepted as the local
draft.** The remaining steps are root's independent 26-test receipt, then the
real 7B coefficient-0 reload once 4B evaluation generation frees GPU 0. Upload
waits for both.
