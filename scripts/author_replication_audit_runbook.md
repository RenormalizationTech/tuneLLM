# Author artifact audit (CPU only)

Run from the project root with the existing environment. This tool uses only
stdlib imports. It does not execute author code, load pickle/tensor files,
install packages, or invoke models/judges. No commands below have been executed
against the network by the worker.

Root reviews/releases each download separately:

```sh
.venv/bin/python -B -m scripts.author_replication_audit gcad-source --allow-download
.venv/bin/python -B -m scripts.author_replication_audit gcad-impolite --allow-download
.venv/bin/python -B -m scripts.author_replication_audit saps-results --allow-download
```

The optional `gcad-impolite` stage archives exactly two pinned CSVs and six
impolite vector blobs as opaque bytes. `gcad-source` selects all pinned `.py`
and `.sh` blobs plus README, requirements, LICENSE, NOTICE, impolite JSONs and
prompts. `saps-results` selects exactly six averaged sarcasm JSONL artifacts
(22,541,097 bytes). All stages require the existing fixed-root metadata to name
the exact commits and be untruncated. GCAD additionally checks `gcad_ref.json`.
The metadata supplied by root is the trust anchor; the tool does not make a
fresh API request to independently authenticate it.

Network requests are HTTPS to `raw.githubusercontent.com` at exact pinned
commit paths, without attached credentials. Redirects are refused. Canonical
size and Git blob SHA-1 are checked, and SHA-256 is recorded. Per-file/stage
caps apply before requests; streamed responses cannot exceed canonical size.
Existing mismatches are never overwritten. Verified existing files can be
reused. Interrupted/failed transfers remain as `.partial` files (and failure
metadata when an exception is caught); they are not silently resumed/retried.
Stop and let the coordinator inspect them. No automatic cleanup is provided.

Offline summary, private first:

```sh
.venv/bin/python -B -m scripts.author_replication_audit summary
```

Default output is the ignored audit root's `saps_summary.json`. It binds the
download manifest, six blobs, pinned source test-set bytes and official oracle
script. The last two already exist in the earlier source-only audit; their
Git blob hashes are checked against the pinned SAPS tree. Input hashes are
checked again before writing. No author module is imported/executed.

Each file must cover all 450 published sentence IDs with consistent ID-to-text
mapping within and across the six files, scale, textual output, four metric keys
and 2/3 contributor counts. Duplicate sentence/layer cells fail. Missing layer cells and null
metric denominators are reported. Missing-cell coverage uses the observed
layer union, not an assumed architectural layer range. Scores of -1 are
retained and counted rather than silently discarded.

The published legacy schema has `type: steer_random` and `scale`, without
`method`. This verified generic label is retained/countable but does not
identify a method: method provenance is the canonical pinned file path only.
Unverified legacy labels fail. Explicit `method` fields, when present, must
match the path. Exact embedded ID/text mappings are compared to the current
released test rows and by text membership separately to train/val/test. A real
mismatch yields `status: test_lineage_mismatch`, without preventing arithmetic
reconstruction over the embedded published IDs/text. Duplicate text strings
under distinct IDs are counted, never collapsed. No normalization, substitution
or reindexing is used to claim fidelity. All current split/instruction hashes
are bound. Saved `prompt_used` metadata is compared exactly against plain and
style instruction strings, with counts/rates/hashes; these metadata comparisons
cannot establish the actual rendered input or explain any discrepancy.

The exact published oracle selects the highest style per **test sentence**,
then averages metrics from that winning row. Both 0 and -1 rank as -1 because
the source uses `value or -1`; tied keys keep the first input row. This is not
fixed validation-layer performance. The differential CPU test uses an
independent literal transcription plus an AST check of the archived expression,
without executing the author script. Raw evaluator judgments cannot be
reconstructed from the averaged files.

Only after root explicitly releases publication:

```sh
.venv/bin/python -B -m scripts.author_replication_audit summary \
  --output results/author_replication_saps_published_01.json --allow-public-output
```

The only allowed public target is that exact path. Public/private summaries
contain aggregate metrics, model IDs and hashes, never original sentences or
generated text. Existing differing summary bytes are also never overwritten.
Successful stages print only counts/bytes/hashes.

Offline verification:

```sh
.venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_author_replication_audit.py
.venv/bin/python -B -m scripts.author_replication_audit --help
```
