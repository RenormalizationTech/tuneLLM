# One final XSTest-safe250 replay

The authoritative design is `docs/xstest_replay_protocol.md`. This wrapper has
only prepare/verify/generate stages, no grader, label import or retry command.
It never calls the parent's writing CLI. All writes stay in the fresh ignored
`outputs/xstest-replay-01` directory. Root owns registration and GPU release.

```bash
.venv/bin/python -m pytest -q tests/test_xstest_replay.py
.venv/bin/python -m scripts.xstest_replay prepare
.venv/bin/python -m scripts.xstest_replay verify
```

Preparation checks the complete parent/protected snapshots, original accepted
grade receipts, retrospective audit bindings, model files and software versions.
It reproduces all 250 native rendered prompts and input IDs using only the cached
tokenizer. Manual left padding and attention masks are reconstructed from the
parent's saved input IDs and batch layout. These are **derived intent**, not old
captured tensors. The new model forward pre-hook only observes inputs: no hooks
modify activations, weights, outputs or generation settings.

After root reviews and commits `results/protocols/xstest-replay-01.json` and
explicitly releases physical GPU0:

```bash
umask 077
.venv/bin/python -u -m scripts.xstest_replay generate --execute-gpu > outputs/xstest-replay-01-generation.log 2>&1
```

GPU0 must match the original UUID with at least 20,000 MiB free and at most 10%
utilization. A failed resource guard does not launch or consume an attempt; wait
for root without changing jobs. Other preflight failures stop. The private
`generation/attempt_started.json` is written immediately before the single
generation-helper call. Any crash thereafter consumes the attempt. No automatic
resume, rerun, package change, higher cap, new grading or new data is authorized.

The exact unchanged loader/generator/decoder perform BF16 SDPA, helpful system,
nonthinking, greedy seed42, eight rows per batch and cap256, including the last
two-row batch. Newly captured prefill input IDs/masks must equal derived intent.
Comparison checks ordered IDs, user UTF-8 hashes, rendered/output UTF-8 bytes,
input/generated IDs, batch membership/positions/padded widths and EOS/cap flags.
Timing/memory are diagnostic and excluded from equality. Missing recorded parent
environment fields are explicitly unavailable, not manufactured matches.

Private `comparison.json` contains field-level differences, first differing
generated-token position (including a sequence-length difference), and mismatch
intersections with the 17 parent refusals and 19 reviewed items. It is written
only after `post_integrity.json` establishes original hashes are unchanged.
Fatal/integrity failures are closed without an equality conclusion. A full match
reuses parent labels by reference only and adds no label evidence; mismatched
outputs inherit no labels and yield no new complete refusal rate. Root projects
only aggregate counts/hashes into public documentation. Either outcome closes
the 4B study. No raw predictions, sequences or individual mismatch IDs are public.
