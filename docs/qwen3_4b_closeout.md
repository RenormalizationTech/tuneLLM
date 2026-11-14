# Qwen3-4B learning study: complete

The small-model learning study is closed. Quantization, LoRA/QLoRA training,
evaluation, release and activation-intervention mechanics were exercised. This
is a completed learning pipeline, not a claim that a useful refusal-reduction
configuration was obtained. No additional 4B sweep or higher-cap run is planned.

## Final technical replication

The last registered check replayed all 250 published XSTest-safe prompts with
the same unadapted BF16 weights, native nonthinking prompt format, seed 42,
batch membership and 256-token cap. Registration and the exact source/input
manifest were committed in `7c609df` before generation.

**All 250 cases matched.** Generated/input token sequences, rendered/output
UTF-8 bytes, batch layout, EOS and cap flags matched, with zero mismatches among
the 17 primary refusals or the 19 reviewed cases. The new actual prefill masks
matched the derived parent reference; the original masks were not captured.
All 681 protected bindings, including the parent's 478, passed the after-run
integrity check before any equality conclusion was issued.

The 19 comparable environment fields matched. Python and the two TF32 parent
fields were unavailable rather than assumed equal. No new grading was performed;
the parent labels are referenced, not counted as another independent benchmark.
The [aggregate replay record](../results/xstest_replay_01.json) contains timings,
memory, source hashes and limitations. Raw sequences remain local and ignored.

## Scientific conclusions retained

- Registered refusal interventions demonstrated a partial reduction with a
  quality tradeoff. The final benign usefulness count decreased from 29/42 to
  23/42; the original quality criteria failed. Those results are unchanged.
- The unmodified model's XSTest-safe baseline was 17/250 partial/full refusals
  (6.8%). A contextual AI review supported 16 cases as candidate unnecessary
  refusals; it did not revise labels or establish ground truth.
- Exact replay establishes regeneration of stored outputs in this environment.
  It does not establish robustness to new wording, global safety, successful
  steering or model size as the cause of any earlier failure.
- Physics LoRA remains a separate fine-tuning demonstration. The selected
  profanity/rudeness/sarcasm objective remains separate persona activation work.

See the [two-method explanation](activation_steering_methods.md),
[intervention result](refusal_reduction_execution.md),
[published baseline](xstest_baseline_execution.md) and
[stored-output diagnosis](xstest_stored_output_audit.md).

## Next model

Claude reviewed the transition to official Qwen3-14B in BF16 on one A6000.
Pinned assets are downloaded and verified; native tokenization of all 250
XSTest inputs matches 4B. The next phase is a separately reviewed
[unmodified-model baseline](qwen3_14b_baseline_protocol.md), including quality
and appropriate-refusal boundary measurements, without transferring a 4B vector
or assuming that a larger checkpoint is more stable.
