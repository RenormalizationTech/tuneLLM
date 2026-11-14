# Final Qwen3-4B check: exact XSTest-safe replay

Status: plan discussed with Claude; code, frozen inputs and committed registration
are required before execution. This is the final scheduled 4B run, not a new
steering study. The user selected published prompts only and asked to finish the
small-model pipeline before progressing to a larger model.

## One bounded question

Can this environment regenerate the stored unmodified-model XSTest baseline?
Replay all 250 original safe prompts in the same source order and the same 32
batches (eight per batch, two in the last). Do not select only the refused cases,
rewrite prompts, add paraphrases, change the output budget or fit any vector.

Parent: `outputs/xstest-baseline-01`, registered in `c5ef01c`.
Parent manifest SHA-256:
`ad8d1ccc729564a7fec7115fb86b0273c0c50e14a7b3c005c5e464487d513fba`.
Parent predictions SHA-256:
`6fbc602f3d35c75f6f67190dc8164440954783c805405932b73e15257cb4250a`.

Use the parent's pinned BF16 Qwen3-4B model, native tokenizer/template, helpful
system message, nonthinking mode, greedy seed 42, batch eight and cap 256.
Retain the same decoding of newly generated tokens with terminal EOS omitted;
do not strip whitespace or generated content. The original software and source
files remain unchanged; use a new thin wrapper and a new output directory only.

## Preparation and execution

Before generation, verify parent snapshots, source hashes, all 478 external
bindings, model files, predictions, accepted labels and completed audit artifacts.
Snapshot the new wrapper/config/runbook and this protocol in a fresh
`outputs/xstest-replay-01/` directory. Register the exact manifest in Git before
launch. Runtime and source binding failures halt; they do not trigger repairs or
extra model calls. No network model download or evaluator call is needed.

CPU checks must reproduce all 250 rendered UTF-8 prompts and input token IDs.
Verify the expected manual left padding and attention masks from the parent's
frozen inputs and batch widths. The parent did not save actual attention-mask
tensors: this is a derived reference, not a comparison to captured old tensors.
Validate the masks/padding delivered to the new generation calls against that
reference without changing hidden states or model weights.

Use physical GPU 0, UUID `GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b`, with the
existing guard of at least 20,000 MiB free and at most 10% utilization. If the
guard fails, wait without launching or disturbing jobs; do not override it.
Record GPU identity, driver, CUDA runtime, Python, Torch/Transformers/tokenizer
versions, BF16/SDPA settings, batch layout, generation time and peak memory.
Compare recorded environment fields with the parent where available. Report
missing parent fields as unavailable rather than inventing equality. A changed
runtime makes a mismatch's cause ambiguous and is not evidence of model size
causing instability.
Record prelaunch environment differences without upgrading, downgrading or
patching packages to restore equality; preserve the existing binding checks.
The single attempt starts at the first generation call. A crash or interruption
after that point consumes it and must be reported; waiting on the guard does not.

## Equality and stop rule

Compare the complete ordered set of IDs, user/rendered-prompt/output UTF-8 bytes,
input/generated token IDs, batch membership and positions, padded widths, EOS
and cap flags. Timings and memory measurements are excluded from equality.
Record per-item differences privately, including the first differing generated
token position (or sequence-length difference). Publish counts/hashes and the
number of mismatches among the 17 parent refusals and the 19 reviewed items;
do not publish raw text or token sequences.

On a full match, reuse parent labels by reference only. This adds no independent
grading evidence and does not change 17/250 (6.8%) or the 16 review-supported
cases. On any mismatch, report it and stop: no rerun, regrading, setting change
or attempt to improve the match. Do not inherit labels for mismatched responses
or calculate a new complete refusal rate. Parent results remain the record.

Hash-check that all original inputs, protocols, outputs, grades and reviews
remain unchanged after the run. Record failures and incomplete execution
honestly; one attempted replay does not authorize another attempt.
Issue an equality/mismatch result only after this after-run integrity check
passes. If it fails, report an integrity failure rather than a replay match.

## Completion meaning

After this replay, close the 4B learning study with the observed result. Existing
interventions demonstrated partial refusal reduction with a quality tradeoff;
no validated useful steering configuration exists. An exact replay validates
artifact regeneration in this environment, not robustness to new wording,
general behavioral stability, global safety or the effectiveness of steering.

No further 4B sweeps, higher-cap runs or paraphrase tests follow. A larger model
gets a separate reviewed protocol and its own unmodified baseline. Larger size
does not guarantee more stable behavior, and all existing 4B results remain as
measured rather than being attributed to size without evidence.
