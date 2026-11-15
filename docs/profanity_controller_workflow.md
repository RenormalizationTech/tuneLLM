# Retained profanity workflow

Use GCAD as the current working profanity reference on Qwen2.5-7B-Instruct.
The [retained profile](../configs/profanity_retained_controller.json) identifies
the exact bank and settings; the [closeout](profanity_method_transfer_closeout.md)
reports the evidence and limits. Fixed projection and adaptive SAPS did not
qualify in this model/task comparison. The earlier Malla-style constant-vector
workflow remains a documented unsuccessful reproduction with missing author
contrastive inputs; its frozen results are preserved.

## Apply this reference

1. Read the retained profile and verify the checkpoint revision, bank SHA,
   tensor/input/runtime bindings and source implementation before loading.
2. Load the pinned native-HF BF16 Qwen2.5-7B checkpoint and its tokenizer in the
   registered environment. The bank uses that model's raw coordinate system.
3. Load the profile's GCAD bank: system-attention delta, positive-system keys
   and average demand. Use the actual reviewed GCAD hook through
   `scripts/profanity_transfer_common.py:gcad_context`, with blocks 9–19,
   scale 1.5, base coefficient 5, inverse false. Remove hooks after generation.
4. The measured setting uses default-system neutral questions, greedy seed 42
   and cap 160. A changed decoding/task setup needs its own evaluation.
5. Report profanity alongside coherence, lexical counts, caps and unwanted
   rudeness/content changes. The reference produced swearing on 14/32 held-out
   questions; response quality and trait independence remain limited.

The profile is a reproducible reference pointer, not a new checkpoint or an
interactive chat command. Existing runner stages are immutable experiment
records; future prompts should use a separate inference invocation and output
folder. Private bank files must be retained locally with their manifest.

## For subsequent models

Discuss scientific choices with Claude first and let the code worker implement
them. Start with the established GCAD extraction and intervention functions.
Create fresh matched positive/clean responses and model-specific delta/key/demand
tensors; do not copy Qwen vectors into a different checkpoint. Keep source,
model, prompt rendering, fit data, sites, strengths, seeds and evaluation split
fixed before outputs. Check mechanics, run a bounded development comparison,
lock one candidate, and confirm it against a baseline and random control on
held-out questions. Retain a setting only under the predefined style/quality
rule; document a null and missing-source limitations without an automatic rescue.

This completes the requested three-method profanity comparison. Joint control
and further model transfer remain separate future decisions.
