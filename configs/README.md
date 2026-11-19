# Experiment configurations

JSON files identify checkpoints, data, decoding and study-specific settings.
Names match their scripts and protocols; numbered or alternative configurations
are retained experimental history, not duplicate current defaults.

Start with [baseline.json](baseline.json) for the original 4B workflow,
[physics.json](physics.json) for the physics corpus/training workflow,
[refusal_7b_transfer.json](refusal_7b_transfer.json) for the fixed 7B transfer,
and [qwen38_27b_nf4.json](qwen38_27b_nf4.json) for the selected 27B loading smoke.
A config's presence does not mean its intervention passed acceptance gates.

[qwen38_sarcasm.json](qwen38_sarcasm.json) freezes the separate NF4 hybrid-model
GCAD sarcasm study. Its [execution record](../docs/qwen38_sarcasm_execution.md)
distinguishes mechanics from quality outcomes; source/config hashes and phase
approvals govern execution.

[qwen38_sarcasm_followup.json](qwen38_sarcasm_followup.json) binds the separate
completed activation/stronger-GCAD follow-up. Its [closeout](../docs/qwen38_sarcasm_followup_execution.md)
separates strong sarcasm at strengths 8/16 from failed coherence gates. A single
prospective registration covers the complete launcher.

Existing configurations may be hash-bound by protocols. Keep them in place and
write a new config for a new experiment. Use the [current roadmap](../docs/project_tracks.md)
to distinguish selected future work from closed studies.


Consumer tuning uses `--strength` in the [published 27B controls runtime](../docs/qwen38_sarcasm_release.md).
It does not require editing any frozen study configuration or refitting the bank.
