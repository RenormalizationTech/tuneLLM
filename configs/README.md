# Experiment configurations

JSON files identify checkpoints, data, decoding and study-specific settings.
Names match their scripts and protocols; numbered or alternative configurations
are retained experimental history, not duplicate current defaults.

Start with [baseline.json](baseline.json) for the original 4B workflow,
[physics.json](physics.json) for the physics corpus/training workflow,
[refusal_7b_transfer.json](refusal_7b_transfer.json) for the fixed 7B transfer,
and [qwen38_27b_nf4.json](qwen38_27b_nf4.json) for the selected 27B loading smoke.
A config's presence does not mean its intervention passed acceptance gates.

Existing configurations may be hash-bound by protocols. Keep them in place and
write a new config for a new experiment. Use the [current roadmap](../docs/project_tracks.md)
to distinguish selected future work from closed studies.
