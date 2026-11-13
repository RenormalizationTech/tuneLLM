# Physics adapter export requirements

The export is a private experimental artifact; it is distinct from the earlier
GSM8K adapter. Prepare a concrete payload and manifest before publication. The
generic publisher is `scripts.publish_release`; publication is a separate step.

## Inputs and guards

- Completed final report from `scripts.summarize_physics`.
- Fixed rank-selection report and its checksum; explicit checkpoint, training
  run, and final-report condition identifying the exported adapter.
- Include the learning-rate selection report with a hash pinned by the final
  report; retain failed LR candidates and their gate results in the card.
- Verified base publication `results/nf4_huggingface_publication.json`, which
  must be private and have passed the pinned-download GPU check.
- If selection falls back to the base, require an explicit descriptive-only
  export flag. State that the adapter failed selection; do not call it a winner.
- Refuse an existing output folder. No upload, deletion, source checkpoint
  modification, or credentials in the staging tool.

## Model payload

Copy the exact selected checkpoint's `adapter_model.safetensors`. In the exported
copy of `adapter_config.json` only, set `base_model_name_or_path` to the private NF4
repository and `revision` to its publication commit. The original Qwen revision
does not exist in the new NF4 repository and must not be used for that dependency.
Record original/exported config hashes and unchanged adapter-weight hash.

Copy tokenizer files from the verified NF4 base, excluding base-model weights and
model config. Include the upstream Apache 2.0 LICENSE. Report rank, alpha, dropout,
LR, selected epoch and step, selected checkpoint's token exposure, and the full
run's three-epoch budget separately. A selected epoch-2 checkpoint has seen two
epochs even though its run completed three.

## Reproduction and attribution

- Include curated evaluation/selection/protocol/corpus metadata, with local
  project paths replaced in export copies. Raw local reports remain untouched.
- Include all 100 paper attributions: authors, title, arXiv ID/version, source URL,
  PRL DOI and verified CC BY 4.0 license.
- Include the exact prepared `train.jsonl`, `validation.jsonl`, `test.jsonl`
  privately under `reproduction_data/`. Their scientific text and file hashes must
  stay unchanged. Confirm no local project paths are present before copying.
- Include frozen benchmark fixtures and writing prompts, with their provenance.
  State their dataset licenses and that the prepared paper text is CC BY 4.0;
  the model's Apache license does not override the data license. Document cleaning
  changes and preserve source attribution.
- Include a code-only source archive of tracked files, omitting Git history,
  model weights, raw outputs, cache and credentials. Redact project paths only in
  metadata/documentation copies; do not edit executable source. Confirm the core
  training modules match the actual run's frozen source hashes.
- Include an environment lock and reproduction guide. Users should start from
  the pinned NF4 dependency and prepared data rather than re-query live sources.
- Package a portable NF4 verification report covering the unchanged published
  model/tokenizer files. Record the parent report hash and explain omission of
  raw machine provenance, which is not a computational dependency.

## Model card and manifest

Use valid HF metadata: base model is the exact NF4 dependency, library PEFT,
text-generation, Apache 2.0, adapter relation. Explain original Qwen lineage and
the separate CC BY data attribution. Lead with the experimental status and the
fact that human writing-quality improvement is not established.

Report objective final results and development selection honestly, including any
gate failures, math regressions, BF16-control differences and limitations. Do not
claim improved PRL acceptance, novelty, human preference, originality, or full
standard benchmark scores. Point to a blinded human-review packet; its ratings
are pending until actually supplied.

Randomize A/B identities without a public deterministic seed. Retain the answer
key locally outside the payload, with its SHA256 commitment in release provenance.
Provide the user only the review packet or standalone review HTML for rating.

Produce `outputs/physics_adapter_release_manifest.json` with explicit `repo_id`,
`private: true`, and file sizes/SHA256. Target:
`RenormalizationTech/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA`.
