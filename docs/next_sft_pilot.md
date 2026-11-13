# Optional backlog: a small task-specific SFT exercise

Status: optional backlog, not the next required milestone. Dataset selection and
protocol registration would precede execution. This document does not register
a training run. The original quantization and QLoRA milestone is complete; the
earlier GSM8K experiment already performed instruction/response fine-tuning.

## Goal

Optionally reuse the pinned Qwen3-4B and verified NF4 base to practice a different
instruction/response task with automatic scoring. The proposed
task is extracting information from supplied text into a fixed JSON schema.
Physics writing and a larger-model exercise remain deferred.

Raw paper training provided examples of domain language. This exercise provides
examples of a requested behavior: instruction plus source text as input, and a
correct JSON response as the target.

## Minimal scope

- One existing model and one fresh LoRA adapter, starting from the unadapted NF4
  base rather than continuing the physics adapter.
- One declared configuration and seed 42; rank/LR/schedule selected for this task.
  The physics rank-128 setting is not an assumed optimum for extraction.
- Physical GPU 1, existing Python/PyTorch/Transformers/PEFT/bitsandbytes stack.
- A small dataset with verifiable field labels and clear reuse terms. The exact
  source, sample count, JSON schema and missing-field policy are still to be set.
- Source-document groups stay disjoint across training, development and final
  testing; deduplication precedes splitting and prompt formatting.
- No default rank/LR grid, extra seed runs or full quantization sweep. Calibration
  is added only for a stated question, with a new bounded protocol.

## Implementation work still needed

Reuse the pinned model/environment and NF4 save/reload helpers in
`scripts/common.py` and `scripts/save_nf4.py`. Adapt the earlier SFT path in
`scripts/train_qlora.py` and the masking/collation in `scripts/training_data.py`
to load the verified saved NF4 base. Add task data preparation, a JSON/schema/
field-value scorer, and the task-specific screen/report block. The physics
orchestration/export guards remain specific to that completed experiment; they
are not a ready-made generic JSON-task runner.

## Data and loss

Store instruction/response records using the model's chat template or a declared
prompt/completion format. Inspect decoded batches before execution. Loss should
cover the desired assistant response and its end token, with prompt and padding
tokens masked. Verify examples with a trusted parser before training.

Targets must follow the declared schema and be supported by the supplied text.
Decide how absent or ambiguous fields are represented before creating labels.
Keep the source revision, sample IDs, transformations and split hashes.

## Evaluation and budget

Record a development base reference with identical prompts, schema and decoding settings.
Reuse cached references only under the compatibility rules in
[the reusable workflow](model_workflow.md).

Primary task measures:

- JSON parse success, including whether extra prose is allowed.
- Schema compliance and correct field types.
- Field-value correctness and complete-record correctness, with normalization
  rules fixed in advance.
- Missing/invented fields, empty outputs and truncation.

Use a small task-specific development block alongside general retention checks.
Freeze the dataset and splits before timing preflights. Training preflights use
training examples; generation preflights use development prompts. Log these
trials, discard their adapter weights and start the actual adapter fresh. Use the
timings to freeze maximum training exposure, schedule, checkpoint rule and final
evaluation budget before the actual optimization run. Final-test examples are
excluded from timing/calibration. The routine final comparison contains the base
and one passing selected adapter. Final
test outcomes report the result and never trigger checkpoint reselection.

Nominate one checkpoint by the registered development criterion. A small,
budgeted extraction-accuracy check may be more suitable than loss alone; declare
its examples, frequency and cost before training. Run the full generation-based
development retention screen on the nominee once. Keep the base if it fails;
do not screen successive epochs until one passes.

## Completion

Save the adapter with its exact NF4 dependency, verify fresh reload, and record
task accuracy, retention changes, actual token exposure, time and memory. A
negative or inconclusive result still completes the learning exercise when
reported honestly. Human or AI writing ratings are not required for this task.
Release selected experimental artifacts privately when a release is requested
under the project's existing publication scope.
