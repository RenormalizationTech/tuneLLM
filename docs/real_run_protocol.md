# First bounded QLoRA experiment

This protocol is fixed before training results are inspected. It follows the
successful two-step pipeline smoke and keeps hands-on learning as the goal.
The exact selected IDs and file hashes will be recorded with the preflight
metadata and committed before the optimizer run.

## Model and training

- Model: Qwen/Qwen3-4B at
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- Load the same NF4/double-quantized base with BF16 compute and FP32 preparation
  of nonquantized modules as the smoke run.
- Use attention LoRA rank 8, alpha 16, dropout 0.05, q/k/v/o projections.
- Select 1,024 prepared GSM8K training rows with seed 42. Check token length
  before selection; explicitly exclude overlength training rows and record them.
  Do not silently truncate supervised answers.
- One epoch, 64 optimizer steps, effective batch 16. Start with per-device
  batch 2 and accumulation 8 on physical GPU 1.
- Learning rate 2e-4, linear decay, warmup ratio 0.05. Record the actual warmup
  step count chosen by Transformers; do not assume rounding behavior.
- Maximum training sequence length 2048, right-padding, assistant-only loss
  including end-of-turn. Base model weights stay frozen.

## Checkpoint selection

Measure reference-answer loss on all 256 validation rows before training and
at steps 16, 32, 48 and 64. Save steps 32 and 64. Choose the saved checkpoint
with lower validation loss; ties choose step 32. This is the only selection
rule. Do not inspect final-test predictions to change it.

Validation loss measures fit to the reference responses and can reflect style
and formatting rather than reasoning. Record loss, gradient norm, learning
rate, memory and elapsed time throughout the run.

## Batching check

Use eight fixed validation rows to compare generation batches of 1 and 8 before
the training experiment. Use explicit non-thinking prompts and greedy decoding,
up to 512 generated tokens. Record exact token agreement, strict/fallback answer
agreement, format/truncation agreement, batch membership and wall time.

Batching may change floating-point behavior. Any disagreement must be reported;
do not search for a favorable batch configuration. Use batch 8 with the same row
order and membership for the final base/adapter/reload comparisons. Generation
uses left-padding and attention masks. Slice at the padded input width and remove
trailing padding before counting or scoring. Training stays right-padded.

## Final evaluations

Freeze all evaluation IDs before training. Evaluate the final conditions after
checkpoint selection:

1. **GSM8K:** 128 source-test rows, excluding the 16 smoke-test IDs. Compare the
   NF4 base with adapters disabled and the selected trained adapter. Use the
   same prompt, batch membership, seed and generation budget for both.
2. **Retention:** 128 questions from the pinned `cais/mmlu` all/test split,
   stratified across subjects excluding names containing `math`,
   `abstract_algebra`, and `high_school_statistics`. Use seed 1234. Present four
   choices and request a single letter, with eight generated tokens maximum.
   This is a small retention probe, not the full standard MMLU evaluation.
3. **Reload:** load the selected adapter onto a fresh identically prepared base.
   Check exact equality of the saved/reloaded PEFT adapter tensors. Reproduce
   generated tokens for the first 16 final GSM8K rows (the same first two batches
   of eight) and the retention fixture. This checks serialization without
   redundantly regenerating all 128 longer math responses. Record this scope
   explicitly and do not claim all 128 math responses were replayed.

GSM8K preparation uses revision `740312add88f781978c0658806c59bc2815b9866`,
strips calculator annotations and adds the same final-answer instruction to
every split. MMLU uses revision `c30699e8356da336a370243923dbaf21066bb9fe`.
Both dataset cards identify MIT licenses. Raw and processed datasets remain in
ignored local folders. Only IDs and provenance are committed.

Report strict/fallback accuracy, format/invalid-answer rate, token lengths,
truncation and paired fixed/broken/unchanged counts. Fallback correctness is the
pre-specified primary paired metric for both GSM8K and the MMLU retention probe;
strict accuracy and format compliance are secondary diagnostics.

GSM8K strict scoring requires the final line to match `#### N`; numeric comparison
removes thousands separators and uses decimal values. Its fallback compares the
last number in the output. MMLU strict scoring requires the entire normalized
output to be one letter A-D, case-insensitive. For non-strict MMLU outputs, the
fallback reads the first standalone uppercase A-D letter in the case-preserved
NFKC text. This avoids turning the English article `a` into a predicted choice.
Fallback scoring is heuristic; report invalid and ambiguous answers separately.

For both primary paired metrics, report an exact McNemar test with unadjusted
exploratory p-values. A nonsignificant result is not proof of equivalence
or retention; a 128-item subset has limited power. Describe observed changes and
uncertainty without claiming improved reasoning from lower loss alone.

## Release interpretation

Successful training, complete provenance and a passing reload establish a working
learning artifact. Whether it is useful also depends on the measured accuracy,
retention and qualitative responses. An experimental adapter may be documented
even without an accuracy gain, but must not be described as a better general
model. Material regressions must be prominent in its card.

Base GGUFs and adapters are separate artifacts with separate cards. No merged
BF16 or merged GGUF model inherits these results without its own evaluation.
Hugging Face uploads require account authentication and the intended repository
visibility. Local training and packaging can finish before those are supplied.
