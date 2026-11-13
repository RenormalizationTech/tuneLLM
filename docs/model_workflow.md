# Reusable model adaptation workflow

Agreed with the user on 2026-11-12. The first implementation uses the pinned
Qwen/Qwen3-4B checkpoint. Learning quantization and fine-tuning remains the main
project goal; scientific-writing gains are an experimental outcome, not assumed.

## Artifact lineage

Preserve the upstream repository, immutable revision, tokenizer, license, package
versions and checksums. Large artifacts, datasets and raw outputs remain under
git-ignored `artifacts/`, `data/`, `outputs/` and `.hf/`.

Produce independent branches directly from the original high-precision weights:

1. BF16 reference checkpoint.
2. GGUF Q8_0 and Q4_K_M for inference. Keep the existing BF16 GGUF reference.
3. bitsandbytes NF4 with double quantization and BF16 compute for QLoRA.

Do not cascade one lossy quantization into another. Save the standalone NF4 base
and tokenizer **before** attaching LoRA. Reload it and verify packed tensors,
quantization metadata, logits and deterministic generation. Record preparation
of unquantized parameters (our PEFT preparation casts them to FP32). Every adapter
must identify the exact saved base and its quantization settings.

## Condensed-matter scientific-writing pilot

Collect approximately 100 PRL condensed-matter **theory** papers, balancing topics
rather than selecting one subfield. Verify PRL publication through APS DOI and
metadata, match the arXiv paper and version, and record the license and source URL.
An arXiv category alone does not establish that a paper is theoretical. Favor
sources with clear reuse permission; a mirrored dataset does not override each
paper's license. Record any shortfall instead of silently changing eligibility.

Prefer LaTeX sources, preserving prose, equations and section structure. Remove
comments, unused files, author/affiliation details, acknowledgments, templates,
publication-status boilerplate and journal labels. Keep attribution and DOI in
the separate provenance manifest. Treat papers and their versions as a single
group for splitting; related duplicates must not cross train/dev/test boundaries.

The first training objective is causal next-token prediction on cleaned paper
text, including its scientific prose and LaTeX. This is domain-adaptive continued
training with LoRA, not the previous GSM8K assistant-only supervised objective.
Measure and report actual token counts. Do not assume 100 papers improve general
physics reasoning or implement a mapping from PRB manuscripts to PRL acceptance.

The first writing evaluations cover titles, abstracts and introductions. Later
editing SFT needs reviewed input/revision pairs and is a separate experiment.

## Baselines, training and selection

Evaluate the saved NF4 base before training. Compare all-linear LoRA ranks 8,
32 and 128, using identical target layers and maximum token exposure. Nominate
each run's checkpoint using development loss; nominated epochs can differ. Adapter storage
of 1–5 GB is an upper allowance rather than a size target. Estimate actual
parameters, checkpoint dtype, optimizer and activation memory before execution.

At rank 32, run a bounded learning-rate pilot (5e-5 and 1e-4), with a recorded
warmup and decay schedule. This pilot registers a three-epoch maximum with
development loss measured after every epoch. Nominate the minimum-loss epoch
(ties prefer earlier), then apply capability/writing gates to that epoch only;
do not search other epochs after a gate failure. Future budget changes require
a new registration. Keep `alpha/r` consistent and record its value. Repeat leading results with
a second seed when practical. Run one BF16 LoRA control with the selected adapter
configuration, data order, token budget and schedule. Compare the repeat/control
at the primary chosen epoch while retaining the same full scheduler horizon.

Freeze source, data IDs/hashes, splitting, development selection rules and final
test protocol before optimization. Use development writing behavior and general
capability checks alongside validation loss. Lower paper loss measures better
fit to the register; it is not proof of better writing. Never select on final-test
outputs. Preserve an unadapted base as a valid outcome if adapters fail checks.

## Evaluation tiers

Prefer a pinned standard harness for supported tasks, especially answer-likelihood
scoring for multiple-choice tasks. This pilot uses a validated custom local
zero-shot MMLU likelihood runner and the pinned official IFEval scorer; report that
scope explicitly. Validate identical prompts, tokenizer/chat
templates, thinking settings, scoring and budgets across backends. Do not present
custom generated-letter accuracy as the standard likelihood benchmark.

- Iteration screen: roughly 150 items spanning math, general knowledge, physics,
  instruction following and writing. It catches large failures, not small gains.
- Candidate evaluation: roughly 400–700 items with increased writing coverage.
- Finalists: broaden only tasks for which more resolution is needed.

Existing physics sets include MMLU college/high-school/conceptual physics and
SciBench. They are retention checks for this writing project. Keep development
and final-test items separate, including when splitting a labeled validation set.
Report per-task results, paired differences, uncertainty, invalid outputs and
truncation; do not hide regressions in a single mean.

For writing, compare identical source material under base, base with a few
examples, and trained adapters. Use held-out papers. Check scientific fidelity
(claims, equations, numbers and citations), clarity, concision, organization,
accessibility and supported framing of significance. Preserve paired outputs in
randomized, blinded review packets. Automatic checks and model ratings are
diagnostics; human preference is pending until actual human review occurs.
Held-out token-weighted loss is a separate objective metric.

Budget evaluations in prompt/output tokens and measure throughput on each model.
Small screens have substantial statistical uncertainty. A larger model can also
be limited by memory and offloading; do not extrapolate runtime from parameter
count alone.

Compare BF16 LoRA with an unadapted BF16 baseline, and NF4 LoRA with the saved
unadapted NF4 baseline. Compare GGUF quantization within the same llama.cpp
backend, retaining its BF16 GGUF reference. A cross-backend difference cannot be
attributed to quantization alone.

## Completion and release

Verify adapter serialization and regeneration on a freshly reloaded saved base.
Report all planned ranks and controls, exceptions, actual time/memory, and limits.
Keep new experimental adapters private by default, consistent with the user's
earlier publication choice. Review concrete model cards and data-license
compatibility before any new upload. Existing published artifacts remain immutable.

Better PRL-like presentation cannot establish novelty, publication suitability
or acceptance. A negative or inconclusive experiment is still a completed,
reproducible learning exercise when reported honestly.
