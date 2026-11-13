# Qwen3-4B PRL-register pilot

This is a learning experiment in condensed-matter scientific writing. The target
is language and structure in titles, abstracts and introductions, not scientific
novelty, journal acceptance, or improved physics problem-solving.

## Data and objective

Select 100 PRL theory papers across the eight available primary condensed-matter
categories. Category balance is approximate and supply constrained. Verify PRL
DOIs through Crossref and CC BY licenses on the versioned arXiv abstract pages.
Use the corresponding source archives. Store raw inputs and cleaned text locally
under ignored directories, retaining attribution, source hashes and review reasons.
The theory labels are AI-reviewed from abstracts, not human-certified.

Freeze a category-stratified 70/10/20 paper split. Keep all versions and chunks
of one paper together. Author overlap is measured and disclosed rather than
claimed absent. Recent paper selection does not prove absence from pretraining.

Remove comments, hidden blocks, author/affiliation commands, funding and back
matter. Retain equations, citations and used LaTeX macro definitions. Record
annotation and long-document flags. Training consumes only the `text` field,
not the metadata. Use causal next-token loss over within-paper chunks of 1024
tokens, with a one-token overlap to preserve every prediction target after the
first paper token. Include all remainder chunks and one EOS at paper end.

## Planned runs

Use the verified saved NF4 base, all-linear LoRA, alpha = 2r, dropout 0.05,
microbatch 1, gradient accumulation 8, BF16 computation, linear learning-rate
decay and 5% warmup. Train at most three epochs, evaluating and saving each epoch.
All NF4 runs receive the same chunks and token exposure. Use seed 42 except the
explicit repeat. Log actual loss tokens, optimizer steps, time and allocator memory.

1. Rank 32, learning rate 5e-5.
2. Rank 32, learning rate 1e-4.
3. Rank 8 at the selected pilot learning rate.
4. Rank 128 at that same learning rate.
5. Repeat the selected configuration with seed 43, reporting stability without
   changing the seed-42 selection.
6. One BF16-base LoRA control with the selected rank and schedule. Frozen BF16
   weights stay BF16; adapters are FP32. NF4 retains the verified PEFT-prepared
   FP32 unquantized modules. This is a comparison of workflow representations.

The repeat and control use the full registered three-epoch schedule; compare their
checkpoint at the chosen NF4 epoch. This avoids changing the learning-rate schedule
by shortening a run to that epoch. Each exact per-run protocol is committed before
optimization, and execution uses the frozen source snapshot.

## Selection and safeguards

Only development data may influence selection. Final-test predictions and paper
loss are withheld until the selected NF4 configuration and control are fixed.

For each run, nominate the epoch with minimum token-weighted development NLL;
ties choose the earlier epoch. NLL is a measure of register fit, not a writing
quality rating. Use 10,000 paired paper-bootstrap resamples, seed 20261112, with
the summed-NLL/summed-token ratio within each draw. This reflects paper clustering.
Apply the screen gates only to that nominated checkpoint. If it fails, the
entire run is ineligible; do not try another epoch to obtain a passing result.

For learning rate, prefer the smaller value when its loss difference from the
best includes zero in the paired 95% bootstrap interval. For rank, prefer the
smallest eligible rank whose loss difference from the best includes zero. These
are conservative pilot selection heuristics, not equivalence tests or proof of
the optimal rank/learning rate.

Screen gates, fixed before training:

- On each 32-question development MC/math task, the net score drops by at most four
  correct answers
  relative to the saved NF4 base.
- On 16 IFEval development prompts, the net score drops by at most three strict
  prompt-level passes. Report gross fixed/broken item counts separately.
- Generation truncations increase by at most two per task relative to the base.
- All 30 development writing outputs are nonempty. New-number/citation flags
  exceed the base-with-few-shot count by at most two; writing truncations increase
  by at most two. Flags are imperfect diagnostics and require human review.

For the writing gate, evaluate the adapter zero-shot and compare it with the
unadapted NF4 base given two training-paper examples. Both receive the same
new-paper evidence and task instruction; only the base receives exemplars.

If no adapter passes, retain the base as the recommended model. Report the best
observed register-fit adapter as experimental if useful, with gate failures
prominently disclosed. No automated score establishes better scientific writing.

## Paired evaluation

The development screen contains 32 general MMLU, 32 physics MMLU, 32 GSM8K,
16 official IFEval and 30 writing tasks. The final screen contains 128 general
MMLU, 96 physics MMLU, 128 GSM8K, 32 IFEval and 60 writing tasks (444 total).
Multiple-choice scores use raw zero-shot MMLU answer-letter likelihoods. This is
a custom local runner with standard prompt construction, not a full published
MMLU leaderboard score. IFEval uses the pinned official strict scorer. URL-fetching
prompts are excluded for offline evaluation. Report subsets and scoring explicitly.

Use greedy non-thinking generation, seed 42 and HF batch 8 consistently; MC batch
16. Caps: GSM8K 512, IFEval 1024, title 48, abstract 256, introduction 768 tokens.
GGUF uses the same input-token IDs and decoding settings, with sequential
llama.cpp requests; disclose backend and batching differences.

Writing prompts give source evidence excluding the title, abstract and a heuristic
introductory passage. Keep whole paragraphs within the evidence budget; record
evidence/reference overlap. Compare base, base with two training-paper examples,
selected adapter and BF16 control. Do not use reference similarity as a primary
quality metric. Blind paired reviews should assess claim fidelity first, followed
by clarity, concision, structure, accessibility and supported significance.
Human ratings remain pending until the user actually reviews them.

Report final results per task, paired item changes and uncertainty. Include paper
NLL and its paper-bootstrap uncertainty separately from writing diagnostics.
Run the reference BF16 and existing Q8/Q4 inference artifacts through the upgraded
screen; reuse their recorded size/speed measurements with their original scope.
Reload-check new adapters, preserve negative results, and make no PRL-acceptance claim.
