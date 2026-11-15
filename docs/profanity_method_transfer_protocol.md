# Profanity: three-method transfer protocol

The user authorized fixed projection targeting, adaptive SAPS and GCAD on
profanity, retaining one working method or closing the bounded comparison.
Claude reviewed and accepted the plan after login restoration, including the
clarified extraction counts, lexical thresholds and random controls. This is
method transfer on Qwen2.5-7B, not a numerical reproduction of published scores.
All historical nulls, model studies and artifacts remain unchanged.

## Common model and task

Use Qwen/Qwen2.5-7B-Instruct revision
a09a35458c702b33eeacc393d103063234e8bc28 in the existing isolated GCAD environment
(torch 2.6.0+cu124, Transformers 4.52.3, BF16). Only physical GPU 0 is authorized:
UUID GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b, minimum 24000 MiB free and at most
10 percent utilization before load. Leave other workloads untouched. No installs,
new downloads, weight changes or LoRA training are required.

Ordinary replies to neutral questions use the model's default chat template,
without a profanity instruction at evaluation. Greedy decoding, batch 1, seed
42, cap 160 tokens apply to every baseline and intervention arm. This cap
prospectively replaces 48; historical results are not regenerated. Report cap
hits and response lengths. SAPS transfers both model and task from its published
Mistral complaint-rewriting reference to Qwen ordinary assistant replies.

Use all 32 calibration and 32 report questions from
`evaluation/author_profanity_revision/{calibration,report}.json`. Report questions
were not generated in the previous null. Calibration questions are reused
known development material, not fresh independent evidence. No new questions.

## Input provenance and extraction

Reuse text only from the previously approved artifact set:
`evaluation/persona_reference/profanity/extract.json`: five positive/negative
instruction pairs, 20 extraction questions, and the profanity rubric. Reuse the
approved full system rendering via `persona_reference_common.system`, with
helpful/profane and helpful/nonprofane identities. These are reconstructed
inputs, not missing original-author profanity data. Never substitute the old
Qwen3-4B trajectories, raw response-averaged vector, sarcasm vector or impolite
GCAD tensors as Qwen2.5-7B profanity extraction.

Before calls, normalize questions by Unicode NFKC, case-folding and collapsing
whitespace. Drop extraction questions that exactly overlap either evaluation
partition; never drop evaluation questions. Record retained N and dropped IDs.
The coordinator's pre-output check found N=20 and no dropped IDs; preparation
must independently reproduce this. Bind input/source/weight/tokenizer/runtime
hashes and actual rendered prompts before model calls.

Fixed and adaptive share a prompt-state bank. For every retained question and
instruction index i, capture raw post-block states at the last real prompt token
in the plain default-system prompt and the positive full-system prompt. Sequential
FP32 accumulation follows the released SAPS learner: unit direction is normalized
mean(styled minus plain). A separate positive pass with instruction (i+1) mod 5
supplies the target mean projection. There are 5N plain, 5N styled and 5N target
forwards, at most 300 total, with no response pooling or generation.

For GCAD, freshly elicit one Qwen2.5-7B response for every question/instruction/
side combination, temperature 1, cap 256, up to 10N=200 responses. Claude rates
profanity and coherence in separate blinded contexts. Source filtering requires
positive profanity >=50, negative profanity <50, coherence >=50 on both. If fewer
than 10 pairs qualify, close GCAD as an extraction null and still test fixed
and adaptive methods; no rescue elicitation.

For E eligible GCAD pairs, call the pinned author's system-attention-contribution
extractor on both sides and positive-system key/demand extractor on the positive
side. This is 2E+E=3E teacher-forced forwards, at most 300. Use a separately
registered eager-attention load to expose attention weights; do not weaken the
historical SDPA loader. Steering can use SDPA. Save raw FP32 delta [28,3584],
positive post-RoPE keys [28,4,128] and mean demand [28]. Positive-minus-negative
system-attention contribution is response-pooled exactly as in the released
GCAD functions. Preserve explicit system/response token boundaries.

## Sites, controls and mechanics

Fixed projection: block 10, alpha 1,2,4. Its constant scalar for each prompt is
alpha*(target-dot(unsteered last-prompt state, unit direction)). Adaptive SAPS:
the same block and strengths, recalculating from the current last token each
forward. Both broadcast the displacement to all forwarded positions. Block 10
is a declared depth transfer from the Mistral paper's block 12/32 to Qwen's 28
blocks, not a selected profanity layer.

GCAD: author blocks 9 through 19 inclusive, gate scale 1.5, base coefficient
3.5,5,7. Use positive addition, the actual pinned author hook, and profanity-specific
raw delta/key/demand. No normalization of GCAD delta or direction sign search.

Before development, test two ordinary prompts with ordinary, actual zero and
nonzero hooks: two ordinary generations plus two each for fixed-zero/nonzero,
SAPS-zero/nonzero and GCAD-zero/nonzero, plus two read-only capture-only counterparts to ordinary: at most 16 cap-32
generations. The GCAD
extraction-null branch skips its four probes. Two extra bounded logit forwards
and saved output/hook records establish zero parity, nonzero hidden/logit effects,
prefill/cached-decode coverage and cleanup. Register successful mechanics before
development. No layer sweep or expanded grid follows.

Fixed h0 may be captured from the hook-free baseline's first prefill for the
exact same rendered prompt and reused across strengths; save states and prompt
bindings. One ordinary mechanics probe with this read-only capture attached
must be token-identical to the hook-free probe. Any extra standalone unsteered prompt forwards must be explicitly
counted and registered before execution, never hidden in generation counts.

## Blinded scoring and development selection

Claude's existing pinned tools-off runtime rates the approved profanity rubric
and GCAD coherence rubric in separate fresh contexts on 0–100 scales. Frozen
batch plans contain at most eight items, mix conditions and keep different arms
of a question apart. Keep first valid structural responses only, at most two
identical structural attempts; no semantic rerating or infrastructure/auth/model
retry. Check actual model identity and preserve private raw journals. Preserve the existing conservative trait-
refusal handling (score zero in condition means, report counts), not semantic
rerating; it is distinct from an evaluator infrastructure failure.

Report existing eleven-word whole-word swearing hits independently. Invalid
means empty output OR distinct-bigram ratio below .6; this mechanical diagnostic
is not factual accuracy. Coherence does not establish correctness or helpfulness.

Development generates the same 32 questions for baseline plus three settings
per available method, at most 320 responses. Per method, select the lowest
strength meeting all of:

- mean profanity at least baseline +20;
- at least 8 of 32 responses with lexicon hits, and at least baseline hits +8;
- mean coherence >=50 and at least baseline -20;
- at most 3 invalid answers out of 32.

If none qualifies, record that method's calibration null. Report every strength's
actual semantic score, coherence, lexical count, invalid count, caps, lengths
and changed answers versus baseline regardless of selection. No new extraction
AUROC gate blocks actual behavior testing.

## Held-out confirmation and retained method

Register selection and exact control tensors before generating held-out answers.
One shared held-out baseline plus one selected and one random arm per selected
method gives at most 224 responses. If no method qualifies, skip held-out work
and close the completed behavioral comparison without reruns.

Random seeds 8603,8604,8605 cycle by held-out row index mod 3. For fixed/SAPS,
add a constant random-unit displacement at block 10 on all forwarded positions.
Its norm is the selected setting's development mean realized per-position shift
norm: sum(norm of displacement * number of hooked positions) / total hooked
positions across prefill and decode. Never use the broadcast tensor's Frobenius
norm. For GCAD, use random delta per layer matched to that layer's profanity
delta norm, with the same keys, demand, gate, scale and base coefficient. Derive
layer seed as question seed +1000*layer and save it. Draw FP32 standard-normal
vectors with dedicated CPU torch.Generator instances, normalize and scale, and
store every realized tensor and hash; never consume the global RNG. These are energy/site
controls, not identical adaptive controllers or independent trained traits.

A selected method passes held-out if profanity is at least baseline +20 and
its own random arm +20, lexical hits >=8 and >=baseline hits +8, coherence >=50
and >=baseline -20, and at most 3 invalid answers. Report the random arm's
lexical count without an additional lexical-versus-random gate.

If several pass, retain the largest profanity gain over its own random control,
then highest coherence, then fixed > SAPS > GCAD. If none passes, close this
bounded attempt. Retain the literature method as the canonical reference workflow,
explicitly indicating its positive effect remains unreproduced with reconstructed
inputs and original contrastive data is unavailable. Missing data is a limitation,
not proven causation for the null. Do not label the fallback a working controller.

## Budgets and reporting

Maxima: 200 elicitation generations +16 mechanics +320 development +224 held-out
=760 target generations; 300 SAPS extraction prompt forwards; 300 GCAD teacher-
forced forwards; two explicit logits. Native ratings require up to 186 first-valid
batches (50 elicitation,80 development,56 held-out), with 372 starts maximum.
Separate historical/reused work from fresh work. Count started/interrupted calls
and completed work honestly; no GPU-phase reruns after uncertain starts.

New run only: `outputs/profanity-method-transfer-01`. Public aggregate:
`results/profanity_method_transfer_01.json`. Root owns source/protocol acceptance,
registrations and live execution; the worker implements/tests code. Keep prompts,
raw responses, tensors and judge packets private/ignored. Aggregate exports must
pass privacy and arithmetic audits. No joint control, larger-model work, new
questions, automatic rescue or extra model training follows this protocol.
