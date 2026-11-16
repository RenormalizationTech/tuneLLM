# Fixed twelve-arm joint GCAD study (prospective)

This is one new descriptive joint-control study on the pinned unadapted
Qwen2.5-7B-Instruct `a09a35458c702b33eeacc393d103063234e8bc28`. It does not
refit banks, optimize joint coefficients, modify earlier decisions, or establish
independent trait knobs. Root reviews/commits the complete source, fixture,
calibration, plans and initial registration before calls. No other experiment is
released by this protocol. The later benign political baseline is separate.

## Immutable inputs and twelve arms

Reuse the three published raw FP32 GCAD banks and frozen runtime from
`outputs/steering-release-staging-03`, checked against its manifest and profiles.
Their provenance/model/tokenizer basis must match; all28 layers are retained,
and only zero-based blocks9–19 are active, scale1.5, inverseFalse. No extraction,
weight edit, normalization, random vector, model download or refit is allowed.

The eight full-factorial settings are the products PF(0,5), impolite(0,3.5),
sarcasm(0,2). Add three half singles(2.5,0,0), (0,1.75,0), (0,0,1) and the half
triple(2.5,1.75,1). All12 arms answer all24 fresh neutral practical-advice
questions once:288 responses, no development/selection/held-out distinction and
no effect-driven stopping. Baseline has a null context, without hooks. Singles
omit all zero-coefficient banks.

The advisor-authored proposal is reviewed before conversion to the tracked
fixture. Preserve its exact question/answer/rubric bytes; record the author,
source hash and side->level/min_score,max_score->band conversion. Check NFKC,
casefold and whitespace-normalized exact overlap against all listed prior
fit/evaluation fixtures plus PM semantic review. This is a lexical check and
bounded review, not proof of semantic independence. Same-family AI fixture and
judge authorship is a limitation. No target/native study calls author fixtures.

## One shared hook and registered mechanics

At each active attention site capture the same native post-RoPE query for every
trait. For each trait use its own saved key and average demand and the unchanged
source gate `2*base*sigmoid(1.5*(demand-average))`. Multiply its raw delta, sum
shifts in canonical alphabetical trait-name order, and add once at the attention
output. Gates do not read another trait's modified attention output. Keep
model-dtype arithmetic. This is not equivalent to asserting that behavioral
outcomes add linearly downstream.

The mechanics budget is20 cap32 generations on the first two frozen questions:
ordinary hook-free, actual all-zero joint, each of three published singles and
matching joint singles, full triple and reversed-input-order triple. Run ten
explicit first-question logit forwards for these ten modes (last-token logits
only). Require exact zero logits/tokens; exact published-single logits/tokens;
exact permutation logits/tokens; nonzero full-triple logit effect and measured
nonzero single shifts. Require all11 sites with no skipped calls, correct
prefill/decode coverage, finite diagnostics and hook/state cleanup. Decode count
is actual suffix length minus one, accommodating immediate EOS without inventing
decode coverage. Root commits mechanics result/run/integrity hashes before main
generation. No retry on a failed or uncertain GPU phase.

## Runtime and fixed generation

Use the existing Python3.11 GCAD environment: torch2.6.0+cu124,
Transformers4.52.3 and accelerate1.7.0. Local pinned BF16 weights, SDPA, native
Qwen default system, EOS-as-PAD left-padding tokenizer. Physical GPU0 UUID
`GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b`, free>=24000MiB/util<=10%, preserve
other jobs. Resource busy before phase creation is WAIT with no writes/calls.

Seed42, B1, main cap160; explicit do_sampleFalse, temperature1, top_p1, top_k0,
repetition_penalty1.05 and use_cacheTrue. Save the complete effective generation
configuration, native input/generated IDs, rendering, EOS/cap, timings and
per-layer/per-phase audits. Repetition/empty checks reuse the frozen ordinary
word-bigram criterion: empty OR distinct bigram ratio<.6. Both invalid components
and fixed-denominator trait scores remain descriptive; no label replacement.

The target ceiling is308 generations (20mechanics+288main), ten explicit logit
forwards, zero TF/MC. Full current-model weights and source/bank/snapshot integrity are checked at
GPU command entry and exit, with one merged traversal per check. No model
weights are traversed per native judgment. Any non-WAIT execution/integrity,
model, auth, isolation or uncertain-call error permanently stops the run.

## Separate native judgments and calibration

Reuse the existing audited tools-off native CLI2.1.323, binarySHA604d29a6869409106b6a1110bdadcc8b72e8654a5a865d23258f938a696e348e,
claude-opus-5-5, medium effort, isolated fresh contexts. Required network calls
are root-released host execution. No dummy audit or new SDK. Exact unchanged
PF, impolite, sarcasm and GCAD coherence strings come from the frozen source
files named in config/proposal; four separate journals, never co-rating.

Freeze eight anchors per metric, four low/four high, broad per-anchor bands.
Calibration passes only if>=7/8 scores are in-band AND high mean minus low mean
>=30 for EACH metric. Trait REFUSAL contributes zero; coherence must always be
numeric. All four calibration streams must finish and root commits the passing
calibration lock before main generation. Failure is a calibration null with no
rubric editing or semantic reroll.

Primary:288 answers x4metrics =144 batches of eight. Duplicates: preselect two
answers/arm using all24 questions once (arm index j gets question2j and2j+1);
three batches/metric =12. Duplicate aliases/journals are separate and never
replace primary first-valid labels. Fixed metric order: PF, impolite, sarcasm,
coherence; complete all primary streams before any duplicates. Payloads contain
only opaque IDs, question, answer, single rubric and schema; no arm, coefficient,
anchor band or design labels. All batches have eight distinct questions.

Total:4calibration+144primary+12duplicate=160 first-valid batches, max320 starts.
At most two identical structural attempts per batch. First valid is immutable;
modelUsage mismatch/absence or auth/infra/isolation/uncertain errors are fatal,
not structural retries. Starts consume budget before invocation; replay does
not consume a new start. Retain all attempts/receipts; validate payload, run
binding, start and actual identity on reads.

## Predeclared descriptive analysis and guide

Report every arm's four fixed-denominator means, trait refusals, cap/empty/
invalid/length and fixed PF lexicon proxy. Quantify 3x4 full-single cross-effects.
Factorial contrasts are **average mixed finite differences** over unspecified
factors, weights `product(2*bit-1)/2**(3-order)`; these are not the usual scaled
effect-coded interaction coefficients. Report all three pair additivity residuals
`pair-baseline - (singleA-baseline) - (singleB-baseline)`, and full/half triple
residuals against their matched singles.

Bootstrap the24 matched questions with all English questions’ arms/scores together,
2000 draws, dedicated seed106012, unadjusted95% percentile intervals for arm
gains and contrasts. This is descriptive uncertainty, not a new selection gate.
Report within-arm Spearman as the primary descriptive correlations; additionally
raw pooled and prompt-centred pooled correlations with explicit condition-
confounding caveats. Constant ranks yield null, not invented correlation.
Duplicates report exact agreement and absolute/signed score differences, without
replacing primary labels or creating a pass threshold.

Assess all three FULL pairs, full triple and half triple as guide candidates.
For each REQUESTED trait require gain>=10 vs baseline AND>=50% of its positive
matched isolated gain. If isolated gain<=0 mark that target N/A, never passing.
Full pairs/triple compare to full singles; half triple to half singles.
Coherence>=50 and>=baseline-20; invalid<=2/24. Off-axis traits are descriptive,
not a rejection gate. No candidate ranking, refit or extra calls. A guide remains
exploratory, with failures and tradeoffs visible; it is not a newly validated
joint-controller release. Coherence is not factual accuracy. Completed primary
results survive descriptive-duplicate failure but execution is then explicitly
incomplete. No conditional publication or downstream experiment is automatic.
