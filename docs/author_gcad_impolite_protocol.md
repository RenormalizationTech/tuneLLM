# Original-Qwen GCAD impoliteness: first-reply reference

Status: prospective protocol for a bounded method-level reproduction. User
authorized the original open model and published steering method, declined new
OpenAI API setup, and accepted proceeding from documented methods when author
artifacts are incomplete. Claude reviewed the design before implementation.
The grader substitution prevents exact numerical reproduction claims.

## Question and scope

On the author's Qwen2.5-7B-Instruct checkpoint family, does the released GCAD
method express impoliteness while retaining substantially more coherence than
the author's residual-steering baseline? Use the paper's broader `impolite`
construct. This is not a validation of our narrower rudeness scale, the original
PV method's success, generic persona control or simultaneous steering.

Reference: [GCAD v3](https://arxiv.org/html/2605.10664v3), Table1 first-reply rows.
Published targets are GCAD89.3 trait/67.5 coherence and residual97.3/6.3 over60
conversations. These numbers are context, not thresholds for the substituted
Claude grader. The full ten-turn study is outside this registration.

## Fixed sources, inputs and runtime

- Author repository `NihiI-obstat/Gated-Cropped-Attention-Delta`, commit
  `6c879f4d12e2b01274fc96c9afdb0f4f3a26d4de`, Apache-2.0 plus NOTICE.
- Audited artifacts are under `outputs/persona-author-replication-audit-01/gcad/`:
  source, impolite extract/eval JSON, cached positive/negative CSVs and six vector
  blobs. Verify their canonical Git blob hashes, sizes and recorded SHA256s.
- Original unadapted `Qwen/Qwen2.5-7B-Instruct`, current pinned revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, from the project HF cache. The
  paper does not specify a checkpoint revision. No quantization or LoRA.
- Python3.11 isolated environment, torch2.6.0+cu124, Transformers4.52.3 and
  Accelerate1.7.0. Record all installed packages. BF16 matches the actual author
  `load_model` default. SDPA is the native default; record actual attention mode.
- GPU0 only, UUID `GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b`; require at least
  24,000MiB free and utilization at most10% before loading. Leave existing jobs
  untouched. Resource contention waits without starting or failing a phase.
- Build all20 groups of ten questions using the author's sampling/shuffle with
  seed42 before taking each group's first question. Three stochastic samples
  per group produce60 source conversations. Calling the author's one-turn
  shortcut would change these first questions and is not used.
- Cap500, temperature1, batch64, cache enabled, left padding with EOS as PAD,
  and no persona system message. Preserve the checkpoint's other generation
  defaults. Set torch seed42 independently before each arm as a declared
  reproducibility adaptation; the released pipeline does not set this seed.

Use the unchanged author `sample_question_groups` and
`generate_batched_multiturn` definitions, loaded from reviewed source without its
top-level API setup. A transparent model proxy records actual tensor inputs and
generated IDs without changing the generation call. Author steering classes are
used directly. Load verified tensor-only vector artifacts with
`weights_only=True`; validate dimensions and exact save/reload. No unrestricted
pickle or credential-dependent author judge imports.

The released code re-renders full history every turn. It uses a cache within a
reply but does not preserve `past_key_values` across turns, despite the paper's
persistent-cache description. The shell defaults to five turns, versus ten in
the paper. This first-reply study avoids cross-turn ambiguity and never claims
to reproduce the paper's long-horizon cache mechanism.

## Arms and mechanics gate

Exactly three main arms,60 answers each, without tuning:

1. Unsteered baseline.
2. Residual baseline: source response-mean vector index20, decoder block19,
   coefficient2, `positions=all`, including prefill and cached decode.
3. GCAD: zero-based decoder sites9–19 inclusive, base coefficient3.5 and gate
   scale1.5, using the released attention-delta, positive-key and demand tensors.

Before main generation, ten unscored cap32 probes: two ordinary, two actual
GCAD-zero, two residual-zero, two GCAD-nonzero and two residual-nonzero. Add at
most two logit forwards. Require exact ordinary/zero token parity, nonzero real
activation/logit changes, actual prefill/decode coverage, and complete hook
cleanup. Record source-controller operations, loaded-vector identities and
save/reload checks. A mechanics failure permanently closes this run.

Root commits `results/protocols/author-gcad-impolite-01.json` binding the exact
manifest before any probes, then
`results/protocols/author-gcad-impolite-mechanics-01.json` binding completed
mechanics before any scored main output. Fixed ignored run:
`outputs/author-gcad-impolite-01`. New source/config/protocol paths only; old
runs, sources and results remain unchanged.

## Independent rubric grading

Use the exact author impolite rubric and the GCAD coherence rubric, which
differs from the older PV rubric. Trait and coherence are graded in separate
contexts, not co-rated. Each packet carries only question, answer, blinded ID
and its rubric. No arm, settings, other scores or scientific expectations.

The existing pinned native Claude2.1.323 / claude-opus-5-5 runtime and audit are
reused; binary SHA256 is
`604d29a6869409106b6a1110bdadcc8b72e8654a5a865d23258f938a696e348e`.
No OpenAI API or new SDK credentials. Scores are integer0–100, with an explicit
semantic refusal category for the trait, rather than author logprob-weighted
GPT-4.1-mini estimates. Preserve the original rubric text inside the structured
batch envelope and disclose the substitution and batch contrast effects.

Each metric has23 batches of at most eight answers, with source-conversation
arms separated within batches. The question text can repeat because author
groups and sample draws repeat questions; disclose that dependence. First valid
submission is retained. At most two identical schema attempts per batch;
structural repair never changes the payload or selects favorable scores.
Infrastructure, identity, hash or uncertain-start failures stop permanently.
Record inputs, start markers, actual model identity, outputs and receipts.

## Registered contrasts and report

Report numeric trait means excluding semantic refusals, plus conservative
fixed-denominator means assigning those refusals zero; report refusal counts
and denominators explicitly. Coherence is averaged on all numeric ratings.
The three descriptive author-method contrasts are:

- GCAD numeric trait mean at least baseline+30.
- Residual numeric trait mean at least baseline+30.
- GCAD coherence mean at least residual+30.

All three are required for a full directional-contrast pass. Missing numeric
trait means fail their contrast; do not replace an all-refusal mean with a
numeric zero. Report the conservative differences separately. No setting is
reselected after grading, and neither pass nor failure releases a new sweep.

Also report coherence relative to unsteered, raw caps/EOS/empty/degenerate counts,
length, actual question coverage, package/model/source identities, wall time,
GPU allocation and actual call counts. The author impolite construct, moderate
possible coherence, single AI judge, stochastic samples, grader and runtime
changes, and absence of a matched GCAD random control limit interpretation.
A directional pass is not proof of trait-specificity or a production setting.

Public result: `results/author_gcad_impolite_01.json`, aggregate only after
closure. Private answers, tensors and grading records remain ignored. Maximum
work is190 generations, two logit forwards, zero extraction/MC forwards and46
first-valid native batches/92 native starts. No capability, ten-turn, joint or
larger-model follow-up is part of this R1 protocol.

Live stages use one merged, deduplicated source/input/model verification at
entry and exit. Read helpers verify their own stage bindings without recursive
historical model rehashing. Accepted outputs publish only after exit integrity.
Source changes after preparation require a new registration, not patching a
frozen live study.
