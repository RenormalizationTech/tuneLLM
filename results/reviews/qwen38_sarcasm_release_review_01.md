# Qwen3.8-27B sarcasm GCAD controls release: review 01 (criteria and draft-doc check)

Advisor review, 2026-11-19. It sets the scientific and portability
acceptance criteria for the experimental HF controls package
`RenormalizationTech/qwen38-27b-sarcasm-gcad-controls`. It also checks the
current guide and card drafts against the evidence.

This is a criteria review. Source and payload signoff will follow in
release review 02, once the worker's runtime, stager, tests and staged
payload exist. The review was read-only. No model or judge call was made,
and no source was edited.

## 0. Evidence the release rests on (fixed; no new science)

**Canonical result and closeout.**
- `results/qwen38_sarcasm_followup_01.json`, sha256 `61362b38…81ad`.
- Outcome `exploratory_null_through_b16`.
- `old_outcome_unchanged: development_null`.
- Independently recomputed in follow-up review 06.

**Bank.**
- `outputs/qwen38-sarcasm-gcad-01` bank, sha256 `4fd9f86b…45de`.
- It is fixed, read-only and shipped byte-identical.

**Sites and scale.**
- Zero-based full-attention layers 23, 27, 31, 35, 39 and 43.
- Gate scale 1.5, coefficient 2b·σ(·), added after `o_proj`.

**Base model: verified in this review.**
- The six runtime files used by the follow-up run are recorded in its
  manifest `model_files`:
  - `model.safetensors`;
  - `config.json`, `generation_config.json`;
  - `tokenizer.json`, `tokenizer_config.json`;
  - `chat_template.jinja`.
- They have the same sha256 as the public
  `RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4` inventory at revision
  `586bb02218d196f485c321bbde3348677c63fc6a`, as recorded in
  `results/qwen38_nf4_public_verification_01.json`.
- `smoke_origin.json` is local-only and not a runtime file.
- The public record's GPU-verified revision is `e7644d6…`, with
  `runtime_files_unchanged: true`. The release must pin `586bb022…`, or
  another revision whose six runtime hashes are identical, and say which.

## 1. Accurate scientific statements (required on card, guide and GitHub closeout)

### 1.1 Status line

The status line must be stated up front, not only in a later section. The
wording should be close to this:

> Experimental style-control demonstration. In the registered exploratory
> test, no strength passed all acceptance gates
> (outcome `exploratory_null_through_b16`). It is not a validated
> controller.

### 1.2 Table columns

Each tested strength gets a table row with these columns:
- SARC;
- coherence;
- impoliteness;
- answers with coherence below 50;
- the gate result.

The values must come from the canonical result.

| b | SARC | Coherence | Impoliteness | Coherence < 50 | Gates |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.00 | 84.28 | 1.9 | 0/32 | — |
| prompted (reference) | 55.56 | 76.00 | 52.1 | 0/32 | — |
| 2 | 0.09 | 84.16 | 2.2 | 0/32 | fail (trait) |
| 4 | 4.97 | 83.44 | 6.8 | 0/32 | fail (trait) |
| 8 | 88.50 | 58.94 | 70.7 | 6/32 | **fail: relative coherence, 58.94 vs ≥ 64.28** |
| 16 | 95.16 | 34.59 | 81.5 | 30/32 | fail: both coherence gates |
| random16 | 0.00 | 84.28 | 2.1 | 0/32 | descriptive only |

### 1.3 The b = 8 default

The default must be labelled an *experimental demonstration default chosen
by the owner*. Its row must state plainly that it failed the registered
relative-coherence gate. It must **not** be called "balanced",
"recommended for production" or "validated".

The current draft's label "Strong; proposed experimental demonstration
default" is acceptable only if the gate failure appears in the same row or
the adjacent sentence.

### 1.4 Low strengths

b = 2 should read "no measurable judged sarcasm (0.09)", not "Low". The draft
row reads only "Low", which suggests a mild effect that was not observed.

### 1.5 The random16 control

The card must state all of the following:
- it is **setting-matched**: same coefficient rule, same per-site ‖Δ‖, same
  K⁺ and D̄;
- it is **not realised-shift-matched**: the decode gate was 0.36–0.50 vs
  0.68–0.87 for b16, and the realised decode shift was smaller;
- it exists only at b = 16, so b = 8 and the lower strengths are not
  direction-tested;
- it uses one random seed set.

The current draft covers most of this. It must add "one random seed set".

### 1.6 Coupling with impoliteness

Impoliteness rises with sarcasm: 70.7 at b8 and 81.5 at b16. The release
must not claim sarcasm is separable from rudeness. It must also not imply
that coherence loss or impoliteness is *necessary* for sarcasm, since that
was not tested either way.

### 1.7 Scope limits

Each item must appear:
- every answer hit the 160-token cap, so behaviour beyond 160 tokens is
  untested;
- greedy decoding with repetition penalty 1.05, batch size 1, native KV
  cache;
- a single-turn English user question with **no system prompt** in the
  steered arms. System prompts, multi-turn chat, sampling and other
  languages are untested;
- 32 fresh, neutral, object-heavy questions, with no independent final
  set;
- NF4 on one GPU stack only;
- scores from an AI judge, with the coherence rubric caveat;
- the bank pooled 5 instruction pairs, and the direction reflects their
  whole wording contrast, not an isolated sarcasm feature.

### 1.8 Untested values

The card must say that finite values in the interior (0, 16) other than 2, 4
and 8 are untested. Values outside [0, 16] are refused.

### 1.9 Old null

The card must state that the old `development_null` is unchanged, and must
make no claim that dose caused it.

### 1.10 Safety and use

The card must warn that higher strengths produce mocking or demeaning text.
The package is for research and demonstration, not deployment to users.

### 1.11 Licence and attribution

The release must carry:
- the base model's licence terms;
- the GCAD port NOTICE and LICENSE;
- a provenance statement that the bank was derived from the base model's own
  generations.

### 1.12 Guide check (this review)

The current guide's numbers check against the evidence:
- the decode ratios 0.13–0.22, 0.97–1.37 and 1.62–2.41 match review 06;
- the 5.34-point miss is correct;
- the b = 8 coefficient examples 2.92, 8 and 13.08 are correct;
- the gate-lowering statement matches review 05.

The equations match the frozen port. No correction is needed beyond adding
the impoliteness column and the scope items in §1.7 where they are missing.

## 2. Portable runtime criteria

**R1. Frozen arithmetic.** The runtime must compute exactly the frozen
`qwen38_gcad_port.Steerer` operations, in the same order and the same dtypes:
- demand from the native post-norm, post-RoPE queries and the bank K⁺;
- σ(1.5(d − D̄));
- c = 2bγ;
- r = fl_m(c·Δ), then a' = fl_m(a + r).

It may be a copied file with a documented transform, an import, or a
faithful re-implementation. In every case it must be proven by R2 and R6.

**R2. CPU parity test.** On a tiny synthetic model with the same module
structure, the portable runtime and the frozen `Steerer` must give
`torch.equal` outputs:
- at b ∈ {0, 0.5, 2, 8, 16};
- for both prefill and decode with a cache;
- with hook cleanup verified.

**R3. Input validation.** The runtime must refuse each of the following:
- a strength that is a bool, non-finite, negative or greater than 16;
- a strength change while generation is in progress;
- a bank whose sha256 is not `4fd9f86b…`, or with wrong shapes or dtypes,
  or non-finite values;
- a model whose geometry does not match: hybrid layer types with full
  attention at those six sites, 24 query heads, 4 KV heads, head width 256
  and the hidden size;
- a base whose six runtime file hashes differ from the pinned set. Any
  override must be labelled unsupported and untested;
- batch size > 1, or an attention mask containing padding. These were
  untested, and the gate would run on pad positions.

**R4. Hook hygiene.**
- Hooks are attached only to the six `self_attn` modules.
- The model's hook signature is restored after normal exit and after an
  exception.
- Nesting or concurrent use is refused.
- b = 0 must register either nothing or an exact no-op, and the parity check
  in R6 must hold.

**R5. Documented interface.**
- The interface must offer strength presets: off = 0, demonstration = 8 and
  extreme = 16.
- A continuous float strength may be offered, labelled untested between the
  tested points.
- The card's example code must be the code exercised by the reload test.

**R6. GPU release smoke (mechanics only; no judging, no scoring).** This
runs on the pinned public base and the staged bank, with the GPU0 UUID
recorded. Each check below must pass:
1. **Off parity.** At b = 0, the logits are `torch.equal` to an unhooked
   forward, and the generated tokens are identical on at least 2 probes.
2. **Frozen equivalence.** At b = 8 and b = 16, the portable runtime's logits
   and greedy tokens are `torch.equal` to the frozen `Steerer` on the same
   probes.
3. **Study reproduction.** Regenerate a small fixed set of saved follow-up
   rows from their frozen `input_ids`, with the same generation config, cap
   160 and batch 1. For example: F01 baseline, b8 and b16, plus one other
   question at b8. The generated IDs must be token-identical to the saved
   rows, on the same GPU and pinned package environment.

   If any row diverges, the release stops and the cause is investigated. It
   must not be relabelled after the fact. The card notes that other hardware
   or library versions may diverge.
4. **Cleanup.** After exit, a forward is `torch.equal` to ordinary.
5. **Telemetry spot check (optional).** Per-token requested and actual
   norms for one probe, against the frozen recording steerer at
   rtol 1e-5.

Smoke outputs are not described as an evaluation, and they are not shown as
quality evidence.

**R7. Environment.** Pin the package versions that the follow-up run's
manifest recorded (`packages`): torch, transformers, bitsandbytes,
accelerate and safetensors. Any deviation needs its own R6 pass, recorded.

## 3. Payload, provenance and publication criteria

**P1. Exact allowlist.** The payload contains only these items:
- the bank;
- the runtime package and an example;
- the card README;
- a METHODS or guide file;
- sanitized aggregate results and metrics in CSV;
- the licence and NOTICE files;
- the requirements and lock files;
- the release manifest.

It must contain no study rows, raw judge replies, activations, random
tensors, `outputs/` paths or symlinks.

**P2. Release manifest.** The manifest binds:
- the controls `repo_id` and the base `repo_id`, revision and six runtime
  file hashes;
- the bank sha256;
- the runtime and port source hashes, and the builder sha256;
- the canonical result sha256 `61362b38…`, the follow-up registration
  `3b10bec5…` and review 06's sha256;
- a per-file sha256 for every payload file.

The required HF commit message carries the manifest sha256, following the
existing release pattern.

**P3. Sanitization and privacy.**
- An automated scan finds no local absolute paths, usernames or email
  addresses.
- It also finds no judge session IDs, cost fields or CLI argv.
- Public results are derived from the canonical result by a deterministic
  transform. The source sha256 is recorded.

**P4. Numbers derived from the result.** The card's numeric table is
generated from, or asserted against, the canonical result by a test. It is
not typed by hand.

**P5. Download and reload.** After upload, the following must hold:
- An anonymous `snapshot_download` of the controls repo at the
  published commit, and of the base at `586bb022…`, succeeds.
- Every downloaded file's sha256 equals the manifest.
- The manifest's own sha256 equals the commit message.
- A fresh process then runs R6.1, R6.2 and R6.4 from the *downloaded* copy:
  - with `python -I`, cwd outside the repository and the repository not
    importable;
  - offline after the download.
- Results equal the local staged smoke.
- The record states that these checks prove file consistency for the
  pinned revision, not publisher identity.

**P6. Tests.**
- **CPU tests:** R2 parity; R3 refusal paths; the allowlist, manifest and
  sanitize logic; card-number consistency (P4); and the preflight, which
  rejects extra, missing or modified files.
- **The GPU smoke is a separate, recorded stage.**

**P7. GitHub closeout.** Docs must link the release and its pinned commit
and the review chain:
- follow-up reviews 01–06;
- release reviews 01–02.

They must keep the original and follow-up outcomes exactly as recorded. No
frozen research file is edited.

## 4. Corrections requested in the HF card draft now

1. In the status line at the top, add the registered-null wording from
   §1.1.
2. Table:
   - add the impoliteness and coherence-below-50 columns;
   - change b = 2 to "no measurable judged sarcasm";
   - add the gate failure to the b = 8 row;
   - add a random16 row.
3. Add "one random seed set". Add the scope items from §1.7: no system
   prompt, single turn, greedy, batch 1, the 160 cap, English, and the
   5-pair bank.
4. Add the safety and use warning and the licence and attribution section.
5. Replace "Ordinary-generation parity verified in mechanics" with a
   reference to the release smoke R6.1, once it exists.

## Next

When the worker's runtime, stager and tests are ready, and a staged payload
exists, send the hashes. Release review 02 will check them against §2 and
§3 and against the R6 smoke record. It is the source and payload signoff
before the PM publishes. After publication, P5 is reviewed in release
review 03.
