# PF transfer primitives: CPU implementation only

This is a helper library, not an executable experiment. It has no model loader,
fixture generator, preparation, manifest writer, GPU command or judge endpoint.
No layer, strength grid, training corpus, test task, token cap or budget is
selected here. Root must settle and register those choices before live work.
The proposal is `docs/profanity_method_comparison_proposal.md`; the user has
authorized trying baseline plus fixed targeting, adaptive SAPS and GCAD, but
the scientific/runtime protocol remains pending. All historical runs stay fixed.

## Explicit interfaces

`mean_prompt_direction(plain, styled, target_styled)` accepts FP32 CPU
`[N,L,H]` last-prompt-token raw post-block captures. It preserves the pinned
SAPS sequential accumulator, normalizes mean(style-plain), and computes the
target projection on a separate styled capture pass. It selects no prompts
or instruction draws. Existing `persona_saps_common.capture` is exposed by
identity as `capture_last_prompt`; it requires B1. These states are not response
means or final-layer-normalized states.

`projection_context` requires the bank, expected model/token/data basis, foreign
tensor denylist, block, alpha, method, h0 and audit explicitly. Fixed targeting
computes FP32 CPU dot(h0,u), converts operands separately to Python floats, then
calculates `alpha*(T-projection)`. Model-dtype multiplication applies that fixed
shift throughout generation. Adaptive uses current last-token projection,
including the source epsilon renormalization and scalar dtype behavior. Its
delta broadcasts over the entire B1 prefix during prefill and the cached token
during decode. Alpha0 installs actual hooks, allowing a genuine zero check.
The unchanged `persona_saps_common` operations implement both contexts.

`gcad_context` requires the verified pinned author factory, explicit layer list,
base coefficient, scale and inverse/debug flags. It passes raw PF delta/key/
demand tensors directly to `MultiLayerGatedSteerer`; it does not normalize,
crop inference tokens, substitute a residual vector, or select layers. The
actual hook captures Q, applies RoPE, compares with positive-system keys, and
adds the gated delta at self-attention output at each prefill/decode position.

## Bank and provenance contract

Banks contain exactly `trait`, `source_trait`, `basis`, `training_data_sha256`,
`representation`, `tensors`, `tensor_sha256`. Both traits must be `profanity`.
The independently approved basis must match exactly and contains `model_id`,
`revision`, `model_source`, `tokenizer_sha256`, `rendering_sha256`,
`coordinate_system="native_hf"`. `model_source` must match the loaded model's
config `_name_or_path`. The surrounding future runner must verify the checkpoint
files, tokenizer/rendering fingerprints and actual model load against its
registration; this tensor helper cannot identify loaded weights from shapes.

Projection banks use unit axes `[L,H]` and targets `[L]` in FP32 CPU storage.
GCAD banks use delta `[L,H]`, key `[L,Nkv,H/Nheads]`, demand `[L]`, also finite
FP32 CPU. All tensor identities must match independently bound hashes. Pass
the downloaded impolite bank's tensor hashes as `forbidden_tensor_hashes` to
reject direct relabelling. Provenance is an audited input contract, not proof
of semantic PF content or author data equivalence. No automatic tag change or
conversion of old impolite/Mistral banks is provided.

## GCAD extraction interfaces and concrete missing assets

Audited root:
`outputs/persona-author-replication-audit-01/gcad`, source revision
`6c879f4d12e2b01274fc96c9afdb0f4f3a26d4de`.

* `core/generate_vec.py:get_prompt_attn_delta_p_and_r` needs rendered
  positive/negative system+user prompts and corresponding saved responses.
  It crops attention contributions to system-key positions, applies `o_proj`,
  then response-pools them; positive-minus-negative means yield the raw delta.
  It needs actual attention weights and therefore eager extraction, not the
  current SDPA-only runtime loader. Future extraction must use a separately
  registered eager load, not weaken that frozen loader's SDPA assertion.
* `scripts/generate_vec_prompt_attn_k.py:compute_k_pos_and_avg_demand` needs
  positive full prompts/responses. It pools post-RoPE system K and response Q
  with the source token weighting, then calculates the mean demand. Its marker
  is the Qwen system-turn end; user-only contrasts without explicit positive
  systems do not establish this required boundary.
* `core/generate_vec.py:get_persona_effective` expects aligned positive/negative
  records with trait/coherence judgments. The source threshold and any adapted
  evaluator/filter must be settled before selecting PF examples. Do not import
  credential-loading pipeline modules merely to reuse the extraction functions;
  a future wrapper can isolate the reviewed definitions with explicit globals.

For the pinned Qwen7, L28/H3584/Nheads28/Nkv4: delta `[28,3584]`, keys
`[28,4,128]`, demand `[28]`. The available author GCAD bank is impolite-only.
There is currently no reconstructed PF GCAD bank. Existing PF PV30 eligible
pairs/18 questions are candidate inputs, not original author PF data; the old
12 response contrasts likewise do not supply these attention/key statistics.
Their use, prompt/system matching and response/token boundaries require an
approved extraction specification. No new fixture or artifact was generated.

Fixed and adaptive can share one PF mean-contrast/unit/target fit, but the
existing response-pooled raw hidden7 direction is not the source SAPS
prompt-state learner. Fresh matching prompt-state captures are needed for
that interpretation. The original TransformerLens/Mistral bank is a different
coordinate system; these native-HF Qwen operations are a declared transfer.

## CPU evidence and remaining release requirements

```sh
.venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_profanity_transfer_common.py
```

Focused tests compare fixed/adaptive math to pinned author hooks on identical
FP32/BF16 tensors and changing states, source sequential fitting, actual tiny
Qwen2 prefill/cache zero/nonzero effects and cleanup, actual pinned GCAD formula
and hooks, provenance/model/token/tensor rejection and explicit parameter
requirements. Tiny random models use no downloaded target weights or GPU.

Before any live comparison: settle and register the extraction inputs/task,
layers/strengths/caps/budgets, common held-out evaluation, model/source/token
fingerprints and judging plan. Checkpoint/runtime and generated-token tracing
helpers can remain read-only imports from the existing stack. Claude login is
currently unavailable; no auth workaround, native invocation or fresh audit is
included here. If all methods fail, retain the canonical reference and disclose
the unreproduced effect and missing-data limitation; no automatic rescue.
