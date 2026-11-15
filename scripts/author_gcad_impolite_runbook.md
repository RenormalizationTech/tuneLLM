# Original-Qwen GCAD R1

CPU implementation only; the worker has not prepared or executed this run.
Root owns registrations and releases. Existing author/audit/PV/SAPS sources and
outputs remain frozen. Fixed output: `outputs/author-gcad-impolite-01`.

## Environments and CPU preparation

GPU commands and CPU preparation use the isolated Python3.11 environment:
`outputs/persona-author-replication-audit-01/envs/gcad/bin/python`, with
torch2.6.0+cu124, Transformers4.52.3, Accelerate1.7.0 and the full package
inventory recorded in `gcad_environment.json`. Native grading/reporting use
the original `.venv/bin/python`; no OpenAI key/SDK is imported. No installation
or download stage exists. Model loading is local-only and remote code disabled.

Root first finalizes the design review and checks the implementation, then:

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_gcad_impolite prepare
.venv/bin/python -B -m scripts.author_gcad_impolite verify
```

Preparation requires the root-supplied canonical `model_manifest.json`, already
verified downloads, the final protocol/review, the existing successful native
audit and pinned binary. It freezes all 20 full sampled ten-question groups
before truncating to first questions, all 60 conversation IDs, exact rendered
bytes and B60 source tensor intent, both grading plans/definitions and vector
shape/dtype/tensor identities. Source closures contain small active code only,
not recursively embedded historical outputs. Model/source hash aliases are
merged, with one traversal each at command entry and exit for live phases.

Initial registration `results/protocols/author-gcad-impolite-01.json` contains
`manifest_sha256`. Root commits that and the exact sources/protocol/review.

## Mechanics, then main (separate releases)

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_gcad_impolite mechanics --execute-gpu
```

Only physical GPU0 is selected (`CUDA_VISIBLE_DEVICES=0`); UUID must match
`GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b`, free>=24000MiB, utilization<=10%.
Resource contention returns a wait without creating a phase. Every other
authorized execution/integrity failure permanently writes `stopped.json`;
there is no generation resume/retry/fallback. Other jobs are never touched.

Five two-question stochastic cap32 batches use the unchanged author generation
function: ordinary, actual GCAD-zero, residual-zero, GCAD-nonzero,
residual-nonzero. Torch seed42 resets for each. Zero uses actual hooks with zero
V, not a bypass. First-prefill last-token logits are observed only in mechanics;
main has no logits observer. Observe only first prefill/decode hidden deltas
while counting every call. Two additional bounded logit forwards (last logits
only) compare baseline/GCAD. Exactly 10 generations, at most 2 logit forwards.
Require zero-token/logit parity, nonzero hidden/logit changes, all-site prefill
and cached-decode coverage, exact tensor-only save/reload and hook cleanup.

After successful exit integrity, root commits mechanics registration with
exact keys:

```json
{
  "manifest_sha256": "SHA256 of manifest.json",
  "mechanics_run_sha256": "SHA256 of mechanics/run.json",
  "mechanics_result_sha256": "SHA256 of mechanics/result.json",
  "vectors_roundtrip_sha256": "SHA256 of mechanics/vectors.pt",
  "integrity_sha256": "SHA256 of mechanics/integrity.json"
}
```

Then root may release:

```sh
outputs/persona-author-replication-audit-01/envs/gcad/bin/python -B -m scripts.author_gcad_impolite main --execute-gpu
.venv/bin/python -B -m scripts.author_gcad_impolite packet
```

Main uses exactly 60 conversations per baseline/residual/GCAD arm, source B64
(one actual batch of 60), cap500, temperature1, cache enabled, and a separate
torch seed42 reset per arm. Original checkpoint top_p0.8/top_k20/repetition
penalty1.05 and other generation defaults are retained. No persona system
message is supplied; the Qwen template still inserts its native helpful system.

Tokenizer PAD is overwritten to EOS151645, exactly as the author loader does.
Generation config retains PAD151643 and EOS[151645,151643]. Actual input IDs,
attention masks and padded continuations are saved; the first generated EOS is
included in the logical suffix, subsequent generation-PAD tokens remain in a
separate raw field. Author decoded answers and retokenized `n_tokens` are
retained unchanged. No old single-EOS/different-PAD validator is used.

Released residual vector index20 is used raw at block19, coefficient2, ALL
positions. Direct author GCAD classes hook attention outputs at zero-based
9..19, base3.5/scale1.5, using stored V/K/average demand. No normalization,
capping, dynamic inference cropping, or regeneration of upstream cropped
attention is performed. The published original method's lazy per-hook device
transfers are left unchanged.
Tensor loading intentionally uses `weights_only=True` rather than the author's
`weights_only=False`: only verified FP32 tensor artifacts are accepted, with
no unrestricted pickle execution. This safer deserialization adaptation does
not normalize, rewrite or change the source tensors. The source pipeline's
single-instruction `ActivationSteererMultiple` and the direct
`ActivationSteerer` used here have an exact hooked-output equivalence CPU test
covering prefill and cached decode.

## Two separate native metric contexts

Root uses the pinned native2.1.323 PATH shim privately, verifying binary
SHA256 `604d29a6869409106b6a1110bdadcc8b72e8654a5a865d23258f938a696e348e`.
Flags/model/effort/tools-off/Git isolation come unchanged from `tone_judge`.

```sh
.venv/bin/python -B -m scripts.author_gcad_impolite judge --metric trait --execute-judgments
.venv/bin/python -B -m scripts.author_gcad_impolite judge --metric coherence --execute-judgments
```

Each stream: 180 blinded items, 22 batches of eight plus one of four (23 total),
no two arms of a source conversation in the same batch. Exact source questions
may repeat because group draws/samples repeat; this dependence is disclosed.
Packets contain only IDs/question/answer, one original verbatim definition and
the structured-output instruction. Neither condition names nor the other
metric's rubric/scores appear. Trait is integer0..100 or explicit REFUSAL
(`score:null, refusal:true`); coherence is always integer0..100. Reasons remain
private. REFUSAL is a valid rating, never a retry trigger.
The verbatim author rubric asks for a number-only answer (or trait REFUSAL);
our outer envelope instead requires structured JSON plus a brief private
reason. This formatting/reason adaptation can affect judgments and prevents
exact evaluator reproduction claims; the underlying original definition is
not edited.

Maximum 46 first-valid batches / 92 starts, including malformed-schema attempts.
Second attempts receive identical payloads and are permitted only for structural
failure. Infrastructure/auth/model/isolation/hash failures or uncertain starts
are fatal. First-valid responses are never re-rated. Root may use `--resume`
only to finish publication/replay of saved responses or genuinely not-yet-started
batches after an interruption without a fatal marker; an accepted response with
a missing receipt is validated then repaired by the unchanged foundation.
No start without a response can be retried. Stray/out-of-budget attempt artifacts
fail. No schema exhausted batch permits later calls. Accepted streams publish
only after exit-integrity checks pass.

## Closure

```sh
.venv/bin/python -B -m scripts.author_gcad_impolite report
```

Writes only the exclusive new aggregate `results/author_gcad_impolite_01.json`
after integrity. Fatal runs remain incomplete even if observed contrasts pass.
Report includes numeric author-style trait means excluding refusals, conservative
fixed-denominator means assigning refusals zero, all coherence ratings, mechanics,
caps/EOS/empty/degeneration/length, source-question coverage and compute/native
costs. The fixed +30 contrasts use exact integer-sum rational comparisons,
not rounded displayed means. Missing numeric means fail, not silently become0.

Limits: Claude integer scores/brief reasons/separate contexts replace author
GPT logprob scores; original-Qwen revision is newly pinned; source impolite is
broader than custom rudeness; batch contrast/dependence remain. First reply
does not test cross-turn cache behavior. No matched random arm, capability,
ten-turn follow-up, R2 or automatic sweep is released by any result.

## Worker CPU evidence

```sh
.venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_author_gcad_impolite.py
.venv/bin/python -B -m scripts.author_gcad_impolite --help
```

Tiny random Qwen2 tests exercise actual author classes in the existing shared
CPU stack; they do not claim numerical identity to a full Qwen7 GPU run in the
separate 4.52.3 environment. Cached tokenizer probes require only existing local
files. No worker preparation, target-model generation, native call or new audit.
