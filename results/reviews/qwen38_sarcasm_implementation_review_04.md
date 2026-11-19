# Qwen3.8-27B sarcasm GCAD: review 04 (synthetic-smoke mechanics signoff)

Advisor review, 2026-11-18, of
`outputs/qwen38-sarcasm-gcad-01/synthetic-smoke/` (written 17:52). The review
was read-only, plus one CPU-only reload of the saved `logits.safetensors` and
`bank.safetensors`. No model, judge or GPU call was made. The plan and reviews
01–03 are unchanged.

## Verdict

**Synthetic mechanics PASS. Signed off against the frozen F4 criteria.**
There is no mechanics reason to withhold elicitation, provided the judge audit
passes. The run used the bound source: the current sha256 values of
`qwen38_sarcasm.py` (`ed2b942d…`), `qwen38_sarcasm_judge.py` (`483c711d…`),
`qwen38_gcad_port.py` (`3d76d814…`) and the config (`0844af59…`) equal those
in review 03. Phase receipt: `1c55497d…c32a`.

## F4 criteria, checked

| # | Criterion | Evidence | Status |
| --- | --- | --- | --- |
| 1 | Ordinary vs b=0: identical tokens, `torch.equal` logits | Both probes have identical 32-token `generated_ids`. On my CPU reload, `torch.equal(logits_0, logits_1)` is True (bf16, shape 1×1×248320). | pass |
| 2 | Post-cleanup: `torch.equal` logits; hook counts restored | `torch.equal(logits_0, logits_3)` is True. The runner raises if the `signature()` of every module's forward and pre-forward hook list differs, both after each `generate` and around the logit block. | pass |
| 3 | b=1 nonzero; finite; per-site ‖Δ‖ > 0 | Max \|Δlogit\| = 2.21875, which I recomputed. The final-norm last position differs (asserted). All 18 bank tensors and 4 logit tensors are finite. ‖Δ‖ per site = 1.23, 2.21, 2.54, 4.94, 2.64, 2.61. The b=1 tokens diverge from ordinary at token 24 (C01) and 26 (C02). | pass |
| 4 | Coverage per site: prefill 1, decode len−1; no hooks elsewhere | Every b=0 and b=1 call at sites 23/27/31/35/39/43 shows `{prefill:1, decode:31, errors:0}` for 32 generated tokens. Each `generate` with a bank asserts that every `*attn*` module in a non-window layer has unchanged hooks during steering. This covers all `linear_attn` modules and the 10 out-of-window `self_attn` modules. | pass |
| 5 | Decomposition, 6 sites × 2 TF forwards | Gated error 0.0066–0.0095 (≤ 0.02). Omitted-gate error 5.00–8.70 (≥ 0.2), with a minimum ratio of 604× the gated error (≥ 10×). Split error 0.0024–0.0032 (≤ 0.01). | pass |
| F5 | Boundaries | Both sides: system 45 tokens, prompt 72, response 17 tokens; 2 prefix checks each. The response IDs on the two sides are identical, as frozen. | pass |
| F6 | Real-load geometry | `load()` runs `validate_loading` and `validate_geometry` on the actual NF4 SDPA load before any phase work. The phase reached completion, so both held. | pass |
| Budget | 6 gen / 2 TF / 4 logit | Matches the plan; the counters are enforced by `reserve()`. | pass |

## Report-only items

- **Cache disclosure.** Cached and full-prefix logits differ by at most
  0.125. This is the registered NF4 limitation, and every compared arm uses
  cached `generate` at batch 1.
- **Memory and time.** Peak allocated memory was 18.0 GB, well within an A6000.
  The phase took 65.7 s, including the model load. Tokens per second cannot be
  separated from the load here (minor; see n1).
- **Synthetic Δ magnitude.** ‖Δ‖ is 1.0–3.2% of the per-site
  attention-output norm (88–156), so even b=1 shifts are small. This
  synthetic Δ comes from identical response bytes, so it is a mechanical
  artifact. It is not evidence about the strength of the real bank.

## Minor notes (no action needed before elicitation)

- **n1.** Record per-stage wall time with the model load separated out, as in
  review 02 m2. Elicitation will give the first real tok/s for the
  runtime-cap forecast.
- **n2.** The hidden-state check compares only the last position. That is
  sufficient here, because the b=1 logits also differ there.

## Remaining gates

1. The judge audit must pass before any judge stream.
2. Elicitation needs PM approval, bound to this result's hash. GPU0 matches
   the smoke runtime.
3. Real-bank mechanics (criteria 1–4, in a fresh process) must pass before
   development.
4. C6 (review 03) must be implemented, CPU-tested and hashed before final
   registration.
