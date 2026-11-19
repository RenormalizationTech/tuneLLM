# Qwen3.8-27B sarcasm follow-up: review 05 (Part A independent check)

Advisor review, 2026-11-19. CPU only. It reads the saved Part A artifacts in
`outputs/qwen38-sarcasm-followup-01/diagnostic/`. The frozen analyzer's
`part_a()` (sha256 `8e13912b…`, unchanged) was run read-only. Its output went
to a scratch file outside the repository, not to the canonical result.
The registered closeout remains the authoritative analysis. No model or
judge call was made, and no source was edited.

Part B was fixed before this review, and nothing here can change it.

## Integrity: verified

**Rows.** 192 rows, with IDs equal to the frozen `inputs.json` replays. Every
row receipt and tensor sha256 matches. There are no failure files.

**Tensor keys per context.**
- S and L (64 each): `block.*`, `attention.*`, `crop.*`, `attention_tokens.*`.
- N (64): `block.*`, `attention.*` only. There is no crop and no per-token
  retention, as specified.

**Fixed tokens.** For all 64 (question, content) groups:
- S and L prompt lengths are equal, and both system spans are 45 tokens;
- the 160 continuation IDs are identical across S, L and N;
- the response masks are identical and all true, with no special tokens.

**Storage.** 1,555,028,999 bytes, under the 2 GiB cap.

**Decomposition across all 768 site-cells.**
- gated error ≤ 0.0106;
- omitted-gate error ≥ 3.54;
- split error ≤ 0.0048.

**Cross-check.** An independent float64 cosine computation without the
analyzer agrees with the analyzer's `cosine_of_mean` to 3 decimals at every
site.

## Results (post-hoc, observational; question-cluster 95% CIs)

**A1. Does the bank direction capture S − L?** It is **positively aligned at
all 6 sites, for both contents**. Every CI lies above 0, and the 3 random
references lie within ±0.03.

| Site | attention I_B | attention I_P | crop I_B | crop I_P | crop S(P) − L(B), bank analogue |
| --- | --- | --- | --- | --- | --- |
| 23 | 0.76 [0.71, 0.79] | 0.88 [0.87, 0.88] | 0.89 | 0.94 | 0.98 |
| 27 | 0.53 [0.47, 0.58] | 0.81 [0.79, 0.81] | 0.72 | 0.91 | 0.98 |
| 31 | 0.38 [0.33, 0.42] | 0.34 [0.31, 0.37] | 0.56 | 0.54 | 0.96 |
| 35 | 0.62 [0.59, 0.63] | 0.58 [0.53, 0.61] | 0.79 | 0.82 | 0.95 |
| 39 | 0.35 [0.31, 0.38] | 0.50 [0.47, 0.52] | 0.58 | 0.81 | 0.92 |
| 43 | 0.43 [0.39, 0.47] | 0.45 [0.40, 0.49] | 0.80 | 0.91 | 0.96 |

The crop cosines are partly alignment by construction:
- the bank was built from cropped attention to the system tokens;
- the S and L system strings are pair 0 of the bank's 5 pairs.

The more informative values are therefore the **full attention-output**
cosines, 0.34–0.88. Even these reflect the whole S/L wording difference, not
pure trait specificity.

**A2. Size of the instruction change, per token, in native units.**

| Site | Per-token ‖S − L‖ / ‖L‖, content B | Content P | Pooled-mean norm / native, B / P | Native ‖L‖ per token |
| --- | --- | --- | --- | --- |
| 23 | 0.10 | 0.22 | 0.035 / 0.12 | 8.9–9.0 |
| 27 | 0.14 | 0.27 | 0.056 / 0.14 | 12.5–13.3 |
| 31 | 0.24 | 0.32 | 0.090 / 0.12 | 11.1–11.4 |
| 35 | 0.21 | 0.31 | 0.072 / 0.11 | 16.9–17.3 |
| 39 | 0.21 | 0.30 | 0.059 / 0.09 | 12.8–13.9 |
| 43 | 0.26 | 0.34 | 0.081 / 0.11 | 14.3–16.0 |

The CIs are tight, with widths of about ±0.01.
- The pooled mean is much smaller than the per-token change. Most of the
  instruction effect varies from token to token, and the consistent (mean)
  component is about 3.5–14% of the native norm.
- For comparison only: the mechanics probes (review 04) gave a b=2 actual
  per-token shift of about 14–37% of native. The conditioning differs:
  Part A holds the continuation fixed, while the probes generate freely.

**A4. Gate under each context.** The sarcastic instruction **lowers** the
GCAD gate relative to the sincere one, at every site and for both contents.
- S − L ranges from −0.015 to −0.075, and every CI lies below 0.
- N gives the highest gate.
- Mean gates are about 0.29–0.41 for content B and about 0.50–0.61 for
  content P.

The gate therefore does not open selectively under the sarcastic instruction.

**A5. Likelihood.** NLL(P | S) − NLL(P | L) = −0.37 [−0.40, −0.35].
NLL(B | S) − NLL(B | L) = +0.15 [+0.14, +0.17]. The instruction effect is
present in the model's predictions, in the expected directions.

**A3. Residual stream (sampled blocks).** In the instruction contrasts, the
norm of the mean grows with depth, from about 0.3–0.4% of the native
residual norm at block 3 to about 9–11% at block 63. It is consistent across
questions: the leave-one-out cosine is 0.81–0.94. The content effect at
fixed S is larger at every depth sampled, from 4% to 28%.

## Interpretation within registered limits

- The data **do not support "wrong direction at these sites"** as a
  description of the old bank. The bank direction is positively and
  substantially aligned with the fixed-token S − L attention change at all
  6 sites.
- The data **do not show that the old strengths were too small relative to
  the mean instruction shift**. The consistent S − L component is about
  4–14% of native per token, and the probes put b=2 at about 14–37% actual
  shift.
- None of this is a causal account of the old null. Part A is observational,
  under fixed tokens, with one instruction pair. Two observations stand out
  as descriptive features worth naming in the closeout:
  - the token-varying structure of the instruction effect, which a constant
    added direction cannot reproduce;
  - the lower gate under S.
- **Part B tests behaviour directly** at b = 2–16, with recorded actual
  shifts and the b=16 random control.
