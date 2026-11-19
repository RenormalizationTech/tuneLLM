# Tunable GCAD on the 27B model

GCAD changes activations during inference while leaving the base model weights frozen. A user can change its base strength without training another model. The extraction bank, intervention sites and gate rule define the controller; the strength controls its nominal amplitude.

The registered outcome is `exploratory_null_through_b16`: no strength passed every trait/quality gate. The owner-selected strength 8 is an experimental demonstration default, not a validated balanced setting.

This guide connects the [general trait-steering theory](trait_foundations.md#gcad-starts-from-the-attention-writer), the [27B implementation](../scripts/qwen38_gcad_port.py) and the [completed activation/strength experiment](qwen38_sarcasm_followup_execution.md). A separate [HF card draft](qwen38_sarcasm_hf_card_draft.md) gives the shorter user-facing explanation. That card is a documentation draft; a standalone 27B controls package has not been published.

## What users would tune

The existing research hook takes `base_coef`, denoted \(b\) below. The release-facing control should expose that same nonnegative strength, fixed for a generation, and retain an explicit off setting. It is a steering amplitude, not a probability of sarcasm, a score out of 100 or a temperature.

| Setting | Measured behaviour on fresh32 | Interpretation for a release |
| --- | --- | --- |
| 0 | Mechanics verified exact ordinary-generation parity | Steering off |
| 2 | Sarcasm 0.09; coherence 84.16 | Essentially no judged sarcasm |
| 4 | Sarcasm 4.97; coherence 83.44 | Limited judged sarcasm |
| 8 | Sarcasm 88.50; coherence 58.94 | Experimental demonstration default; fails relative coherence 58.94 < 64.28 |
| 16 | Sarcasm 95.16; coherence 34.59 | Stronger experimental setting with substantial quality loss |

The baseline scores 0.00/84.28; the explicit-instruction reference scores 55.56/76.00. Values between the tested strengths are configurable in the research hook but have not been evaluated. Strength 8 is the owner-chosen experimental demonstration default, not a validated balanced setting. No strength passed every registered acceptance gate.

Only \(b\) needs to be exposed as the ordinary user control. For the measured 27B profile, keep the following fixed:

- the fitted bank of raw deltas, positive keys and demand centres;
- zero-based full-attention sites 23, 27, 31, 35, 39 and 43;
- gate scale 1.5 and addition after each native attention output projection;
- the pinned NF4 base and compatible runtime. Other checkpoints or inference backends are separate compatibility/evaluation questions.

Changing a strength takes effect through the inference hook; changing a config field without that runtime does not implement GCAD. Start a fresh generation when changing strength. Removing the hook later does not reverse effects already propagated into cached states. The experiments apply steering throughout prefill and decode, not only to visible answer tokens.

## Extracting the attention writer

At a selected layer \(\ell\), let \(A^{(\ell,h)}_{tj}\) be the ordinary causal attention weight from query token \(t\) to source token \(j\). It is normalized over the full allowed prefix. Let \(\mathcal S\) be the system-token span, including the rendered chat-template boundaries.

The 27B model has 24 query heads, four KV heads and head width 256. For the pinned contiguous GQA layout,

\[
\pi(h)=\left\lfloor\frac{h}{6}\right\rfloor,
\qquad h=0,\ldots,23.
\]

Unlike an attention block without output gating, this model multiplies the concatenated head output by a native featurewise sigmoid gate. Writing this gate as \(g^{\rm native}_{\ell,t}\), the cropped system contribution is

\[
a^{\rm sys}_{\ell,t}
=W_{O,\ell}\left[
 g^{\rm native}_{\ell,t}\odot
 \operatorname{concat}_{h=0}^{23}
 \left(\sum_{j\in\mathcal S}
 A^{(\ell,h)}_{tj}v^{(\ell,\pi(h))}_{j}\right)
\right].
\]

The output projection is bias-free in the pinned model. Cropping retains a contribution to the full attention output: it does not renormalize system-only weights. For fixed queries, keys, values and native gate, the system and complementary-source contributions sum to the full output in exact arithmetic. The measured reconstruction checks allow the model-dtype rounding differences.

For eligible matched pair \(i\), response positions \(R_i^+\) and \(R_i^-\) come from the exact saved generated IDs. The bank stores

\[
\Delta_\ell=\frac1n\sum_{i=1}^n
\left[
\frac1{|R_i^+|}\sum_{t\in R_i^+}a^{{\rm sys},+}_{\ell,t}
-
\frac1{|R_i^-|}\sum_{t\in R_i^-}a^{{\rm sys},-}_{\ell,t}
\right],
\qquad n=90.
\]

Response means and pair means accumulate in FP32. This is an unnormalized direction in the attention output's residual coordinates. Its magnitude therefore matters. The contrast comes from matched sarcastic/sincere generations; it is not an independently isolated pure sarcasm feature.

The bank also stores positive-system mean keys \(\bar k^+\), weighted by system-token counts, and a demand centre constructed from positive response-token-weighted mean queries and those keys. These use post-normalization, partially position-rotated Q/K from the native model. Delta's equal-pair averaging and the key/query token weighting are deliberately different operations.

## Two different gates

The native output gate above belongs to the base model. GCAD introduces a separate scalar gate at each selected layer and token:

\[
d_{\ell,t}=\frac1{24}\sum_{h=0}^{23}
\frac{\widetilde q_{\ell,t,h}^{\mathsf T}
\bar k^+_{\ell,\pi(h)}}{\sqrt{256}},
\qquad
\gamma_{\ell,t}=\sigma\left(1.5(d_{\ell,t}-\bar d_\ell)\right),
\]

\[
c_{\ell,t}=2b\gamma_{\ell,t},
\qquad
a'_{\ell,t}=a_{\ell,t}+c_{\ell,t}\Delta_\ell.
\]

Here \(\widetilde q\) is the native post-normalization, post-RoPE query. The native output gate is already included in \(a\) and the extracted crop; it must not be confused with \(\gamma\). Injection is after `o_proj`, before the subsequent residual/MLP computation. Linear-attention blocks have no direct GCAD hooks, although later blocks can receive changed activations indirectly.

For nonnegative \(b\), the scalar coefficient lies between 0 and \(2b\), and equals \(b\) at \(d=\bar d\). For example, with \(b=8\), demand differences \(-1,0,+1\) give coefficients approximately \(2.92,8,13.08\). These are multipliers on a bank vector, not percentages of the native activation.

The gate is a query/key compatibility proxy, not a confidence score or a calibrated detector of when sarcasm is appropriate. Position-rotated key averages and changed trajectories both affect it. In fixed-token replay, the sarcastic system actually lowered this gate relative to the literal system at all six sites; the method does not assume that an explicit sarcasm instruction must open the gate further.

## Nominal strength versus realised shift

With a frozen incoming state, increasing \(b\) scales the algebraic requested addition linearly. Across a complete generation, changed earlier activations alter later queries, gates, native output norms and selected tokens. Doubling \(b\) therefore does not guarantee twice the realised shift or twice the behavioural effect.

In exact arithmetic the requested norm is \(c\|\Delta\|_2\). The NF4/BF16 runtime distinguishes that algebraic value from the rounded addition operand and the measured output difference. With \(\operatorname{fl}_m\) denoting model-dtype rounding,

\[
r_{\ell,t}=\operatorname{fl}_m(c_{\ell,t}\Delta_\ell),
\qquad
a'_{\ell,t}=\operatorname{fl}_m(a_{\ell,t}+r_{\ell,t}),
\]

\[
\rho^{\rm actual}_{\ell,t}
=\frac{\|\operatorname{FP32}(a'_{\ell,t})-
\operatorname{FP32}(a_{\ell,t})\|_2}
{\|\operatorname{FP32}(a_{\ell,t})\|_2}.
\]

Zero native denominators are excluded and counted. Requested operand norms, actual norms and algebraic values are recorded separately. Ratios are per token, with prefill and decode separate. A sequence Frobenius norm is not a per-token denominator; dividing it by the square root of sequence length gives an RMS token norm, which still need not equal the mean token norm.

The measured decode actual/native ratios span 0.13–0.22 at strength 2, 0.97–1.37 at 8 and 1.62–2.41 at 16, across the six sites. These values describe the generated trajectories, not a universal conversion from strength to style.

The random16 reference uses one seed set and matches the nominal coefficient rule and raw delta norms, but its queries/gates diverge during generation. Its realised decode shifts are smaller. It is a setting-matched reference, not an energy-matched proof of general direction specificity. No random control was run at strength 8.

## Why style and answer quality can move together

The addition enters the same residual computation that supports the answer's content. It is not a separate tone-only output filter. Large shifts can change topic selection, emphasis and the next-token distribution as well as wording. This provides a mechanism by which style control can also affect answer quality; it does not establish a complete causal explanation of a particular score change.

Sarcasm can be coherent, helpful and on-topic. Our coherence rubric asks whether the answer makes sense in the question's context. It should not penalize sarcasm or rudeness by themselves, and it is separate from factual correctness, though the historical rubric's hallucination wording makes that separation imperfect. It is an AI judgment after generation, not a decoding constraint.

The observed tradeoff is sharper than a generic claim that sarcasm lowers coherence: strength 8 produces strong judged sarcasm but misses the relative-coherence rule by 5.34 points; at 16, 30/32 answers have coherence below 50. Impoliteness also increases. The experiment does not show that incoherence or impoliteness is necessary for sarcasm, or that the old low-strength null was caused solely by dose.

The full [HF card draft](qwen38_sarcasm_hf_card_draft.md#adjusting-sarcasm-strength) includes per-arm impoliteness, counts below coherence 50 and gate results. All fresh answers were capped at 160 tokens. The steered arms used single-turn English questions, no system prompt, greedy decoding with repetition penalty 1.05, batch one and native KV caching. Other languages, added system prompts, multi-turn chats, sampling and longer answers are untested. The profile is exploratory, NF4-only and based on 32 questions on one GPU/software stack, without an independent final set. See the [canonical result](../results/qwen38_sarcasm_followup_01.json), [independent review](../results/reviews/qwen38_sarcasm_followup_review_06.md) and [plot](assets/qwen38_sarcasm_followup.svg) for the complete evidence.

The method derives from the pinned GCAD implementation; the [port notice](../scripts/qwen38_gcad_port_NOTICE) and [license](../scripts/qwen3_gcad_port_LICENSE) retain attribution. The [general GCAD discussion](gcad_and_combined_control.md) covers other models and combined controls; numeric strengths do not transfer directly between banks or checkpoints.
