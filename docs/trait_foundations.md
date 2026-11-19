# Theoretical background for trait steering

Trait steering modifies inference-time activations to influence response style.
Activation projections and gate values specify internal control signals;
trait expression, coherence and semantic preservation are measured at the
output level.

The formulation builds on [attention and residual states](learning_foundations.md#attention-and-the-residual-stream)
and [steering geometry](refusal_foundations.md).
The [trait results](steering_reference.md) and
[GCAD implementation discussion](gcad_and_combined_control.md) connect these
operators to the measured experiments.

## A direction and a target coordinate

Let \(h_i^+\) and \(h_i^-\) be styled and plain activations for matched inputs.
A projection-targeting bank can estimate

\[
u=\frac{\mathbb E_i(h_i^+-h_i^-)}
{\|\mathbb E_i(h_i^+-h_i^-)\|_2},\qquad
T=\mathbb E_i(u^\top h_i^+).
\]

The unit direction defines an axis; \(T\) defines a target signed coordinate.
In the pinned SAPS implementation, extraction uses last-prompt-token residual
states. A styled target is not a calibrated probability of sarcasm.

A fixed-projection baseline uses an unsteered reference \(h_0\):

\[
h'=h+\alpha(T-u^\top h_0)u.
\]

The scalar is computed once from that reference. It is then a fixed
displacement, even when later token states differ.

## SAPS as a local feedback rule

The corresponding adaptive rule uses the current state:

\[
h'=h+\alpha(T-u^\top h)u.
\]

Define the current coordinate \(p=u^\top h\). Since \(u^\top u=1\),

\[
p'=(1-\alpha)p+\alpha T,\qquad
p'-T=(1-\alpha)(p-T).
\]

At \(\alpha=1\), this edit places the measured coordinate exactly at \(T\).
That statement assumes an exactly unit direction and exact arithmetic; the
implementation's epsilon normalization and dtype casts make it approximate.
For \(0<\alpha<1\), it moves partway there; above 1 it overshoots.
If the **same isolated scalar edit** were applied repeatedly without other
computation, its error contracts for \(0<\alpha<2\).
A decoder adds nonlinear layers, new tokens and a changing context between
edits, so this is not a stability theorem for generation.

**Numerical illustration.** For \(p=2,T=5\), alpha 1 adds \(3u\), reaching coordinate 5.
Alpha 1.4 adds \(4.2u\), reaching 6.2. Bigger coefficients can overshoot a target
even though the direction is unchanged.

[Naganna et al., SAPS](https://aclanthology.org/2026.findings-emnlp.1594/)
introduces adaptive projection steering. The
[pinned author implementation](https://github.com/SharathHN/saps-steering/blob/c56fb28ac0d14c8fd70035e035c5452e3c6c6ccc/src/steering.py)
used here measures the last forwarded token and broadcasts its shift across
forwarded positions. Therefore, the equation controls that measured
last-token coordinate; it does not independently set every prefill token to
\(T\). During cached decoding there is normally one new forwarded position.
For a non-last prefill position \(j\), the same broadcast gives
\(p'_j=p_j+\alpha(T-p_{\rm last})\), not independent targeting of \(p_j\).
A post-block edit can affect later layers' computed keys and values; it does
not retroactively alter KV entries already computed before the hook.

## GCAD attention-delta extraction { #gcad-starts-from-the-attention-writer }

GCAD stands for Gated Cropped Attention-Delta. At one layer, let
\(A^{(h)}_{tj}\) be head \(h\)'s ordinary causal attention weight from query
position \(t\) to source position \(j\). With system-prompt positions
\(\mathcal S\), define

\[
a^{\rm sys}_{\ell,t}
=W_{O,\ell}\operatorname{concat}_{h=1}^{H_q}
\left(\sum_{j\in\mathcal S}
A^{(\ell,h)}_{tj}v^{(\ell,g(h))}_j\right).
\]

Here \(H_q\) is the number of query heads, \(v_j\) is a value vector,
\(W_O\) the output projection, and \(g(h)\) maps a query head to its shared
KV head under grouped-query attention. This expression isolates a linear
contribution to the attention output; any output-projection bias is separate.
For the pinned contiguous head layout, zero-based query head \(h\) maps to
\(g(h)=\lfloor h/(H_q/H_{\rm kv})\rfloor\), assuming
\(H_q\) is a multiple of \(H_{\rm kv}\).
Values include any value-projection bias; cropping weights that contribution
by the attention mass on the selected system positions.

The attention weights were normalized over the **full allowed prefix**.
Cropping to \(\mathcal S\) does not renormalize them.
For example, system weights \(0.2,0.1\) and values \(2,4\) contribute
\(0.2(2)+0.1(4)=0.8\) in a scalar illustration.
Renormalizing them to sum to 1 would produce \(2.667\), a different quantity.

The bank's raw contrast is

\[
\Delta_\ell
=\mathbb E_+\!\left[\frac1{T_i}\sum_{t\in R_i}
a^{\rm sys}_{\ell,t}\right]
-\mathbb E_-\!\left[\frac1{T_i}\sum_{t\in R_i}
a^{\rm sys}_{\ell,t}\right],
\qquad T_i=|R_i|.
\]

The response-position mean is taken within each sample before averaging
samples. Cropping is an **extraction** operation. It does not discard
user/response attention sources during the ordinary steered forward pass.
[Kang et al., section 4](https://arxiv.org/html/2605.10664v3#S4)
motivates this attention-level contrast.
The [pinned extractor](https://github.com/NihiI-obstat/Gated-Cropped-Attention-Delta/blob/6c879f4d12e2b01274fc96c9afdb0f4f3a26d4de/core/generate_vec.py)
records the concrete pooling and eager-attention path. A different attention
kernel during inference need not reproduce extraction arithmetic exactly.

## GCAD gate { #the-gate-is-a-compatibility-proxy }

A representative demand statistic and injection rule in our pinned runtime are

\[
d_{\ell,t}=\frac1{H_q}\sum_{h=1}^{H_q}
\frac{\widetilde q_{\ell,t,h}^{\top}\bar k^+_{\ell,g(h)}}{\sqrt{d_h}},
\qquad
c_{\ell,t}=2b\,\sigma\!\left(\zeta s(d_{\ell,t}-\bar d_\ell)\right),
\]

\[
a'_{\ell,t}=a_{\ell,t}+c_{\ell,t}\Delta_\ell,\qquad
\sigma(x)=\frac1{1+e^{-x}}.
\]

The current query \(\widetilde q\) is post-RoPE, \(\bar k^+\) is a stored
positive-system key average, \(d_h\) is head width, and \(\bar d_\ell\)
is a stored mean demand. The key/demand pooling weights follow the pinned
extractor: system-token counts for keys and positive response-token counts
for demand. Those pooling rules need not equal the sample weighting for
\(\Delta_\ell\).
The default sign is \(\zeta=+1\). The code's inverted option uses \(\zeta=-1\),
reversing which demands get a larger gate; it does not negate \(\Delta_\ell\).

For nonnegative base strength \(b\), positive sharpness \(s\), and default
\(\zeta=+1\), the
coefficient is between 0 and \(2b\), with value \(b\) at mean demand.
The stored keys already include positional rotation; averaging them gives
a heuristic compatibility statistic, not an exact attention probability,
confidence or detector of whether a token “needs” the trait.
Position changes can also change this compatibility score. A systematic
position-dependent drift in a particular steered run is a hypothesis to test,
not a result established by our trait evaluations.

**Numerical illustration.** For \(b=2,s=1.5\), mean demand gives \(c=2\).
A demand difference \(+1\) gives \(c\approx3.270\);
a difference \(-1\) gives \(c\approx0.730\).
The algebraic requested displacement has norm \(c\|\Delta_\ell\|_2\).
Model-dtype rounding can make the realised output difference differ from this
value. Equal \(b\) values do not match the strength of banks with different delta
norms. The separately measured requested and actual norms are defined below.

The precise gate, GQA mapping and insertion site are implementation-specific;
see the
[pinned GCAD author code](https://github.com/NihiI-obstat/Gated-Cropped-Attention-Delta/blob/6c879f4d12e2b01274fc96c9afdb0f4f3a26d4de/core/gated_steerer.py)
and [local mathematics record](gcad_and_combined_control.md).
GCAD injects after the attention output projection, before the subsequent
residual/MLP computation. It is not the same site as post-block residual steering.
The output is already in residual coordinates: adding \(c\Delta\) there
also adds that displacement to the next residual sum. The distinction is
the extraction, gate and location before the MLP, not an unrelated vector space.

### Nominal strength and realised shifts { #nominal-strength-versus-realised-shift }

The user-facing strength \(b\) scales the requested addition for a fixed
incoming state. Across a generation, earlier edits change later queries,
gates, native activations and selected tokens. Doubling \(b\) need not double
the realised shift or the behavioural effect. Strength is not a sarcasm
probability, temperature or percentage of native activation.

In finite precision, distinguish three quantities. With \(\operatorname{fl}_m\)
denoting model-dtype rounding, define

\[
r_{\ell,t}=\operatorname{fl}_m(c_{\ell,t}\Delta_\ell),
\qquad
a'_{\ell,t}=\operatorname{fl}_m(a_{\ell,t}+r_{\ell,t}).
\]

The algebraic norm is \(c\|\Delta\|_2\); the requested operand norm is
\(\|r\|_2\); the actual shift is measured from the rounded outputs in FP32:

\[
\rho^{\rm actual}_{\ell,t}
=\frac{\|\operatorname{FP32}(a'_{\ell,t})-
\operatorname{FP32}(a_{\ell,t})\|_2}
{\|\operatorname{FP32}(a_{\ell,t})\|_2}.
\]

Zero denominators are excluded and counted. Report ratios per token, separating
prefill and decode. A sequence Frobenius norm is not a per-token denominator;
dividing it by the square root of sequence length gives an RMS token norm,
which need not equal the mean token norm. Matching raw delta norms and nominal
strengths also does not match realised shifts along different trajectories.

### Porting GCAD to gated attention { #gcad-on-the-27b-hybrid-model }

The preceding crop assumes an attention writer without a native output gate.
When a base model includes a featurewise gate, extraction must retain it. Let
\(g^{\rm native}_{\ell,t}\) denote that gate, held at its ordinary full-prefix
value while cropping source positions. The system contribution becomes

\[
a^{\rm sys}_{\ell,t}=W_{O,\ell}\left[
g^{\rm native}_{\ell,t}\odot
\operatorname{concat}_{h=1}^{H_q}
\left(\sum_{j\in\mathcal S}
A^{(\ell,h)}_{tj}v^{(\ell,g(h))}_j\right)
\right].
\]

This native featurewise sigmoid gate belongs to the base model. It is distinct
from GCAD's scalar demand gate, which controls the added delta. For fixed
queries, keys, values and native gate, system and complementary crops sum to
the full bias-free writer in exact arithmetic. Cropping never renormalizes
attention weights or recalculates the native gate from a shortened context.

In the [27B NF4 port](../scripts/qwen38_gcad_port.py), query/gate components
are de-interleaved, Q/K use native normalization and partial RoPE, and 24 query
heads share four KV heads of width 256: zero-based \(g(h)=\lfloor h/6\rfloor\).
The output projection is bias-free. Response and matched-pair means accumulate
in FP32; exact saved response IDs define the extraction positions. The fitted
bank is an equal-pair mean of 90 eligible sarcastic-minus-sincere contrasts,
while key and query summaries use their registered token weighting.

Only full-attention blocks **23, 27, 31, 35, 39 and 43** receive direct hooks;
linear-attention blocks remain unhooked. GCAD adds the raw delta after
`o_proj`, with gate scale 1.5 and default sign. Later blocks can still receive
changed activations indirectly. These are model-specific port choices, not
universal GCAD settings. The [27B results](qwen38_sarcasm_results.md) report
their measured effect, and the [usage guide](qwen38_sarcasm_release.md) exposes
only strength as an ordinary user control.

### Why style and answer quality can move together

GCAD edits the same residual computation that supports answer content; it
does not apply a tone-only output filter. A shift can change topic, emphasis
and token selection along with wording. This provides a possible mechanism
for quality costs, without establishing the cause of a particular null result.

Sarcasm can be coherent and helpful. The coherence rubric asks whether an
answer makes sense in the question's context; sarcasm or rudeness alone should
not lower it. It is an AI judgment after generation, not a decoding constraint
or a general factual-correctness test. The historical rubric's hallucination
wording also makes the separation from correctness imperfect. Trait scores,
coherence and cross-trait effects must be measured separately.

## Interactions between combined controls { #why-combining-trait-controls-can-fail }

At one writer, suppose several trait perturbations sum to
\(\delta a=\sum_k c_k\Delta_k\). Then

\[
\|\delta a\|_2^2
=\sum_k c_k^2\|\Delta_k\|_2^2
+2\sum_{i<j}c_ic_j\Delta_i^\top\Delta_j.
\]

Alignment changes total perturbation energy. Even orthogonal deltas can affect
the same output behavior after nonlinear computation. A local description is

\[
\delta s_j\approx\nabla_a s_j^\top\delta a,
\]

for a differentiable score surrogate \(s_j\). Actual AI rubric scores need not
be differentiable; this equation is an explanatory local model, not an
estimated gradient from our experiment.

Hook order can matter for mixed operators. Let the coordinate-setting operator be
\(S_T(h)=h+(T-u^\top h)u\); then

\[
S_T(h+w)=S_T(h)+(I-uu^\top)w,
\]

which differs from \(S_T(h)+w\) whenever \(u^\top w\ne0\).
By contrast, summing fixed contributions at one shared writer is commutative
in exact arithmetic. Later layers, recomputed gates, finite precision and
autoregressive feedback still create interactions.

The [completed joint study](gcad_and_combined_control.md#completed-joint-phase-and-next-stage)
qualified none of five combined candidates. Direction geometry helps formulate
hypotheses; it does not replace matched-task measurements of all target traits,
coherence and meaning. The separate [27B sarcasm studies](qwen38_sarcasm_results.md)
are complete: stronger settings induced sarcasm but none passed every quality gate.

## Interpretation of control signals

A gate value \(c=3\) at base strength \(b=2\) specifies the multiplier of an
attention-output delta. A probabilistic interpretation would require separate
calibration against observed response behavior.

Orthogonality removes the cross-term between two deltas in squared perturbation
norm. It is insufficient to establish behavioral independence after downstream
computation. The [trait results](steering_reference.md) report the observed
coherence costs and cross-effects.
