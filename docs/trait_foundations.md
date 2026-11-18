# Mathematical foundations of trait steering

Trait steering aims to change how an answer is written while retaining its
usefulness and meaning. A style score is an output measurement; an activation
projection or gate is an internal control signal. They are not interchangeable.

Prerequisites: [attention and residual states](learning_foundations.md#attention-and-the-residual-stream)
and [steering geometry](refusal_foundations.md). Then read the
[measured trait references](steering_reference.md) and
[GCAD implementation discussion](gcad_and_combined_control.md).

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

**Worked example.** For \(p=2,T=5\), alpha 1 adds \(3u\), reaching coordinate 5.
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

## GCAD starts from the attention writer

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

## The gate is a compatibility proxy

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

**Worked example.** For \(b=2,s=1.5\), mean demand gives \(c=2\).
A demand difference \(+1\) gives \(c\approx3.270\);
a difference \(-1\) gives \(c\approx0.730\).
The actual displacement is \(c\|\Delta_\ell\|_2\). Equal \(b\) values do not
match the strength of banks with different delta norms.

The precise gate, GQA mapping and insertion site are implementation-specific;
see the
[pinned GCAD author code](https://github.com/NihiI-obstat/Gated-Cropped-Attention-Delta/blob/6c879f4d12e2b01274fc96c9afdb0f4f3a26d4de/core/gated_steerer.py)
and [local mathematics record](gcad_and_combined_control.md).
GCAD injects after the attention output projection, before the subsequent
residual/MLP computation. It is not the same site as post-block residual steering.
The output is already in residual coordinates: adding \(c\Delta\) there
also adds that displacement to the next residual sum. The distinction is
the extraction, gate and location before the MLP, not an unrelated vector space.

## Why combining trait controls can fail

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
coherence and meaning. The separate 27B sarcasm study remains deferred.

## Check your understanding

**Question:** Can a GCAD gate value \(c=3\) be interpreted as a 75% probability
of sarcasm when \(b=2\)?

**Answer:** No. It is a multiplier of a raw attention-output delta, not a
calibrated output probability.

**Question:** If two deltas are orthogonal, are their induced traits independent?

**Answer:** No. Orthogonality removes their cross-term in squared perturbation
norm, but says nothing sufficient about nonlinear downstream behavior.

Continue to [trait results](steering_reference.md) to compare these mechanisms
with what the project actually observed.
