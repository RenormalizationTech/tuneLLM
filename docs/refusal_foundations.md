# Theoretical background for refusal steering

Activation steering intervenes on an internal state during inference while
preserving the stored model weights. Its effect depends on the intervention
operator, direction, placement and token scope, together with the subsequent
nonlinear and autoregressive computation.

The [shared notation](learning_foundations.md) defines the residual states and
linear algebra used below. Intervention sites and measured outcomes are specified
in the [implementation](activation_steering_methods.md) and
[7B results](refusal_7b_transfer_execution.md).

## Estimating a direction from contrasting examples

At one fixed model, layer and token-selection rule, estimate

\[
v=\frac1{N_+}\sum_{i=1}^{N_+}h_i^+
-\frac1{N_-}\sum_{i=1}^{N_-}h_i^-,
\qquad
u=\frac{v}{\|v\|_2}.
\]

The plus/minus labels identify extraction conditions; their orientation
determines the sign. A zero or numerically tiny \(v\) cannot supply a reliable
unit direction. Paired contrasts can instead average
\(h_i^+-h_i^-\) over matched questions.

This is a difference of means. It may encode topic, prompt style, length and
other differences between the samples. Under a shared spherical Gaussian
model, the mean difference points along the linear class discriminant; with
general shared covariance \(\Sigma\), the discriminant direction is instead
proportional to \(\Sigma^{-1}v\), when invertible. This statistical illustration
explains why an unwhitened contrast is not automatically an optimal classifier
or a pure behavioral feature.

[Arditi et al., sections 2.3–2.4](https://arxiv.org/html/2406.11717v3#S2)
uses mean differences and interventions for refusal. Related
[Contrastive Activation Addition](https://arxiv.org/html/2312.06681v4#S3)
uses contrastive activation vectors for other behaviors.
Their empirical findings motivate tests on specified models; they do not
establish one universal refusal direction.

## Fixed addition and coefficient units { #fixed-addition-and-the-units-of-strength }

A fixed edit is

\[
h'=h+\alpha v=h+\beta u,\qquad
\beta=\alpha\|v\|_2.
\]

The coefficient \(\alpha\) multiplies a **raw** vector; \(\beta\) measures
displacement along a **unit** vector. Equal numerical coefficients do not
imply equal edits across vector banks, layers or models. The perturbation norm
is \(|\alpha|\|v\|_2\).

For the recorded 4B vector, \(\|v\|\approx61.19\); alpha \(-0.5\)
therefore corresponds to unit-axis displacement about \(-30.60\).

If the contrast points toward the positive extraction condition, negative
alpha moves oppositely. That sign convention says where the activation moves,
not what a response is guaranteed to do. The downstream computation is nonlinear.

A useful descriptive diagnostic is

\[
\rho(h)=\frac{\|\alpha v\|_2}{\|h\|_2},
\]

for nonzero \(h\). It compares edit size to activation size.
It is not the percentage of refusal removed or a predictor of quality.

## Projection removal { #projection-removal-is-a-different-operator }

For unit \(u\), directional ablation uses

\[
P=I-uu^\top,\qquad h'=Ph=h-u(u^\top h).
\]

Because \(u^\top u=1\),

\[
P^2=P,\quad P^\top=P,\quad u^\top Ph=0,\quad
\|Ph-h\|_2=|u^\top h|.
\]

These are exact-arithmetic identities. Projection removes whatever signed
component is present. Fixed subtraction translates every point by the same
amount, even when its original projection is negative.

**Geometric comparison.** Let \(v=(2,0)\), \(u=(1,0)\), and use a fixed edit
\(h-0.5v\).

| Input \(h\) | Fixed subtraction | Projection removal |
| --- | --- | --- |
| \((3,4)\) | \((2,4)\) | \((0,4)\) |
| \((-3,4)\) | \((-4,4)\) | \((0,4)\) |

For a negative initial projection, fixed subtraction increases its magnitude,
whereas ablation sets the coordinate to zero.

## Intervention placement { #placement-determines-what-is-removed }

In a pre-normalized block, write the attention and MLP writers as \(a,m\).
Projecting only the block input does not prevent later writers from adding
a component in direction \(u\). By contrast,

\[
h_0=Ph,\qquad r=h_0+P\,a(h_0),\qquad
h_{\rm out}=r+P\,m(r)
\]

keeps each residual sum orthogonal to \(u\), in exact arithmetic.
This explains why input, attention-writer and MLP-writer interventions
are distinct from one output edit.

The [4B implementation](activation_steering_methods.md) used these three
sites in every block for projection removal, versus one block-output hook
for fixed subtraction. The two arms therefore changed coverage as well as
operator. FP32 projection followed by BF16 casts gave only approximate
orthogonality. Inference-time hooks are also different from editing stored
weights to eliminate a writer direction.

At a single linear writer, the exact-arithmetic identity
\(P(Wx+b)=(PW)x+Pb\) explains when projecting an output can be represented
by editing that writer's weights **and bias**. Equivalence of a whole network
also requires matching every relevant site, including incoming residuals,
and the same arithmetic. A hook at another side of a normalization or nonlinear
operation is not covered by this identity.

## Local linearization and its limits { #why-a-linear-edit-can-have-nonlinear-consequences }

For a downstream logit map \(z=f(h)\) and a sufficiently small edit \(\delta h\),

\[
f(h+\delta h)\approx f(h)+J_f(h)\delta h.
\]

Different output logits have different sensitivities to the same direction.
A state shift can change style, refusal and factual content together.
If the chosen token changes, future context also changes, and this local
approximation no longer describes the entire response trajectory.

A KV cache stores computed states for reuse. An edit can change cached states,
later states, or both, depending on hook placement. Cache reuse is not proof
that steering affects only the visible current token. Token scope must specify
prefill positions and cached decoding separately.

For an activation-edit intervention, a behavioral contrast estimates the
effect of **that operator under that protocol**. It does not prove the vector
is the unique representation of refusal. Prompt confounding and output
judgments remain separate questions.

## Controls and recorded outcomes { #controls-and-what-our-results-establish }

A zero-strength edit should reproduce the chosen baseline path within its
registered tolerance; removing hooks
should restore it. Nonzero edits should change the intended internal state.
These are mechanical checks. Matched prompt, random-direction and held-out
comparisons address behavior and specificity.

The [7B experiment](refusal_7b_transfer_execution.md) found a dose/usefulness
tradeoff. A weaker selected-case diagnostic rescued two benign answers, while
the original fixed-workflow qualification remained null.
That diagnostic also contained one repetitive output, and refusal reduction
occurred on unsafe contrasts too; it was not a benign-specific control.
Refusal rate and useful-answer rate are separate outcomes; improving one is
not algebraic evidence of improving the other.

The [source references](references.md#activation-steering) and
[implementation explanation](activation_steering_methods.md) identify how our
data, layer outputs and scopes differ from the cited papers.

## Magnitude matching across vector banks

For raw banks with norms 10 and 40, a shared coefficient of \(0.5\) produces
displacement norms 5 and 20. Coefficient comparisons therefore require the
vector normalization and effective perturbation to be specified.
Matching geometric magnitude controls one aspect of the comparison;
behavioral specificity and response quality remain empirical questions.
