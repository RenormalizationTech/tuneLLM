# Mathematical foundations for tuning LLMs

Tuning becomes easier to reason about when we separate three operations:
changing the stored weights, learning a weight update, and changing activations
during a forward pass. Quantization does the first, LoRA/QLoRA the second,
and the steering methods here the third. This guide supplies the notation
needed to understand all four research sections.

Start with vectors, matrix multiplication, derivatives and conditional
probability. The examples below are teaching calculations, not new model
experiments. Each section's **Mathematical foundations** page continues from
here; its implementation and results pages explain what this project tested.

| Section | First principles | Practical question |
| --- | --- | --- |
| [Quantization](quantization_foundations.md) | Discrete approximations and propagated error | What changes when weights use fewer bits? |
| [LoRA and QLoRA](qlora_foundations.md) | Low-rank updates and gradients | What is learned, and where does the memory go? |
| [Refusal steering](refusal_foundations.md) | Vector geometry and interventions | How do subtraction and projection differ? |
| [Trait steering](trait_foundations.md) | Attention, gates and feedback | Why can stronger style reduce quality? |

## Shapes and notation

We use column vectors. A linear layer has weight matrix
\(W\in\mathbb R^{d_{\rm out}\times d_{\rm in}}\), input
\(x\in\mathbb R^{d_{\rm in}}\), and output \(Wx\).
Actual frameworks often store a batch as rows and compute \(XW^\top\);
this is the same map in a different layout.

| Symbol | Meaning |
| --- | --- |
| \(\theta\), \(\phi\) | All model parameters, or the subset being optimized |
| \(h_{\ell,t}\) | Hidden activation at layer \(\ell\), token position \(t\) |
| \(v\), \(u=v/\|v\|_2\) | A raw direction and its unit-length version |
| \(\nabla_\phi L\) | Derivatives of loss with respect to trainable parameters |
| \(\mathbb E\), \(\|\cdot\|_2\), \(\|\cdot\|_F\) | Expectation, Euclidean vector norm, Frobenius matrix norm |

Notation is local to each equation: Adam's second moment \(v_t\) and attention
values \(v_j\) are different objects from the steering direction \(v\).

For \(h=(3,4)\), \(\|h\|_2=5\). For unit \(u=(1,0)\),
\(u^\top h=3\) is a **signed coordinate**, not a probability.
Cosine similarity instead divides the dot product by both norms and lies
between \(-1\) and \(1\) for nonzero vectors.

## From logits to a generated sequence

A decoder assigns a distribution to the next token. If its logits are
\(z\in\mathbb R^{|\mathcal V|}\), where \(\mathcal V\) is the vocabulary,

\[
p_\theta(y_t=k\mid x,y_{<t})
=\frac{\exp z_k}{\sum_{j\in\mathcal V}\exp z_j},
\qquad
p_\theta(y\mid x)=\prod_{t=1}^{T}p_\theta(y_t\mid x,y_{<t}).
\]

The softmax converts unrestricted scores into probabilities. The sequence
product says that earlier generated tokens become context for later tokens.
Greedy decoding chooses the largest logit at each step; it does not generally
find the most probable complete sequence. Temperature changes sampling
probabilities at inference, not the weights learned during training.

An early output change can therefore change the rest of a response. Equal
first tokens do not prove equal logits or equal later outputs. These
autoregressive ideas underlie modern language-model training; see
[Radford et al., Language Models are Unsupervised Multitask Learners, section 2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

## What a supervised loss actually rewards

For example \(i\), let \(n_i\) be the number of supervised answer tokens.
Teacher forcing supplies the reference prefix \(y_{i,<t}\):

\[
\ell_i=-\frac{1}{n_i}\sum_{t=1}^{n_i}
\log p_\theta(y_{i,t}\mid x_i,y_{i,<t}).
\]

The prompt is context; the supervised targets can exclude prompt and padding
positions. Masking their direct loss does **not** stop gradients flowing through
the computations that use prompt information. Logits at one position predict
the next token, so a one-position shift error trains a different objective.

Two common reductions are

\[
L_{\rm example}=\frac1N\sum_i\ell_i,\qquad
L_{\rm token}=\frac{\sum_i n_i\ell_i}{\sum_i n_i}.
\]

If one answer has 10 tokens and another 100, the example mean gives each
answer half the weight; the token mean gives the second ten times the weight.
Our 27B adapter optimization used equal-example means; development checkpoint
selection used token-weighted answer/EOS NLL. The
[training records](../results/qwen38_qasper_training_01.json) preserve those choices.

For one target token \(k\), differentiating softmax cross-entropy gives

\[
\frac{\partial(-\log p_k)}{\partial z_j}
=p_j-\mathbf1[j=k].
\]

Gradient descent increases the target's relative logit when its probability
is low. This rewards the reference continuation, which need not be the only
good answer. A probability of \(0.5\) gives loss \(0.693\) nats; \(0.25\)
gives \(1.386\). Perplexity is \(\exp(L_{\rm token})\), meaningful for a
specified tokenizer, target mask and data distribution.

## Optimization and memory

Plain gradient descent uses
\(\phi_{t+1}=\phi_t-\eta_t\nabla_\phi L_t\).
AdamW instead combines gradient moments with decoupled weight decay:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]
\[
\phi_{t+1}=(1-\eta_t\lambda)\phi_t
-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon},
\qquad
\widehat m_t=\frac{m_t}{1-\beta_1^t},\quad
\widehat v_t=\frac{v_t}{1-\beta_2^t}.
\]

Here \(g_t\) is the gradient, \(\eta_t\) the learning rate,
\(\lambda\) weight decay, and \(\epsilon>0\) a numerical stabilizer.
Squares and divisions are elementwise; moments start at zero.
These are the mathematical update rules, not an exact specification of every
mixed-precision kernel. See
[Kingma and Ba, Adam](https://arxiv.org/abs/1412.6980) defines the moments;
[Loshchilov and Hutter, AdamW](https://arxiv.org/abs/1711.05101)
separates weight decay from that adaptive update.
The displayed decay factor uses PyTorch's learning-rate-times-decay convention.
Our 27B trainer applied decay 0.01 to trainable adapter parameters, not to the
frozen base.

Gradient accumulation sums appropriately scaled microbatch gradients before
one optimizer step. With equal microbatches, \(B_{\rm effective}=B_{\rm micro}KD\)
for \(K\) accumulation steps and \(D\) data-parallel workers.
A partial final group needs its actual denominator. This identity does not
guarantee identical random draws, floating-point reductions or generated tokens.
Gradient clipping rescales an excessive gradient norm; it does not repair
bad labels or the wrong loss mask.

Freezing weights removes their parameter gradients and optimizer states;
it does not remove the forward computation or all saved activations.
Activation checkpointing saves some activation storage by recomputing it
during backward. Optimizer checkpoints, which save training state for
resumption, are a different use of the word “checkpoint.”

## Attention and the residual stream

For a standard attention layer, collect token queries, keys and values
as rows of \(Q,K,V\). Causal attention is

\[
A=\operatorname{softmax}_{\rm row}
\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),\qquad O=AV,
\]
\[
M_{tj}=\begin{cases}0&j\leq t\\-\infty&j>t.\end{cases}
\]

Each row of \(A\) is a probability distribution over allowed source positions.
The scale controls dot-product magnitude; the mask prevents reading future
tokens. Multi-head outputs are concatenated and output-projected. A schematic
pre-normalized decoder block is

\[
r_\ell=h_\ell+\operatorname{Attn}_\ell(\operatorname{Norm}(h_\ell)),\qquad
h_{\ell+1}=r_\ell+\operatorname{MLP}_\ell(\operatorname{Norm}(r_\ell)).
\]

This locates a residual-output intervention versus an attention-writer
intervention. It is an explanatory architecture, not the exact block diagram
of every model. The 27B checkpoint used here is hybrid: not every block uses
ordinary full attention. See
[Vaswani et al., Attention Is All You Need, section 3](https://arxiv.org/html/1706.03762v7#S3)
for the attention mechanism and our [27B record](qwen38_release_plan.md) for the tested architecture.

### Normalization and positional information

RMS normalization, with learned gain \(g\) and stabilizer \(\epsilon\), is

\[
\operatorname{RMSNorm}(h)
=g\odot\frac{h}{\sqrt{\|h\|_2^2/d+\epsilon}}.
\]

Unlike mean-centering LayerNorm, it rescales without subtracting the coordinate
mean. Consequently, an additive edit made before normalization is generally
not equivalent to adding the same vector after normalization.
[Zhang and Sennrich, RMSNorm](https://arxiv.org/abs/1910.07467) is the source.

Rotary position embeddings apply position-dependent rotations to queries and
keys. In the ideal rotation notation,
\((R_tq)^\top(R_jk)=q^\top R_{j-t}k\), connecting dot products to relative
positions. See [Su et al., RoFormer](https://arxiv.org/abs/2104.09864).
An average of keys rotated at different positions is not the key of a single
actual token. This matters for interpreting the GCAD gate.
Grouped-query attention shares each KV head across several query heads;
[Ainslie et al., GQA](https://aclanthology.org/2023.emnlp-main.298/) describes it.
Some architectures also normalize queries/keys before rotation; the exact
ordering belongs to the tested implementation.

## Why lower loss is not the whole evaluation

Training minimizes a reference-based surrogate. The intended outcome may
instead be grounded answers, good abstracts, useful responses or coherent style.
Our PRL result illustrates the distinction: lower development
(checkpoint-selection-set) NLL and higher final reference
overlap accompanied worse blinded AI preference. Our QASPER result improved
the registered answer and evidence metrics. See the
[completed comparison](showcase.md#qwen38-27b-portfolio-summary).

For paired scores \(s_i^{\rm adapter},s_i^{\rm base}\),

\[
\widehat\Delta=\frac1n\sum_i
(s_i^{\rm adapter}-s_i^{\rm base}).
\]

Questions from one paper are dependent. A paired paper bootstrap resamples
papers with replacement, keeps both arms paired, and recomputes the chosen
question-weighted statistic for each resample. It does not create new
independent questions, fix data leakage, or remove judge bias.
[Efron, Bootstrap Methods: Another Look at the Jackknife](https://doi.org/10.1214/aos/1176344552)
introduces bootstrap resampling; the paper grouping here follows this study's data structure.

Choose settings on development data, then freeze them before final evaluation.
A passed point-estimate retention rule is not a proof of statistical
noninferiority. A small paired interval supports a claim on the sampled task
and protocol, not every future prompt.

## Check your understanding

**Question:** Two models have the same greedy next token. Must their
cross-entropy be equal?

**Answer:** No. If the correct token is largest with probabilities \(0.6\) and
\(0.9\), both choose it, but losses are \(0.511\) and \(0.105\) nats.

**Question:** Does assigning zero loss to prompt tokens make the model ignore
the prompt?

**Answer:** No. Answer-token predictions still depend on the prompt through
the forward computation and its gradient paths.

## Reading map

Continue with the first foundation page in the section you want to understand.
Each includes primary references, a worked example and its connection to the
project. The [reference index](references.md) collects papers and software;
the [reproduction guide](reproduce.md) describes actual runtime requirements.
