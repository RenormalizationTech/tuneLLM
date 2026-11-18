# Mathematical foundations of LoRA and QLoRA

Fine-tuning changes a model's parameters to improve an objective on examples.
LoRA restricts the trainable change to a low-rank factorization. QLoRA adds a
quantized frozen base to reduce storage requirements. Neither method guarantees
that fitting references better will improve the task metric you care about.

Prerequisites: [losses and gradients](learning_foundations.md) and
[quantization](quantization_foundations.md). Then read the
[workflow](model_workflow.md) and [27B task results](qwen38_release_plan.md).

## A low-rank weight update

For a frozen \(W_0\in\mathbb R^{d_{\rm out}\times d_{\rm in}}\), write

\[
y=W_0x+\gamma BAx,\qquad
A\in\mathbb R^{r\times d_{\rm in}},\quad
B\in\mathbb R^{d_{\rm out}\times r},\quad
\gamma=\alpha/r.
\]

The update \(\Delta W=\gamma BA\) has rank at most \(r\).
The base matrix is not assumed low-rank. Different targeted matrices each get
their own factors; a nonlinear network of low-rank updates is not just one
rank-\(r\) map. The original
[LoRA paper, section 4.1](https://arxiv.org/html/2106.09685v2#S4.SS1)
introduces this parameterization. Low-rank task updates are a useful modeling
hypothesis, not a theorem that every task has a low-rank optimum.

**Worked example.** One hypothetical \(4096\times4096\) matrix contains
\(16,777,216\) entries. At \(r=16\),

\[
P_{\rm adapter}=r(d_{\rm in}+d_{\rm out})=131,072,
\qquad
P_{\rm adapter}/P_{\rm base}=0.0078125.
\]

That is 0.78125% of this matrix's count. Total model counts require summing over
every targeted matrix; they are not obtained by multiplying this percentage
by the model's advertised size.

Rank limits expressivity, but more rank also adds trainable state and possible
overfitting. Alpha sets the branch's scale; rank, initialization, learning rate
and target modules must be interpreted together. The
[PEFT LoRA configuration reference](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora)
also documents dropout, target selection and alternative scaling.
Our registered 27B runs used \(\alpha/r=32/16=2\), not the rsLoRA
\(\alpha/\sqrt r\) convention.

## How gradients reach the adapter

Let \(g=\partial L/\partial y\) for one example, omitting dropout for clarity.
The chain rule gives

\[
\nabla_B L=\gamma g(Ax)^\top,\qquad
\nabla_A L=\gamma B^\top g x^\top,\qquad
\nabla_x L=(W_0+\gamma BA)^\top g.
\]

The shapes of these derivatives match \(B,A,x\), respectively.
Although \(W_0\) is frozen, its value participates in the input gradient.
Setting the **entire forward pass** to no-gradient mode would prevent the
desired adapter training.

A common initialization makes \(B=0\) and \(A\) nonzero/random. Then the
initial update is zero, the first task-loss gradient for \(A\) is zero, and
\(B\) can receive a nonzero gradient. If both factors were zero, both task-loss
gradients would be zero. This follows directly from the equations; it explains
why “start with no model change” does not mean initializing both factors to zero.
Zero task-loss gradient does not exclude an AdamW decay update to nonzero \(A\).
[LoRA](https://arxiv.org/html/2106.09685v2#S4.SS1) describes the zero-update initialization.

With dropout probability \(p\), the train-time branch is
\(\gamma BA\,\mathcal D_p(x)\), where \(\mathcal D_p\)
randomly masks/rescales inputs; dropout is disabled for evaluation. The formulas
above apply to the deterministic branch or conditional on the dropout operation
with its corresponding chain rule.
For inverted dropout, \(\mathcal D_p(x)=m\odot x/(1-p)\) with
\(m_j\sim\operatorname{Bernoulli}(1-p)\) and \(0\leq p<1\);
the rescaling preserves each coordinate's expectation.

The factorization is not unique: \(BA=(BC)(C^{-1}A)\) for invertible
\(C\in\mathbb R^{r\times r}\). Consequently, penalties or optimizer choices
on the factors can matter even when two factorizations initially give the
same effective update.

## What changes in QLoRA

Let \(q_0=Q(W_0)\) denote stored quantization codes and state, and
\(\widetilde W_0=D(q_0)\) their reconstructed numerical weights:

\[
y=\widetilde W_0x+\gamma BAx,\qquad
\min_{A,B} L(\widetilde W_0,A,B),\qquad q_0\ \text{fixed}.
\]

The trainable factors receive gradients; the frozen codes are not optimized.
There is no need to differentiate rounding in order to learn \(A,B\).
Backpropagation through the frozen computation is still necessary to train
adapters earlier in the network.
[QLoRA](https://arxiv.org/html/2305.14314v1#S3) combines this idea with
NF4, double quantization and paged optimizers. Adopting its frozen-base approach
does not mean every local run uses every optimizer feature from the paper.

Storage, compute and optimizer-state precisions are separate choices. In this
project the saved native NF4 screen retained some floating base weights in
BF16, while PEFT preparation for the matched 27B training/science/retention arms
cast floating base weights and trainable adapter parameters to FP32.
The quantized branch used BF16 computation under the registered BF16 autocast;
FP32 parameter storage does not imply every operation executes in FP32.
See the [published loading recipe](../releases/Qwen3.8-27B-tuneLLM-QASPER-QLoRA/README.md);
do not replace that recipe with “everything is four-bit.”

## Where memory goes

An accounting model is

\[
M_{\rm peak}\approx
M_{\rm frozen\ base}+M_{\rm adapter}
+M_{\rm adapter\ gradients}+M_{\rm optimizer}
+M_{\rm activations}+M_{\rm workspace}.
\]

The terms depend on the runtime and lifetimes of tensors. For \(P_a\) FP32
adapter parameters, parameters, gradients and two FP32 Adam moments alone
require approximately \(16P_a\) bytes, before buffers and implementation
details. Freezing the base saves its trainable state, not all activation memory.

The selected 27B inventory targeted 496 linear modules and contained
116,727,808 trainable parameters. At four bytes per parameter, their bare
parameter payload is \(466,911,232\) bytes; the safetensors file additionally
contains metadata. The
[training report](../results/qwen38_qasper_training_01.json) records the
actual inventory rather than assuming every matrix has the same dimensions.

Longer contexts can make activations dominate memory. Gradient checkpointing
trades recomputation for activation storage; gradient accumulation reduces
the microbatch requirement while preserving the registered loss reduction.
Neither changes the number of trainable parameters.

## Targets and masks define what the adapter learns

Our two 27B adapters had independent training data and weights:

- QASPER: numbered paper context plus question → structured answer, abstention
  decision and paragraph evidence IDs.
- PRL: a fixed body excerpt → a reference abstract.

Answer-only loss assigns supervision to output tokens, including structural
tokens where registered. It does not explicitly optimize factual entailment.
A model can learn reference phrasing, brevity or formatting while getting worse
on a preference rubric. Evidence IDs and abstention therefore need their own
evaluation, not just lower answer-token NLL.
[Dasigi et al., QASPER](https://arxiv.org/abs/2105.03011) motivates
paper-grounded questions and evidence; our custom subset/scorer is documented separately.

The [QASPER final result](../results/qwen38_qasper_final_01.json) passed
fixed task and retention gates. The
[PRL result](../results/qwen38_prl_final_01.json) failed abstract-quality
gates despite lower development (checkpoint-selection-set) NLL. These outcomes test a workflow; they do not
validate a universal rank or learning-rate choice.

## Merging and changing the base

Without dropout, and in exact arithmetic, LoRA on one fixed numerical base
can be written as a merged matrix:

\[
W_{\rm merged}=\widetilde W_0+\gamma BA.
\]

Changing from \(\widetilde W_0=D(Q(W_0))\) to the original \(W_0\)
changes the base. Quantizing the merged result changes it again:

\[
D(Q(W_0+\gamma BA))
\ne D(Q(W_0))+\gamma BA
\quad\text{in general}.
\]

Thus adapter reload, higher-precision merging and merged GGUF export are
different verification tasks. This project published separate adapters;
it did not claim that an unmeasured merged Q4 model reproduces their results.

## Check your understanding

**Question:** With \(B=0\), \(Ax\ne0\) and \(g\ne0\), which factor can receive
a nonzero first task-loss gradient?

**Answer:** \(B\). Its gradient is \(\gamma g(Ax)^\top\);
the task-loss gradient for \(A\) contains \(B^\top=0\).

**Question:** Does doubling rank always double the applied perturbation?

**Answer:** No. With fixed alpha, \(\gamma=\alpha/r\) halves, while the factors
and the allowed update space also change. The actual perturbation is
\(\|\gamma BAx\|\), not rank alone.

For the concrete optimization, masking and checkpoint decisions, continue to
[the model workflow](model_workflow.md) and [registered 27B plan](qwen38_release_plan.md).
