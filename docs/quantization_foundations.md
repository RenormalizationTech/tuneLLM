# Theoretical background for quantization

Quantization represents high-precision weights using a finite codebook and
associated reconstruction parameters. Storage is reduced at the cost of
approximation error whose effect depends on the input distribution, model
computation and inference backend.

The [shared notation](learning_foundations.md) defines the linear algebra
used below. The [4B measurements](benchmark_summary.md),
[conversion workflow](execution.md) and [27B comparison](qwen38_quantization_results.md)
specify the evaluated implementations. The uniform quantizer is an analytical
model; exported formats can mix quantizer types and tensor precisions.

## Uniform quantization and reconstruction

A common affine scheme stores an integer \(q\), positive scale \(s\), and
integer zero point \(z\):

\[
q=\operatorname{clip}
\left(\operatorname{round}(w/s)+z,q_{\min},q_{\max}\right),\qquad
\widehat w=s(q-z).
\]

The endpoints allow at most \(2^b\) codes for \(b\) bits. Some symmetric
schemes reserve a code or use asymmetric endpoints, so bit width alone does
not determine the exact grid. Reconstruction gives an approximation
\(\widehat w\), not recovery of the original number.
[Jacob et al., section 2](https://arxiv.org/html/1712.05877v1#S2)
describes affine quantized arithmetic.

For nearest rounding without clipping, each scalar has error
\(|w-\widehat w|\leq s/2\). The often-used variance \(s^2/12\)
additionally assumes error is uniformly distributed over a rounding interval;
it is a model, not a guarantee for neural weights.

**Numerical illustration.** Take \(s=0.1,z=0\), codes \(-7,\ldots,7\).
For \(w=0.26\), \(q=3\), \(\widehat w=0.3\), and error is \(0.04\).
For \(w=1.0\), clipping gives \(q=7,\widehat w=0.7\);
error is \(0.3\), so the no-clipping bound no longer applies.

A single large outlier can force a large scale for a whole tensor. Using
separate scales for blocks or channels reduces that coupling, at the cost
of storing additional metadata.

## Error propagation through linear layers { #why-equally-small-weight-errors-can-behave-differently }

Let the reconstructed matrix be \(\widehat W=W+E\). For a fixed input \(x\),

\[
\delta y=(\widehat W-W)x=Ex,\qquad
\|\delta y\|_2\leq\|E\|_2\|x\|_2.
\]

The matrix norm here is the spectral/operator norm. Two errors with the same
Frobenius norm can affect different input directions. With a calibration
matrix \(X\) whose columns are layer inputs, an activation-aware objective is

\[
\min_{\widehat W\in\mathcal Q}\|(W-\widehat W)X\|_F^2.
\]

The set \(\mathcal Q\) contains matrices representable by the chosen quantizer.
This connects error to inputs rather than treating every weight equally.
[GPTQ, section 3](https://arxiv.org/html/2210.17323v2#S3)
uses this layer-reconstruction perspective. It is background theory:
this project did not run GPTQ.

For small perturbations and a differentiable downstream computation, a
first-order approximation is \(\delta z\approx J\,\delta h\), where \(J\)
is its Jacobian. The approximation becomes unreliable for large perturbations
or when generated tokens change. There is no universal conversion from
“weight error of 1%” to “task accuracy loss of 1%.”

## NF4 and scale quantization { #nf4-and-quantizing-the-scales }

NF4 uses a nonuniform 16-value codebook designed around approximately
normally distributed weights. A conceptual block encoding is

\[
s_b=\max_{i\in b}|w_i|,\qquad
q_i=\arg\min_{k\in\{0,\ldots,15\}}|w_i/s_b-c_k|,
\qquad
\widehat w_i=s_b c_{q_i}.
\]

Here \(c_k\) is a codebook value; an all-zero block needs a zero-scale
special case. This notation describes the idea, not the exact library
packing or boundary rules. More resolution is allocated in dense parts of
the assumed distribution; this is not equal spacing or ordinary four-bit
floating point.

Double quantization also compresses the scales. In a simplified accounting,
a 32-bit scale for each block of 64 weights adds \(32/64=0.5\) bits per weight.
Compressing that metadata reduces overhead; it does not turn activations,
gradients and optimizer states into four-bit values.
[Dettmers et al., QLoRA, section 3](https://arxiv.org/html/2305.14314v1#S3)
introduces NF4 and double quantization. These are distributional design choices,
not a theorem of lossless model compression.

## Storage and compute precision { #storage-precision-is-not-compute-precision }

For \(P\) uniformly quantized weights, a storage estimate is

\[
M_{\rm payload}\approx \frac{Pb}{8}\ {\rm bytes}
+M_{\rm scales}+M_{\rm unquantized}+M_{\rm metadata}.
\]

A hypothetical 27 billion weights at exactly four bits occupy 13.5 GB
(decimal), or about 12.57 GiB, **before** overhead. Real models retain some
tensors at higher precision and add quantization state. This is why “27B
at four bits” does not predict a file's exact size.

A kernel can unpack or dequantize weights into BF16/FP16 computation, often
in tiles rather than materializing a full high-precision copy. Other kernels
use integer operations for parts of the computation.
[bitsandbytes' four-bit layer documentation](https://huggingface.co/docs/bitsandbytes/reference/nn/linear4bit)
distinguishes storage type and computation type. Lower file size does not
by itself imply lower latency.

Peak GPU memory also includes activations, temporary buffers and the KV cache.
For ordinary full-attention layers, an illustrative cache estimate is

\[
M_{\rm KV}\approx
2L_{\rm attn}BT\,n_{\rm kv}d_h\,b_{\rm elem}.
\]

The factor 2 counts keys and values; \(L_{\rm attn}\) counts full-attention
layers, \(B\) sequences, \(T\) stored tokens, \(n_{\rm kv}\) KV heads,
\(d_h\) head width, and \(b_{\rm elem}\) bytes per element.
Grouped/multi-query attention shares KV heads; see
[Shazeer, Fast Transformer Decoding](https://arxiv.org/abs/1911.02150) for
multi-query attention and [Ainslie et al., GQA](https://aclanthology.org/2023.emnlp-main.298/)
for the grouped-query generalization.
This is not a complete cache estimate for hybrid models with recurrent state.

## NF4 and Q4_K_M in this project

GGUF is a model container; Q4_K_M names a llama.cpp quantization recipe.
It can mix tensor precisions and carries block metadata. It is not NF4
renamed, nor does the label mean every stored number uses four bits.
Use the pinned [llama.cpp quantizer source](https://github.com/ggml-org/llama.cpp/blob/8ca5a8008b57aa39f2601546c0d07912045080a5/tools/quantize/quantize.cpp)
and the [published card](../releases/Qwen3.8-27B-tuneLLM-GGUF/README.md)
for the actual representation and settings.

The [27B screen](../results/qwen38_27b_quantization_screen_01.json)
compares whole workflows. Q4 and BF16 GGUF share a backend; there was no BF16
Transformers reference isolating the effect of NF4. A different batch size,
template, kernel or precision can also change outputs. Similar small-subset
counts do not prove equivalent models.

## Sensitivity and evaluation notes

**Scale sensitivity.** Without clipping, the nearest-rounding bound scales
linearly with the quantization step: doubling the step from \(s\) to \(2s\)
raises the bound from \(s/2\) to \(s\).

**Input dependence.** For \(E=\operatorname{diag}(0.1,0)\), the inputs
\(x_1=(10,0)\) and \(x_2=(0,10)\) produce errors \(Ex_1=(1,0)\) and
\(Ex_2=(0,0)\). A fixed weight error can therefore have substantially different
effects across inputs.

Bit width, reconstruction error, compression ratio and task performance measure
different properties. The project evaluates each exported runtime on fixed
tasks and verifies its reload separately.
