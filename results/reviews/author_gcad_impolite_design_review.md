# Original-Qwen GCAD R1 review

Claude and the coordinator reviewed the prospective first-reply design before
any target output. The user authorized the original open model and author
method, with existing Claude evaluation replacing unavailable OpenAI API
grading. Exact numerical reproduction is not claimed.

## Scientific design accepted

The fixed three-arm design uses60 source conversations each for baseline,
residual and GCAD; ten unscored mechanics probes precede180 scored generations.
Separate trait/coherence contexts produce46 first-valid native batches with
at most92 starts. The full20x10 source question plan is sampled before taking
first questions, preserving the paper's first-reply sampling distribution.
Three registered30-point directional contrasts, refusal-excluded means and
conservative fixed-denominator means are declared before outputs. No R2,
capability run, random GCAD construct, rescue sweep or joint control is released.

## Source checks resolved

- The actual loader defaults to BF16, with native SDPA under the pinned
  Transformers4.52.3 runtime. Python3.11.18 and torch2.6.0+cu124 are isolated from
  the shared project environment.
- No explicit system message still yields Qwen's exact default helpful system
  prompt through its chat template; rendered bytes and actual token IDs are
  retained. Temperature1 overrides the checkpoint0.7 default, while top_p0.8,
  top_k20 and repetition_penalty1.05 remain source defaults.
- Source residual index20 maps to zero-based decoder block19, coefficient2,
  all forwarded positions. GCAD directly uses zero-based sites9–19.
- Author tokenizer PAD becomes EOS151645, while generation PAD151643 and both
  EOS IDs151645/151643 remain checkpoint defaults. Actual mask and completion
  padding must be validated separately.
- Published V/K/demand tensors have the expected FP32 dimensions, were loaded
  tensor-only and are not renormalized. Cropping belongs to upstream extraction;
  this run reuses the source bank without regenerating cropped attention.
- Source generation reconstructs history each turn and caches only within a
  reply. First-reply scope does not reproduce the paper's cross-turn KV claim.
- The author's generation definitions and steerer classes are used without
  credential-dependent pipeline imports. A recording proxy preserves arguments
  and returned tensors. Logit instrumentation is limited to last-token probes.

## Implementation review

Accepted before preparation. Claude independently checked the complete runner,
common module and author hooks, then journal/report/publication paths. The
coordinator also reviewed the active code and independently reran all31 CPU
tests (31 passed in5.22s). Tests include real author-hook zero parity and
single-versus-multiple residual-wrapper equivalence on prefill and cached decode.
CLI imports/help work in the isolated author environment.

Tensor-only loading replaces the author's unrestricted pickle loading without
changing verified tensors. The grading envelope requests JSON plus a brief
reason despite the retained author definition asking for a number only; this
is explicitly part of the evaluator adaptation.

There are no known scientific or implementation blockers. Claude identified
an operational constraint: wrong-interpreter or uncommitted-registration
preconditions currently arm the permanent stop. The coordinator will use the
specified interpreter and commit both required registrations in sequence;
this behavior is retained and no execution failure will be retried.

Initial manifest registration must be committed before mechanics; a separate
mechanics lock must be committed before scored generation. This review does
not itself release either stage. Historical frozen files remain unchanged.
