# Original-Mistral sarcasm: released SAPS method

Prospective design discussed with Claude. Reproduce the released method on
`mistralai/Mistral-7B-Instruct-v0.1`, revision
`ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca`, using the pinned author repository
`saps-steering` at `c56fb28ac0d14c8fd70035e035c5452e3c6c6ccc` and its actual
TransformerLens2.17 runtime. This is a new method-level run. The audit found
that public result logs do not match the released data and prompt metadata;
our arithmetic reconstruction does not resolve their generation lineage.
New results cannot be claimed as an exact numerical reproduction.

## Runtime and direction

Use the isolated SAPS environment, original author prompt builder, activation
capture, instruction selector, direction learner, hook factories and answer
stripping. Pin local model files, disable remote code/network model loading,
and preserve the author's loader processing defaults. No native-HF port and
no direction copied from Qwen are permitted.

The author's `torch_dtype=float16` is interpreted by TransformerLens's backward
compatibility path as its model dtype. Record actual config and parameter
dtypes, fold/centering flags, normalization after loading, warnings, BOS
settings and actual rendered prompt token IDs. Preserve the author's BOS
behavior, even if the rendered chat template plus default tokenization adds
two BOS tokens. Do not silently repair runtime quirks.

Seed Python and torch42 before the actual model loading, matching
`compute_direction.py`. Keep the resulting instruction draws, including any
RNG consumption during loading. Direction learning uses the released500
training sentences and random instruction mode:500 plain and500 styled prompt
forwards, then500 independently drawn styled target forwards. The author
learner produces per-layer unit mean-difference directions and mean styled
target projections. Save and verify tensor roundtrips before generation.
No training generation or fine-tuning occurs.

## Fixed generation design

Use all450 released test sentences in their original order, with the plain
rewrite instruction in every arm. Fix zero-based block12 from the paper
before generation; no per-sentence layer oracle, validation sweep or
post-hoc layer choice. Four arms:

- source SAPS alpha0 baseline, verified against hook-free generation;
- fixed projection, alpha1.0;
- adaptive SAPS, alpha1.0;
- adaptive SAPS, alpha1.4.

This includes the paper's fixed1.0/adaptive1.4 comparison and a matched-alpha
comparison. Fixed projection captures the unsteered last-token residual once
per test prompt and reuses its scalar adjustment throughout generation.
That requires450 additional prompt forwards. SAPS recomputes the adjustment
from the current last-token projection at every forward call. The actual
source adds that shift to the entire residual tensor, broadcasting it across
all prefill positions at batch size1; preserve this behavior rather than
substituting a last-position-only hook.

Use a separate model load and source-style invocation/output file per arm, in
the listed order, with the source random instruction mode recorded explicitly.
Use source greedy generation, batch size1, cap60, actual source cache/EOS/BOS
settings, and author answer stripping. Save full token IDs, prompt boundaries,
logical continuations, cap/EOS flags, raw rendered output and stripped answer.
No explicit sarcasm instruction is added during test generation.

Ten unscored cap16 mechanics generations and at most two additional logit
forwards test finite projections and actual SAPS-alpha0/fixed-zero token parity
against hook-free generation, nonzero hidden/logit changes,
prefill/cached-decode coverage and cleanup. These use separately frozen
ordinary prompts, never test-score selection. Retain and count their prompt
projection captures separately. Commit successful mechanics and direction
bindings before scored generation. Any execution/integrity failure stops
without a rescue sweep.

## Evaluation and interpretation

Use the existing pinned native Claude tools-off grader, with author
sarcasm/entailment/grammaticality/clarity definitions on the original0–5 scales.
This replaces the public logs' three-judge average; evaluator substitution
and four-metric co-rating are explicit limitations. Predetermine blinded
20-item batches,90 first-valid batches total; avoid placing different arms
of the same sentence together. At most two identical structural attempts
per batch,180 starts total, no semantic rerolls. Infrastructure, identity or
uncertain-start errors stop the run.

Each grading item includes the original complaint sentence and the author-stripped
generated sentence, labelled as in Appendix C. Entailment compares the two;
sarcasm is judged from the generated sentence alone. Preserve Appendix C hard
constraints: truncated or non-terminated sentences cannot score above3 for
sarcasm. Report cap hits per arm beside both sarcasm threshold counts.

Report all four metric distributions, paired sentence-level differences,
caps/empty outputs and degeneracy indicators. Descriptive directional checks,
reported regardless of outcome:

1. Adaptive1.0 mean sarcasm is at least0.5 above baseline.
2. Adaptive1.0 mean grammaticality is at least fixed1.0.
3. Adaptive1.4 mean sarcasm exceeds adaptive1.0.

These checks do not select settings, release another experiment or establish
robust everyday persona control. Include counts at sarcasm>=3 and>=4
separately: the author's level3 allows partial irony; clearly self-contained
sarcasm starts at4. Report meaning preservation alongside style changes.
Public Table1 means and our arithmetic reconstruction are context, with
lineage/evaluator limitations next to any comparison. No random-direction
control, broad capability benchmark, joint trait experiment or larger-model
transfer is released here.

Maximum scored generation is1800 answers. Add ten mechanics generations,
1950 main/extraction prompt forwards, separately counted mechanics projection
captures, two bounded logit forwards, zero MC and90 first-valid native
batches. Exact stage budgets, paths, runtime identity, released artifact
hashes and protocol registrations must be reviewed and frozen before calls.
This prospective document does not itself authorize a live stage release.
