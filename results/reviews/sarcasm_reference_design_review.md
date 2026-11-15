# SAPS isolated sarcasm design review

Claude reviewed the concrete bounded protocol before implementation and found
its accounting correct: at most1242 generations,1500 direction-learning prompt
forwards, two logit probes,64 MC items,73 first-valid batches and146 starts.
The coordinator accepts the design with the corrections below. No SAPS model or
grading output informed these choices. Code acceptance and exact committed
registration remain required before live execution.

Accepted source choices: unchanged author train/validation strings and a fixed
100-of450 test subset, fresh native HF residual coordinates, B1 source broadcast
semantics, greedy cap60, original four rating definitions archived from AppendixC,
separate calibrated custom ratings and a single native Claude replacing the
authors' three-judge average. The released test-example oracle table is not a
deployment benchmark. Dataset transformation lineage and hostile/profane source
language are limitations, not hidden filtering criteria.

Required corrections adopted in the protocol:

- Skip any method lacking a quality-eligible validation configuration; choose
  the winner only among eligible settings. Never substitute a quality failure.
- Baseline h0 reuse requires a read-only first-prefill, last-token capture at
  all four blocks, separately on validation and test, before any fixed arm.
  Use the author's FP32 CPU dot and Python-scalar coefficient arithmetic;
  zero parity covers capture-hook attachment.
- Fixed random uses the same per-case c. Adaptive random uses the same scalar
  controller along a random direction, not trajectory-norm equality. Report
  prefill/decode shift distributions and a report-only random/real mean ratio
  flag outside0.8–1.25. No second random construction is added.
- Source RNG draws are reconstructed, not claimed identical: author seeding
  precedes model loading, which could consume Python randomness.
- Custom content means faithful paraphrase; baseline off-target scores are
  reported because source complaints can themselves contain profanity/hostility.

The review accepted elicitation gating before directions, validation-only
selection, a0.5 style-gain threshold, baseline/random comparisons, fixed ties,
no report reselection and capability screening framed as off-task transfer.
This small single-judge method adaptation does not validate generic assistant
persona transfer, statistical significance or independent simultaneous control.

## Early operations and rubric inspection

Claude compared the new pure operations and rubric with the archived source and
AppendixC. It found the four criteria and general rules identical after whitespace
normalization, and the B1 controller/broadcast operations consistent with source.
Its CPU checks confirmed the reconstructed RNG draw equivalence and scalar dtype
promotion. Require a zero-dimensional adaptive target: a length-one target would
promote the BF16 controller calculation. The initial fixed-tensor suggestion
was superseded by the source-scalar correction below. This is early review, not full runner
acceptance or a claim that Claude ran the tests.

Reports must distinguish twenty-item co-rating contrast effects from the original
single-item three-judge procedure, and sentence-alone reference sarcasm from the
custom contextual construct. Packet labels must distinguish source complaint and
generated sentence. Shift norms are per position: broadcasting over S prefix
positions multiplies the full-prefix Frobenius norm by sqrt(S), and squared norm
by S. The coordinator corrected the advisor's initial S-times L2 wording before
implementation acceptance.

## Main runner review

Claude subsequently reviewed the main runner and runbook and accepted the
implementation within its disclosed adaptations, with no actionable blocker.
It executed no tests, model or API calls in that review; worker CPU results
remain independently recorded. It checked validation/test-specific read-only
baseline captures, case-specific constants,1500 direction forwards across
all36 post-block sites, source/output packet labels, separate original/custom
rubrics, absent-method branches, committed selection before test, controller
random diagnostics and the conditional off-task capability screen. Pending
wording fixes were per-position norm labels and naming the ordinary-with-capture
versus actual-zero parity probe accurately. The optional additional SAPS lexical
analysis was not added; source-language confounds and custom baseline scores
remain explicitly reported. Passing CPU checks and exact registration still
precede live release.

The worker's differential test later disproved the advisor's earlier claim that
a zero-dimensional FP32 fixed coefficient multiplies BF16 identically to a
Python scalar. The worker corrected the hook; the coordinator additionally
checked author generate.py and found its subtraction occurs after separately
converting the FP32 CPU dot and target to Python scalars. Claude agreed to match
these exact operations and test coefficient construction plus the BF16 shift
end to end before freeze. Adaptive tensor arithmetic is unchanged. No generation,
selection or budget change follows from this pre-registration correction.

The worker then froze the corrected SAPS implementation after24 focused CPU
checks passed (9.15 seconds). Checks include exact author hook operations in
FP32/BF16, end-to-end Python coefficient construction including an FP32
subtraction regression, cached generation/broadcasts, last-block versus final
norm, reconstructed learning equations/draws, branch plans, budgets, grading
journals, null reports and privacy. No SAPS preparation, model/GPU/grading calls
or frozen PV edits occurred during implementation.
