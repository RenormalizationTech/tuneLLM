# Isolated sarcasm: SAPS reference adaptation

Status: Claude reviewed the bounded design; implementation and exact manifest
registration still precede live execution. Complete the first Persona Vectors
rudeness study before this study. Joint control remains deferred for discussion.

Reference: [SAPS paper](https://aclanthology.org/2026.findings-emnlp.1594/) and
[author code](https://github.com/SharathHN/saps-steering), archived at commit
`c56fb28ac0d14c8fd70035e035c5452e3c6c6ccc` in the local source audit.
Before registration archive the paper's Appendix C judge criteria.
Use the released 500 train and 50 validation strings. Select 100 of 450 test
strings with seed 702 before outputs; the first 20 of that fixed ordering form
the separate custom-rating subset. Keep released text unchanged, including
hostile or profane input language, and disclose its provenance limitations.

Use the existing pinned Qwen3-4B native HF BF16/SDPA model, nonthinking, seed42,
greedy cap60, cache enabled and batch ONE. GPU0 resource rules match the PV
protocol. This ports the method into a fresh native HF residual basis; neither
TransformerLens vectors nor a numerical identity claim transfer across backends.
Use literal author plain/style instructions and user-only template rendering.

First generate validation baseline and prompted style-index0 control: 100
answers. Co-rate the author's four 0..5 scales: sarcasm, entailment,
grammaticality and clarity. Before any direction work, the control must have
at least25/50 style scores>=3, mean style>=3 and mean grammar/clarity>=3.5;
otherwise close as elicitation_null.

For each of all500 training strings capture last-prompt-token post-block states
under plain and a randomly chosen style instruction. Mean(style-plain), then
normalize, separately at each of all36 blocks. A second500 styled-prompt pass
estimates each direction's target projection. Total:1500 prompt forward
sequences, not1000. Freeze 1000 instruction draws from random.Random(42), first
500 and then500 continuing the stream. These draws reconstruct the algorithm;
the author seeds before model loading, so exact original RNG identity is not
claimed. Stack captures before one CPU transfer per forward.

Compare two author methods, with zero-based blocks11/15/19/23:

- Fixed projection: c=alpha*(target-h0 dot u), then add c*u at every forwarded
  position, including the full prefill. Alpha=1 only.
- SAPS: at each forward, delta=alpha*(target-h_last dot u)*u, broadcast across
  that forward's positions. Alpha=1 or1.4. Cached decode has one new position;
  prefill uses its last position's correction across the full prefix. Alpha1.4
  intentionally overshoots the target.

Reuse baseline h0 only with a read-only first-prefill hook, position-1, capturing
all four blocks separately on validation and test. Baseline runs before fixed
arms; compute c in FP32 and add in model dtype. Ten unscored cap32 generation
probes (four ordinary/four actual zero/two nonzero), plus two logit forwards,
check parity including the capture hook, source-equivalent tensor operations,
cache coverage and cleanup before selection. Save exact vector round-trips.

Validation adds600 generations: four blocks times three settings times50.
Reuse the first baseline/control; never regenerate them. Eligible configurations
need mean grammar/clarity>=3.5, entailment>=3 and empty/degenerate/capped counts
no greater than baseline+3 each. Choose each method's highest mean style, then
smaller alpha, then earlier block. A method without an eligible setting is
recorded absent and its test arm skipped, never replaced by a quality failure.
Choose an overall winner among eligible settings, ties preferring fixed after
the other tie breaks. Require style gain>=0.5 over baseline, else selection_null.
Commit selection and vector bindings before test.

Test up to five arms on100 untouched strings: baseline, best fixed, best SAPS,
saved random direction for the locked winner, and prompted style-index0 control.
No test-set layer oracle or reselection. For fixed, random uses the same per-case
c times an independent unit vector. For SAPS, random uses the same scalar rule
alpha*(target-h dot u) but adds along random unit r; this matches the controller,
not the different trajectories' actual norms. Seed r with7200+block. Report real
and random per-step shift-norm distributions separately for prefill and decode;
flag random/real mean outside0.8–1.25 as norm-mismatched, report-only. Do not also
run a second random-control construction.

Reference report gates: winner mean style>=baseline+0.5 and random+0.5, the
validation quality floors, and positive control at least50% scores>=3. Report
all arms even on a failed gate. Separately apply the calibrated three-axis
rubric to the same20 test strings times up to five arms (100 existing answers).
Here content means faithful paraphrase of the source sentence. Report baseline
and steered profanity/rudeness alongside sarcasm; input language confounds these
scores. Do not claim independent trait control.

Only after the reference report passes, run the existing fresh matched48-item
baseline/winner capability screen:32 generations plus64 MC. This is off-task
transfer, not the original paraphrase benchmark. Use the PV capability tolerances.

Maximum1242 generations=100+10+600+500+32;1500 direction-forward sequences,
two unscored logit forwards and64 MC. Author-scale grading uses batches up to20
distinct source strings:5 initial+30 selection+25 test. Custom uses up to8:
13 batches. Maximum73 first-valid batches/146 invocations; missing method arms
and early nulls reduce actual work. Freeze anonymous distinct-source plans and
seeds before outputs. One pinned native Claude replaces the paper's three-judge
average; co-rating the four original scales is a disclosed adaptation. Keep
custom judgments in separate contexts without earlier scores. Apply the same
tools-off identity checks, two identical-payload structural attempts, first-valid
retention and permanent infrastructure/hash/uncertain-start stops as PV.

All source, template, rubric, partition, runtime, actual intervention, judgment
and cost bindings need CPU acceptance and committed registration. The worker
implements; the coordinator reviews and releases live stages. Preserve prior
studies. Consolidate the isolated workflow after closure; no combined-control
experiment follows automatically.
