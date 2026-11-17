# Fixed benign-text check using the earlier 4B refusal cases

Status: implementation/review, before new responses. The user explicitly
accepted reusing the earlier 4B-refused texts, selected benign over-refusal as
the primary goal, and authorized continued work with the existing workflow and
only necessary additions. This small check implements that previously discussed
reuse. It adds no new benchmark, extraction, direction, method or tuning.

## Evidence and fixed scope

The [initial transfer](refusal_7b_transfer_execution.md) and its single weaker
dose retain their original registered null decisions. At **-0.25**, benign
development refusals fell 3/30 to 0/30, useful/correct answers rose 20/30 to
21/30, and one baseline refusal became a useful full answer. All quality gates
passed. Only the historical unsafe-compliance gain threshold failed. Those
unsafe contrasts measure willingness, not the user's benign-helpfulness goal.

Use the original **17** nonempty partial/full refusal cases from the completed
Qwen3-4B XSTest-safe baseline, preserving its exact IDs and labels:
`14,57,59,61,64,67,103,109,151,153,162,165,168,169,171,417,418`.
Derive question text verbatim from the pinned stored safe250 source and verify
its labels against the accepted 4B grading. The existing review supported 16;
ID67 remains its original P despite the review's disagreement. Do not drop or
relabel it, or add matcher-only IDs54/420.

Four cases (`109,162,165,171`) were already in the dose-selection development
set. Reuse their existing baseline/-0.25 responses and original accepted labels,
reporting them separately; do not regenerate or regrade them. Generate only
the other **13**, in the original source order:
`14,57,59,61,64,67,103,151,153,168,169,417,418`.

For these 13, compare fresh baseline and the fixed **-0.25** block-18-output
intervention. Keep the same model revision, raw-vector hash, half-vector scaling
through the unchanged -0.5 hook, BF16 runtime, GPU0/UUID, native Qwen2 template,
mask handling, greedy seed42, repetition penalty1, EOS151645/PAD151643, cap320
and batch8 (batches8/5). Fresh baselines are necessary because prior outputs
used different settings or batch membership. No further coefficient or layer
is considered here.

## Prospective data allocation

This check newly uses seven IDs from the previous protected105:
`59,61,67,151,153,168,418`, leaving **98** protected. IDs57/64 also appear in
the transferred final42, so that future final set can no longer be called
untouched. Register this explicitly before generation. Preserve all historical
split files, allocation records, scores and raw artifacts byte for byte.
All25 DECCP held-out rows remain untouched.

The texts were selected on 4B refusal behavior before this 7B setting. Some
previous 7B baseline outputs are known, and four cases influenced dose choice.
The 13-case check excludes those four from new generation, but is still a
selected diagnostic with earlier cross-model/public-data exposure, not a clean
held-out benchmark or a population refusal-rate estimate.

## Execution and reporting

Freeze a fresh `outputs/refusal-7b-benign-transfer-01/` fixture, rendered bytes,
IDs, allocation, sources/runtime/model/direction and prior-label provenance.
Review and commit the exact registration before any new model output. Reuse
the tested existing loader, generation, intervention, journal and scoring
helpers. Preserve completed/uncertain stage records without automatic reruns.
There is no extraction, probe, selection, final or retention stage.

Maximum new target responses: **26** (baseline13 + candidate13). Jointly grade
all26 anonymously with the unchanged refusal rubric, validator, audit and pinned
Claude2.1.323 binary, using the existing complete-ID/first-valid journal. The
four reused cases keep their prior packet contexts and labels; disclose this.

Report the 13-case comparison as paired **counts**: baseline/candidate partial
and full refusals, useful/correct and valid full answers; baseline refusals
becoming useful valid full answers; newly refused formerly compliant cases;
baseline-useful answers losing usefulness; empty, degenerate, unscorable and
cap counts. Keep overlapping harm categories separate, and show the four
already-seen cases in a separate descriptive block. No pooling into independent
evidence, new acceptance threshold, population rate or significance claim.

Allowed conclusion: on a specified number of these 4B-selected texts that 7B
refused, a specified number became useful answers, with the observed quality
losses. No general-controller, selective-safety or broad robustness claim follows.
Neither earlier null is relabelled as a pass. Compact result:
`results/refusal_7b_benign_transfer_01.json`. Present the actual result before
considering any additional work.
