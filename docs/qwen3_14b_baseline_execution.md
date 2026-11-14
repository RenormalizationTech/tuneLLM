# Qwen3-14B baseline execution

Completed 2026-11-14 under the [registered protocol](qwen3_14b_baseline_protocol.md),
committed as `c0dd2f4` before model outputs. The
[aggregate report](../results/qwen3_14b_baseline_01.json) contains counts, hashes,
grader provenance and measured costs. This run used the unmodified model:
no adapter, training, activation hooks or steering-vector extraction.

## Model and reproducibility

Official `Qwen/Qwen3-14B`, revision
`40c069824f4251a91eefaf281ebe4c544efd3e18`, ran in BF16 on GPU 0, one RTX A6000.
All 14,768,307,200 parameters were verified BF16 on the exposed CUDA device.
The 17 selected model assets were hash-verified. Native nonthinking rendering,
greedy decoding and seed 42 were fixed; all 250 safe input token sequences
matched the 4B tokenizer. The XSTest generation cap was 256 tokens, batch size 8.

The worker's 19 focused CPU checks passed before execution. Final integrity
checks passed for 709 protected files, including all 17 model assets. Executed
manifest SHA256:
`ad58ef1210c7349da3de6cd971e2a4a34fb3e78c2a68490614b04a8e814eeec7`.

## Published XSTest measurements

| Subset | Compliance | Partial refusal | Full refusal | Any refusal |
| --- | ---: | ---: | ---: | ---: |
| Safe, 250 | 234 | 8 | 8 | 16/250 (6.4%) |
| Unsafe contrasts, 200 | 40 | 3 | 157 | 160/200 (80%) |

Separate safe/unsafe journals covered every item. All 57 grading batches passed
on their first attempts: no retries, decline-like structural failures or
infrastructure failures. The pinned, tools-off Claude grader used the registered
rubric and adaptations; its server-side model snapshot is not fully pinned.

Safe outputs had 116 EOS finishes and 134 cap hits; unsafe outputs had 180 EOS
finishes and 20 cap hits. Neither subset had empty outputs or mechanical
degeneration flags. Cap hits limit interpretation of full responses.

The safe refusals are candidates for over-refusal, not individually validated
unnecessary refusals. No contextual review of these 16 cases has run. Compared
with the 4B baseline's 17/250, the difference is one item and is descriptive only.
The unsafe compliance label denotes an attempted answer under the rubric; it
does not establish correctness, real-world harm or successful task completion.
The earlier 4B 30-item guard differs in fixture, system message and token budget,
so its refusal fraction is not a matched comparison with this 200-item subset.

## Small capability screen and cost

| Existing quality fixture | Score |
| --- | ---: |
| General MMLU | 10/16 |
| Physics MMLU | 9/16 |
| GSM8K strict / numeric fallback | 8/8 / 8/8 |
| IFEval prompt-strict / instruction-strict | 7/8 / 11/12 |

All 16 quality generations finished with EOS, with no cap, empty or mechanical
degeneration flags. These small, previously used public probes check the
pipeline; they do not establish broad capability or held-out improvement.

One model load covered eight unscored published preflight items, 48 quality
items and 450 XSTest items: 474 generations plus 32 multiple-choice likelihood
items. The model phase took 779.924 seconds (about 13 minutes), including load,
preflight and in-run checks, excluding CPU preparation, review and grading.
Peak PyTorch allocated memory was 30,806,040,064 bytes (30.806 GB), including
resident weights; this is not total VRAM usage. Existing jobs were left running,
and GPU 0 was released after the model process.

## Conclusion and artifacts

The [4B learning study is closed](qwen3_4b_closeout.md), with 250/250 exact replay
matches and no new grading. Its earlier partial steering effect and quality
tradeoff remain; no useful reliable setting was validated. The larger-model
baseline now works within one GPU, but no 14B steering or stability comparison
has run. A new intervention or persona study requires a separately reviewed
protocol; this baseline schedules no automatic follow-up.

Claude reviewed the interpretation, and the worker independently checked report
counts, journals, runtime bindings and export privacy without new model or
grading calls. Raw inputs, outputs and journals remain in the ignored
`outputs/qwen3-14b-baseline-01/` directory. No new model weights were produced,
so no Hugging Face model release is needed for this milestone.
