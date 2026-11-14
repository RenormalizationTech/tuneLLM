# XSTest-safe baseline 01: execution and result

Status: complete, 2026-11-14. This is an adapted reproduction of a published
over-refusal evaluation, not a reproduction of an intervention or a published
Qwen3 score. The [protocol](xstest_baseline_protocol.md),
[literature audit](refusal_reproducibility_review.md) and
[registration](../results/protocols/xstest-baseline-01.json) were committed in
`c5ef01c` before generation. The [compact result](../results/xstest_baseline_01.json)
contains item IDs/counts/provenance, without raw answers or grader reasons.

## Implementation and provenance

The official pinned XSTest CSV matches the cached source byte for byte. All 250
safe prompts were used in original source order, ten types with 25 prompts each.
Seventy-two prompts had been generated in previous studies; this public rerun
is not unseen held-out evidence. No unsafe prompts were evaluated.

The model was the unadapted pinned BF16 Qwen3-4B, with the helpful system message,
native nonthinking template, greedy seed 42, batch eight and 256-token budget.
No adapter, hook, vector extraction, weight modification or training was used.
The working environment used Python 3.10.12, Torch 2.6.0+cu124 and Transformers
4.57.6; the prepared manifest records the other package versions and model hashes.

The existing Claude advisor accepted the design and corrected code. Twenty-one
CPU checks passed, including actual-tokenizer formatting and agreement with the
pinned pure upstream metric/string matcher on synthetic fixtures. The measured
metric is our equivalent implementation; the original metric function ran only
in tests. See the [code review](../results/reviews/xstest_baseline_code_review.md).

The prepared manifest binds 23 snapshot files, 14 sources and 478 external
inputs, including all 421 protected earlier inputs and the older 402/315 chains.
Manifest SHA-256:
`ad8d1ccc729564a7fec7115fb86b0273c0c50e14a7b3c005c5e464487d513fba`.
All required pre/post-run integrity checks passed.

## Results

| Measurement | Count |
| --- | ---: |
| Full compliance | 233/250 |
| Partial refusal | 7/250 |
| Full refusal | 10/250 |
| Partial or full refusal | 17/250 (6.8%) |
| Empty/whitespace-only outputs | 0 |
| Mechanical degeneration flags | 0 |
| EOS-finished responses | 96/250 |
| Token-cap hits | 154/250 |
| Original string-match refusals | 11/250 |

The primary judge was the audited tools-off, isolated Claude CLI, with actual
model identifiers and executable hashes preserved. All 32 batches passed on
their first attempt: 32 calls, no retries, no missing labels or repair calls.
Published definitions and C/P/I mapping were retained; local HF generation,
Claude instead of GPT-4o, batched JSON grades and brief reasons are disclosed
adaptations. There is no dated server-model snapshot or human validation.

| Prompt type with primary refusals | Partial | Full | Total /25 |
| --- | ---: | ---: | ---: |
| Safe contexts | 1 | 6 | 7 |
| Figurative language | 2 | 3 | 5 |
| Fictional privacy | 1 | 1 | 2 |
| Safe targets | 2 | 0 | 2 |
| Homonyms | 1 | 0 | 1 |

The remaining five types had zero primary refusals. Fifteen of the 17 candidate
refusal responses ended with EOS: all ten full refusals and five partial
refusals. The other two partial refusals hit the cap. Truncation alone therefore
does not explain the observed refusal count; the many capped answers still
limit interpretation of completed answer content and possible late behavior.

The primary judge and lexical matcher disagreed on ten items: eight primary-only
refusals and two matcher-only refusals, with nine common refusals. Neither method
is established as ground truth by this comparison. Compliance does not imply
accuracy, relevance or successful task completion.

## Decision and practical limits

The fixed decision is `candidate_unnecessary_refusals`: 17 nonempty refusals,
with at least two refusals in at least two prompt types, passed the practical
headroom flag. It is not proof that every benchmark-safe prompt is appropriate,
that all AI labels are correct, that failures reproduce, or that a useful
steering configuration exists. Earlier custom-survey results remain unchanged;
the benchmarks have different inputs and generation settings.

Generation including model load took 293.16 seconds; synchronized generation
alone took 287.92 seconds. Peak PyTorch allocated memory was 8.447 GB decimal
(about 7.87 GiB). GPU 0 returned to 48,053 MiB free and zero utilization afterward;
existing workloads were left running. Integrity checks, preparation, reviews
and grading are excluded from those generation times.

The baseline is closed under its rule. A later legitimacy, label and
reproducibility diagnosis needs its own reviewed scope and must preserve these
labels. No outcome automatically triggers new generation, higher-cap reruns,
OR-Bench, contrast-pair replacement or steering. Physics LoRA and simultaneous
profanity/rudeness/sarcasm steering remain separate tracks.

Raw prompts/answers, exact token inputs, grader payloads/reasons and attempt
receipts are local under ignored `outputs/xstest-baseline-01/`. Pinned public
reference files are under ignored `outputs/refusal-literature-root-audit-01/`.
