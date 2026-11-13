# Qwen3-4B layer-23 refusal follow-up

Status: completed with no qualifying setting. The layer-23 intervention raised
benign refusals as dose increased, but the strongest setting failed the fixed
quality gate and weaker settings missed the refusal threshold. The committed
null decision skipped final generation and closes this refusal pilot. The user selected this single follow-up after
Claude accepted the completed layer-17 null. The fixed design is in
[the separate protocol](refusal_late_protocol.md).

This is an exploratory follow-up motivated by an already-seen layer-23
development projection AUC of 0.974. It tests layer 23 only at raw strengths
0.25, 0.5 and 1. The original layer-17 null remains unchanged; this is not
a continuation of its search or a replacement result. Development data and
prior outcomes have been seen. No final response has been generated or judged.

The model, prepared data, greedy nonthinking generation, seed 42, batch 8,
cap 320 and physical GPU 0 are unchanged. The saved layer-23 vector is reused.
Development defaults and their existing AI ratings are reused exactly and
bound by hash; only new candidate responses are judged. This avoids repeated
generation and classification of unchanged defaults. The original baseline
labels came from a different packet context, which limits comparability.

New layer-23 zero and nonzero probes must pass before the 90 candidate
development responses. A committed decision precedes the optional 228-response
final comparison. A null closes this refusal pilot. No negative steering,
training, NF4 transfer, public vector release or new model-weight publication
is included.

AI rubric classifications measure a harmful-minus-benign prompt-class
intervention, not a pure refusal feature or a safety improvement. Usefulness
ratings are indicative because responses are frequently truncated and steering
can change length; cap counts must accompany them.

The exact source/config/input snapshot and baseline reuse are registered in
`results/protocols/refusal-run-02.json`. The unchanged audited judge path
passed binding/version checks; no extra dummy model call was made. A new
anonymous-ID seed 20261115 avoids reusing ID strings across studies, while
the data seed and all prepared prompts remain unchanged.

The source and run registration were committed in `057d7d3` before the probes.
The actual layer-23 zero-vector hook reproduced baseline tokens exactly. The
nonzero 0.25 probe recorded prefill/decode hooks, maximum hidden-state delta
1.640625 and last-prompt logit delta 4.4785. Both eight-response probes hit the
320-token cap on every item. The exact mechanics/budget report was committed
in `414832c` before development generation; its rough remaining projection
was 636.3 seconds, excluding loading and judging.

## Development result and stopping decision

| Safe development condition | Primary refusals / 30 | Useful and correct / 30 | Cap hits / 30 | Degenerate / 30 |
| --- | ---: | ---: | ---: | ---: |
| Reused baseline | 1 | 11 | 27 | 0 |
| Strength 0.25 | 4 | 10 | 25 | 0 |
| Strength 0.5 | 7 | 7 | 21 | 0 |
| Strength 1 | 24 | 0 | 5 | 6 |

The primary numerator counts valid partial and full refusals, excluding
mechanically degenerate responses on the fixed denominator of 30. There were
no missing, empty or AI-flagged unscorable responses. Strength 0.25 had two
full and two partial refusals; strength 0.5 had seven full refusals; strength
1 had 23 full and one partial refusal among its 24 valid outputs. Six further
outputs at strength 1 were mechanically degenerate and excluded.

No setting met both the refusal target (at least 15/30 and six above baseline)
and quality gates (no empty outputs and no more degeneration than baseline).
Strength 1 met the refusal target but failed the quality gate; the other
strengths passed quality but missed the refusal target. The locked null in
`dd88d1d` skipped all 228 final responses. No final-set or steered unsafe
score is reported. No further layer or strength search follows this result.

The dose-related increase is a descriptive exploratory signal, not evidence
of no effect. It suggests a refusal-versus-quality tradeoff under this
prompt-class intervention. It does not establish a usable setting or a pure
refusal feature. Development was already seen and there is no final
confirmation. Usefulness fell as dose increased, but frequent and unequal
truncation limits interpretation. The secondary prefix matcher disagreed with
AI on 2/30, 4/30 and 15/24 valid candidate outputs, respectively, and never
overrode the primary measure.

## Judging, reuse and cost

All 90 new candidate items were classified in five fresh-context batches by
the unchanged registered Claude adapter. Every batch was valid on its first
attempt. The 60 baseline outputs and their 60 existing labels were reused,
with no new baseline generation or judgment. Reused baseline labels came from
the original mixed packet; new candidate labels came from a separate packet.
The baseline unsafe count of 28/30 is reused descriptive context, not a
follow-up intervention result.

New generation consisted of 16 hook-probe responses plus 90 development
responses, for 106 total. GPU-stage wall time including loading was
38.24 seconds for the probes and 192.34 seconds for development, totaling
230.58 seconds. Peak PyTorch allocation was 8.52 GB. These costs exclude
coding, CPU preparation, review and Claude judging. All jobs used the same
physical GPU 0; existing workloads were left running.

The full compact result is [`results/refusal_02.json`](../results/refusal_02.json),
with exact reuse/source hashes and the committed decision in
[`results/protocols/refusal-selection-02.json`](../results/protocols/refusal-selection-02.json).
Raw follow-up artifacts are ignored under `outputs/refusal-02/` and
`outputs/refusal-judgments-dev-02/`. No new model weights were created.
