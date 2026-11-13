# Qwen3-4B layer-23 refusal follow-up

Status: implementation frozen; 13 new CPU checks passed and Claude accepted
the method review. No follow-up generation or response judgments have run. The user selected this single follow-up after
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
