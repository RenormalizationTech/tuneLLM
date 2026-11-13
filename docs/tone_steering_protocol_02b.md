# Tone-02B: the single registered rewrite fallback

Claude approved this concrete fallback after tone-02A's mechanical eligibility
upper bound was 31, triggering the condition registered in the
[tone-02 addendum](tone_steering_protocol_02.md). No development/final generation
or experiment AI judging has occurred. Preserve all tone-01 and tone-02A sources,
configuration, protocols and outputs. The original core pipeline remains exact.

## Construction and immutable inputs

Prepare a fresh `outputs/tone-02b` run with
`configs/tone_steering_02b.json`, the original core sources/fixture/rubric, and
this addendum. Commit a separate registration before B generation, binding the
prepared manifest, independent rewrite implementation, configuration and exact
tone-02A neutral response file.

The neutral source is `outputs/tone-02/extraction/neutral.json`, SHA256
`3805d89738b7f2eb0ee34184d60cd9edfc02cd904da70b2836c7f6c6c610c441`.
Copy its bytes unchanged as B's neutral file, retaining its original generation
metadata. Validate that all 80 scenario IDs, order and original inputs match the
prepared B fixture. Generate on physical GPU 0, the device used for those neutral
responses, while leaving existing workloads untouched.

Use pinned BF16 Qwen3-4B, greedy nonthinking decoding, seed 42, batch eight and
the same 256-token cap. Exactly one warm rewrite is generated per scenario; no
sampling, repeated attempts or selection among generations.

The fixed system is the tone-02 neutral system:

> Answer the user's request accurately in two or three sentences. Preserve the stated facts and do not invent details.

The user message uses this fixed template, with the original scenario and
neutral output substituted verbatim. Set N to the neutral output's whitespace
word count plus three.

```text
Original scenario:
{scenario}

Neutral message:
{neutral}

Rewrite this message in a warm, encouraging, considerate tone. Keep every stated fact and the same requested message; add no facts, flattery or extra claims. Use at most {N} words.
Return only the rewritten message.
```

Record N, actual word count/compliance, rendered prompt, input/generated token
IDs and normal generation diagnostics for each item. The word budget is an
instruction with reported compliance, **not an additional hard eligibility
gate**. This clarifies tone-02's earlier word "limiting" before B generation;
the original 0.85–1.25 token window remains the hard length criterion.

## Unchanged analysis and stop rule

The hard eligibility rules, 50-pair requirement and fixed round-robin order are
unchanged. Compute the mechanical upper bound before judging. If fewer than 50
can qualify, stop tone steering. If mechanically feasible, only use separately
reviewed, hash-registered fresh-context judging after the final helper passes its
context audit. If fewer than 50 then pass all AI/content/warmth checks, stop.
No further tone construction attempts are authorized by this pilot.

Warm members are conditioned on their neutral partner; disclose this asymmetry.
For extraction, re-encode both under the same tone-02 neutral context. The
development/final default and warm-prompt baseline remain exactly the tone-02
systems. Use the original gates, timing-budget registration, development-only
selection and final lock with unchanged BF16-to-NF4 transfer.

A stop is a contrast-construction outcome, not evidence that steering cannot
work. Consult Claude before the separately registered refusal phase. No model
weights are trained or changed by this fallback.
