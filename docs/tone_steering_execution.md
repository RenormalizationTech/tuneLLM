# Written-tone steering execution

## First extraction attempt: insufficient matched pairs

The registered `tone-01` attempt generated neutral and warm responses for 80
synthetic everyday scenarios using unadapted BF16 Qwen3-4B. Generation completed
on the user-authorized physical GPU 0 in 56.36 seconds including model loading,
with 8,268,976,128 bytes peak PyTorch allocation. Existing jobs were untouched.
See the [registration](../results/protocols/tone-01-registration.json) and
[feasibility result](../results/tone_01_feasibility.json).

The warm system prompt mostly produced longer responses: mean lengths were
16.675 neutral and 36.9625 warm tokens, and the median per-pair length ratio was
2.457. Only five pairs met the registered 0.85–1.25 length window; all five also
passed the other mechanical filters. Seventy-two pairs passed both responses'
literal fact checks and all 80 were nonempty and uncapped.

Full eligibility also requires AI content/task-completion approval and a strict
warmth preference. These additional checks can only remove pairs, so no more
than five could qualify against the required 50. The coordinator stopped before
experiment judging, vector extraction, development selection or final
generation. It would be wasteful to judge all pairs when the mechanical upper
bound already makes the registered requirement impossible.

This is a failed **contrast-pair construction** attempt, not a negative result
about activation steering. No vector was evaluated and no weights were trained
or changed. The frozen source/configuration/fixture and original protocol remain
intact. Raw responses are local under `outputs/tone-01/extraction/`; only compact
measurements and hashes are tracked in Git.

The coordinator consulted Claude and registered a separate length-controlled
follow-up before generation, preserving the first result and all original
eligibility rules.

## Symmetric sentence format: fallback triggered

Tone-02A requested two or three sentences under both systems, with the warmth
instruction added only to the warm system. It reused the same 80 extraction
scenarios, model, decoding and eligibility rules. See its
[registration](../results/protocols/tone-02-registration.json),
[addendum](tone_steering_protocol_02.md) and
[mechanical result](../results/tone_02a_feasibility.json).

The 160 responses completed in 54.42 seconds including model load. Mean lengths
were 30.3125 neutral and 37.425 warm tokens, with median pair ratio 1.22739.
Thirty-five pairs passed the length window and 76 the literal checks; their
intersection left a mechanical upper bound of 31, below the required 50.
All 80 pairs were nonempty and uncapped. Again, no experiment AI judging,
steering-vector extraction, development selection or final generation occurred.

This triggered the single conditional rewrite fallback registered before A's
outputs existed. Claude reviewed the concrete
[B construction](tone_steering_protocol_02b.md): reuse A's exact neutral bytes
and generate one constrained warm rewrite per scenario, under the unchanged
hard token window and content checks. B's word budget is a construction
instruction with measured compliance, not an additional eligibility gate.
If B cannot provide 50 eligible pairs, the tone pilot stops; there is no further
construction attempt in this protocol.

## Rewrite fallback: tone construction stopped

The [registered B fallback](../results/protocols/tone-02b-registration.json)
copied A's neutral responses exactly and generated 80 warm rewrites in 22.46
seconds including model load, on the same GPU 0. See the
[B result](../results/tone_02b_feasibility.json).

All 80 followed the maximum-word instruction, but 31 were shorter than 0.85
times their neutral partner; none exceeded 1.25. Forty-nine passed the token
window and 76 the literal checks, leaving a mechanical upper bound of 46.
Mean warm length was 26.225 tokens versus the reused neutral mean of 30.3125;
median pair ratio was 0.880476. All outputs were nonempty and uncapped.

| Construction | Mechanical upper bound /80 | Median length ratio | Outcome |
| --- | ---: | ---: | --- |
| Tone-01, concise systems | 5 | 2.457 | Stopped before judging |
| Tone-02A, symmetric sentence format | 31 | 1.227 | Registered fallback triggered |
| Tone-02B, one warm rewrite | 46 | 0.880 | Registered tone construction stopped |

The 50-pair threshold was a declared design choice, not a universal scientific
minimum. We preserve its stop rule for this pilot; a later smaller-set study
would require its own protocol and could not turn this outcome into a success.
There were no experiment AI ratings, steering vectors, development selection,
final generations or NF4 transfer. Tiny CPU hook tests validate implementation
mechanics and do not demonstrate an effect on the 4B model.

Across the three constructions, 400 new responses were generated and 80 neutral
responses reused. Measured generation-stage wall times total about 133 seconds,
including each model load and excluding preparation, coding, review and any
judging. All existing jobs remained untouched, and the project released GPU 0.
The coordinator has consulted Claude about the next phase; no additional tone
construction is running.
