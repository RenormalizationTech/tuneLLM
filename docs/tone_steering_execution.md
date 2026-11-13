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

The coordinator has asked Claude to review a separately registered
length-controlled follow-up. Any new attempt must preserve the first result,
declare its construction before generation and keep final outcomes out of
calibration. No follow-up result is claimed here.
