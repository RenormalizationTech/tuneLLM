# Tone-02: bounded length-controlled contrast construction

Reviewed with Claude after the registered tone-01 feasibility failure, before
any development or final generation. The original
[protocol](tone_steering_protocol.md), frozen at commit `7c0bc1f` with SHA256
`9635e862d8c826c418543aa0c98c7352f2a1c0286e5900ce53ee72447dab518c`, continues
to define data separation, extraction, judging, gates, selection and final
evaluation. This addendum changes only the contrast construction and shared
response format. Tone-01 sources, artifacts and result remain unchanged.

## Attempt A: symmetric sentence budget

Use a fresh `outputs/tone-02` preparation/manifest and registration with
`configs/tone_steering_02.json`. Reuse the same pinned BF16/NF4 dependencies,
synthetic fixture, 80 round-robin extraction candidates, rubric, core sources,
GPU 0 and seed 42. All sources/configuration/addendum are committed before
generation.

Neutral system prompt:

> Answer the user's request accurately in two or three sentences. Preserve the stated facts and do not invent details.

Warm system prompt: the exact neutral text followed by:

> Use a warm, encouraging, considerate written tone within the same two or three sentences, without flattery or extra factual claims.

Generate one greedy response under each system for every extraction scenario.
The same new neutral system becomes the default and common extraction context
for development/final comparisons; the new warm system is their prompting
baseline. No development/final output was used to choose this change.

Keep the original 0.85–1.25 token-length window and 50-pair requirement. Perform
the mechanical feasibility audit before AI judging. If at least 50 pairs remain
mechanically possible, judge the frozen pairs and follow the original selection
pipeline. Fewer than 50 AI-approved eligible pairs still stops that attempt;
do not silently relax a gate or cherry-pick generations.

## One conditional fallback B

Only if A's mechanical eligibility upper bound is below 50, allow one
separately sealed rewrite construction. Retain A's neutral responses. Generate
one warm rewrite per scenario, with Qwen3-4B BF16, preserving the same facts and
requested task and limiting the rewrite to at most the neutral response's
whitespace-word count plus three. Still require the original token-length
window, content/task checks and strict warmth preference; a word budget does
not replace these gates.

Before B generation, commit its exact rewrite prompt, new implementation hash,
input response hashes, configuration and manifest in a separate registration.
The original core sources remain unchanged. Each scenario gets one attempt;
never select among multiple valid generations. Re-encode both retained responses
under the same new neutral context for extraction. The neutral response is now
an input to warm construction, so disclose that asymmetry.

If B cannot provide 50 eligible pairs, report the failure and stop tone steering.
Do not create a third construction or change the acceptance window. Consult
Claude before proceeding to the separately registered refusal phase.

## Reporting

Report the tone-01 failure, the reason for this format change, filter attrition
per family and any subsequent AI rejection. The experiment targets warmth under
a fixed sentence budget, which is narrower than unconstrained warm writing.
Keep final-test outcomes out of construction, timing and checkpoint selection.
This addendum does not register new LoRA training or authorize altered weights.
