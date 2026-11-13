# Benign helpfulness Phase A: design review

Date: 2026-11-13. The user authorized autonomous continued work. The coordinator
discussed this new scope with the existing Claude scientific advisor before
assigning implementation to the dedicated Codex worker.

The coordinator limited the scope to legitimate requests, benign over-refusal
and response quality. No optimization of harmful-request compliance, activation
intervention, reuse of the earlier source-class vector or weight change is
part of this study. The original failed steering configuration and all earlier
results remain unchanged. This scope change was stated to the user before work.

Claude recommended a baseline-first gate because prior benign refusal rates
were low. The coordinator adopted a 36-item baseline survey instead of
immediately registering a three-condition experiment. Claude accepted this
scope with refinements: seal a separate held-out fixture before outputs; define
refusal separately from incorrectness and unnecessary caveats; exclude invalid
outputs from the headroom numerator; and check all 24 sensitive-request refusal
labels with primary judge labels frozen. The coordinator accepted these changes.

The fixed protocol is [benign_helpfulness_protocol.md](../../docs/benign_helpfulness_protocol.md).
It requires at least 33/36 EOS finishes, then at least four valid partial/full
refusals among 24 sensitive requests to indicate readiness for a separately
registered comparison. It registers no automatic candidate or held-out run.
The earlier final set cannot become an unseen test again.

The study is AI-authored and AI-reviewed, with no claimed human assessment.
It cannot establish behavior on harmful requests, preserved safety boundaries,
PRL writing quality or population-level over-refusal rates. Physics anchors
measure supplied content and constraints. A floor result is an expected possible
outcome, not a reason to replace the fixture after inspecting responses.

The advisor's initial all-item audit found no illegitimate request and checked
the basic physics arithmetic. It requested a chronology/causation correction
in held-out item HS09, word-number spacing corrections, simpler anchors and
more salient benign-sensitive cues. The coordinator selected that revision
instead of accepting a low-trigger fixture whose likely floor would be less
informative. All corrections and re-sealing precede generation; no model output
guided them.

Claude re-audited all 72 revised items and accepted legitimacy, salient benign
cues, family counts, spacing, three checkable anchors per item, the chronology
fix and physics results. Development SHA-256:
`3570e0ced65b3fdd8077f962800c3ae42ba8be011009c222ad8eafa8d4594768`.
Held-out SHA-256:
`bc5c8ee16159f6889f73ecbaebbbe5ce5da0a04c32c1008bc3b7fd8dd067eecf`.
Many prompts explicitly state benign intent or exclude harmful instructions;
this may reduce refusal headroom. A floor result cannot generalize to all
ambiguous sensitive requests. This limitation is accepted before generation.

Fixture/code acceptance and execution results will be recorded separately
after their checks. This design record alone does not authorize unreviewed
model execution.
