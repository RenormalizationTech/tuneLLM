# Persona P1: design review

Date: 2026-11-14 UTC. The user chose to finish the baseline's 16 unaccepted
ratings and then move to tone/persona work. The grading completion finished,
was reviewed by Claude and published as `8affb3b` before this implementation.

The coordinator discussed the next study with Claude. Claude proposed a
prompt-only comparison before optional activation steering: unchanged BF16
Qwen3-4B under a common accurate/calibrated system and that same text plus one
fixed direct/composed persona block. The coordinator accepted two matched
conditions, 24 development and 24 sealed held-out items, four six-item families,
one blinded paired grading pass and fixed development-to-held-out gates.

The coordinator tightened quality preservation from net change to at most one
gross paired quality loss, so gains cannot cancel deterioration on previously
acceptable answers. Claude accepted this clarification, the separate overall
and insufficient-evidence calibration gates, and null wording that means
not demonstrated rather than impossible. Claude noted the low power and
possible grading variability of this strict small-sample rule; the protocol
discloses those limits.

Claude reviewed the protocol and requested that the judge receive the frozen
rubric/persona definition as well as requests, anchors and anonymous responses.
That clarification was applied. Both agents agreed that proper uncertainty
is valid task fulfillment, requested artifact voice is not invented assistant
experience, and invented user/speaker facts remain quality failures. A common
system controls template differences, and user messages must be byte-identical
between conditions. No new CLI flags or dummy audit calls are registered.

The study uses no activation capture, hooks, refusal vectors, LoRA, training or
weight modifications. Its purpose is a prompt-level reference and feasibility
test. It cannot establish emotional states or activation control. Claude will
independently audit all 48 fixture items before they are frozen. Code and
fixture acceptance and execution outcomes are separate later checks.

See [protocol](../../docs/persona_prompt_protocol.md). The advisor's full
planning response is retained privately at
`outputs/persona-prompt-advisor-plan.md`.

Claude's first all-item audit found correct quantitative answers, valid family
counts/anchors and no illegitimate content. It identified a calibration-design
weakness: all uncertainty items named missing evidence and all pushback items
asked for a plan check. Before outputs, the coordinator accepted a fixed half
without those explicit hints in both splits, with cue groups reported only
descriptively. Conditional uncertainty estimates with stated assumptions and
generic courtesy/sign-offs were clarified to avoid inappropriate error flags.
Initial drafts remain private; revised fixtures require re-audit and re-sealing
before the snapshot or generation.

The revised audit confirmed correct counts, cue balance and unhinted tasks. It
requested final small anchor corrections in PD09 (allow any plausible unmeasured
alternative influence) and PH07 (do not assume an unstated absence of scale;
permit an explicit same-scale conditional conclusion). The invalidity rubric
was also clarified so accurate correction of a flawed method/claim is valid
pushback rather than refusal, and conditional-conclusion handling applies beyond
uncertainty estimates. These are pre-output measurement fixes; no response
informed them. Claude requested only a final hash/diff check after those edits.

Claude's final hash/diff check accepted the exact PD09/PH07 anchor changes and
the two rubric-field changes, with no further prompt or count revisions. Final
development SHA-256:
`b0537177afbdf8fd213813b250c1d9d0f81f05bdbd465422a798dfc982f89b4b`.
Held-out SHA-256:
`4799e84e2325b1f4d4b21a59973dfbb405ec2000c6df6c8fa3f07128eec96191`.
Rubric SHA-256:
`0790d1f9b77f7bc812cb7bef99aa6637fd7b59e4b125dd0ad058c65c8a36b595`.
The seal matches; draft versions remain private with restricted permissions.
Fixture/rubric acceptance precedes any model output or registered execution.

The coordinator reviewed the completed runner and requested corrections to
failure reporting, persistent generation stops and recorded GPU identity checks.
The worker applied them before the snapshot. All 25 focused CPU checks passed,
including exact prompts/fixture seals, balanced anonymous pairs, schema rejection,
gross-loss and calibration boundaries, committed held-out lock guards, source
mutation, first-valid receipt replay, exhaustion/fatal stops and export of a
mechanical-null result. Fixture checks passed with no normalized exact overlap
against the historical sources. These are code/integrity checks, not evidence
of a persona effect; no target-model or grading call has occurred at this stage.
