# Profanity reference-grid closeout

The isolated profanity grid closed as `selection_null`. All 21 quality-eligible
settings scored zero on friendly, helpful swearing, as did baseline. The single
nonzero mean (0.1 at author layer 20/block 19, coefficient 2) failed coherence
(59.5). No setting qualified for held-out, random/prompt, custom or capability
testing; ten report questions remain ungenerated.

This applies the Persona Vectors general trait procedure to newly generated
profanity artifacts, not released author profanity data or a numerical reproduction.
Root and Claude accepted the first valid artifact set before promotion and model
elicitation; all raw provenance remains retained. Thirty filtered pairs spanning
18 questions supplied all 60 native response trajectories.

Each cell is **friendly-swearing / coherence**, on 0–100 scales over ten fixed
development questions. A dagger marks a quality failure.

| Author layer (block) | Coefficient 0.5 | Coefficient 1 | Coefficient 2 |
| --- | ---: | ---: | ---: |
| 4 (3) | 0.0 / 77.0 | 0.0 / 79.6 | 0.0 / 78.2 |
| 8 (7) | 0.0 / 81.2 | 0.0 / 74.6 † | 0.0 / 78.5 |
| 12 (11) | 0.0 / 80.4 | 0.0 / 79.7 | 0.0 / 76.9 |
| 16 (15) | 0.0 / 80.4 | 0.0 / 79.4 | 0.0 / 69.3 † |
| 20 (19) | 0.0 / 76.7 | 0.0 / 76.5 | 0.1 / 59.5 † |
| 24 (23) | 0.0 / 78.4 | 0.0 / 78.5 | 0.0 / 66.8 † |
| 28 (27) | 0.0 / 77.7 | 0.0 / 77.0 | 0.0 / 70.5 † |
| 32 (31) | 0.0 / 77.8 | 0.0 / 79.5 | 0.0 / 74.9 † |
| 35 (34) | 0.0 / 78.5 | 0.0 / 79.6 | 0.0 / 76.5 |

Baseline coherence was 81.7; nine of ten answers reached cap 512. Capped counts
are therefore weak quality checks here. The registered coherence floor and
empty/degenerate checks remained unchanged.

The fixed eleven-word lexical diagnostic found 0 matching responses among all 280
grid outputs, versus 85/100 positive and 0/100 negative elicitation responses.
This descriptive counter neither changes selection nor covers every form of
profanity; word occurrence and helpful swearing are different constructs.

Cost: parent 210 generations/60 teacher-forced responses/25 native calls;
child 280 generations/zero TF/35 calls. Combined 490 generations/60 TF/60 calls,
zero MC; artifact generation was one separate native call. All grading batches
were first-attempt valid. Grid generation took 2,041.34 seconds, whole stage
2,046.91 seconds. The parent was prospectively handed off before steering
and remains amended_before_steering, with its original selection unrun.

Limits: nine sites and three coarse strengths, ten development questions, greedy
decoding and one AI judge do not establish impossibility or general validity.
The friendly-swearing rubric is narrower than lexical profanity; co-rated quality
may have unknown dependence on style. Frequent truncation limits interpretation.
No effective isolated control or cross-trait independence is claimed. Combined
control remains deferred; further search requires a new scientific decision.

The worker independently audited exact scores, integer selection, first-valid
receipts/model identity, provenance, lexical coverage, privacy, costs and skipped
stages: PASS. Cap flags were genuine (no PAD hits): baseline 9/10 and 25 settings
10/10. One generic sentence in the generated aggregate's limits says historical
parent null; that wording applies to the rudeness profile only. This profanity
parent is amended_before_steering, as correctly recorded in its status field.
The same Claude foundation model generated the new trait artifacts and graded
answers, creating generator/judge circularity; independent human validity has
not been established.

Claude reviewed this closeout. At coefficient 2, shift/per-token-residual norm
ratios were approximately 0.12–0.27, versus 0.24–0.48 for rudeness. Raw-vector
coefficients therefore do not match perturbation size across traits. These
results cannot distinguish an insufficient tested magnitude, suppression by the
model, or an ineffective direction; no larger-magnitude claim follows. Profanity
also had fewer retained pairs (30 versus 66) and modest overall positive
elicitation (trait 38.72, coherence 74.38), limiting extraction evidence.
The 85/100 prompted lexical observation uses extraction questions and explicit
systems; the 0/280 grid observation uses selection questions without those
systems. They are context, not a matched prompt-versus-steering comparison.
No prompted control on selection/report questions was generated.

See the [aggregate](../results/persona_pv_grid_profanity_01.json),
[lexical diagnostic](../results/diagnostics/persona_pv_grid_profanity_lexical_01.json),
[grid protocol](persona_pv_grid_protocol.md) and
[artifact protocol](profanity_reference_protocol.md).
