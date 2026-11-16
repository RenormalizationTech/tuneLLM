# Sarcasm GCAD lower-strength study: scientific result review

Reviewer: Claude advisor, read-only. Source: `results/sarcasm_gcad_low_strength_01.json`
(status `complete`, descriptive status `complete`). I checked the reported
figures and gate arithmetic against the result file. The worker audits
execution integrity separately. I made no calls.

## Verdict

**A narrow, bounded pass.** GCAD at base coefficient 2 (blocks 9–19, scale 1.5,
the parent's sarcasm bank) is retained as the sarcasm control for
Qwen2.5-7B-Instruct ordinary replies, at greedy decoding and cap 160. It comes
with a substantial coherence cost; this does not show that quality is solved.
The parent study's development null for {3.5, 5, 7} is unchanged.

## Development (re-graded together in new batches)

| Arm | Sarcasm | ≥50 /32 | Coherence | Gate result |
| --- | ---: | ---: | ---: | --- |
| Baseline | 0.0 | 0 | 85.75 | — |
| Prompted (reference) | 61.66 | 27 | 76.44 | — |
| GCAD 1 | 0.16 | 0 | 86.06 | Fails sarcasm gain and ≥50 count |
| **GCAD 2** | 49.75 | 16 | 74.69 | **Passes all; lowest qualifier** |
| GCAD 3 | 83.41 | 32 | 61.31 | Fails relative coherence (< 65.75) |

There is a large observed step from coefficient 1 to 2: sarcasm goes from
about 0 to 50, then to 83 at coefficient 3. No finer grid was tested, so the
shape of the curve between these points is unknown.

## Held-out (primary decision)

The 32 held-out questions are reused project material, held out only for this
sarcasm selection. The answers are freshly generated.

| Arm | Sarcasm | ≥50 /32 | Coherence | Invalid | Capped |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.06 | 0 | 89.59 | 0 | 32 |
| Prompted (reference) | 66.47 | 29 | 77.13 | 0 | 10 |
| **Selected GCAD 2** | **63.78** | **29** | **70.72** | 0 | 27 |
| Random control | 0.0 | 0 | 87.72 | 0 | 32 |

**Gates, all passing:**

- Sarcasm 63.78 ≥ baseline + 20 (20.06) and ≥ random + 20 (20.0).
- 29 of 32 answers at 50 or above, against a requirement of 8.
- Coherence 70.72 ≥ 50, and ≥ baseline − 20 (69.59): **a margin of 1.13
  points.**
- 0 invalid answers. Profanity lexicon hits were 0 in every arm.

## H14 sensitivity (descriptive; primary decision unchanged)

Excluding H14, which overlaps topically with extraction question x18, the
descriptive means and counts are nearly unchanged: selected GCAD 2 has sarcasm
63.84, coherence 70.68 and 28 of 31 answers at 50 or above; baseline coherence
is 89.48. No gate is recomputed on the 31 items, and the primary 32-question
decision is unchanged. Full semantic independence between the extraction and
held-out questions is not established.

## Strengths

- A clear confirmatory effect on held-out questions not used for selection:
  sarcasm close to the prompted reference (96% of the prompted gain), with
  29 of 32 answers clearly sarcastic.
- **Specificity against the random control.** The random delta, with the same
  sites, gate formula and per-layer raw delta norms, produced zero sarcasm and
  near-baseline coherence.
- The selection was prospective, with unchanged gates, no reselection on
  held-out results, and the parent null kept intact.

## Limits

1. **The quality margin is narrow.** Coherence fell 18.9 points against
   baseline, 1.13 points inside the gate. The prompted reference lost 12.5
   points, so GCAD costs more coherence than prompting at similar sarcasm.
   Coherence is not correctness or helpfulness; neither was measured.
2. **A large step between tested strengths.** Sarcasm rose from about 0 at
   coefficient 1 to about 50 at 2 in development, and no finer grid was
   tested. The selected strength also differed between development (49.8) and
   held-out (63.8). Robustness across other prompt sets, system messages or
   sampling is not established.
3. **The random control matched the raw delta, not the realized energy.** It
   matches the selected setting's per-layer raw delta norms and gate formula
   exactly. Its realized per-position norm was 3.10 against 4.04, about 23%
   lower, because the gate depends on the activations. Realized energy was not
   matched.
4. **Truncation.** 27 of 32 selected answers hit the 160-token cap, as did all
   baseline and random answers. Coherence was judged on truncated text.
5. **Impoliteness is coupled (descriptive only).** Mean impoliteness: baseline
   1.6, selected 38.8, random 1.7, prompted 49.9. The correlation between
   sarcasm and impoliteness within the selected arm is r = 0.64. The pooled
   r = 0.93 is confounded by differences between conditions and is not
   evidence of a shared mechanism. GCAD sarcasm was descriptively less impolite
   than prompted sarcasm at similar intensity. This is an isolated-trait study
   in which coupling is allowed. The control is not established as independent
   of, or disentangled from, other traits, and independence was never a goal
   or a gate.
6. **Validity:** one Claude judge family, which also generated and reviewed
   the artifact; a reconstructed artifact; 32 ordinary questions; one
   checkpoint. This is a method transfer, not a reproduction of the
   original-Mistral SAPS study.

## Relationship to other results

- The parent's development null for {3.5, 5, 7} stays a null for its grid.
- The original-Mistral SAPS sarcasm reference remains a separate,
  model-specific result.
- The refusal headroom check (semantic count 3 of 100; secondary prefix
  count 5) closed separately. It neither tests nor rules out GCAD refusal
  reduction in general.

## Cost

230 target generations, 2 logit forwards, and 88 first-valid native batches
using 88 starts. No extraction, teacher-forced or MC work.
