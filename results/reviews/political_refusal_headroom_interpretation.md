# Political-refusal headroom: interpretation and scope recommendation

Reviewer: Claude advisor, read-only. This reads only the aggregate results in
`results/political_refusal_recovery_headroom_01.json`. I made no calls, wrote
no code and changed nothing frozen. Nothing here is evidence of GCAD success
or failure, because no steering output exists.

## What headroom shows

The frozen mode is **C (coverage)**. These figures are on the 32 development
target items (16 canonical questions × EN/ZH).

| | Baseline | Prompted (P+), reference only |
| --- | --- | --- |
| FR / PR / EV / S | 0 / 0 / 3 / 29 | 0 / 0 / 0 / 31 (+1 invalid) |
| Coverage (all items) | .516 | .625 (+.109) |
| Coverage, EN / ZH | .625 / .406 | .562 / .688 |
| Valid substantive, EN / ZH | 13 / 8 | 12 / 13 |
| Contradicted facts | 4 | 4 |
| Control coverage | .688 | .625 |
| EN cap hits; mean tokens EN / ZH | 3 of 24; 381 / 270 | 10 of 24; 450 / 359 |

**Reading:**

- **There is no explicit refusal in this pool.** FR + PR = 0, and there are
  only 3 evasions. Hard or explicit refusal reduction **cannot be measured
  here**.
- **The measurable phenotype is omission of the core facts.** It is
  concentrated in **Chinese** (ZH coverage .41 against EN .63).
- **Coverage depends on wording.** The P+ instruction raised ZH coverage
  sharply, but EN coverage and the controls fell. The EN answers also got
  longer and hit the cap more often. Coherence is not graded yet.
- These are headroom and reference descriptions only. They don't predict
  steering.

## Is this a genuine scope fork that warrants the user's input?

**Yes.** This is not a request for approval of something already decided.

1. **The user's stated central goal** is to test GCAD refusal reduction on
   7B, finish quickly, and then run the fixed 4B control. On this pool the C
   branch can test only **information-access and omission** (coverage,
   mainly in Chinese). That is a different construct, and the protocol's
   claim labels already forbid calling it refusal reduction.
2. **The remaining cost is substantial and uneven:**
   - up to ~702 generations and ~196 first-valid native batches;
   - most of it is **building and reviewing the fit and intervention runner**
     (the GCAD, fixed and projection hooks, plus mechanics). That is the slow
     part, not GPU time.
3. **The user's original brief included omission and framing as phenotypes,
   so option A is within the requested scope.** Whether it is worth the time
   is a priority judgement only the user can make.

Stopping now is **neutral with respect to GCAD selection**, because no
steering output exists. If the user picks B, record it as "C branch
available, not executed: a scope decision made before any steering output."
It is not a null and not a failure.

## Options to put to the user (short)

- **(A) Continue the registered C study, labelled as Chinese-language
  information access and omission.**
  - It adds a fourth individual control target.
  - It does not answer hard refusal reduction.
  - Cost: implementing the runner plus about 700 generations.
- **(B) Close 7B explicit-refusal reduction as not measurable on this pool,
  and move to the 4B control and finalization.**
  - The fastest route to the user's stated endpoint.
  - The refusal row in the release says: "7B explicit refusal: no headroom
    on XSTest-safe (3/100) or on the political pool (FR + PR 0/32). GCAD
    refusal reduction not tested."
- **(C) Choose a different published benign over-refusal pool** that elicits
  explicit refusals from Qwen2.5-7B, for example an OR-Bench-Hard sample.
  This needs a new headroom-first protocol and adds the most time.

**Advisor's lean:** given the user's emphasis on finishing 7B quickly and on
refusal specifically, **B**. Choose A only if the user values the
information-access and omission target as a deliverable in its own right.
Choose C only if hard refusal on 7B is essential and the extra time is
acceptable.

Root should put this as one short question to the user, with these three
options and the cost of each. No fit or steering work should start before the
answer.

## Whatever is chosen

- The headroom result and the mode C decision stay frozen and are published
  as recorded.
- No claim about GCAD, fixed or projection is made from headroom.
- The protocol's claim labels apply. If A runs, any gain is a "political-
  reference coverage gain", and the report must note that baseline hard
  refusal was 0.
