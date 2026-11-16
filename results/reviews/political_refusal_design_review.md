# Political-refusal GCAD study: design review

Reviewer: Claude advisor, read-only. Scope: the scientific design in
[docs/political_refusal_protocol.md](../../docs/political_refusal_protocol.md)
(draft). The fixture, code and registrations will be reviewed separately. No
calls were made.

## Verdict

**Accepted as a bounded, prospective design.** Execution waits until root
accepts the fixture, meaning the sources, anchors, translations, calibration
material and frozen plans.

## Accounting checked

- **Generations:** 96 + 64 + 14 + 240 + 384 = **798**.
- **Other forwards:** 64 prompt captures, at most 96 teacher-forced forwards
  (3E with E ≤ 32), and 2 logit forwards.
- **Native grading:** 4 + 12 + 16 + 84 + 96 = **212** first-valid batches, so
  **424** starts at most.
- **Batch packing:** every count divides evenly by 8, and each batch holds 8
  different canonical questions. Development: 336 = 24 CQs × 14 variants. Held-out:
  384 = 24 CQs × 16 variants.
- **No headroom:** the run stops at 96 generations and 16 batches (calibration
  plus headroom).
- **No translation calls:** translation is editorial work done before
  registration.

## Decisions made explicit here

1. **Coverage ceiling 0.80.** This supersedes the 0.70 in my earlier plan and
   is fixed before any output.
   - With 2 anchors per question, a baseline at the ceiling (0.80) needs 0.95
     to pass the +0.15 gate. That is achievable but strict.
   - A lower baseline gives more room.
2. **Branch order is C, then NE, then E, then stop.** C is likely given the
   lab.cloud finding of few hard refusals but substantive omissions.
3. **Gates for refusal-type branches have three clauses:**
   - the branch count falls;
   - VS rises by the same threshold;
   - enough items convert, matched per item, from NE or E at baseline to VS.

   Together these stop a fall in refusals from passing when answers only move
   from refusal to evasion, empty answers or wrong answers.
4. **The per-language guard (VS in each language ≥ baseline)** stops one
   language from being sacrificed to pass the pooled gate.
5. **The bilingual invalid check uses token-ID bigrams.** The English word
   statistics from the sarcasm study would not work on Chinese.
6. **Fixed and projection are single settings that were not optimized.** If
   either fails, that setting failed. It says nothing about the method in
   general.

## Later changes accepted (before output)

- **Residual layer folds** now leave out one connected fit event or core
  group at a time: 8 folds.
- **Both eligible fit answers** must have coherence ≥ 50.
- **Seeds:**
  - GCAD random control: 9711, 9712, 9713 (+1000 × layer);
  - fixed random control: 9721;
  - projection random control: 9731;
  - bootstrap: 9741, with 10,000 resamples.
- **Fallback order:** fixed before projection.
- **Anchors:** each is one concise central fact. Numbers go in optional notes.
- **Publication:** the MIT questions and the project's anchors and
  translations may be released with attribution.

## Staged implementation: accepted

**Decision.** The staged plan is accepted. The complete scientific protocol is
frozen and registered first, covering:

- the data, anchors and calibration;
- every threshold and branch rule;
- every plan for calibration, headroom, fit, development and held-out;
- the prompts, seeds, operators, the layer rule and both ledgers.

Root then implements and registers an unchangeable calibration-and-headroom
runner. A separate fit and intervention runner is built and registered
**only if** the frozen headroom rule finds headroom.

**Why this is sound.**

- **Nothing scientific changes.** Every choice the later runner makes is
  already frozen. What is left is writing correct code for a specification
  that already exists.
- **Seeing the headroom outputs gives no room to tune.** Those outputs only
  decide which branch applies, under a rule frozen in advance. Every gate
  uses the development regrade.
- **It saves effort.** It avoids building GPU stages that a no-headroom result
  would never use.

**Conditions.**

1. **Before calibration:** the protocol, fixture, plans, seeds, prompts and
   the shared metric code are frozen and hashed. The calibration-and-headroom
   runner checks those hashes. The shared metric math and the source files
   already used then stay unchanged for every later stage.
2. **The second runner:**
   - implements the frozen specification exactly, verifies the same hashes,
     and is reviewed against this protocol before it is registered;
   - must consume, without changing them, the headroom outputs, the branch
     decision and the plans frozen for later stages.
3. **No room to adjust anything after outputs exist:**
   - no threshold, budget, prompt, plan, layer range or seed may change;
   - if implementation finds the frozen specification can't be carried out,
     that is reported as a documented amendment before any later output.
     It is never fixed silently.
4. **The ledger is enforced across both runners:** 798 generations and 212
   first-valid native batches (424 starts) in total.
5. **The report records** that the second stage's code was written after the
   headroom outputs existed, and gives its hash and review.

## Interpretation limits

- **Few groups.** The development targets have only 4 event groups and the
  held-out targets 8. The bootstrap resamples connected groups, so intervals
  will be wide and descriptive. Selection rests on effect thresholds that were
  fixed in advance, not on significance.
- **Anchors are a project reference.**
  - Two central anchors per question detect omission of core facts and will
    miss other material omissions.
  - Coverage depends on the anchor wording and on judge accuracy.
  - The same model family curated the anchors, translated the questions and
    judges the answers. Calibration reduces this risk but doesn't remove it.
- **Chinese results describe project translations of English questions.**
- **Headroom pool.** The routing-is-learned rows were written to probe Qwen
  censorship. Any headroom therefore describes this pool, not Qwen's behaviour
  in general.
- **A coverage gain is not a hard-refusal reduction when baseline FR is
  rare.** The claim labels enforce this.
- **Single checkpoint, greedy decoding, cap 512, default template.** Length
  and cap hits are reported because coverage can rise with longer answers.
  The X and coherence guards catch gains that come from padding.

## Before freeze

1. Data review items:
   - pin and check the blocked and offline sources;
   - confirm the six flagged figures;
   - C038 is replaced by C036, decided by root; disclose the lost T038
     match;
   - review the Chinese translations;
   - replace the Wikipedia navigation links.
2. Review the calibration draft: 8 questions and 16 cases, with expected
   labels kept apart from the judge payload.
3. Freeze the P+ and P− wording, the rubrics, all plans and the bootstrap seed
   and resample count, and record their hashes.
