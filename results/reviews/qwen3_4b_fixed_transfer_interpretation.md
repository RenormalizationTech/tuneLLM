# Qwen3-4B fixed transfer: interpretation review

Reviewer: Claude advisor, read-only. I read only
`results/qwen3_4b_fixed_transfer_01.json`. I made no model or judge calls and
changed no frozen file.

## Integrity

- **Status:** `complete`.
- **Budgets used:**
  - generations 952 of 952;
  - teacher-forced passes 513 (3 × (45 + 63 + 63)) of at most 900;
  - logit forwards 5 of 5;
  - grading: 252 of 252 batches valid first time from 252 starts, so no
    retries.
- **Gates:** the decisions come from the frozen gate keys, and the margins in
  the aggregate match recomputation from the condition means.
- **No rule was applied after the outcomes.**

## Interpretation

1. **The fixed 7B settings transferred as steering. They did not transfer as
   quality-qualified controllers.**
   - Each GCAD arm shows a large trait effect: PF +37.3, SARC +56.1 with
     22/32 ≥ 50, and impolite +78.9 numeric.
   - Each random-Δ arm stays at baseline (0, 0 and 1.4).
   - Mechanics passed for all three banks.

   So the null concerns **quality at a fixed setting**, not the absence of
   an effect.
2. **The quality failures differ in severity.**
   - **PF** collapsed: coherence 20.6 and 6/32 mechanically invalid outputs.
     At b = 5, this bank and window produced a severe quality loss. Neither
     the dose response nor the cause was tested.
   - **SARC and impolite** failed only the gate allowing a coherence loss of
     at most 20 points, by **3.1** and **5.3** points. These are narrow
     misses. They are still failures, and the 7B SARC pass itself had only a
     1.1-point margin.
3. **Impolite.** The author-effect rule, the unchanged 7B rule, passed. The
   new 4B qualification, declared before any output, failed on coherence
   relative only. Its random-gain, absolute-coherence and invalid clauses
   passed.
   - Report this as `author_effect_only`. It is not a working controller
     and not a method failure.
   - Do not apply the qualification back to 7B. 7B's descriptive loss of 6.8
     points is shown only as context.
4. **Coupling.** SARC raised descriptive impoliteness from 1.3 to 32.3, the
   same direction as on 7B (38.75). It remains descriptive only.
5. **Attribution: none.** The carried-over b is not a matched strength.
   - The 4B Δ norms, the 14-block window, Qwen3's missing default system
     message and the model itself all differ from 7B.
   - The realized shift norms were lower than on 7B for PF and SARC, yet
     quality fell more, so a simple "more energy" story is not supported
     either.
   - None of these factors was varied, so **no cause and no model-size
     effect** can be claimed.

## Required reporting

- **Statuses:** PF `heldout_failure`, SARC `heldout_failure`, impolite
  `author_effect_only`. **No 4B controller qualified.**
- **No 4B bank and no new 4B Hugging Face steering repository.** Existing
  4B quantization and physics repositories are unaffected.
  - Publish the 7B controllers only, in
    `RenormalizationTech/qwen2.5-7b-gcad-steering-controls`.
  - Put all 4B outcomes and the carried-forward 4B refusal row (a partial
    effect with a failed quality rule) in the GitHub aggregate and closeout,
    linked from the 7B card's related results.
- **Required sentence:** "Outcomes are model-specific. Differences between
  checkpoints do not establish an effect of model size, and a 4B null does not
  show that the method fails on small models."
- **No rescue or retuning** within this study. Any lower-strength 4B work
  would be a new, separately registered study, which is not recommended under
  the current scope.
- **Execution notes to keep:**
  - the Qwen3 `q_norm`/`k_norm` port, whose Qwen2 path is bit-identical;
  - the BF16 model and gate, with FP32 banks;
  - the execution-only `no_grad` extraction wrapper, with bit-identical CPU
    evidence;
  - the impolite sub-batches of 20;
  - decoding pinned to the 7B values;
  - the system-message difference.

**Verdict:** I accept the closeout as written in
`docs/qwen3_4b_fixed_transfer_closeout.md`. The worker can finalize.
