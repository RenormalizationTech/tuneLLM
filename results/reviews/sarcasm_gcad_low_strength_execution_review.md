# Sarcasm GCAD lower-strength execution review

Verdict: **ACCEPT — no stored-data blocker found.** This is a CPU/read-only audit of the completed records, with no target-model loading, inference, native grading, new tests, or changes to frozen artifacts. The four new closeout/profile files are derived documentation only.

## Audit scope and probe correction

The earlier audit probe raised an assertion before compaction; its exact expression/traceback was not recoverable. It is **not identified as a study defect**, and no specific cause is claimed retrospectively. Fresh named checks supersede that incomplete probe. Follow-up probe errors (`hits` helper location and `mean_sarcasm` versus the actual H14 `mean_sarcasm_conservative` schema) were audit-script assumptions, not stored-data failures. They required no study change.

The completed checks used actual schemas and expected-versus-actual comparisons. Model-file verification streamed hashes only; CPU deserialization was limited to steering/control/logit tensors with `weights_only=True`.

## Bindings and receipts

- Current/snapshot source, input, model and protected parent bindings passed `Context.verify(models=True)`: 20 active source entries, 1249 protected external entries, 13 model-file entries and 30 snapshot entries. Initial, mechanics and selection registrations matched their exact recomputed records and committed authority.
- Completed mechanics/development/heldout file hashes, run hashes, exit-integrity records, GPU UUID, BF16/SDPA, seed 42 and batch 1 settings verified. The final report reproduced **exactly** from accepted journals.
- New native streams contained **20/20/16/16/16** accepted batches (development trait/coherence; heldout trait/coherence/impoliteness): **88 first-valid batches, 88 starts, all attempt0, actual model `claude-opus-5-5`**. The existing strict reader checked payload, binding, start/receipt/raw hashes, complete IDs, model identity and first-valid reuse. No rerating or retries.
- The exact64 parent development records (baseline32 + prompted32), including token/output bytes, were reused. All160 development items received new child grades; no parent grades entered child selection.
- The parent's report/selection/stop and bank remain bound unchanged; original status is `development_null`, with no parent heldout outputs. The bank is the original SARC bank, not a PF bank.

## Fresh-work accounting

| Stage | Generation-start records | Explicit logit-start records | All model-start records |
| --- | ---: | ---: | ---: |
| Mechanics | 6 | 2 | 8 |
| Development | 96 | 0 | 96 |
| Heldout | 128 | 0 | 128 |
| Total | **230** | **2** | **232** |

The 232 raw files reconcile to 230 generations **plus**2 explicit logits. Fresh prompt/teacher-forced/MC counts are 0. The 64 reused answers are not fresh generations. The parent's 466 generations,300 prompt forwards,261 teacher-forced forwards,2 logits and 115 native starts remain historical costs, not child work.

## Mechanics, controls and selection

Actual saved ordinary/zero generated IDs and logits were exactly equal. Nonzero GCAD changed logits and hidden states; all 11 blocks 9–19 had prefill/cached-decode coverage and hook cleanup. Per-answer cached call/position counts were reconciled with raw token lengths. Saved settings and all 128 heldout before-call setting hashes matched the recorded inputs.

The raw FP32 bank shapes are delta 28×3584, key 28×4×128, demand 28; tensor hashes and finite values verified. All **352** heldout random layer tensors (32×11) were independently regenerated exactly with dedicated CPU generators, seeds9603/9604/9605 by row and `seed+1000*block`. Raw norms match the real delta; keys/demand/gate are unchanged. Controls metadata and selection-lock hashes match.

Development metrics and gates were recomputed from all 320 fresh ratings. GCAD 1 fails target gain/count; GCAD 2 passes every gate; GCAD 3 fails relative coherence. Thus **GCAD 2 is the lowest and only qualifier**, committed before heldout generation. No heldout-driven reselection occurred.

## Final arithmetic

| Arm | SARC mean | ≥50 /32 | Coherence | Capped /32 | Mean tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.06250 | 0 | 89.59375 | 32 | 160.00000 |
| prompted | 66.46875 | 29 | 77.12500 | 10 | 120.43750 |
| selected_gcad | 63.78125 | 29 | 70.71875 | 27 | 156.96875 |
| random_gcad | 0.00000 | 0 | 87.71875 | 32 | 160.00000 |

All arms have 0 empty/invalid answers,0 trait REFUSAL labels and 0 fixed-lexicon hits. Selected gains are **63.71875** versus baseline and **63.78125** versus random; 29/32 meet the ≥50 threshold. Coherence 70.71875 clears 50 and baseline−20=69.59375, but **drops 18.875 and clears the relative floor by only 1.125**. All six primary gates pass. Coherence is not factual accuracy or helpfulness.

Actual position-weighted hidden shifts are **4.0366705750** selected versus **3.1004033846** random. The control matches raw layer norms/sites/gates, not realized energy; this limits mechanistic attribution.

H14 exclusion was recomputed from 31 rows: selected SARC 63.8387096774, coherence 70.6774193548,28/31 ≥50. It is descriptive only; all 32 primary rows and the primary decision remain unchanged. Descriptive impoliteness means and within/pooled correlations were independently recomputed: selected 38.75, within-selected r 0.6370922702; pooled r 0.9318638164 is condition-confounded, not causal or independence evidence.

## Privacy, limitations and anchors

The public aggregate passes the existing privacy guard; no raw prompts, answers, grading reasons, token arrays, credential or home-path strings are exported. Heavy truncation, one judge family/artifact, reused project questions and topical H14 overlap limit generalization. No accuracy, capability, joint-control or universal-quality claim is supported.

- Report SHA256: `1cd1cb405af1e2f05631a2a73d916eaea0c8f0430460ba39265531925eb8e54a`.
- Manifest SHA256: `e916394e9425dfbcdfe43415617a4745fb047c1397498022800cf2733207552a`.
- Bank SHA256: `6b0dd37a5f825a4daa72d48e56e2e47df67dd449db007ab4f03ebbc238a4fcc2`.
- Selection SHA256: `7b79093f93d98d19350b2261ed6301b50c21a6dbfa7e63cc6c959d9424ee32b1`.
- Controls SHA256: `0fe6b4f6f5241b89cfbd8276122fdfb5f361183bad4fc3edf1b1a79858d8f368`.
- Controls metadata SHA256: `c3fda982096996b1aecf0a96a75baba92e98d1128329d931847958aa6aae7a55`.

The independent scientific interpretation is recorded separately in [the scientific review](sarcasm_gcad_low_strength_scientific_review.md).
