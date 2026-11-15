# Three-method profanity transfer: execution review

Status: complete; GCAD strength 5 retained. This review is after all outputs
and decisions; the prospective protocol and source snapshots remain unchanged.

## Independent worker audit

The existing Codex worker independently recomputed the report, every development
and confirmation gate, selected strength and preferred method. All pass for
GCAD 5; fixed/SAPS do not qualify. Model/source/snapshot hashes, stage integrity,
committed registrations, first-valid receipts, control regeneration, private
prompt/token traces and public-export privacy checks pass.

Verified completed totals: 632 generations, 300 prompt forwards, 240 GCAD teacher-
forced forwards, 2 logits, 0 MC. There are 186 first-valid grading batches from 187
starts. The only structural retry was development trait batch 8, with unchanged
payload and no semantic rerating. Recorded GPU stages total 3146.033 seconds.

The worker created only the new static retained-controller profile, binding the
exact bank/settings/checkpoint/source/runtime/data/registrations and measured
held-out scope. It validated profile values and hashes against the completed
study. No additional model, judge, install or download calls occurred.

## Claude scientific review

The existing advisor separately reviewed the final saved evidence. It confirmed
the decision rule: selected profanity 31.125 exceeds baseline/random 0 by 20,
lexical 14/32 exceeds both absolute 8 and baseline + 8, coherence 82.6875 exceeds 50
and baseline 85.34375 - 20, and invalid count 0 is within 3. GCAD 7 failed development
relative coherence and was not reselected on held-out results.

The review supports retaining GCAD for this model/task and marking this bounded
study complete. The claim is moderate profanity control on 32 ordinary questions,
with a single adapted Claude judge and reconstructed inputs. It is not a rescue
or reproduction of the previous Malla-style/PV nulls, nor independent control
of all three traits. Coherence does not establish accuracy or helpfulness.

Material limits: swearing lexicon coverage 14/32, variable answer lengths and
truncation, possible coupled rudeness, and filtered held-out grading batches
averaging 3–4 items rather than development's 8. Raw random-delta norms were
matched, but realized mean shifts were 6.584 versus 7.610, about 13.47 percent
lower under random; comparison is approximate in realized energy. Root computed
exact descriptive distributions from saved grades: median profanity 35, 9/32
scores >=50, 27/32 higher than baseline/random and 5 ties. No new ratings or
selection changes were introduced by these summaries.

## Coordinator acceptance

The coordinator independently reviewed implementation and reran all 33 focused
CPU tests (6.60 seconds), accepted the source/protocol before live calls, checked
prepared input plans, registered every release, and verified the final aggregate
against the frozen conditions. Earlier extraction/behavioral nulls and all raw
historical sources remain unchanged. Private trajectories, banks and journals
stay in the ignored output folder. The result and reusable profile are accepted;
no new experiment is released by this review.

See [closeout](../../docs/profanity_method_transfer_closeout.md),
[aggregate](../profanity_method_transfer_01.json), and
[retained profile](../../configs/profanity_retained_controller.json).
