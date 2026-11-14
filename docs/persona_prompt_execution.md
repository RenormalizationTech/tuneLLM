# Persona P1 execution record

The user selected grading completion followed by tone/persona work. The benign
baseline completion finished and was published as `8affb3b`. This separate
prompt-only pilot was discussed with Claude, then registered in `6677110`
before model or grading outputs. See the [protocol](persona_prompt_protocol.md),
[registration](../results/protocols/persona-prompt-01.json) and
[design/fixture review](../results/reviews/persona_prompt_review.md).

## Preparation

Claude independently checked all 48 authored items, their anchors and final
rubric. Revisions improved unprompted uncertainty/pushback coverage and removed
overly narrow anchors and invalidity ambiguities before generation. Each split
has four families of six items, with three anchors each. Six tasks per split
are unhinted, six retain explicit checks and twelve are not in those diagnostic
groups. Cue groups are descriptive, not additional gates.

The worker implemented the new runner. The coordinator reviewed paired input
identity, anonymization, scoring, lock semantics, source bindings and fatal-stop
behavior. All 25 focused CPU checks passed, including schema rejection,
balanced A/B positions, gross-loss/calibration boundaries, held-out lock guards,
source mutation, first-valid journal replay and export of mechanical-null
results. Historical normalized exact-overlap checks passed; they do not prove
semantic independence.

The verified manifest binds 15 active sources, 21 snapshot files, 421 external
bindings and 12 cached model files, including the earlier 402/315 protected
inputs. Manifest SHA-256:
`615c542af1535bd318bdc65195f7c71a15b606c76312c55f5f1cf93c2274e379`.
The coordinator independently rechecked the snapshot/model bindings before
registration. Input documents remain frozen afterward.

An unused first preparation snapshot was archived because the coordinator
appended final code-review notes while preparation was in flight. Only that
new review-document hash differed; prior protected inputs matched. The guard
stopped preparation before any model or judge output. Its snapshot/log/incident
are retained privately under `outputs/persona-prompt-preparation-drafts/`.
The final snapshot uses the completed review notes; source, fixtures, rubric and
prompts were not changed for this rebuild, so unchanged passing tests were not
repeated.

## Registered comparison

Both conditions use unchanged cached BF16 Qwen3-4B with the same user messages,
greedy nonthinking decoding, seed 42, batch 8 and cap 768. C uses the common
accuracy/calibration system prompt; P adds the frozen direct/composed persona
block. This is a matched prompt-strategy comparison, not a change to weights or
activation control. The earlier no-system baseline is not reused as C.

Development generates 24 answers per condition. Held-out generates another 24
per condition only after all development gates pass and its decision is
committed. GPU 0's UUID/resource guard passed at the coordinator's pre-release
check: 48,053 MiB free and 0% utilization. Existing jobs remain running.

The judge sees anonymous balanced A/B pairs, anchors and the frozen paraphrased
persona rubric. It uses the pinned audited CLI and fresh tools-off contexts,
eight pairs per batch, explicit ID/count checklists and first-valid freezing.
Maximum six grading invocations per split, twelve total. Style, quality,
calibration, experience and invalidity remain separate measurements.

## Outcome

Claude accepted the final registration before development release. Both
conditions generated their 24 development answers: 24/24 EOS finishes each,
zero cap hits, empty outputs or mechanical-degeneration flags. The mechanical
gate passed. Generation took 64.12 seconds including loading, with peak PyTorch
allocation 8.75 GB decimal (8.14 GiB), including model weights and excluding
other processes.

The anonymous packet had 24 pairs with candidate positions balanced 12/12;
private source/cue metadata was excluded. All three eight-pair grading batches
passed on their first attempts, with observed `claude-opus-5-5` identity and no
retries. The coordinator independently recomputed the metrics from accepted
labels and private keys; they match the saved development decision.

**Development did not pass the registered rule.** P won 15/24 style comparisons,
below 16 required, and had two gross paired quality losses, above one allowed.
The other gates passed. The three quality gains cannot cancel those losses.
The pilot closed; held-out generation/grading was skipped, no passing lock was
issued, and there was no prompt revision, coefficient search or extra call.

| Descriptive AI assessment | Common system C | Persona system P |
| --- | ---: | ---: |
| Style preference | 5 wins | 15 wins |
| Quality acceptable | 17/24 | 18/24 |
| Overconfident/unsupported, overall | 5/24 | 4/24 |
| Overconfident/unsupported, insufficient evidence | 2/6 | 2/6 |
| Fabricated-experience flags | 0 | 0 |
| AI-invalid / mechanical-degeneration flags | 0 / 0 | 0 / 0 |
| Mean output words | 139.6 | 106.2 |

Four style comparisons were ties. Two C-acceptable answers became P-unacceptable;
three changed in the other direction, yielding a descriptive net gain of one.
The registered gross-loss guard protects already acceptable answers rather
than allowing gains elsewhere to cancel their loss. No threshold was changed
after seeing the result. P won 9/12 comparisons when shown as A and 6/12 when
shown as B; this small position breakdown is descriptive, not another gate.

The family breakdown limits the interpretation: P won 11/12 physics and
practical-writing pairs, versus 4/12 insufficient-evidence and pushback pairs
(C won five of those, with three ties). Both gross quality losses, PD15 and
PD17, were pushback tasks. Thus the pooled preference does not establish a
consistent benefit on the uncertainty/correction behaviors central to the
persona. P was about 24% shorter on average; the rubric rejects length alone
as a merit, but a contribution from length cannot be ruled out.

This is a directional prompt-style signal, with slightly higher aggregate
AI-acceptable count and no increased calibration flags. It is not an accepted
quality-preserving configuration under the fixed rule. The narrow misses do
not prove causal quality harm, lack of any style effect, or that this persona
is unreachable. No held-out confirmation exists.

See the [public result](../results/persona_prompt_01.json). The saved private
development-decision SHA-256 is
`7264e2f991991a9b2e16d79c63109a9b7737e01afb46aca0a8a8bb9e6958692b`.
Actual new model budget was 48 responses and three grading invocations; the
held-out fixture remains unused. No activation capture, steering vector, new
adapter, weights or Hugging Face model release followed.

Claude's final read-only review accepted the counts, frozen thresholds,
provenance/privacy and development stop. Its suggested conclusion is reflected
here: directional style preference without meeting this specification, with
small-sample and grading limits explicit. No follow-up run was launched.

The worker's final read-only close-out confirmed source/snapshot and saved
prediction/decision/receipt bindings, exactly three first-valid invocations,
no held-out directory or passing lock, and the existing 421/prior402/315
verification evidence. Large files were not rehashed again without a new
concern. Project GPU memory was released: GPU 0 returned to 48,053 MiB free
and 0% utilization, with other jobs untouched.

## Interpretation limits

These are small custom AI-authored tasks and one AI grading pass, without
external fact checking or human validation. Rating style and quality together
may create halo effects. A strict gross-quality-loss threshold can fail through
response/grading variability; such a result is not proof of causal harm.
The dated server-side judge snapshot is unknown, and the historical isolation
audit lacked an executable digest. The installed pinned native binary is checked
before/after calls. A prompt-level pass would not establish activation control,
internal emotional states, general capability preservation or performance on
harmful requests. Development outputs are now seen and cannot serve as an
unseen test; the held-out set remains unused.
