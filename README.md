# tuneLLM

A hands-on project for learning open-source LLM quantization and fine-tuning.
Research into behavioral representations and activation steering is an optional
extension.

The project has separate tracks: **physics LoRA/QLoRA demonstrates fine-tuning**;
**refusal and persona activation steering study inference-time behavior control**.
Persona system prompting is a reference/control for the latter, with no weight
training or activation intervention. The behavioral goal is reliable **individual
control of refusal, profanity, rudeness and sarcasm**. GCAD is preferred where
it works; each trait may retain its own intervention and tuned setting.
Cross-effects describe trait coupling, with strong correlations reported rather
than requiring independent knobs. Simultaneous control is experimental. Preserve
the completed 4B results as history; larger-model work requires discussion. Generic physics questions in
a behavior evaluation do not make it physics fine-tuning. See the
[project map](docs/project_tracks.md), [four-behavior references](docs/steering_reference.md)
and [GCAD discussion](docs/gcad_and_combined_control.md).

**Live work (2026-11-16):** the user selected a
[minimal transfer of the completed 4B refusal workflow to Qwen2.5-7B](docs/refusal_7b_transfer_execution.md).
It reuses the original pairs, splits, two interventions and quality checks,
with a fixed relative-depth layer and newly extracted 7B direction. The fixed
transfer is complete: subtraction reduced benign refusals 3/30→0/30 but safe
usefulness fell 20/30→13/30 with one degeneration; projection removal preserved
19/30 usefulness but missed the original mechanism-gain threshold. Neither
qualified, so final and retention were skipped. Only one subtraction case was
a paired useful benign rescue. The single -0.25 calibration completed: benign
refusals 3/30→0/30, useful/correct 20/30→21/30, no invalid outputs, one useful
paired rescue. It passed quality but missed the original unsafe-contrast gain
gate; final/retention were skipped. A fixed-setting check on the earlier
4B-refused benign texts is under review. Both registered nulls remain unchanged.
The cancelled OR-Bench/new-method expansion is unused.

**Completed status (2026-11-16):** the
[fixed-setting Qwen3-4B GCAD transfer](docs/qwen3_4b_fixed_transfer_closeout.md)
is complete. All three style directions induced their traits, but none passed
every quality gate. Profanity lost substantial coherence; sarcasm and
impoliteness narrowly exceeded the allowed coherence loss. The conventional
refusal record is carried forward without new compute. No 4B controller is
relabelled as working, and no retuning is active. The retained 7B style banks
are [published with runtime and evaluation summaries](https://huggingface.co/RenormalizationTech/qwen2.5-7b-gcad-steering-controls).
The original 23-file isolated release was verified at its pinned revision;
its banks/runtime remain unchanged in the additive joint release below. Existing quantization models and
the public physics adapter remain separate.

**Phase 1 complete: joint 7B study and public guide.** All 12 fixed arms were
measured on the same 24 AI-authored English advice prompts. **0/5 combined
candidates met all criteria; all five failed a coherence point-estimate rule.**
Strong cross-effects were observed, not independent controls. Full-triple
coherence was16.08 vs baseline80.08; half-triple57.12 still missed the allowed
20-point loss. Its paired CI crosses the boundary and does not create a pass.

PF means **friendly-rubric swearing**, not literal swearing frequency. PF solo
scored36.38 vs full-triple6.25, while literal lexicon hits rose14/24→24/24.
The separately registered [post-hoc lexical audit](results/joint_gcad_lexical_audit_01.json)
documents that construct confound; original grades/gates remain unchanged.
The [public measured guide](https://huggingface.co/RenormalizationTech/qwen2.5-7b-gcad-steering-controls/blob/e1e56e6384214be873984f5fb9421ff6c61b9e85/joint_controls/GUIDE.md)
and [publication receipt](results/steering_publication_joint_04.json) cover all
34 anonymously verified payload files, both manifest hashes and preserved
isolated banks/runtime. No new GPU check accompanied publication.

**Phase 2 baseline complete:** the [DECCP screen](docs/deccp_headroom_closeout.md)
did not meet prespecified headroom criteria. English targets had1 explicit
refusal+7 grader-classified evasions/36 (22.2%), versus0/11 genre controls; the22.2pp gap passed,
but the25% and12-item conditions failed. Chinese targets had0 refusals+5
evasions/11 (45.5%), descriptive without Chinese controls. Explicit refusal was
rare; soft evasion is distinct, and missing facts were not scored as censorship.
All58 responses completed;25 held-out rows remain ungenerated. No intervention
was tested or refusal model published. The requested refusal-steered artifact
remains incomplete. The reviewed recommendation is to close the conventional7B
attempt; a separate Chinese evasion study is pending the user choice already
asked asynchronously. No automatic extension is released. Larger-model discussion is allowed; execution requires user discussion/decision.

The preceding Hugging Face-only transition is complete. NF4 and PRL Writing QLoRA are
now public; the GSM8K adapter repository was deleted after a verified local
archive. See the [transition receipt](results/hf_visibility_transition_01.json).
No new7B refusal model or controller is claimed; the completed benign screen
does not reopen earlier frozen refusal experiments.

[Verified local cleanup](results/storage_cleanup_20261116_01.json) reclaimed
49.576 GiB: redundant verification caches, two duplicate PRL staging weights,
and Qwen3-14B/Mistral-7B weight blobs. Their tokenizer/config metadata remain;
all 18 physics optimizer states, checkpoints, active 7B/4B weights, environments
and the GSM8K archive were retained. Future model options are Qwen3.8-27B and
Gemma 4, not another Qwen3-14B download or study; larger-model work awaits
user discussion.

The [refusal reproducibility review](docs/refusal_reproducibility_review.md)
compares the original author code with published over-refusal benchmarks. The
completed [adapted XSTest-safe baseline](docs/xstest_baseline_execution.md)
found 17/250 partial/full refusals on the unmodified model. These are candidate
over-refusals, not a validated steering result; 154 answers
hit the fixed 256-token cap.
The [stored-output review](docs/xstest_stored_output_audit.md) supported 16 cases
as candidate unnecessary refusals, with original grades preserved. Robustness
to new wording remains untested.

The [4B refusal study is closed](docs/qwen3_4b_closeout.md): its final replay
matched all 250 published baseline cases token for token, with no new grading.
That historical closeout does not close the new 4B persona pipeline.
The [Qwen3-14B BF16 baseline is complete](docs/qwen3_14b_baseline_execution.md)
on one A6000: candidate refusals on published safe prompts were 16/250 (6.4%),
and partial/full refusals on the separate unsafe contrasts were 160/200 (80%).
All 57 grading batches passed on their first attempts. The model phase took
about 13 minutes with 30.8 GB peak allocated memory. These are unmodified-model
measurements; reliable refusal steering and greater stability from size remain
unestablished. See the [registered protocol](docs/qwen3_14b_baseline_protocol.md).
The completed 14B baseline records are retained. Its cached weights have been
removed while tokenizer/config metadata remain; Qwen3-14B is not a future target.

The [trait-rating calibration passed](docs/persona_rating_calibration_completion_execution.md):
both blinded passes matched all 48 authored presence labels for each trait.
This supports limited repeatability on clear examples. The
[isolated 4B profanity pilot closed with a development null](docs/profanity_steering_execution.md):
all six settings produced 0/16 profane outputs despite changing answer tokens.
[Rudeness and sarcasm also closed with development nulls](docs/isolated_persona_traits_execution.md):
31/32 and 32/32 pairs qualified, but every setting produced 0/16 usable target
outputs. The interventions changed answer tokens without demonstrating target
control in this range. The user then selected the
[literature-first isolated workflow](docs/persona_literature_workflow.md): released
Persona Vectors impoliteness, SAPS sarcasm, and an explicitly adapted PV profanity
profile. The source-tested implementations are complete. Both full PV grids
closed with development nulls. SAPS sarcasm completed a narrow held-out
paraphrase reference pass (27/100 reaching the author sarcasm threshold ≥3 versus 0 baseline/random),
with reduced meaning preservation and small capability checks passing. See the
[reference execution record](docs/persona_reference_execution.md). Joint control is deferred for later discussion as a separate
experimental project. Those earlier studies did not establish usable control for every trait; the subsequent 7B references and fixed-setting 4B result give the current evidence. See
[the current project map](docs/project_tracks.md) and
[local artifact retention](docs/workspace_maintenance.md).

The [original-model method references](docs/author_model_reproduction_plan.md)
were completed before the fixed-setting 4B transfer. A verified GCAD impoliteness reference uses Qwen2.5-7B;
SAPS uses Mistral-7B-v0.1. The profanity route follows the documented paper method
with explicitly reconstructed missing inputs. Existing Claude evaluation replaces
unavailable author graders with clear disclosure. See the
[source audit](results/reviews/author_model_source_audit_01.md).
The first [original-Qwen GCAD reference](docs/author_gcad_impolite_closeout.md)
completed: impoliteness1.62→77.37/100 with coherence82.68→75.90; residual
addition produced84.52/7.83. All registered contrasts passed under the disclosed
Claude grading adaptation.
The [profanity paper-method attempt](docs/author_profanity_method_closeout.md)
stopped at direction validation with reconstructed minimal-edit pairs; it did
not test steering. The approved [short-answer revision](docs/author_profanity_revision_closeout.md)
also closed at the row-level gate, while paired-question sensitivity reached1.0.
The [existing-direction behavioral test](docs/author_profanity_direction_test_closeout.md)
is complete: all six strengths produced 0/32 lexical-swearing responses and
0/10 semantic profanity, matching baseline. The verified intervention changed
wording but did not establish profanity control; no held-out arm qualified.
The [three-method profanity transfer](docs/profanity_method_transfer_closeout.md)
is complete. **GCAD strength 5 is retained** on Qwen2.5-7B: held-out profanity
31.13/100 and lexical swearing 14/32 versus 0 for baseline and random control;
coherence 82.69 versus baseline 85.34. Fixed projection and SAPS did not qualify.
See the [retained workflow](docs/profanity_controller_workflow.md).
The [original-Mistral sarcasm reference](docs/author_saps_mistral_closeout.md) is
complete: SAPS1.4 reached clear sarcasm in276/450 rewrites versus0 baseline,
while meaning preservation fell4.11→2.98/5. Fixed and adaptive control show
different style/quality trade-offs.

The [Qwen sarcasm transfer](docs/sarcasm_method_transfer_closeout.md) closed with
a development null: GCAD 3.5 raised sarcasm to 87.1/100 but reduced coherence
from 85.9 to 52.4; no tested SAPS or GCAD setting met both requirements.
The separate [lower-strength study](docs/sarcasm_gcad_low_strength_closeout.md)
retained **GCAD strength 2**: held-out sarcasm 63.78/100 versus 0.06 baseline
and 0 random, with 29/32 answers scoring at least50. Coherence fell
89.59→70.72, only1.125 points inside the allowed loss. Impoliteness also rose
1.63→38.75; this coupling is reported, not rejected. See the
[sarcasm workflow](docs/sarcasm_controller_workflow.md).
The [Qwen7 refusal headroom screen](docs/refusal_headroom_closeout.md) found
3/100 unnecessary refusals, below the predeclared five-question threshold.
The secondary phrase matcher counted five; classification sensitivity is
disclosed. No GCAD refusal controller was tested or retained.

The [7B political-information screen](docs/political_refusal_headroom_status.md)
found no explicit refusal headroom on its development targets. The user closed
the available coverage branch before any fit or steering, then selected the
completed [Qwen3-4B fixed transfer](docs/qwen3_4b_fixed_transfer_closeout.md).
Its trait effects and quality failures are reported without a model-size causal
claim. The7B banks and descriptive joint guide are public; the new DECCP
baseline screen is complete with a failed headroom gate. Larger models remain
not released for execution before user discussion/decision.

For adapter-training tasks, the routine workflow policy is **one base and one
adapter configuration**, seed 42,
a small development screen, and one fixed final comparison. Compatible baseline
results can be reused. Multi-rank/LR sweeps and extra controls are calibration
experiments rather than default steps.

The original quantization and QLoRA milestone is complete. Benchmark reports and
Hugging Face cards are updated. The registered tone pilot stopped during
contrast construction: its three mechanical upper bounds were 5, 31 and 46 pairs
against the declared 50-pair requirement. No steering vector or final tone test
was produced; see the [execution record](docs/tone_steering_execution.md).
The positive refusal pilot also completed with a null: at fixed layer 17,
benign refusals were 1/30 at baseline and 2/30 at strengths 1 and 2; strength 4
degenerated on all 30 prompts. No setting qualified, so final generation was
skipped. See the [execution record](docs/refusal_execution.md) and
[compact result](results/refusal_01.json). This does not rule out refusal
control at other layers. The user-selected layer-23 follow-up also completed: refusals rose to 4/30,
7/30 and 24/30 at strengths 0.25/0.5/1, but the strongest setting had six
degenerate outputs and no setting passed both gates. Final generation was
skipped and the positive-intervention pilot is closed. See the
[follow-up record](docs/refusal_late_execution.md).
Management plans are discussed with Claude before implementation; each study
records whether review was available. The dedicated Codex worker implements code.

The earlier refusal-reduction study assessed willingness and answer quality
separately; persona style was deferred. The completed positive probes
did not test reduction. The completed comparison of baseline, single-block
subtraction and all-block projection removal was registered in
[the reduction protocol](docs/refusal_reduction_protocol.md). Claude declined
reviewing this design; the user authorized proceeding without that review.
The coordinating Codex owns acceptance and the dedicated worker implements
and tests. The completed reduction study selected block-23 subtraction on
development. On the final set, valid answers on the refusal-triggering class
increased from 1/30 to 7/30, but benign useful/correct answers fell from 29/42
to 23/42 with three mechanical degeneration flags across candidate outputs.
The tiny capability screen stayed within its tolerances, while the final
response-quality criteria failed. This is a measured tradeoff, not a validated
usable configuration. See the [execution record](docs/refusal_reduction_execution.md)
and [compact summary](results/refusal_reduction_summary_03.json).
The [two-method guide](docs/activation_steering_methods.md) explains constant
vector shifts versus all-block projection removal, with formulas, diagrams and
code links. A [stored-output diagnosis](docs/refusal_output_diagnosis.md) reviewed
19 existing pairs without generation or changes to scores. Its qualitative
findings do not support a token-cap increase as a sufficient fix: apparent
errors, repetition and drift also need attention.

The [benign helpfulness baseline survey](docs/benign_helpfulness_protocol.md)
generated all 36 fresh legitimate requests within its token budget, without
empty or mechanically degenerate outputs. Primary grading stopped incomplete:
the first 20-item batch passed, but the second omitted one required rating on
both permitted attempts. The historical incomplete report claims no formal
refusal-headroom or quality score.
See the [execution record](docs/benign_helpfulness_execution.md) and
[incomplete report](results/benign_helpfulness_01.json). The sealed held-out set
is unused. The user then authorized a separately registered
[grading-only completion](docs/benign_helpfulness_completion_execution.md).
Both eight-item batches passed first try, retaining the original 20 labels.
The [combined report](results/benign_helpfulness_completed_01.json) records zero
refusals among 24 legitimate sensitive requests (`insufficient_headroom`),
101/108 anchors satisfied and 11/36 apparent error/unsupported-claim flags.
These AI assessments are descriptive, not verified factual accuracy.
That baseline survey used no activation intervention or candidate system prompt.

The user-selected [persona P1 pilot](docs/persona_prompt_execution.md) is also
complete. A fixed system prompt was preferred for style on 15/24 development
pairs, below the required 16, with two gross AI-judged quality losses versus
one allowed. Aggregate acceptable answers were 17/24 for the common system and
18/24 for the persona, but three gains could not cancel two previously
acceptable answers becoming unacceptable. Other gates passed. Held-out was
skipped as registered. The [result](results/persona_prompt_01.json) is a
directional prompt-style signal, not an accepted quality-preserving persona
or activation-control demonstration. No further run is active.

The [benchmark summary](docs/benchmark_summary.md) collects the completed
quantization and physics-adapter scores, with the earlier GSM8K experiment
reported separately.
The [JSON instruction-tuning exercise](docs/next_sft_pilot.md) is optional backlog:
the earlier GSM8K QLoRA run already covered instruction tuning.
Physics writing remains a lightweight exploratory exercise; further training and
writing assessment are deferred. Later Claude/Codex ratings will be identified
as AI judgments.

The current reusable workflow is documented in
[docs/model_workflow.md](docs/model_workflow.md). The completed Qwen3-4B physics-writing exercise uses
a saved NF4 base, a balanced corpus of condensed-matter theory PRLs, and writing
evaluations for titles, abstracts and introductions, with one BF16 LoRA control.
The verified standalone base is recorded in
[results/nf4_checkpoint.json](results/nf4_checkpoint.json).

The physics pilot completed six training runs, matched BF16/NF4 controls and
fixed broad evaluation on a licensed 100-paper corpus. Its selected rank-128,
epoch-1 adapter reduced held-out paper-text NLL from 1.97494 to 1.73395 but
reduced numeric GSM8K accuracy from 117/128 to 111/128. Human writing-quality
improvement remains unestablished. See the [full physics result](results/physics_prl_final.json)
and [execution record](docs/physics_prl_execution.md). A blinded local review is
available at `outputs/physics-writing-review/review.html`; assessment is deferred
and human review is optional for the learning milestone.
The [public physics adapter](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA)
is at revision `b9f7ff0222d9e4070ee88f89e4f00ff096e20359`. Its
[public NF4 base](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-NF4)
has current card revision `2ae7eec7ac38290ea28efefd1a06e779027aafc7`; the adapter
still pins base revision `e9af208e44e083b77702a71d8aa65b7d97f78ce6`.
The [visibility-transition receipt](results/hf_visibility_transition_01.json)
records anonymous dependency verification and unchanged weights. This used
small-file downloads and large-file LFS metadata against previously verified
hashes, with no new weight download or GPU check. The prepared test split and
evaluation fixtures are public, so future reuse is not clean held-out testing.
Original [publication checks](results/physics_huggingface_publication.json)
remain historical evidence.


Initial workflow:

1. Run a reproducible small-model inference baseline and fixed evaluation.
2. Convert to GGUF, quantize to Q8_0 and Q4_K_M, and measure the tradeoffs.
3. Fine-tune with QLoRA on a small licensed dataset; reload and evaluate the adapter.
4. Publish selected validated artifacts and model cards to our existing
   Hugging Face organization.

The GitHub repository and Hugging Face organization already exist. The first
artifacts include [public GGUFs](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-GGUF),
the public NF4 base and PRL Writing QLoRA adapter. The earlier GSM8K adapter
repository is deleted; its negative learning result and verified local archive
are preserved in the [transition receipt](results/hf_visibility_transition_01.json).
The organization is
[RenormalizationTech](https://huggingface.co/RenormalizationTech).
See [plan.md](plan.md) for priorities and the optional research backlog, and
[docs/execution.md](docs/execution.md) for environment setup and observed results.

The first model is `Qwen/Qwen3-4B`, pinned in `configs/baseline.json`.
Two BF16 runs produced identical tokens and scored 7/8 on a basic pipeline smoke
fixture. This fixture is not a capability benchmark; compact results are in
[results/initial_baseline.json](results/initial_baseline.json).
BF16, Q8_0 and Q4_K_M GGUF files passed CPU prompt comparisons and short GPU
benchmarks; sizes, checksums and measurements are in
[results/initial_quantization.json](results/initial_quantization.json).

A two-step QLoRA smoke run also passed fresh-base adapter reload with identical
generated tokens. Results are in
[results/initial_qlora_smoke.json](results/initial_qlora_smoke.json). This validates
the workflow; it does not establish a useful fine-tuned model.

A bounded run on 1,024 GSM8K examples completed 64 steps with a frozen,
committed protocol. The selected adapter improved answer-format compliance
from 107/128 to 128/128 but reduced numeric accuracy from 121/128 to 101/128.
It is documented as an experimental learning result. See
[the full result](results/qlora_real_01.json),
[the fixed protocol](docs/real_run_protocol.md), and
[release details](releases/README.md). Uploaded file hashes and pinned-revision
load checks passed; see [publication verification](results/huggingface_publication.json).

After installing the environment described in the execution record:

```bash
.venv/bin/python -m scripts.env_check --output outputs/environment.json
.venv/bin/python -m scripts.download_model --config configs/baseline.json
.venv/bin/python -m scripts.hf_baseline --output outputs/my-baseline
```

Run GPU commands with access to the NVIDIA devices. The baseline defaults to
physical GPU 1 and refuses to overwrite an existing run directory.

Large local files belong in ignored `artifacts/`, `data/`, `outputs/`, and `.hf/`
directories. Source code, configurations, small evaluation fixtures, and compact
results belong in Git. The Hugging Face cache path will be set explicitly by the
execution workflow; merely creating `.hf/` does not configure the cache.

Roles: coordinating Codex manages planning and acceptance, Claude advises on
planning and science, and a dedicated Codex worker implements and tests.
