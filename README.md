# tuneLLM

A hands-on project for learning open-source LLM quantization and fine-tuning.
Research into behavioral representations and activation steering is an optional
extension.

The project has separate tracks: **physics LoRA/QLoRA demonstrates fine-tuning**;
**refusal and persona activation steering study inference-time behavior control**.
Persona system prompting is a reference/control for the latter, with no weight
training or activation intervention. The next milestone is isolated activation
steering of **profanity, rudeness and sarcasm**, with simultaneous control deferred for later discussion and literature review. Complete this pipeline on **Qwen3-4B before further
14B work**. Generic physics questions in a behavior
evaluation do not make it physics fine-tuning. See the
[project map and selected persona targets](docs/project_tracks.md).

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
The completed 14B baseline is retained; further 14B experiments are deferred
under the user's revised 4B-first persona sequence.

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
experimental project; the full persona-control objective remains unachieved. See
[the current project map](docs/project_tracks.md) and
[local artifact retention](docs/workspace_maintenance.md).

The current priority (2026-11-15) is [original-model method reproduction](docs/author_model_reproduction_plan.md)
before further4B transfer. A verified GCAD impoliteness reference uses Qwen2.5-7B;
SAPS uses Mistral-7B-v0.1. The profanity route follows the documented paper method
with explicitly reconstructed missing inputs. Existing Claude evaluation replaces
unavailable author graders with clear disclosure. See the
[source audit](results/reviews/author_model_source_audit_01.md).

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
The [private physics adapter](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA) is published with its
[pinned private NF4 base](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-NF4).
[Publication checks](results/physics_huggingface_publication.json) passed.


Initial workflow:

1. Run a reproducible small-model inference baseline and fixed evaluation.
2. Convert to GGUF, quantize to Q8_0 and Q4_K_M, and measure the tradeoffs.
3. Fine-tune with QLoRA on a small licensed dataset; reload and evaluate the adapter.
4. Publish selected validated artifacts and model cards to our existing
   Hugging Face organization.

The GitHub repository and Hugging Face organization already exist. The first
artifacts are published: [public GGUFs](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-GGUF)
and a [private experimental adapter](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-GSM8K-QLoRA).
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
