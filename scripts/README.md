# Scripts and entry points

Run modules from the repository root with the environment specified in the
[reproduction guide](../docs/reproduce.md). There is no single verified Python
environment for every study.

| Purpose | Entry point |
| --- | --- |
| Original baseline and GGUF/NF4 workflow | `hf_baseline.py`, `benchmark_screen.py`, `gguf_eval.py`, `save_nf4.py`; [model workflow](../docs/model_workflow.md) |
| Physics QLoRA | `physics_train.py`, `run_physics_final.py`; [execution record](../docs/physics_prl_execution.md) |
| Small pinned public refusal bundle | `setup_refusal_artifact.py`; [download and CPU checks](../docs/reproduce.md#download-the-published-refusal-experiment-small-files-only) |
| 7B refusal experiments | `refusal_7b_transfer.py`, `refusal_7b_halfdose.py`, `refusal_7b_benign_transfer.py` and matching runbooks |
| Trait experiments | `persona_reference.py`, `sarcasm_gcad_low_strength.py`, `joint_gcad.py`; [trait evidence](../docs/steering_reference.md) |
| 27B NF4 loading smoke | `qwen38_nf4_smoke.py --inspect`; [separate environment](../docs/reproduce.md#separate-qwen38-27b-nf4-environment) |
| Portfolio figure | `plot_portfolio_results.py`; saved inputs in `docs/assets/` |
| Website | `docs_hooks.py`, `check_docs_site.py`; [build guide](../docs/website.md) |
| Release preparation | `stage_*release.py`, `publish_release.py`; [release index](../releases/README.md) |

`*_runbook.md` documents a specific completed run; `*_common.py` and `*_hooks.py`
provide shared mechanics. `joint_release_runtime/` holds runtime source for the
published joint-control package. Historical runners are kept at their recorded
paths because configs, protocols and source hashes depend on them. Read a
runbook's prerequisites before execution; do not rerun a frozen study or upload
an artifact just because its script is present.

Portable CPU tests live under [tests/](../tests/conftest.py), not here.
