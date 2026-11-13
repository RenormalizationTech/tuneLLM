# Physics-writing workflow: execution record

This record distinguishes completed preparation from later training/evaluation
results. The experiment design is in [physics_prl_protocol.md](physics_prl_protocol.md)
and the reusable process is in [model_workflow.md](model_workflow.md).

## Completed preparation

- Original checkpoint: `Qwen/Qwen3-4B`, revision
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- Standalone base: `artifacts/nf4/Qwen3-4B`, about 3.21 GiB. NF4/double
  quantization, BF16 compute, FP32 nonquantized modules. All 1,659 state tensors,
  two last-token-logit probes and generated probe IDs matched on a stored-state
  reload with requantization forbidden.
- Data: 100 PRL-associated condensed-matter theory arXiv source versions, each
  with Crossref publication verification and an authoritative arXiv CC BY license
  check. Eight primary categories are represented. AI abstract review retained
  142 of 190 shortlisted candidates; exclusions and source failures are retained.
- Splits: 70 training, 10 development, 20 final-test papers, category stratified.
  Five heuristic author identities overlap across splits; no group-disjoint
  generalization claim is made.
- Training exposure: 550 within-paper chunks, 522,380 prediction targets per epoch,
  sequence length 1024, 69 optimizer steps per epoch, 207 steps over three epochs.
- Tokenizer parity: the saved NF4 tokenizer and original upstream tokenizer give
  identical IDs on all 100 cleaned papers. A generic loader regex warning was
  recorded, but no tokenizer rewrite was applied.
- Writing tasks: 30 development and 60 test tasks covering titles, abstracts and
  introductory passages; two-example prompting baselines use training papers only.
- Initial audited source/protocol commit: `0ed4b7a`; gate clarifications before
  optimization: `75d51c7`.

See `results/nf4_checkpoint.json`, `results/physics_prl_corpus.json`,
`results/physics_prl_publication_verification.json`, and the per-run committed
protocols under `results/protocols/` for checksums and exact inputs.

## Acquisition details and limits

The convenience source is `scholarweave/arxiv-latex`. Its hosted filter endpoint
was only partially indexed, so its apparent small counts were not used as corpus
counts. We read metadata columns from the recorded parquet shards instead of
downloading the full LaTeX dataset, then downloaded selected source archives
directly from versioned arXiv URLs.

The exact reviewed shortlist is recorded in `results/physics_prl_candidate_index.json`.
Primary categories and abstract review provide theory eligibility; this is not
human expert certification. The final corpus has 13 papers in each category except
materials science (9), reflecting source availability and cleaning exclusions.
Every development category is represented.

Four arXiv titles differ substantially from their Crossref publication titles.
Matching authors and research topics confirm the association; the audit records
each case. These are arXiv versions associated with PRLs, and can precede final
publisher wording. They are not a paired PRB-to-PRL revision dataset.

Cleaning removes title-block identity commands even when they follow the abstract,
hidden/comment blocks, acknowledgments/funding, and back matter. Used macro
definitions and scientific notation are retained. Six sources contain visible
revision-color markup, and one long main text was inspected through its conclusion.
Raw archives are retained locally; TeX is never executed. Five Unicode replacement
characters in proper names on two training papers were retained in the frozen
pilot; inspected equations and numbers were unaffected. The cleaned macro subset
is intended as training text and is not guaranteed to compile. See
`results/physics_prl_corpus_quality.json`.

## Development baselines

The definitive NF4 nonwriting baseline used batch 8 and completed in about
342 seconds:

| Task | Result |
|---|---:|
| General MMLU, raw zero-shot likelihood | 20/32 |
| Physics MMLU, raw zero-shot likelihood | 22/32 |
| GSM8K, last-number extraction | 29/32 |
| GSM8K, strict final-answer format | 26/32 |
| IFEval, official strict prompt scoring | 13/16 |

The two-example writing baseline completed all 30 tasks in about 288 seconds.
All outputs were nonempty, none hit the generation cap, and one received an
automatic new-number/citation flag. Human ratings remain pending. There were no
exact 25-word-shingle matches to the local training corpus; this diagnostic does
not establish originality or absence of memorization.

The earlier batch-4 benchmark was an interrupted preflight and is excluded from
paired comparisons. A CPU-only Q4 GGUF API preflight verified input-token parity,
answer-letter probability extraction and generation; its arithmetic probe returned
42. It is not a GPU performance or general capability result.

## Reproduction entry points

Install the pinned project requirements in a local environment with compatible
CUDA access; large files stay in ignored directories.
The upstream model and prepared GSM8K splits are prerequisites; their acquisition
and preparation commands are in [the original execution record](execution.md#qlora-preparation).

```bash
.venv/bin/pip install -r requirements-benchmark.txt
.venv/bin/python -m scripts.download_model --config configs/baseline.json
.venv/bin/python -m scripts.setup_benchmark_sources
.venv/bin/python -m scripts.prepare_benchmark_screen
.venv/bin/python -m scripts.save_nf4
```

The scripts refuse to overwrite existing checkpoints, prepared paper splits and
run directories. Reuse the existing verified checkpoint, or choose fresh paths
when reproducing. The initial GGUF conversion procedure remains in `execution.md`.

To reconstruct acquisition in a fresh checkout:

```bash
.venv/bin/python -m scripts.fetch_physics_candidates --candidate-index results/physics_prl_candidate_index.json
.venv/bin/python -m scripts.prepare_physics_corpus verify-dois
.venv/bin/python -m scripts.prepare_physics_corpus sources
.venv/bin/python -m scripts.prepare_physics_corpus verify-licenses
.venv/bin/python -m scripts.prepare_physics_corpus prepare
.venv/bin/python -m scripts.prepare_writing_fixture
```

Source availability and mirror metadata can change. The original archives and
frozen inputs, rather than a fresh live query, define this experiment. Recreate
and review exclusions before registering a new run if acquisition differs.

Before optimization, record the NF4 development reference and the two-example
writing reference under the names consumed by the selection runner:

```bash
.venv/bin/python -m scripts.benchmark_screen --representation nf4 \
  --fixture data/benchmark-screen-01/development.json \
  --tasks general_mc physics_mc gsm8k ifeval --output outputs/physics-screen-nf4-dev-02
.venv/bin/python -m scripts.benchmark_screen --representation nf4 \
  --fixture data/benchmark-screen-01/development.json \
  --writing data/benchmark-screen-01/writing_development_fewshot.json \
  --tasks writing --output outputs/physics-writing-nf4-fewshot-dev
```

These are fresh-run examples; existing output folders are never overwritten.
Archive the source/fixture hashes and register the protocol before optimization.

Prepare a run on CPU, review its exact protocol, and commit a byte-identical copy
before execution:

```bash
.venv/bin/python -m scripts.physics_train --output outputs/new-r32 --rank 32 --learning-rate 5e-5 --epochs 3
# Copy outputs/new-r32/protocol.json to a fresh tracked protocol path and commit it.
.venv/bin/python -m scripts.physics_train --run outputs/new-r32 --protocol results/protocols/new-r32.json
```

`physics_selection` reads development reports only. `physics_eval` refuses test
loss under pilot protocols. Final-test authorization requires a separate frozen
evaluation preparation after model selection. `benchmark_screen` writes individual
outputs and scores; `build_writing_review` creates an offline blinded review with
a separate identity key. No human judgments are synthesized.

After both registered learning-rate runs complete, the screen runner nominates
each run's minimum-loss epoch, checks unchanged inference dependencies and frozen
fixtures, and runs the development comparisons:

```bash
.venv/bin/python -m scripts.run_development_screens \
  --runs outputs/physics-r32-lr5e5 outputs/physics-r32-lr1e4 \
  --stage lr_pilot --mapping outputs/physics_lr_mapping.json --execute
.venv/bin/python -m scripts.physics_selection \
  --mapping outputs/physics_lr_mapping.json --output results/physics_lr_selection.json
```

Use the analogous rank-stage mapping after the registered rank runs complete.
These commands never read final-test outputs. `publish_release` uploads only an
explicit, hash-reviewed manifest to the authorized repository and visibility;
it refuses to overwrite an existing repository by default and verifies the
revision-pinned remote bytes before reporting publication success.

After rank selection and the complete seed-43/BF16 control runs, prepare the fixed
final matrix using their actual run paths:

```bash
.venv/bin/python -m scripts.run_physics_final --repeat outputs/SELECTED-SEED43 --control outputs/SELECTED-BF16
# Review and commit results/protocols/physics-final-eval-{nf4,bf16}.json.
.venv/bin/python -u -m scripts.run_physics_final --execute
.venv/bin/python -m scripts.summarize_physics --mapping outputs/physics_final_mapping.json
```

The runner fixes the primary selected epoch for both controls, checks complete
training exposure, and refuses changed inputs or result folders without matching
invocation records. It evaluates five conditions on the 20 held-out papers; four
HF conditions receive the 444-item screen, the two-example NF4 writing reference
receives 60 tasks, and BF16/Q8/Q4 GGUF receive the 384 nonwriting items. BF16 LoRA
is compared with its own unadapted BF16 base; GGUF quantization is compared within
the llama.cpp backend. The repeat's paper loss is reported separately without
reselection. The final review's random answer key is kept outside the published
payload with a SHA256 commitment; keep an offline backup before collecting ratings.

## Results

The rank-32, 5e-5 pilot completed all 207 steps and 1,567,140 prediction targets
in 22m14s including epoch validation/saving. Peak PyTorch allocation was 7.08 GiB.
Its development losses were 2.2268125 before training and 1.9777026, 1.9657982,
1.9670665 after epochs 1, 2 and 3. Epoch 2 was nominated for the learning-rate
screen described below. Final-epoch adapter tensors, probe logits and token
sequences matched exactly on a fresh-base reload.

The private NF4 base is published at revision
`e9af208e44e083b77702a71d8aa65b7d97f78ce6`. All 16 downloaded file hashes and
both original GPU verification probes match. See `results/nf4_huggingface_publication.json`.

The second rank-32 pilot (1e-4) also completed the identical three-epoch budget.
Its best development loss was 1.9676020 at epoch 1. Its nominated writing screen
hit three generation caps, exceeding the registered increase limit of two. The
5e-5 pilot passed all registered development gates and was selected before the
rank comparison. The exclusive decision is `results/physics_lr_selection.json`
(commit `9d60703`); ranks 8 and 128 use 5e-5 with the same maximum exposure.
These small development checks do not establish improved human-rated writing.

The rank-8 and rank-128 runs have now completed at 5e-5, seed 42. Both began at
the identical token-weighted development NLL of 2.2268125. Their saved curves and
minimum-NLL epoch nominations are:

| Rank | Epoch 1 NLL | Epoch 2 NLL | Epoch 3 NLL | Nominated epoch / step |
|---|---:|---:|---:|---:|
| 8 | 2.0046957 | 1.9825771 | 1.9786641 | 3 / 207 |
| 128 | 1.9626056 | 1.9730483 | 2.0037009 | 1 / 69 |

These are token-weighted paper-text losses. Per-paper NLL and chunk-level target
counts remain in each run's `validation/step-*.json` reports. Epoch nomination
uses the registered minimum-development-NLL rule and precedes screen gates; it
does not establish a writing-quality gain.

| Rank | Trainer wall time | Peak PyTorch allocation | Adapter weight bytes |
|---|---:|---:|---:|
| 8 | 1,331.195 s (22m11s) | 6.28 GiB | 66,126,768 |
| 128 | 1,371.527 s (22m52s) | 10.34 GiB | 1,057,033,224 |

Wall time covers the whole Trainer interval, including epoch validation and
checkpoint saves. Peak allocation includes loaded weights and training/validation
allocations on the selected GPU; it is not total VRAM or other-process memory.
The adapter byte counts refer to `adapter_model.safetensors`; each run's three
epoch checkpoints and final adapter have that same weight-file size.

Ranks 8, 32 and 128 received equal maximum exposure: the same 550 training chunks
and all remainder chunks, 522,380 prediction targets per epoch, three epochs,
207 optimizer steps, 1,650 microbatches and 1,567,140 total prediction targets.
Their nominated checkpoints nevertheless have different exposure: rank 8's
epoch 3 has 1,567,140 targets, rank 32's epoch 2 has 1,044,760, and rank 128's
epoch 1 has 522,380. Actual exposure assertions passed for both new runs.

Both new runs also passed fresh-base final-adapter replay: adapter tensors,
probe logits and generated token IDs matched exactly, with zero maximum logit
difference. These serialization probes cover the final epoch. Raw logs are
`outputs/physics-r8-training.log` and `outputs/physics-r128-training.log`; complete
metadata is in the corresponding `outputs/physics-r8-lr5e5/run.json` and
`outputs/physics-r128-lr5e5/run.json`.

**Development rank selection is fixed: rank 128 at 5e-5, epoch 1 / step 69.**
`outputs/physics_rank_mapping.json` identifies rank 8 epoch 3, the reused rank 32
epoch 2 screen, and rank 128 epoch 1. All three ranks passed the registered gates. Rank 32 minus rank 128 development
NLL was 0.0031926 (paired paper-bootstrap 95% interval 0.0017998 to 0.0045028);
rank 8 minus rank 128 was 0.0160585 (0.0132842 to 0.0188015). The registered
smallest-compatible-rank heuristic therefore selected 128. See
`results/physics_rank_selection.json`. This heuristic is not proof of practical
writing superiority. The seed-43 repeat and BF16 control will use the full
three-epoch schedule but be compared at the primary epoch-1 checkpoint. Final
comparisons remain pending. Human writing-quality improvement is not established.

Selection has visible tradeoffs. The selected rank-128 checkpoint scored 28/32
on development GSM8K (base 29/32), 11/16 on strict IFEval (base 13/16; rank 32
12/16), and produced three automatic writing-fidelity flags versus one for the
two-example base. Its two writing truncations and two extra flags each sit at
the registered increase limit. All three ranks had two writing truncations.
The 0.0032-nat/token development-loss advantage over rank 32 is small; these
screens do not establish that the larger adapter writes better for people.

The selected rank-128 configuration's NF4 seed-43 repeat has completed all 207
steps and 1,567,140 prediction targets. Its epoch losses were 1.9624437,
1.9723484 and 2.0057803, close to the primary seed-42 trajectory. Trainer wall
time was 1,371.993 seconds and peak PyTorch allocation was 10.34 GiB. All final-
epoch adapter tensors, probe logits and generated IDs matched on fresh-base
replay. Comparison remains fixed at epoch 1 / step 69. The BF16 control also completed the full exposure, with development losses
2.1754497 before training and 1.9413228, 1.9539306, 1.9851287 after epochs 1–3.
Its Trainer interval was 1,087.157 seconds (18m07s), with peak PyTorch allocation
13.76 GiB. All final-epoch reload probes matched exactly. Comparison remains
fixed at epoch 1. Final-test evaluation is next.
