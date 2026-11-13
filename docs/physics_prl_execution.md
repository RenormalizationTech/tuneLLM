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
Raw archives are retained locally; TeX is never executed.

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

## Results

The rank-32, 5e-5 pilot completed all 207 steps and 1,567,140 prediction targets
in 22m14s including epoch validation/saving. Peak PyTorch allocation was 7.08 GiB.
Its development losses were 2.2268125 before training and 1.9777026, 1.9657982,
1.9670665 after epochs 1, 2 and 3. Epoch 2 is nominated for the screen; this is
not yet a model selection. Final-epoch adapter tensors, probe logits and token
sequences matched exactly on a fresh-base reload.

The private NF4 base is published at revision
`e9af208e44e083b77702a71d8aa65b7d97f78ce6`. All 16 downloaded file hashes and
both original GPU verification probes match. See `results/nf4_huggingface_publication.json`.

Remaining training and final comparisons are in progress. This section will be replaced
with the completed run registry, selected checkpoints, final tests, release
revisions and limitations when those stages finish.
