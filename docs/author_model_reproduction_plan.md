# Original-model steering reproduction

User decision, 2026-11-15 (updated after evaluator/artifact clarification): establish published reference effects on the authors'
models and tasks before further Qwen3-4B transfer, new method comparisons or joint
control. The completed 4B adaptations remain unchanged. The coordinator owns
planning and execution releases, Claude reviews scientific choices, and the
Codex worker implements and tests code.

## Source and target matrix

| Trait | Reference and original model | Initial reproduction target | Readiness |
| --- | --- | --- | --- |
| Rudeness, using the broader author construct `impolite` | [GCAD](https://arxiv.org/html/2605.10664v3), Qwen/Qwen2.5-7B-Instruct | First-reply GCAD and residual-steering rows of Table 1; ten-turn endpoints require a separately reviewed follow-up | [Completed bounded reference](author_gcad_impolite_closeout.md): all3 directional contrasts passed; GCAD trait77.37/coherence75.90 versus baseline1.62/82.68 |
| Sarcasm | [SAPS](https://aclanthology.org/2026.findings-emnlp.1594.pdf), mistralai/Mistral-7B-Instruct-v0.1 | Reconstruct released result tables first; then author-runtime fixed-layer generation at layer12, fixed alpha1 and SAPS alpha1.4 | [Completed reference](author_saps_mistral_closeout.md):90 first-valid batches; clear sarcasm276/450 at SAPS1.4 vs0 baseline, with meaning cost; published scores not reproduced |
| Profanity | [Malla et al. v4](https://arxiv.org/abs/2609.06951v4), Qwen/Qwen2.5-7B-Instruct | Appendix E profanity row: layer7, coefficient6, self-effect +1.44 on the author 0–10 scale, coherence0.96 | Two preserved extraction nulls; [actual direction test](author_profanity_direction_test_closeout.md) completed: baseline and six strengths all 0/32 lexical swearing and 0/10 semantic profanity; no held-out setting qualified |

The PV paper's Appendix G, Figures18–19, concerns prediction of fine-tuning
shifts. It is not evidence of a successful inference-time impoliteness steering
curve. Its available impolite artifacts do not settle that gap. GCAD supplies a
direct target but introduces a different, attention-level method; call it a
GCAD reproduction, not a successful PV reproduction.

## Pins and access

- GCAD author repository: [NihiI-obstat/Gated-Cropped-Attention-Delta](https://github.com/NihiI-obstat/Gated-Cropped-Attention-Delta), commit `6c879f4d12e2b01274fc96c9afdb0f4f3a26d4de`.
- SAPS author repository: [SharathHN/saps-steering](https://github.com/SharathHN/saps-steering), commit `c56fb28ac0d14c8fd70035e035c5452e3c6c6ccc`.
- Qwen checkpoint: `a09a35458c702b33eeacc393d103063234e8bc28`, four safetensors files,15,231,271,888 bytes; current authenticated HEAD succeeds, ungated Apache-2.0.
- Mistral checkpoint: `ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca`, two safetensors files,14,483,498,040 bytes; HEAD succeeds, ungated Apache-2.0.
- Llama-3.1-8B-Instruct is currently gated with HEAD403. No licence is accepted on the user's behalf. It is not required for the selected ungated references.

The papers do not pin these current checkpoint revisions. Record that provenance
limit. Source trees, primary papers, source-archive inventories and access
evidence are kept under ignored `outputs/persona-author-replication-audit-01/`.
Download only allowlisted pinned files, verify Git blob SHA1, size and SHA256,
and retain the original bytes. Binary vectors are not loaded during source audit.

## Milestones

1. **Published-result reconstruction, no GPU or judge calls.** Recompute SAPS
   per-layer means and the authors' per-sentence oracle table from the released
   averaged ratings. Report sentence coverage, missing/duplicate rows and whether
   two or three evaluators contributed. The paper's fixed-layer numbers and its
   oracle numbers are distinct targets. This verifies released arithmetic, not
   new model generations or independent judgment.
2. **Author mechanics and environment.** Use separate environments rather than
   modifying the shared `.venv`. SAPS documents Python3.11 and pins
   TransformerLens2.17.0, torch2.6.0, Transformers4.57.6 and NumPy1.26.4. Match
   the actual author prompt rendering, BOS behavior, residual basis, precision,
   generation and hooks. CPU and zero/nonzero probes precede scored generation.
3. **One bounded reference per available trait.** Commit exact inputs, code,
   environment, checkpoint, settings and evaluation plan before target outputs.
   Match author settings and sample counts where supplied; label smaller diagnostics
   and reconstructed data. Original paper rubrics are primary. Do not import our
   4B coherence floor or call a changed evaluator an exact score reproduction.
4. **Transfer later.** Only after interpreting the original-model results,
   consider the Qwen3-4B failures, profanity method-family comparison, narrower
   rudeness or generic assistant sarcasm. Joint control remains deferred.

## Concrete GCAD target and evaluator dependency

Table1 reports60 conversations (20 question groups, three samples), temperature1,
cap500 per reply, with a persistent KV cache for the full10-turn study. Qwen
GCAD uses layers9–19, base coefficient3.5, gate scale1.5; the residual reference
uses author layer20 and coefficient2. Author code must resolve exact indexing,
turn construction, seed handling and precision before registration.

Published impolite first-reply scores are GCAD89.3 trait/67.5 coherence and
residual97.3/6.3. Ten-turn GCAD averages are90.8/66.9. These are targets from the
paper, not our observations and not proof of high-quality rudeness control.
The author construct and moderate coherence remain explicit.

The main judge is `gpt-4.1-mini-2025-04-14`, with separate trait and coherence
calls per reply. A first-reply reproduction of the two published conditions is
120 generations and240 judgments before any retry allowance. A full ten-turn
comparison is1,200 generations and2,400 judgments, not automatically released
by this plan. Exact request construction, output limits and a cost cap must be
reviewed before live API calls. The user declined new OpenAI API setup. The target remains open Qwen, while
existing pinned Claude grading replaces the evaluator with explicit disclosure.
The prospective first-reply protocol now adds an unsteered baseline:180 scored
generations,360 logical score slots in46 blinded native batches, plus ten short
mechanics probes. See [R1 protocol](author_gcad_impolite_protocol.md). No paid
OpenAI API calls are planned.

## Sarcasm evaluation dependency

Paper Table1, Mistral fixed layer12, reports fixed-projection means
sarcasm3.12/entailment4.04/grammar4.50/clarity4.62 at alpha1, versus
SAPS3.89/3.70/4.60/4.68 at alpha1.4. Oracle means are a separate result.
The author judges are Gemma4-31B-IT, Qwen3.5-27B and GPT-OSS-120B, averaged;
individual judge logs and the executable judge pipeline are absent from the
pinned release. Their weights and inference serving cannot be assumed to fit
the existing single48-GB workflow. Source-result reconstruction completed in
[the arithmetic report](../results/author_replication_saps_published_01.json);
fresh scoring will use the existing Claude setup, as authorized, with an
explicit evaluator adaptation. Exact numerical reproduction is not claimed.
The six canonical released logs contain450 consistent IDs but448 distinct
strings, with zero exact overlap against any released train/val/test split.
Their prompt metadata matches style instructions rather than the plain
instruction used by released generation code. Actual rendered inputs remain
unverified. Stored-score reconstruction is arithmetic only, not a rerun.

## Profanity artifact search and stopping condition

The primary arXiv PDF and source archive were fetched. The source archive has
18 members: manuscript, figures, style/bibliography and result-table files.
It contains no executable experiment code or full prompt dataset. The manuscript
claims full code and288 contrastive triples in supplementary material but has no
external artifact link. Public title/arXiv-ID repository searches returned no
verified author repository. This establishes an artifact gap in the checked
locations, not that no release exists anywhere.

The user has no supplement and authorized proceeding from the paper. Rebuild
missing12 contrastive triples, neutral questions and judge envelope through a
separately frozen, reviewed process. Preserve the paper's definitions, response
pooling, layer-selection band, raw-vector intervention and calibration rules;
record every unspecified choice. Call this a paper-method reimplementation
with reconstructed inputs, not an original-artifact or numerical reproduction.
The primary self-judge can run on unsteered Qwen locally; Claude can provide an
independent secondary report. Contacting authors is not authorized or needed
for this selected route.

## Recorded implementation progress

GCAD source acceptance and its initial registration were committed before
mechanics; successful mechanics were separately locked before main generation.
Ten probe generations, two logit forwards and180 main generations completed.
The main GPU stage used82.46 seconds including model loading, with72.27 seconds
in generation. Both metric streams passed exit integrity:46 first-valid batches,46 starts,
no retries. All three predeclared contrasts passed; see the
[closeout](author_gcad_impolite_closeout.md). No target outputs selected settings.

The two original model snapshots and separate Python3.11 environments are ready.
No shared training environment was changed. SAPS requires an original-runtime
run because the released result logs have unresolved data/prompt lineage gaps.

Profanity implementation passed28 CPU tests; one reviewed generated artifact
was frozen before24 teacher-forced responses. Primary AUROC0.13194 failed the
registered0.85 gate, closing extraction before mechanics/calibration. This is
a direction-validation null, not a measured steering failure. See the
[closeout](author_profanity_method_closeout.md).

## Resource and preservation policy

Only GPU0 is authorized, subject to free-memory/utilization checks and an
unquantized-model memory preflight. Leave existing jobs untouched. Space was
approximately2.9TiB free at inspection. Keep weights, environments, raw source
archives and answers in existing ignored directories. Preserve all earlier
models, data, source files, protocols, nulls and successful results.

This is a reviewed-source preparation plan, not a registration of a live model
experiment. Stage-specific registrations and manifests follow source acceptance.
