# Gemma 4 31B QASPER study: final outcome audit (review 02)

Advisor audit, 2026-11-20. CPU only: no model, generation, judge or GPU calls,
and no edits to R0, the amendment, any source or any earlier review. Saved rows
were rescored with the frozen Qwen per-item scorer. Aggregation, gate checks
and a paper bootstrap with a different seed were written separately in the
advisor's scratchpad.

**Verdict: every registered task and retention gate passes.** The result is
reproduced from the saved rows, and its provenance is intact.

There is one material interpretation limit. The strict baseline is a format
floor: **all 403 base outputs are wrapped in ```` ```json ```` fences**, which
the frozen strict parser rejects, so base strict F1 is 0. The fairer content
comparison is the lenient diagnostic, which is not a registered gate. It still
shows a clear answer gain and no evidence gain.

## Provenance

| Item | SHA256 |
| --- | --- |
| `results/gemma4_qasper_final_01.json` | `a149de809997eb8ec18b670a4c1d32eca13edf014bd07fc28333abad3d50b39f` |
| `outputs/gemma4-qasper-study-01/analysis-amendment-01.json` | `b90d9e36d4e9f3d48c956f9330e164d309cd034e7294563bd4865f351fc1e95a` |
| `outputs/gemma4-qasper-study-01/closeout-amendment-01-status.json` | `54aea3a1ba70d9f3c7d889690e1a08cafd81ea62f9c253f36bfc531052a4db21` |
| amendment registration (commit `ff0d7b8`) | `29f41ff850a58d1ca5d23e6e443253d7341cc2ec84c4079776813a0a270862b2` |
| R0 (commit `7bc2866`) | `ba1983fdb9e6ca1e11ce9648e69a4c057832d6524b5014cd99cabe3fca4ba9f0` |
| completion seal | `97aed854a630993bbcd94967da8b56faf24249f5281260ba716b0e0caa73e861` |

- **Amendment inputs:** all 35 sources and fixed inputs registered in the
  amendment are unchanged. It binds amendment review 01 (`f1d5c9b1…`).
- **R0 sources:** all unchanged.
- **Original failure kept:** `pipeline-status.json` still reads
  `implementation_or_infrastructure_stop`.
- **Analysis source:** the amended analyzer `78506e39…`. The final file's
  report fields equal the analysis file.
- **Sealed files:** the final-arm and retention registration, result and
  prediction hashes match the completion seal.
- **Identical prompts:** base and adapter prompts are byte-identical for all
  403 rows (both use the base template). The prompt arm is an alias of base.
- **Generation endings:** all 806 final generations ended on a stop token, and
  none hit the token limit.

## Registered task gates (strict parser)

Denominators reproduced: 403 records, 309 primary, 267 evidence-strict.

| Gate | Registered limit | Observed | Pass |
| --- | --- | --- | --- |
| Answer F1 gain vs each baseline, primary | ≥ 0.05 | +0.6401 (adapter 0.6401; base and prompt 0.0000) | yes |
| Paired paper bootstrap 95% CI (109 papers, 10,000 draws, seed 20261117) | lower bound > 0 | [0.5884, 0.6922]; advisor's own seed: [0.5870, 0.6911] | yes |
| Evidence F1 loss vs better baseline, evidence-strict | ≤ 0.02 | +0.7472 (gain) | yes |
| Invalid JSON, all 403 | ≤ 2% | 5/403 = 1.24% | yes |
| False-unanswerable increase, 293 answerable primary | ≤ 0.05 | +0.0137 (4/293 vs 0/293) | yes |

## Registered retention gates (fixed net-loss limits)

| Task | Base | Adapter | Net | Limit | Pass |
| --- | --- | --- | --- | --- | --- |
| General MC (MMLU) | 108/128 | 105/128 | −3 (3 broken, 0 fixed) | 5 | yes |
| GSM8K | 126/128 | 126/128 | 0 | 5 | yes |
| IFEval strict | 26/32 | 26/32 | 0 | 3 | yes |

- **How each task was checked:**
  - **MC:** recomputed by the advisor's own argmax over the saved
    letter log-likelihoods.
  - **GSM8K:** recomputed with the frozen `evaluation/gsm8k.py`.
  - **IFEval:** the analyzer re-ran the official scorer and matched
    `scores.json`. The advisor checked the saved per-item counts.
- **Broken MC items:** `mmlu-test-01154`, `mmlu-test-00168`,
  `mmlu-test-09529`.

## Diagnostics and interpretation (not gates)

| Primary (n = 309), lenient parser | Base | Adapter |
| --- | --- | --- |
| Answer F1 | 0.3338 | 0.6401 |
| Evidence F1 | 0.7232 | 0.7172 |
| Invalid | 0 | 3 |
| False-unanswerable (of 293) | 12 | 5 |
| Answer length, median words (non-abstaining) | 35 | 3 |

- **Lenient answer gain:** +0.3063, paper bootstrap 95% CI [0.2513, 0.3617].
- **Lenient evidence, evidence-strict primary (n = 267):** base 0.7335,
  adapter 0.7472.
- **Prompt selection:** all three templates (base, verify, schema) scored 0.0
  strict on the 35 development questions, because every output was fenced.
  The tie rule chose base. The prompt arm therefore gives no information about
  prompting.
- **Interpretation:**
  - The strict +0.64 combines two effects: removing the code fences (format)
    and shorter, reference-style answers (style).
  - The lenient +0.31 isolates the answer-style and content gain.
  - Evidence selection is essentially unchanged by training (−0.006 to
    +0.014 lenient).
  - The registered conclusion holds as specified. Any write-up should report
    the lenient comparison next to the strict gate and should not describe the
    gain as improved document grounding.
  - The strict base score of 0 must not be read as "the base model lacks the
    knowledge". Leniently parsed, base already reaches answer F1 0.334 and
    evidence F1 0.723. A substantial part of the strict gain is format
    compliance.
  - This study supports no broad Gemma-versus-Qwen ranking. The two families'
    strict baselines fail for different reasons (Qwen base strict 0.457), and
    the prompts, templates and parser behavior are not matched across
    families.
- **Unanswerable questions:** strict unanswerable accuracy on the 16
  all-unanswerable primary rows is 0.50 for the adapter. Lenient accuracy is
  0.6875 for base and 0.50 for the adapter. This is a very small sample and
  needs no gate action, but it should be noted.

## Limitations to carry into any report

- **Not a fresh held-out test:** this is a reused cross-family project
  evaluation set (Qwen IDs) and custom text-only QASPER, not an official
  benchmark result. The public papers may be in pretraining.
- **Single run:** one configuration and one seed, with no quality-driven
  reruns. The selected checkpoint is step 61, the lowest development NLL
  (0.4207, 0.3543, 0.3469 at steps 21, 41 and 61).
- **Strict gate measures format:** the strict-parser gate largely measures
  format compliance against a fenced-output base. Lenient scores are
  diagnostic only and do not change the registered result.
- **Retention limits:** retention gates are fixed point-estimate loss limits,
  not non-inferiority. The samples are small: 128, 128 and 32.
- **Amended closeout:** closeout ran under analysis amendment 01, a
  validation-label correction only. The original failed closeout is kept.
- **No upload:** no Hugging Face publication was made. Any upload is a separate
  decision.
