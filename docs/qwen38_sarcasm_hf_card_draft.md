# HF card draft: 27B sarcasm GCAD controls

Status: **documentation draft for an experimental controls package**. In the registered exploratory study, no strength passed all acceptance gates (`exploratory_null_through_b16`), despite strong sarcasm at 8/16. This is not a validated controller. The standalone 27B runtime/package is not yet staged or published. No new HF repository or working download command is claimed here. The text below is intended for the eventual model card; exact download/runtime examples should be added and checked against the packaged interface before upload.

## What this controls package does

GCAD (Gated Cropped Attention-Delta) adds a learned style direction to selected attention outputs during generation. The direction is extracted from matched sarcastic and sincere generations. A token-dependent gate varies the addition as the model generates its reply. **The base model weights stay unchanged; strength is adjustable without retraining.**

This is an experimental inference control for the pinned Qwen3.8-27B NF4 base. It needs both the steering bank and compatible inference runtime; loading the base weights alone does not enable GCAD.

## Adjusting sarcasm strength

The planned CLI exposes `--strength`, which sets the runtime's base coefficient `base_coef` (b). Choose it before starting a fresh generation. Keep the bank, six selected layers and gate scale fixed to reproduce this profile.

Strength 0 is off. The proposed packaged interface accepts finite strengths from 0 through 16, with **8 as the owner-chosen experimental demonstration default** and 16 as the extreme preset. Strength 8 failed the registered relative-coherence gate; it is not a validated balanced setting.

| Strength/arm | Sarcasm | Coherence | Impoliteness | Coherence <50 | Registered gates |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 0.00 | 84.28 | 1.91 | 0/32 | Reference |
| Prompted reference | 55.56 | 76.00 | 52.09 | 0/32 | Reference |
| 2 | 0.09 | 84.16 | 2.22 | 0/32 | Fails trait gates; essentially no judged sarcasm |
| 4 | 4.97 | 83.44 | 6.75 | 0/32 | Fails trait gates; limited judged sarcasm |
| 8 | 88.50 | 58.94 | 70.72 | 6/32 | Fails relative coherence: 58.94 < 64.28 |
| 16 | 95.16 | 34.59 | 81.53 | 30/32 | Fails both coherence gates |
| Random16 | 0.00 | 84.28 | 2.09 | 0/32 | Descriptive reference only |

Higher settings can amplify sarcasm and impoliteness while reducing how directly and clearly the answer addresses the question. Strength is not a sarcasm probability, and doubling it does not guarantee twice the effect. Only 0 was mechanically checked as off, and 2/4/8/16 were behaviourally evaluated; intermediate settings have not been evaluated. A balanced default is not established.

Internally, the runtime adds `2 * b * sigmoid(1.5 * (demand - centre)) * delta` at each selected layer/token. The gate changes with the model state, so actual shifts vary across tokens and settings. Users normally tune b, rather than changing the bank, layers or gate scale.

## Measured scope

Results use 32 fresh, neutral, object-heavy English questions per arm, an AI judge and 160-token answers; all replies reached that cap. The bank pools 90 eligible matched pairs across five instruction pairs, using the base model's own generations. It captures their whole wording contrast, not an isolated pure sarcasm feature. Steered arms use a single user turn with no system prompt, greedy decoding, repetition penalty 1.05, batch one and native KV caching. Multi-turn conversations, added system prompts, sampling, other languages and answers beyond 160 tokens are untested. Baseline sarcasm/coherence was 0.00/84.28; an explicit sarcasm instruction gave 55.56/76.00. Strong settings produce sarcasm, but no tested strength passes every registered trait/quality gate. This package should be presented as an exploratory style-control demonstration, not a validated general-purpose controller.

At strength 16, a setting-matched random direction scored zero sarcasm. It uses one random seed set. Its realised shifts differ along its generated trajectory; it is not an energy-matched control, and there was no random control at strength 8 or the lower strengths. Coherence loss is an observed tradeoff, not a defining requirement of sarcasm. Impoliteness is a separate measured outcome; sarcasm was not shown separable from rudeness. The historical coherence rubric separates coherence from correctness, but its hallucination wording makes that separation imperfect. The old `development_null` remains unchanged; this experiment does not establish dose as its cause.

For equations, architecture details and measured shift norms, see the [GitHub theory and tuning guide](https://github.com/RenormalizationTech/tuneLLM/blob/main/docs/qwen38_sarcasm_gcad_guide.md). The [full experiment record](https://github.com/RenormalizationTech/tuneLLM/blob/main/docs/qwen38_sarcasm_followup_execution.md) includes all gates and limitations.


## Intended use and attribution

This is for experimental style-control research and creative demonstrations. Strong settings can produce mocking or demeaning text; the study does not establish a general-purpose quality-preserving controller. Only the pinned NF4 model and recorded one-GPU software stack were tested.

The controls package must preserve the Apache-2.0 GCAD port licence/NOTICE and the base model's applicable licence terms. The bank is derived from that base model's own generated responses; raw generations and judge sessions are not part of the proposed public payload.
