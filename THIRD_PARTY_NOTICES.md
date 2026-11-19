# Third party materials and attribution

Original tuneLLM code and documentation are licensed under [Apache 2.0](LICENSE).
Third-party code, data, quotations and model artifacts retain their own terms.
The root license does not replace the licenses listed below. Base weights are
obtained separately; this Git repository contains no model weight files.

| Material | Use in tuneLLM | Terms and retained attribution |
| --- | --- | --- |
| Qwen and Mistral checkpoints | Base models for inference, adaptation and steering | Their upstream model cards/licenses; exact revisions are in each experiment profile and HF release. |
| GCAD author implementation | Adapted attention hooks, direction extraction and gating | Apache 2.0; [license](scripts/qwen3_gcad_port_LICENSE), [Qwen3 notice](scripts/qwen3_gcad_port_NOTICE), [Qwen3.8 notice](scripts/qwen38_gcad_port_NOTICE). |
| Persona Vectors | Author prompts, impoliteness fixtures and an adapted profanity construction | Apache 2.0; [rudeness attribution](evaluation/persona_reference/rudeness/ATTRIBUTION.md) and [profanity attribution](evaluation/persona_reference/profanity/ATTRIBUTION.md), with local licenses in those directories. Generated profanity artifacts are project adaptations, not released author data. |
| SAPS released prompts and train/validation/test data | Original-model sarcasm reference and method comparison | Authors' MIT license; [license](evaluation/persona_saps/LICENSE), [attribution](evaluation/persona_saps/ATTRIBUTION.md), pinned author commit `c56fb28ac0d14c8fd70035e035c5452e3c6c6ccc`. |
| SAPS Appendix C rubric | Adapted AI-judge evaluation | Naganna et al. (2026), [paper](https://aclanthology.org/2026.findings-emnlp.1594/), Appendix C; CC BY 4.0 under the [ACL publication terms](https://aclanthology.org/2026.findings-emnlp.1594/). The extracted rubric is distinct from the MIT prompt-data license. Structured Claude scoring is a project adaptation. |
| XSTest | Refusal evaluation prompts and source IDs | CC BY 4.0; [retained license](results/refusal_source_licenses/xstest_LICENSE.txt), source revision and attribution in [the source audit](docs/refusal_protocol.md#pinned-sources-and-storage). |
| AdvBench | Harmful extraction instructions | MIT repository terms; [retained license](results/refusal_source_licenses/advbench_LICENSE.txt) and [source audit](docs/refusal_protocol.md#pinned-sources-and-storage). |
| Databricks Dolly 15k | Benign extraction instructions | CC BY-SA 3.0; [pinned dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a). Raw Dolly data are not bundled here. The published refusal vector has [separate vector terms](https://huggingface.co/RenormalizationTech/qwen2.5-7b-refusal-steering-experiment/blob/3e8a151667a8fef0ebba552a3f3d74317b1c093d/VECTOR_LICENSE.md). |
| DECCP and routing-is-learned | Political-information baseline fixtures, translations and reference annotations | Apache 2.0 / MIT respectively; [fixture attribution, source revisions and license copies](data/steering/political_refusal_01/README.md#source-attribution-and-licences). The [smaller DECCP screen](evaluation/deccp_headroom/README.md) also retains its source license. |
| Physics-paper corpus | Domain-text QLoRA training | Each of the 100 papers was verified as CC BY 4.0; prepared data and individual author/source attributions accompany the [physics adapter](https://huggingface.co/RenormalizationTech/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA). Model Apache terms do not replace paper terms. |
| MMLU, GSM8K and IFEval | Evaluation only | MIT / MIT / Apache 2.0 as recorded in the released `benchmark_dataset_licenses.json`; pinned datasets and scorers are listed in [the benchmark summary](docs/benchmark_summary.md). |

## SAPS provenance scope

The SAPS train, validation, test and prompt files are redistributed under the
explicit MIT license accompanying the authors' `data/prompts` directory at the
pinned revision. The authors' generation lineage and the original sources of
individual transformed sentences are not fully documented. This project does
not invent row-level provenance or claim exact reproduction of published
numerical results. The existing attribution and hash-bound fixtures remain
unchanged. The paper-derived rubric is attributed separately above.

## Historical records and generated examples

Frozen protocols and receipts retain historical machine paths, private-status
labels and source hashes. They describe those runs; [current reproduction
instructions](docs/reproduce.md) and the [release index](releases/README.md)
describe current entry points and artifact availability. Machine paths are not
required download locations.

The [showcase examples](docs/showcase.md) contain explicitly selected, unedited
excerpts of saved model responses and separately labeled interpretation. XSTest
prompt text retains CC BY 4.0 attribution. Model outputs and AI-judge labels are
not human-verified factual ground truth. No new target-model generations were
made for the showcase.
