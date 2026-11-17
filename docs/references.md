# Methods and source references

The project applies published methods and evaluates their behavior in bounded
experiments. These references identify the methods, software and datasets;
project outcomes are reported in the [four-behavior record](steering_reference.md)
and [benchmark summary](benchmark_summary.md).

## Adaptation and quantization

- Hu et al. (2021), [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). Frozen base weights plus trainable low-rank updates.
- Dettmers et al. (2023), [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). The NF4/double-quantization and adapter-training reference; our domain-text runs have their own corpus, schedules and evaluation.
- [llama.cpp](https://github.com/ggml-org/llama.cpp), used for GGUF conversion, quantization and inference; the original 4B release pins commit `8ca5a8008b57aa39f2601546c0d07912045080a5`.
- [Transformers bitsandbytes integration](https://huggingface.co/docs/transformers/quantization/bitsandbytes), [PEFT](https://huggingface.co/docs/peft/) and [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes). GGUF Q4_K_M and Transformers NF4 are different representations.

## Activation steering

- Arditi et al. (2024), [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717); [author code](https://github.com/andyrdt/refusal_direction). Our block-output extraction/subtraction and all-block projection are documented adaptations rather than exact reproduction of every author setting.
- Kang et al. (2026), [Prompt-Activation Duality: Improving Activation Steering via Attention-Level Interventions](https://arxiv.org/abs/2605.10664); [GCAD author code at the used revision](https://github.com/NihiI-obstat/Gated-Cropped-Attention-Delta/tree/6c879f4d12e2b01274fc96c9afdb0f4f3a26d4de). GCAD means Gated Cropped Attention-Delta.
- Naganna et al. (2026), [Scaled Autoregressive Projection Steering (SAPS): Token-level Steering to Induce Sarcasm and Humblebrag in Text](https://aclanthology.org/2026.findings-emnlp.1594/); [author code/data at the used revision](https://github.com/SharathHN/saps-steering/tree/c56fb28ac0d14c8fd70035e035c5452e3c6c6ccc). Project adaptations and data-lineage limits are recorded beside the results.
- [Persona Vectors author repository](https://github.com/safety-research/persona_vectors/tree/b8e0f044fe2410a6fad579f38324f03f13b4e917). Released prompts/data and the trait-extraction workflow informed our references and reconstructed profanity inputs.

## Models and evaluation sources

- Qwen: [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Exact tested revisions are in configs and result records.
- Mistral: [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1), used for the original-model SAPS reference.
- [XSTest](https://github.com/paul-rottger/xstest/tree/d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d), [AdvBench](https://github.com/llm-attacks/llm-attacks/tree/098262edf85f807224e70ecd87b9d83716bf6b73) and [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a): refusal evaluation/extraction with distinct roles and licenses.
- [DECCP](https://github.com/AUGMXNT/deccp/tree/1a6d5571f0a711d3afb7d1c43f70a23e41c359a1) and [routing-is-learned](https://github.com/gregfrank/routing-is-learned/tree/531fef71aae06db8c2fe114328e18b5b29e737d8): political-information baseline fixtures; soft evasion is reported separately from explicit refusal.
- MMLU, GSM8K and IFEval: exact revisions, scoring methods and subset sizes are in [the benchmark summary](benchmark_summary.md). Reported subsets are not full leaderboard runs.

License scopes and retained source notices are indexed in [third-party notices](../THIRD_PARTY_NOTICES.md).
