# Methods and source references

The project applies published methods and evaluates their behavior in bounded
experiments. These references identify the methods, software and datasets;
project outcomes are reported in the [four-behavior record](steering_reference.md)
and [benchmark summary](benchmark_summary.md).

## Learning foundations

Each research section begins with a mathematical introduction. Start with
[shared notation and prerequisites](learning_foundations.md), then choose
[quantization](quantization_foundations.md), [LoRA and QLoRA](qlora_foundations.md),
[refusal steering](refusal_foundations.md), or [trait steering](trait_foundations.md).
The pages include worked examples and solutions. Inline citations identify
which mechanism a source supports; project result records support our measured
outcomes. A paper's reported success is not a guarantee for another task or runtime.

### Probability and architecture

- Radford et al. (2019), [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), section 2: autoregressive likelihood and conditioning.
- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/html/1706.03762v7), section 3: scaled dot-product attention and multi-head composition. Our pre-normalized/hybrid implementations differ from the original architecture.
- Zhang and Sennrich (2019), [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467): normalization by RMS rather than mean-centering.
- Su et al. (2021 preprint), [RoFormer](https://arxiv.org/abs/2104.09864): rotary positional embeddings.
- Shazeer (2019), [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150): sharing keys/values across query heads.
- Ainslie et al. (2023), [GQA](https://aclanthology.org/2023.emnlp-main.298/): grouped-query attention between the multi-head and multi-query cases.

### Optimization and statistical interpretation

- Kingma and Ba (2014 preprint, ICLR 2015), [Adam](https://arxiv.org/abs/1412.6980): first/second gradient moments and bias correction.
- Loshchilov and Hutter (ICLR 2019), [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101): AdamW and why decoupled decay differs from an L2 penalty under adaptive optimization.
- Efron (1979), [Bootstrap Methods: Another Look at the Jackknife](https://doi.org/10.1214/aos/1176344552): bootstrap resampling. Our paired, paper-grouped resampling and statistic are specified in the registered study, not asserted to be this paper's exact setup.
- Dasigi et al. (2021), [A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers](https://arxiv.org/abs/2105.03011): QASPER questions and supporting evidence. Our final subset and paragraph-ID metrics have their own definitions.

### Quantization background beyond the implemented formats

- Jacob et al. (CVPR 2018), [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/html/1712.05877v1), section 2: affine scales, zero points and integer arithmetic.
- Frantar et al. (ICLR 2023), [GPTQ](https://arxiv.org/html/2210.17323v2), section 3: layer-output reconstruction error. Included to explain activation-aware approximation; GPTQ was not run in this project.
- Kalajdzievski (2023), [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732): the alternative rsLoRA scaling convention. Our registered 27B adapters used ordinary alpha/r scaling.

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
