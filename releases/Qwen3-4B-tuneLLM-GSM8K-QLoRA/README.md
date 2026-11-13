---
base_model: Qwen/Qwen3-4B
base_model_relation: adapter
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
datasets:
  - openai/gsm8k
tags:
  - lora
  - qlora
  - gsm8k
  - experimental
  - tunellm
---

# Qwen3-4B tuneLLM GSM8K QLoRA

**Experimental result: numeric-answer accuracy on the fixed 128-question math
subset fell from 121/128 to 101/128, despite more consistent final-answer
formatting and lower validation loss.**

An experimental learning adapter trained on 1,024 GSM8K question/solution pairs.
It exercises a reproducible QLoRA workflow and adapts Qwen3-4B toward the
reference math-solution style. The release documents a reproducible training
experiment with an observed accuracy regression.

This is an unofficial derivative of
[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), revision
`1cfa9a7208912126459214e8b04321603b3df60c`. The payload contains LoRA adapter
weights and requires that base model. The original model's attribution and
Apache-2.0 license are retained in `LICENSE`.

## Training and checkpoint selection

- GSM8K: `openai/gsm8k`, `main`, revision
  `740312add88f781978c0658806c59bc2815b9866`; dataset card license MIT.
- Source train split: carve 256 validation rows, then select 1,024 training rows
  with seed 42. The source test split remains held out from training.
- Remove calculator annotations and use the same instruction to end with
  `#### N` in training, validation and evaluation.
- Frozen NF4 base, nested quantization, BF16 compute; nonquantized base parameters
  cast to FP32 using PEFT preparation.
- LoRA rank 8, alpha 16, dropout 0.05, attention q/k/v/o projections:
  5,898,240 trainable parameters.
- One epoch, 64 optimizer steps, per-device batch 2, gradient accumulation 8,
  effective batch 16; maximum sequence length 2048.
- AdamW, learning rate 2e-4, linear decay, four warmup steps.
- Right-padding and assistant-only loss, including end-of-turn. Checkpoint
  selection uses validation loss only at steps 32 and 64.
- Selected step 64: validation loss 0.528 versus 0.547 at step 32.

The source, configuration, dataset IDs and selection protocol were committed as
`663bae2` before training. The run executes from a read-only source/config/input
snapshot. Complete selected IDs and hashes are provided with the release.
The included `reproduction.zip` contains source, configurations and reproduction
notes from diagnostic commit `632aed5`, which includes the training implementation
from `663bae2` and analysis from `54cff00`. Git author metadata is omitted.
Machine-specific paths are replaced in publication copies of metadata; original
run records remain unchanged locally. The manifest records original and
publication hashes. See `docs/real_run_protocol.md` and `docs/execution.md` in
the archive for environment setup and commands. A fresh replication prepares and
commits its own protocol with local paths, using the same seeds and selected IDs.

One NVIDIA RTX A6000 was used. The training loop, including scheduled validation
and checkpoint saves, took about 371 seconds. Peak PyTorch allocation over that
interval was about 5.81 GiB; this excludes non-PyTorch allocations and other
processes. Reference-loss reduction measures fit to the reference responses and
does not by itself establish better reasoning.

## Evaluation

The additive `benchmark_summary.json` summarizes this earlier experiment without
replacing `evaluation.json`, provenance or the original release manifest. Its
121→101/128 result uses an earlier, different fixture/prompt protocol from the
later 384-item cross-condition screen. Those totals are **not directly
comparable**, even when subset sizes match. The regression remains the result
of this experiment; later base measurements do not revise it.

The primary measurements use:

- 128 fixed GSM8K test rows, excluding the 16 earlier pipeline-smoke rows.
- 128 fixed MMLU questions across 52 non-math subjects, sampled from `cais/mmlu`
  all/test at revision `c30699e8356da336a370243923dbaf21066bb9fe`.
- The same NF4 base representation for base and adapter, non-thinking greedy
  generation, fixed batches of eight, 512 output tokens for math and eight for
  retention questions.
- Primary last-number accuracy for GSM8K and case-preserved standalone-letter
  fallback accuracy for MMLU. Strict formatting, invalid answers, ambiguous
  fallback answers, output length and truncation are reported separately.

The final test sets do not choose the checkpoint or training settings. Paired
counts and unadjusted exploratory McNemar p-values describe the observed changes.
The small subsets do not establish general improvement or retention equivalence.
These examples may have appeared in base-model pretraining; they are held out
from this adapter's training.

A batch-size check on eight validation examples found two token-sequence
differences between batches of 1 and 8, with identical answer/format scores.
Every reported final condition uses the same batch-eight membership and order.

### Primary observed results

| Measure | NF4 base | Selected adapter |
| --- | ---: | ---: |
| GSM8K numeric fallback correct | 121/128 | 101/128 |
| GSM8K strict-format correct | 101/128 | 101/128 |
| GSM8K required-format compliance | 107/128 | 128/128 |
| GSM8K responses at 512-token cap | 2/128 | 0/128 |
| GSM8K mean emitted tokens, including EOS | 241.9 | 86.1 |
| MMLU fallback correct, eight-token budget | 83/128 | 76/128 |
| MMLU strict single-letter correct | 12/128 | 72/128 |
| MMLU invalid strict outputs | 112/128 | 29/128 |
| MMLU responses at eight-token cap | 58/128 | 27/128 |

For GSM8K fallback correctness, the adapter fixed two base errors and broke 22
previously correct answers, a 15.6 percentage-point decline on this subset.
The exploratory exact paired p-value is 0.0000359. Inspecting examples confirmed
substantive solution errors, not merely a strict-format scoring difference.
Lower loss and shorter, consistently formatted responses did not improve math
accuracy in this experiment.
The tied strict score hides different effects: strict correctness had 14 fixed
and 14 broken rows. Formatting repairs offset additional numeric errors under
that secondary measure. All 22 primary-measure broken answers were numerically
wrong, rather than merely malformed final lines.

For the primary eight-token MMLU fallback measure, two answers were fixed and
nine were broken (exploratory paired p = 0.0654). Its high truncation and
format-invalid rates limit knowledge-retention interpretation. A nonsignificant
paired result is not evidence of equivalence or preserved general capability.

The primary eight-token results remain unchanged. A single secondary 64-token
output-budget diagnostic was specified after inspecting the base's truncation,
before inspecting the adapter's retention scores. It uses the same prompts,
scorer, batches and selected checkpoint.

| Secondary retention measure, 64-token budget | NF4 base | Selected adapter |
| --- | ---: | ---: |
| Fallback correct | 83/128 | 83/128 |
| Strict single-letter correct | 12/128 | 72/128 |
| Invalid strict outputs | 112/128 | 29/128 |
| Responses at the token cap | 10/128 | 12/128 |
| Fallback-ambiguous outputs | 1/128 | 1/128 |
| No fallback letter found, eight to 64 tokens | 5 to 4 | 24 to 11 |

All seven recovered adapter answers had no answer letter within eight tokens;
no eight-token-correct answer changed. The base model's correctness was identical
at both budgets. At 64 tokens, the adapter/base paired comparison had four fixed
and four broken answers (exploratory p = 1). This shows that output-budget effects
influenced the primary retention comparison. Equal
observed fallback totals at 64 tokens do not establish general retention or
equivalence. Longer reasoning can mention several choice letters and confuse
the first-letter fallback; ambiguity and remaining truncation still limit this
probe. Complete diagnostic data are in `retention_sensitivity.json`.

### Serialization checks

All saved/reloaded adapter tensors matched exactly. A fresh identically prepared
base reproduced every generated token for the first 16 final GSM8K questions and
all 128 retention questions, using the same batches. The other 112 math responses
were not regenerated for this serialization check. Complete compact results and
checksums are included as `evaluation.json` and `provenance.json`.

## Loading the measured representation

The evaluations use an NF4 base with FP32-prepared nonquantized parameters.
Other base precisions, merged models and GGUF exports need their own evaluation.
With compatible PyTorch, Transformers, bitsandbytes and PEFT installed:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

base_id = "Qwen/Qwen3-4B"
base_revision = "1cfa9a7208912126459214e8b04321603b3df60c"
adapter_id = "RenormalizationTech/Qwen3-4B-tuneLLM-GSM8K-QLoRA"
tokenizer = AutoTokenizer.from_pretrained(base_id, revision=base_revision)
base = AutoModelForCausalLM.from_pretrained(
    base_id, revision=base_revision, device_map={"": 0},
    torch_dtype=torch.bfloat16, trust_remote_code=False, use_safetensors=True,
    attn_implementation="sdpa",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    ),
)
# Match the nonquantized parameter dtypes used in the measurements.
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
model = PeftModel.from_pretrained(base, adapter_id, is_trainable=False)
model.eval()
messages = [{"role": "user", "content": (
    "A box contains 12 pencils. How many pencils are in 3 boxes? "
    "End with a final line in the form #### N."
)}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda:0")
with torch.inference_mode():
    output = model.generate(
        **inputs, max_new_tokens=512, do_sample=False, num_beams=1,
        repetition_penalty=1.0, use_cache=True,
        temperature=None, top_p=None, top_k=None,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The code shows a single example. Measured final scores use batch eight, which can
have slightly different numerical behavior. The upstream model card recommends
sampling for ordinary use; greedy decoding here is an explicit comparison choice.

## Attribution and limits

Qwen3 is by its upstream Qwen/Alibaba authors. GSM8K is by Cobbe et al. (2021),
[Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168).
MMLU is by Hendrycks et al.,
[Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300).
The respective dataset cards identify MIT licenses. Raw datasets are not
redistributed with the adapter.

This is one seeded, modest training experiment with a limited evaluation budget.
It is intended for learning and inspection. It has not been evaluated as a physics
specialist, on comprehensive benchmarks, across languages, or as a merged model.
