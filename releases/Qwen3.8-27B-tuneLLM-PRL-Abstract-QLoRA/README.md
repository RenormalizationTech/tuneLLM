---
license: apache-2.0
library_name: peft
pipeline_tag: text-generation
base_model: RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4
base_model_relation: adapter
tags:
- qlora
- text-only
- research
---
# Qwen3.8-27B-tuneLLM-PRL-Abstract-QLoRA

**EXPERIMENTAL; NOT RECOMMENDED**

Experimental labels do not establish improved capability. Gate-passing labels apply only to the recorded task, prompts and small subsets.

This package contains a separate adapter over [RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4/tree/e7644d6f288cb8a9ba51856ca2c9fa68946f70fe), pinned to `e7644d6f288cb8a9ba51856ca2c9fa68946f70fe`.
Modification notice (Apache-2.0): tuneLLM converted the upstream Qwen checkpoint to a text-only NF4 base by removing vision/MTP components and quantizing its language weights, then trained this separate task-specific QLoRA adapter. The adapter targets QASPER answer/evidence JSON or PRL-associated body-to-abstract generation, as identified by this package. These modifications are not an upstream Qwen release.
The original selected adapter tensor file is copied byte-for-byte. The copied configuration changes only its portable base repository and revision; config_diff.json records the hashes and field differences.

57 training and 9 development papers passed both AI claim-audit passes. Final evaluation uses 40 CC-BY arXiv papers associated with PRL, fresh to this project; they are not proven absent from pretraining. The old 20 test papers remain historical diagnostics.
All 57 selected training papers and all 40 fresh evaluation papers have cond-mat categories. Author overlap was not excluded. Paper/version and near-duplicate exclusions do not establish author-held-out or pretraining-clean generalization.
The task produces a grounded abstract of at most 200 words from a fixed body excerpt that may omit the middle. Judging uses two fresh contexts from the same AI family as the advisor; this is not independent human scientific validation.

Development judging may be incomplete: the registered fallback selects base and aliases identical baselines when the frozen selection record so indicates. This establishes no stronger-prompt claim. See the prompt selection provenance.

Actual development choice: **base**; identical-baseline alias: **True**.
Registered fallback reason: incomplete development judgments; no extra calls.
**For this abstract task, recommend the unadapted [RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4/tree/e7644d6f288cb8a9ba51856ca2c9fa68946f70fe). This adapter failed the completed registered science gates.**

Preference gives half credit to ties, averages two passes per paper, then averages papers; it is not a raw win rate.

| Comparison | Adapter preference score | 95% paper-bootstrap CI | Unsupported-claim mean difference | 95% difference CI |
|---|---:|---|---:|---|
| adapter vs base | 0.175000 | [0.075, 0.2875] | 0.025000 | [0.0, 0.075] |

The zero-increase unsupported-claim point gate failed near zero and its interval includes zero; this does not demonstrate a large factual regression.

Secondary custom diagnostics (not metrics from a ROUGE package; no gate or selection effect). LCS/token F1 compare with the reference abstract; novel unique 8-grams compare with the supplied excerpt.

| Arm | Mean words | Novel unique 8-gram fraction | Custom whitespace LCS F1 | Normalized token F1 |
|---|---:|---:|---:|---:|
| base | 176.925000 (n=40) | 0.985603 (n=40) | 0.205140 (n=40) | 0.382270 (n=40) |
| adapter | 131.325000 (n=40) | 0.525501 (n=40) | 0.244157 (n=40) | 0.414167 (n=40) |
| fixed_copy | 134.000000 (n=40) | 0.002458 (n=38) | 0.136275 (n=40) | 0.248110 (n=40) |

Means retain all 40 papers except undefined novel-8-gram fractions for outputs shorter than eight words; each cell reports its denominator. Fixed-copy empty cases are retained. Full per-paper and reviewer-agreement diagnostics are in secondary_diagnostics.json.

Source overlap increased while judged quality fell. The causal mechanism is unproven; lexical overlap does not establish support for a claim.

| Retention task | Base correct | Adapter correct | Net change |
|---|---:|---:|---:|
| general_mc | 110 | 110 | 0 |
| gsm8k | 114 | 120 | 6 |
| ifeval | 23 | 24 | 1 |

Retention gates allow at most five fewer numeric GSM8K and general-MMLU correct answers, and three fewer strict IFEval prompt passes. Numeric GSM8K uses last-number fallback; strict-format results are separately retained. Physics MMLU, if present, is diagnostic. Point limits do not establish non-inferiority.
The NF4 smoke's cached/full-prefix numerical discrepancy remains unresolved. Matching two smoke token sequences does not establish cache equivalence. Compared arms use identical PEFT preparation and cached generation. Science generation uses batch1; retention generation uses batch4 with identical fixed groups across base/QASPER/PRL, and retention MC likelihood scoring remains batch1.

Batch4 is not token-equivalent to batch1: 2/4 invented preflight probes differed, while disabled and zero-active adapters agreed within each batch size. Retention comparisons require the exact same batch4 groups and padding across all retention arms. Final science generation remains a distinct batch1 measurement; these results must not be pooled. This is descriptive evidence, not a demonstrated explanation of the batching difference.

## Load the portable adapter

Use the exact environment in environment.lock.txt. This recipe restores stored NF4 directly; it supplies no fresh quantization configuration. PEFT k-bit preparation casts nonquantized floating base parameters to FP32, as in both measured base and adapter arms. The tiny task example below is invented and contains no test-paper text. Resolve the adapter revision once after publication, then retain the printed immutable SHA for reproduction.

```python
import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from peft import PeftModel, prepare_model_for_kbit_training
base_id = "RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4"
base_revision = "e7644d6f288cb8a9ba51856ca2c9fa68946f70fe"
adapter_id = "RenormalizationTech/Qwen3.8-27B-tuneLLM-PRL-Abstract-QLoRA"
adapter_revision = HfApi().model_info(adapter_id).sha  # Resolve once; retain this immutable SHA.
print("Resolved adapter release:", adapter_revision)
tokenizer = AutoTokenizer.from_pretrained(base_id, revision=base_revision, trust_remote_code=False)
base = Qwen3_5ForCausalLM.from_pretrained(base_id, revision=base_revision, dtype=torch.bfloat16,
    device_map={"": 0}, attn_implementation="sdpa", use_safetensors=True, trust_remote_code=False)
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False})
base.config.use_cache = False
model = PeftModel.from_pretrained(base, adapter_id, revision=adapter_revision,
    is_trainable=False, autocast_adapter_dtype=True)
model.eval()
user = 'Write a grounded physics abstract in at most 200 words using only the supplied paper body excerpt, which may omit its middle and nonprose material. Preserve quantities and limitations; do not invent claims. Return only the abstract.\n\nWe study a synthetic two-site lattice model. Exact diagonalization gives a gap of two energy units. This calculation concerns only the two-site system; no thermodynamic-limit conclusion is established.'
prompt = tokenizer.apply_chat_template([{ "role": "user", "content": user}],
    tokenize=False, return_dict=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda:0")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    tokens = model.generate(**inputs, use_cache=True, do_sample=False, num_beams=1, max_new_tokens=512,
        temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, logits_to_keep=1)
print(tokenizer.decode(tokens[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# For matched base comparisons, use this same prepared model:
# with model.disable_adapter():
#     base_tokens = model.generate(**inputs, ...same settings...)
```

This staging step performed no model load or generation. Portable-config runtime replay remains a separate pinned-download check before claiming a verified adapter publication.

## Attribution and reproduction

Model/code: Apache-2.0. QASPER and paper texts retain CC-BY-4.0 terms; these are separate from model licensing. Retention data: MMLU/GSM8K MIT; IFEval and its Google Research scorer Apache-2.0. See data_licenses.json, provenance.json and paper_attributions.json where applicable.
The source/environment snapshots and selected checkpoint metadata preserve the original experiment. Sanitized metadata paths use placeholders; input hashes always describe the original local bytes. No base weights or merged adapter export is included.

## Published-artifact verification

Anonymous download of commit `adcae12715bf1a946b0424cef4038dd9cac8fc8d` verified every published file. The downloaded model then passed the recorded GPU functional reload check. For adapters, the documented PEFT loading route restored the selected weights exactly. See `verification.json` for the scope and original commit. This documentation update records that earlier check; runtime files retain their verified hashes. The original strict cache-equivalence failure remains unresolved.
