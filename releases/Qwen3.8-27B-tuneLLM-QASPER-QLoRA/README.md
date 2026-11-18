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
# Qwen3.8-27B-tuneLLM-QASPER-QLoRA

**Passed registered task and retention gates on fixed subsets**

Experimental labels do not establish improved capability. Gate-passing labels apply only to the recorded task, prompts and small subsets.

This package contains a separate adapter over [RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4/tree/e7644d6f288cb8a9ba51856ca2c9fa68946f70fe), pinned to `e7644d6f288cb8a9ba51856ca2c9fa68946f70fe`.
Modification notice (Apache-2.0): tuneLLM converted the upstream Qwen checkpoint to a text-only NF4 base by removing vision/MTP components and quantizing its language weights, then trained this separate task-specific QLoRA adapter. The adapter targets QASPER answer/evidence JSON or PRL-associated body-to-abstract generation, as identified by this package. These modifications are not an upstream Qwen release.
The original selected adapter tensor file is copied byte-for-byte. The copied configuration changes only its portable base repository and revision; config_diff.json records the hashes and field differences.

**Baseline execution disclosure: the original 10000-second phase failed after 395/403 questions. The eight never-previously-evaluated suffix questions were completed once in a separate phase capped at 900 seconds.**

All 403 questions have first-time coverage through two sequential GPU0 executions. The original failed receipt is preserved. The total 12-hour QASPER evaluation pool and all quality gates remain unchanged. This is not an uninterrupted original-cap run. See baseline_completion.json and baseline_completion_amendment.json for the segment receipts, exact suffix IDs and source hashes.

Trained on 482 QASPER questions; checkpoint selection used 101 development questions. Final: 403 questions, 309 primary and 267 evidence-strict primary. Prompt selection used a separate frozen development prefix.
Public NLP papers may have appeared in pretraining. Paragraph-ID evidence metrics and the text-only primary subset are custom; this is not the full official benchmark or a physics-specialization result.
One configuration and seed were evaluated for one epoch. The prompt baseline was the best of three fixed templates on 35 development questions. Answer F1 measures reference-answer overlap; interpret the gains under this protocol rather than as a general scientific-comprehension result.

| Arm | Strict primary Answer F1 | Lenient primary Answer F1 | Invalid strict JSON / all403 | Invalid lenient / all403 |
|---|---:|---:|---:|---:|
| base | 0.456660 | 0.456660 | 1 / 403 | 0 / 403 |
| prompt | 0.566795 | 0.583710 | 10 / 403 | 0 / 403 |
| adapter | 0.690066 | 0.690066 | 0 / 403 | 0 / 403 |

| Adapter comparison | Strict gain (95% paper CI) | Lenient gain (95% paper CI) |
|---|---|---|
| vs base | 0.233406 (0.188524, 0.280809) | 0.233406 (0.188524, 0.280809) |
| vs prompt | 0.123271 (0.083021, 0.165413) | 0.106356 (0.068216, 0.146589) |

Selected prompt prefix SHA256: `98d4f02210af9a47a5b7e5ae0bab7a30eee897e236e1ada70aba76d96f29e9ce`.

Full strict/lenient groups, evidence scores, invalid JSON, false abstentions, paired paper-bootstrap intervals and gate outcomes are in science_report.json.

Gold-unanswerable detection is a small descriptive cohort: adapter 11/16, base 12/16. These counts do not establish improved unanswerable detection.

Figure/table cases are outside the text-only primary gate. Missing visual inputs and empty-text-evidence scoring complicate their evidence scores; subgroup results remain descriptive.

| Retention task | Base correct | Adapter correct | Net change |
|---|---:|---:|---:|
| general_mc | 110 | 109 | -1 |
| gsm8k | 114 | 119 | 5 |
| ifeval | 23 | 26 | 3 |

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
adapter_id = "RenormalizationTech/Qwen3.8-27B-tuneLLM-QASPER-QLoRA"
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
user = 'Answer using only the numbered paper paragraphs. Return only JSON with keys "answer" (string), "unanswerable" (boolean), "evidence" (array of paragraph IDs). For insufficient evidence return an empty answer and evidence array with unanswerable=true.\n\n[p000000] Results\nIn a synthetic experiment, the blue sample cooled to 2 K.\n\nQuestion: What temperature did the blue sample reach?'
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

Anonymous download of commit `d598923f01d6b0ff233ab950becbd63a5babab67` verified every published file. The downloaded model then passed the recorded GPU functional reload check. For adapters, the documented PEFT loading route restored the selected weights exactly. See `verification.json` for the scope and original commit. This documentation update records that earlier check; runtime files retain their verified hashes. The original strict cache-equivalence failure remains unresolved.
