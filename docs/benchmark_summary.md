# Benchmark summary

Completed fixed-subset computational measurements. Human writing assessment is deferred and is not required to complete this reporting phase.

| Condition | General /128 | Physics /96 | GSM8K /128 | IFEval /32 |
| --- | ---: | ---: | ---: | ---: |
| nf4_base | 85 | 63 | 117 | 24 |
| nf4_adapter | 93 | 67 | 111 | 23 |
| bf16_base | 92 | 63 | 119 | 24 |
| bf16_control | 95 | 64 | 114 | 25 |
| bf16_gguf | 92 | 62 | 119 | 25 |
| q8 | 89 | 61 | 121 | 25 |
| q4 | 89 | 61 | 116 | 24 |

Each full condition has 384 items. General/physics use custom zero-shot MMLU likelihood subsets; GSM8K uses last-number fallback; IFEval uses official prompt-level strict scoring. These are not full leaderboard scores.

Saved NF4 and BF16 Transformers are different workflow representations; GGUF uses llama.cpp with batch one, while Transformers generation uses batch eight. Greedy decoding, non-thinking mode and seed 42 are fixed. GSM8K/IFEval output caps are 512/1024 tokens. Close counts do not establish equivalence or a universal quantization verdict.

The earlier GSM8K adapter experiment reported 121→101/128 under a different protocol and must not be compared directly with this matrix. The physics adapter’s 117→111/128 math regression remains reported; lower paper NLL is not evidence of better writing. Human quality assessment is deferred, with no required pending human action for these computational results.

The JSON companion records exact dataset/scorer revisions, evaluated implementation commits and hashes, published GGUF weight hashes, paired fixed/broken counts and exploratory unadjusted McNemar tests. Equal totals and nonsignificant tests are not equivalence findings. IFEval language-detection random-state sensitivity further limits close comparisons.

Historical evaluation.json, provenance, publication records and release manifests are unchanged. Per-repository documentation updates are staged separately and require a reviewed, current parent commit before application.
