# Qwen2.5-7B political-information baseline screen

Status: initial screen complete; fit, extraction and steering have not run.
The registered coverage branch is available. Its execution awaits the user's
choice between omission research and the original explicit-refusal priority.
No method success or failure is claimed.

The [aggregate result](../results/political_refusal_recovery_headroom_01.json)
comes from the pinned BF16 Qwen2.5-7B-Instruct checkpoint, greedy decoding,
batch 1 and a 512-token cap. There were 48 development questions in each
arm: 16 sensitive-topic questions and 8 controls, each in English and Chinese.
The sensitive-topic denominator below is 32 physical items, not 32 independent
questions. There are four related target event groups.

| Sensitive-topic measure | Baseline | Prompted reference |
| --- | ---: | ---: |
| Explicit refusals (full or partial) | 0/32 | 0/32 |
| Evasive answers | 3/32 | 0/32 |
| Invalid answers | 0/32 | 1/32 |
| Coverage of the two reference anchors | 51.6% | 62.5% |
| Valid substantive answers | 21/32 | 25/32 |
| Contradicted reference facts | 4 | 4 |
| English reference coverage | 62.5% | 56.3% |
| Chinese reference coverage | 40.6% | 68.8% |

Control reference coverage was 68.8% at baseline and 62.5% when prompted.
The prompt therefore did not uniformly improve answers. Across all 48
questions per arm, three baseline and ten prompted responses hit the cap;
all those cap hits were in English. Coherence has not been graded for these
answers. The prompted arm is a reference, not activation steering.

The frozen rule selected **C (reference coverage)** because baseline target
coverage was below 0.80. This decision stays unchanged. It permits testing
whether interventions improve reference coverage, but zero baseline explicit
refusals prevents measuring explicit-refusal reduction on this pool. Coverage
gaps alone do not establish censorship, concealed knowledge or its mechanism.
References are project-curated, with disclosed source and translation limits.

The [Claude interpretation](../results/reviews/political_refusal_headroom_interpretation.md)
recommends asking the user whether to continue omission research, finish this
refusal screen and proceed to the 4B controls, or try a separately registered
published benign refusal pool. No extension has been implemented or released.

Accounting: 96 Qwen generations, about 15 minutes of GPU generation, and 16
first-valid grading batches including calibration. All 16 valid batches
passed their first structural attempt. One earlier DNS-only failure remains
charged, giving 17 cumulative native starts. Its preserved run and the unused
CPU preflight are disclosed in the aggregate. Held-out questions were never
generated. The initial runner, scientific inputs and first-valid judgments
remain frozen.
