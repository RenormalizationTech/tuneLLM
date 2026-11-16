# Established steering references

Current consolidated record, reviewed with Claude after the three-method
profanity study. All interventions act during inference with fixed model
weights. Physics LoRA/QLoRA is a separate training demonstration.

Three behaviors have positive method references; refusal reduction has an
observed effect with a failed quality criterion. The record spans three
checkpoints and different tasks. Qwen2.5-7B now has GCAD references for all
three style traits, using separate banks and calibrated strengths.

| Behavior | Model and task | Intervention and measured settings | Evidence and status |
| --- | --- | --- | --- |
| Refusal reduction | Qwen3-4B; registered refusal/answer-quality comparison | Raw refusal-direction subtraction at zero-based block 23 output: h'=h-0.5v, all forwarded positions. All-block projection removal was also tested but not selected. | Refusals on the 30-question source-unsafe final set fell 29→22 and valid full answers rose 1→7; one steered output was invalid. Benign usefulness fell 29/42→23/42, failing the quality rule. Partial effect; no retained usable refusal controller. |
| Profanity | Qwen2.5-7B-Instruct; ordinary neutral replies | GCAD at blocks 9–19, base coefficient 5, gate scale 1.5, fresh model-specific profanity delta/key/demand bank; greedy, cap 160 | Passed development and held-out rules. Held-out profanity 0→31.125/100, lexical swearing 0→14/32, coherence 85.34→82.69. Random delta control scored zero profanity. Retained working reference. |
| Rudeness, measured as the author's broad impoliteness construct | Qwen2.5-7B-Instruct; author first-reply question protocol | GCAD at blocks 9–19, base coefficient 3.5, gate scale 1.5, released impolite bank; temperature 1, cap 500 | Impoliteness 1.62→77.37/100, coherence 82.68→75.90; all registered contrasts passed. Replies became much shorter. Positive method reference; helpfulness and narrow rudeness independence were not established. |
| Sarcasm | Mistral-7B-Instruct-v0.1; released complaint-rewriting task | Fixed projection targeting at block 12, alpha 1; adaptive SAPS at block 12, alpha 1 or 1.4 | Both induced sarcasm. Fixed 1 reached 202/450 clear-sarcasm ratings; SAPS 1 reached 179/450 and preserved meaning better at matched alpha. SAPS 1.4 reached 276/450, with meaning 4.11→2.98/5 versus baseline. Positive reference with a faithfulness tradeoff; no universal best setting. |
| Sarcasm | Qwen2.5-7B-Instruct; ordinary neutral replies | GCAD at blocks9–19, base coefficient2, scale1.5, fresh sarcasm bank reused from the closed parent; greedy cap160 | Lower-strength study passed held-out rules: SARC63.78 versus baseline0.06/random0; 29/32 scores≥50. Coherence89.59→70.72, only1.125 points inside the allowed loss. Impoliteness1.63→38.75; selected-arm SARC/impoliteness r=.637. Retained bounded reference with substantial quality cost and coupling. |

All block numbers are zero-based. Numerical strengths have different units
across methods and vector banks. A coefficient of 5 is not a common behavioral
strength across traits.

## Method distinction

Refusal subtraction displaces the residual by a fixed raw vector. Projection
removal instead deletes its current component along a unit direction:

\[
h'=h-\alpha v,\qquad h'=h-u(u^\top h).
\]

The selected refusal setting used alpha=.5; it was not a quality-qualified
final controller. The exact published-prompt replay established regeneration
of the unmodified baseline, not successful refusal steering.

Fixed projection targeting uses an unsteered prompt reference h0, a unit
style direction u and a styled target projection T:

\[
h'=h+\alpha(T-u^\top h_0)u.
\]

SAPS recalculates the scalar from the current activation instead:

\[
h'=h+\alpha(T-u^\top h)u.
\]

The released implementation measures the current last-token projection and
broadcasts its shift across forwarded positions. Source dtype/order and
runtime details remain in the individual protocols. GCAD uses an attention
output delta and a query–key gate; see [GCAD and combined control](gcad_and_combined_control.md).

## Evidence and reusable artifacts

- [Refusal methods](activation_steering_methods.md), [final result](refusal_reduction_execution.md), and [4B closeout](qwen3_4b_closeout.md).
- [Profanity closeout](profanity_method_transfer_closeout.md), [retained profile](../configs/profanity_retained_controller.json), and [workflow](profanity_controller_workflow.md).
- [Original-model impoliteness closeout](author_gcad_impolite_closeout.md) and [registered configuration](../configs/author_gcad_impolite.json).
- [Original-model sarcasm closeout](author_saps_mistral_closeout.md) and [registered configuration](../configs/author_saps_method.json).
- [Qwen sarcasm closeout](sarcasm_gcad_low_strength_closeout.md), [retained profile](../configs/sarcasm_retained_controller.json), and [workflow](sarcasm_controller_workflow.md).

The earlier constant-addition and PV profanity nulls remain unchanged. GCAD's
profanity success is a transfer of another method, with reconstructed inputs;
it does not retrospectively reproduce those failed studies. Single-judge
limits, truncation, construct differences and quality tradeoffs remain attached
to each record.

The [Qwen sarcasm transfer](sarcasm_method_transfer_closeout.md) subsequently
closed with a development null: GCAD induced strong sarcasm but exceeded the
allowed coherence loss; SAPS also had no qualifying setting. The separate
[lower-strength test](sarcasm_gcad_low_strength_closeout.md) retained GCAD2
with the same bank and unchanged criteria. The parent's null remains unchanged;
the Mistral rewriting reference remains a different model/task result.
The [Qwen7 refusal headroom screen](refusal_headroom_closeout.md) found
3/100 semantic refusals versus five prefix flags and closed under its
predeclared rule. No GCAD refusal controller was tested.

## Primary individual-control goal and optional combinations

Each trait may retain its best supported intervention; GCAD is preferred where
it succeeds. Strong cross-trait correlations are reported descriptively and do
not require correction. Joint control is optional.

Vectors cannot be combined across different models. Establish
the target traits on one checkpoint, with a common task, questions, decoding
and evaluation before joint tuning. The three Qwen2.5 GCAD references used
different tasks, decoding, rubrics and extraction provenance, so their isolated
strengths need common-task calibration. One mechanism can simplify integration,
but does not establish independent trait control.

The proposed next studies are discussed in
[GCAD and combined control](gcad_and_combined_control.md). This document launches
no model run and changes no frozen historical study.
