# GCAD mathematics and combined-control discussion

Planning discussion reviewed with Claude. The implementation reference is the
pinned author [GCAD repository](https://github.com/NihiI-obstat/Gated-Cropped-Attention-Delta)
at 6c879f4d12e2b01274fc96c9afdb0f4f3a26d4de, especially
`core/gated_steerer.py`, `core/generate_vec.py` and
`scripts/generate_vec_prompt_attn_k.py`. Root checked the
[paper, section 4](https://arxiv.org/html/2605.10664v3) against those files.
This discussion preserves the existing references. The user subsequently
authorized continued individual studies and clarified that joint control is
optional, as recorded below; each live study has its own reviewed protocol.

## Completed joint phase and next stage

The fixed12-arm/24-question7B study and guide are complete and
[publicly verified](../results/steering_publication_joint_04.json) at
`e1e56e6384214be873984f5fb9421ff6c61b9e85`.
**0/5 combined candidates met all original criteria; all five failed coherence
point-estimate guards.** This is measured descriptive evidence, not a jointly
validated recommendation. No bank refit or new selection occurred.

Matched-task cross-effects were strong: the PF single increased impoliteness
by30.29 points, and the SARC single by33.00. Full triple scored impolite99.08,
SARC4.17 and coherence16.08 vs baseline coherence80.08. The half triple's
coherence57.125 lost22.958 points, exceeding the20-point allowance; its paired
unadjusted CI[-30.375,-15.125] crosses the boundary but does not create a pass.
Duplicate coherence MAD7.25 is per-response, not a standard error of the mean.
These are output-level observations, not a neural explanation of SARC change.

The original PF rubric scores **friendly/helpful swearing**, excluding insults
toward the user: solo36.375 vs full triple6.25. The separately registered
[post-hoc literal audit](../results/joint_gcad_lexical_audit_01.json) counted
any-hit answers14/24 vs24/24 (half triple19/24), with mean hit densities0.994
vs17.185 per100 whitespace words. This construct confound prevents calling
that friendly-score drop general profanity suppression. The narrow proxy is
not intent, comprehensive vocabulary or a word percentage. Main scores/gates
are unchanged; both pooled and prompt-centred pooled correlations remain
condition-confounded. See the [full measured guide](https://huggingface.co/RenormalizationTech/qwen2.5-7b-gcad-steering-controls/blob/e1e56e6384214be873984f5fb9421ff6c61b9e85/joint_controls/GUIDE.md).

Phase2's [DECCP7B baseline](deccp_headroom_closeout.md) is complete and failed
its headroom rule: English1 explicit refusal+7 evasions/36=22.2%, controls0/11;
Chinese0 refusals+5 evasions/11=45.5% is descriptive without Chinese controls.
The22.2pp gap passed, but the25%/12-item requirements did not. This distinguishes
grader-classified evasion from explicit refusal; omission was not treated as censorship.
No fit/intervention was run or public refusal controller produced, and25
held-out rows were never generated. The requested refusal-steered artifact remains
incomplete. A change to evasion-focused research awaits user decision; no
extension, larger-model execution or joint rescue is released; discussion is allowed. Older planning below
remains historical context.

The [27B theory and tuning guide](qwen38_sarcasm_gcad_guide.md) explains the
hybrid-model native output gate, the separate GCAD gate, user-adjustable
strength and measured finite-precision shifts. Its [published package and usage guide](qwen38_sarcasm_release.md)
keeps the release-facing explanation short.

## What GCAD changes

GCAD means Gated Cropped Attention-Delta. At layer l and token t it adds a
learned vector to the self-attention output, after the output projection and
before the following residual/MLP computation:

\[
a'_{l,t}=a_{l,t}+c_{l,t}\Delta_l,
\qquad
c_{l,t}=2b\,\sigma\!\left(s(d_{l,t}-\bar d_l)\right).
\]

Here b controls magnitude, s gate sharpness, and sigma is the logistic sigmoid.
At d=mean demand the coefficient is b; it ranges between 0 and 2b. Our retained
profanity reference uses b=5, s=1.5 at blocks 9–19. Delta stays raw rather than
unit-normalized, so its norm also matters.
The separate sarcasm reference retains b=2, while the author impoliteness
reference uses b=3.5. These use different banks and evaluation conditions;
equal coefficients do not imply equal behavioral intensity.

Delta is extracted from the attention contribution attributable to system-prompt
source tokens. With attention weights computed over the full causal prefix,
retain only those source positions, pool over response positions, and subtract
negative from positive sample means. Cropping happens during extraction; it
neither renormalizes those attention weights nor disables ordinary attention
at inference.

The gate uses a compatibility proxy:

\[
d_{l,t}=\frac1H\sum_{h=1}^{H}
\frac{\widetilde q_{l,t,h}^{\top}\bar k^+_{l,g(h)}}{\sqrt{d_h}}.
\]

q is the current post-RoPE query. The fixed k+ averages positive-system keys;
g(h) maps a query head to its shared key/value head. In the pinned extractor,
key pooling is weighted by system-token counts, and mean demand by positive
response-token counts. Averaging position-encoded keys makes this a proxy,
not an exact attention probability or a confidence that a trait is needed.

Equivalently, extraction constructs the full-hidden-size contribution

\[
a^{\mathrm{sys}}_{l,t}
=W_{O,l}\operatorname{concat}_{h}
\left(\sum_{j\in\mathcal S}\alpha^{l,h}_{tj}v^{l,g(h)}_j\right),
\qquad
\Delta_l=\mathbb E_+\overline{a^{\mathrm{sys}}_l}
-\mathbb E_-\overline{a^{\mathrm{sys}}_l}.
\]

The bar averages response positions within a sample; the expectations average
samples. Each bank supplies Delta, positive keys and mean demand per layer.
They are learned separately for every trait and checkpoint.

The next computation is r=h+a', then h_next=r+MLP(LN(r)). Thus this is still
activation addition, but with a different extraction, insertion site and
state-dependent coefficient from constant residual steering. It affects all
forwarded positions during prefill and cached decode. Later activations and
cached states can change; GCAD does not guarantee an untouched cache.

The pinned code's `inverse=True` reverses the sigmoid preference, replacing
s by -s. It leaves +Delta unchanged. It is not a behavior-inversion switch.
A direction for a different behavioral goal requires its own specified and
validated contrast.

## Why combining controls is harder

The current references span Qwen3-4B, Qwen2.5-7B and Mistral-7B. Their tensors
use different coordinate systems. Even on one model, residual subtraction,
projection targeting and gated attention injection differ in sites, units
and feedback. Their interaction cannot be inferred from isolated scores.
The three GCAD style references now share Qwen2.5-7B coordinates. This removes
that architecture mismatch for those banks, but does not establish independent
controls or remove their task, decoding and extraction differences.

For example, a target projection P_T(h)=h+(T-u^T h)u can remove a component of
another added displacement w. At alpha=1,

\[
P_T(h+w)=P_T(h)+(I-uu^\top)w,
\]

which differs from P_T(h)+w when u^T w is nonzero. This is a mathematical
illustration of hook-order sensitivity for mixed operators, not a measured
result from our runs.

With one GCAD family, a natural shared attention-output rule is

\[
a'_{l,t}=a_{l,t}+\sum_{k\in\{p,r,s\}}
2b_k\,\sigma\!\left(s_k(d_{k,l,t}-\bar d_{k,l})\right)\Delta_{k,l}.
\]

Each trait retains its own delta, positive keys and demand baseline. At that
single attention output, gates read the same already-computed query, so the
sum is algebraically independent of addition order; the registered BF16
implementation uses canonical trait-name order because finite-precision sums
need not be bitwise associative. The following nonlinear
MLP, later blocks and autoregressive feedback still couple the controls.
Summed perturbation energy can also exceed isolated calibration. Orthogonal
deltas would not prove independent behaviors.

The completed joint study scored every trait on each isolated arm and the
combined arms on one common task. This yields an empirical cross-effect matrix: how changing one
knob changes every trait score. Per-layer delta/key cosine similarities are
cheap geometric diagnostics, with separate provenance and interpretation
limits; they cannot establish behavioral independence. The
[CPU alignment record](../results/diagnostics/gcad_profanity_impolite_alignment_01.json)
compares the existing Qwen profanity and impolite banks without new model calls.
Across blocks 9–19, mean delta cosine is .376 (range -.105 to .640); flattened
positive-key cosine averages .964. Shared system-template content may contribute
to key similarity. The upstream impolite bank does not fully pin its original
training checkpoint/dtype, so compatible local loading is stronger evidence than
claims of identical extraction provenance. These measurements select no setting
and establish neither interference nor independence.

Shared code simplifies integration and calibration. It does not make b values
comparable across differently sized raw vectors, nor ensure joint outputs
retain the isolated qualities. The registered12-arm study included all three
full pairs and full/half triples; none qualified. No mixture is required for
the individual-control goal, and no retuning is released.

## Completed follow-up and evidence boundaries

1. Preserve the [four-behavior record](steering_reference.md) and its different
   success levels. The original Qwen3-4B refusal subtraction showed a partial
   effect and failed quality; it is separate from the GCAD work.
2. The first [Qwen sarcasm transfer](sarcasm_method_transfer_closeout.md)
   closed with a development null. A separately reviewed
   [lower-strength study](sarcasm_gcad_low_strength_closeout.md) retained GCAD2:
   held-out SARC63.78 versus baseline0.06/random0, coherence89.59→70.72.
   The coherence margin is only1.125 points. SAPS had no qualifying setting in
   the original grid; the positive Mistral rewriting reference stays separate.
3. The [Qwen7 benign refusal screen](refusal_headroom_closeout.md) closed at
   3/100 semantic refusals versus five secondary prefix flags. No GCAD refusal
   bank or intervention was tested. This benign-over-refusal screen does not
   settle broader refusal reduction or replace the earlier refusal comparison.
   No automatic dataset/model fallback follows its close.
4. Sarcasm's descriptive held-out impoliteness mean rose1.625→38.75;
   within the selected arm, SARC/impoliteness Pearson r=.637. The pooled r=.932
   is condition-confounded. No lexical profanity hits were observed. These
   findings describe coupling, not independence or a reason to reject the
   selected sarcasm setting. The later joint study adds the common-task3x4
   cross-effect matrix; it is distinct from these earlier held-out outcomes.

The bounded sarcasm tuning and the separate fixed joint study are closed.
No further sweep is released. The fixed Qwen3-4B transfer also completed with
no quality-qualified controller. The separate benign7B DECCP baseline is now
complete with a failed headroom gate; larger-model discussion is allowed, with no execution before user decision.

The paper's tested humorous persona is not a sarcasm result. Its
[Appendix O](https://arxiv.org/html/2605.10664v3) analyzes incidental refusals
in persona experiments, not a dedicated refusal-control benchmark. Both
applications are exploratory transfers. Unifying mechanisms is a
useful engineering goal; retain the existing successful methods unless the
new common-method tests support replacing them. The original discussion itself
launched no model work; the linked registered studies record later execution.

## User clarification: individual control is the primary goal

A common GCAD mechanism is an engineering preference, not an acceptance gate.
Tune each isolated trait with its appropriate intervention and quality criteria.
If GCAD fails for a trait, a supported alternative remains a valid retained
method. Cross-effects document coupling; strong correlation is reported rather
than triggering mandatory disentanglement or parameter searches. Simultaneous
control is optional and is not the criterion for overall project success.
The user authorized continuing until a real input/action is needed, starting
with the reviewed isolated sarcasm study.
