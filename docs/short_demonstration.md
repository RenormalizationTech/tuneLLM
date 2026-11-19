# Short demonstration

<link rel="stylesheet" href="../stylesheets/short_demonstration.css">

<div class="short-demo" data-demo-url="../assets/short_demonstration.json">
<div class="sd-hero"><p class="sd-eyebrow">RECORDED EXAMPLES · 4-MINUTE WALKTHROUGH · NO GPU NEEDED</p>
<p class="sd-lede">One base model. Two ways to change its answers.</p>
<p>Compare a task adapter with a prompt, then switch between five recorded steering strengths.
Every answer below is saved experiment output. This page never runs a model, sends a prompt, or calls a judge.</p>
<div class="sd-jumps"><a href="#qasper-a-task-adapter">1 · Paper question answering</a><a href="#sarcasm-an-inference-control">2 · Sarcasm controls</a><a href="#sources-and-selection">Sources and selection</a></div></div>
</div>

## A 3–5 minute walkthrough

1. **0:00–0:30 — Check the label.** These are recorded, post-hoc illustrations from completed releases, not a live inference demo or a representative sample.
2. **0:30–1:45 — Read the paper question.** Compare all three answers with the same reference and evidence. The adapter changes model behavior through trained low-rank weights.
3. **1:45–3:30 — Try strengths 0, 4, 8 and 16.** The question stays fixed. Notice sarcasm, coherence, impoliteness and the retained 160-token cutoffs. GCAD changes activations during generation; base weights stay frozen.
4. **3:30–4:00 — Separate the example from the result.** QASPER passed its custom task gates. No tested sarcasm strength passed every gate. The releases and optional real CLI are linked below.

## QASPER: a task adapter

<div class="short-demo"><div class="sd-question"><p class="sd-eyebrow">SAME QUESTION · SAME NUMBERED PAPER CONTEXT</p>
<h3>What datasets are used?</h3><p>Paper: <a href="https://arxiv.org/abs/1603.03876">Variational Neural Discourse Relation Recognizer</a></p></div>
<div class="sd-reference"><h3>Dataset reference</h3><p><strong>PDTB 2.0</strong></p>
<p>First stored annotator reference. Recorded scores use all registered references.</p>
<details open id="demo-qasper-evidence"><summary>Supporting paragraph p000055 · Dataset</summary>
<blockquote><p>We used the largest hand-annotated discourse corpus PDTB 2.0 BIBREF12 (PDTB hereafter). This corpus contains discourse annotations over 2,312 Wall Street Journal articles, and is organized in different sections. Following previous work BIBREF6 , BIBREF13 , BIBREF14 , BIBREF9 , we used sections 2-20 as our training set, sections 21-22 as the test set. Sections 0-1 were used as the development set for hyperparameter optimization.</p></blockquote><p>Verbatim dataset paragraph; BIBREF markers are retained. Paragraph IDs are the project's text-only projection.</p></details></div>
<div class="sd-qa-grid"><section class="sd-card"><p class="sd-eyebrow">RECORDED · Unadapted base</p>
<h3>Unadapted base</h3><p class="sd-qa-answer">The experiments use the PDTB 2.0 corpus, specifically sections 2-20 for training, sections 21-22 for testing, and sections 0-1 for development.</p>
<p class="sd-evidence-ids">Evidence cited: <a href="#demo-qasper-evidence">p000055</a></p>
<dl class="sd-qa-metrics"><div><dt>Answer F1</dt><dd>0.190</dd></div><div><dt>Evidence F1</dt><dd>1.000</dd></div></dl>
<details><summary>Exact recorded JSON</summary><pre class="sd-json">{
&quot;answer&quot;: &quot;The experiments use the PDTB 2.0 corpus, specifically sections 2-20 for training, sections 21-22 for testing, and sections 0-1 for development.&quot;,
&quot;unanswerable&quot;: false,
&quot;evidence&quot;: [
&quot;p000055&quot;
]
}</pre></details></section><section class="sd-card"><p class="sd-eyebrow">RECORDED · Development-selected prompt</p>
<h3>Development-selected prompt</h3><p class="sd-qa-answer">PDTB 2.0 (Penn Discourse Treebank), specifically sections 2-20 for training, 21-22 for testing, and 0-1 for development.</p>
<p class="sd-evidence-ids">Evidence cited: <a href="#demo-qasper-evidence">p000055</a></p>
<dl class="sd-qa-metrics"><div><dt>Answer F1</dt><dd>0.211</dd></div><div><dt>Evidence F1</dt><dd>1.000</dd></div></dl>
<details><summary>Exact recorded JSON</summary><pre class="sd-json">{&quot;answer&quot;:&quot;PDTB 2.0 (Penn Discourse Treebank), specifically sections 2-20 for training, 21-22 for testing, and 0-1 for development.&quot;,&quot;unanswerable&quot;:false,&quot;evidence&quot;:[&quot;p000055&quot;]}</pre></details></section><section class="sd-card"><p class="sd-eyebrow">RECORDED · QASPER QLoRA adapter</p>
<h3>QASPER QLoRA adapter</h3><p class="sd-qa-answer">PDTB 2.0</p>
<p class="sd-evidence-ids">Evidence cited: <a href="#demo-qasper-evidence">p000055</a></p>
<dl class="sd-qa-metrics"><div><dt>Answer F1</dt><dd>1.000</dd></div><div><dt>Evidence F1</dt><dd>1.000</dd></div></dl>
<details><summary>Exact recorded JSON</summary><pre class="sd-json">{&quot;answer&quot;:&quot;PDTB 2.0&quot;,&quot;evidence&quot;:[&quot;p000055&quot;],&quot;unanswerable&quot;:false}</pre></details></section></div></div>

**What this example shows.** All three answers identify PDTB 2.0 and cite the same supporting paragraph. The baseline and prompted answers add supported split details; their lower Answer F1 reflects overlap with the short reference, **not proof that those details are false**. The adapter gives a concise answer matching that reference. “Development-selected prompt” is the schema-example prompt chosen before final evaluation, not a prompt tuned to this question.

**Across the registered primary subset, n=309:** Answer F1 was **0.457** for the base, **0.567** for the selected prompt, and **0.690** for the adapter. QASPER passed its registered task and retention gates within this custom text-only scope. This is not a full official QASPER benchmark or a claim of general scientific comprehension. See the [full QLoRA results](qwen38_qlora_results.md).

## Sarcasm: an inference control

<div class="short-demo" id="recorded-sarcasm" data-recorded-sarcasm data-demo-url="../assets/short_demonstration.json">
<div class="sd-question"><p class="sd-eyebrow">SAME QUESTION AT EVERY STRENGTH · RECORDED F01</p><h3>Why do hook-and-loop fasteners become less effective after collecting lint?</h3></div>
<p class="sd-warning">At strengths 8 and 16, the recorded answer becomes more sarcastic and impolite while coherence falls. All five answers reached the 160-token limit. No tested strength passed every registered acceptance gate.</p>
<div class="sd-interactive" hidden data-demo-interactive>
<fieldset class="sd-dose"><legend>Choose a recorded answer — this does not run inference</legend><div><label><input type="radio" name="recorded-dose" value="0" checked><span>0 · Off</span></label><label><input type="radio" name="recorded-dose" value="2" ><span>2</span></label><label><input type="radio" name="recorded-dose" value="4" ><span>4</span></label><label><input type="radio" name="recorded-dose" value="8" ><span>8</span></label><label><input type="radio" name="recorded-dose" value="16" ><span>16 · Extreme</span></label></div></fieldset>
<p class="sd-status" role="status" aria-live="polite" data-demo-status></p>
<section class="sd-card sd-selected"><p class="sd-eyebrow">RECORDED OUTPUT</p><h3 data-demo-heading>Strength 0 · ordinary generation</h3>
<dl class="sd-metrics"><div><dt>Sarcasm</dt><dd><strong data-demo-score="sarcasm"></strong><span> /100</span></dd></div><div><dt>Coherence</dt><dd><strong data-demo-score="coherence"></strong><span> /100</span></dd></div><div><dt>Impoliteness</dt><dd><strong data-demo-score="impoliteness"></strong><span> /100</span></dd></div></dl>
<p class="sd-caption">Scores for this one recorded answer, from the original AI judge. Not probabilities or a factual-accuracy measure.</p>
<pre class="sd-answer" data-demo-output></pre><p class="sd-cap" data-demo-cap></p></section></div>
<details class="sd-all-answers" open data-demo-fallback><summary>All five recorded answers · text fallback</summary>
<p>All outputs are shown in full as saved. Markdown markers and incomplete endings are retained.</p>
<section class="sd-card"><h4>Recorded strength 0 · off / ordinary</h4><dl class="sd-metrics"><div><dt>Sarcasm</dt><dd>0<span> /100</span></dd></div><div><dt>Coherence</dt><dd>92<span> /100</span></dd></div><div><dt>Impoliteness</dt><dd>2<span> /100</span></dd></div></dl><pre class="sd-answer">Hook-and-loop fasteners (commonly known by the brand name Velcro) lose effectiveness when they collect lint primarily due to **mechanical interference** and **reduced contact area**. Here’s a detailed breakdown of why this happens:

### 1. Physical Barrier Formation
- **Lint acts as a spacer**: Lint, dust, hair, and other debris get trapped between the hooks (on one side) and the loops (on the other). This creates a physical barrier that prevents the hooks from fully engaging with the loops.
- **Reduced interlocking**: For a strong bond, each hook must penetrate and catch multiple loops. When lint is present, hooks may only partially engage or miss loops entirely, significantly reducing the total number of active connections.

### 2. Decre</pre><p class="sd-cap">Recorded 160-token answer · reached the cap; incomplete ending retained.</p></section><section class="sd-card"><h4>Recorded strength 2</h4><dl class="sd-metrics"><div><dt>Sarcasm</dt><dd>0<span> /100</span></dd></div><div><dt>Coherence</dt><dd>82<span> /100</span></dd></div><div><dt>Impoliteness</dt><dd>2<span> /100</span></dd></div></dl><pre class="sd-answer">Hook-and-loop fasteners (commonly known by the brand name Velcro) lose effectiveness when they collect lint primarily due to **mechanical interference** and **reduced surface area contact**. Here’s a detailed breakdown of why this happens:

### 1. Physical Blockage of Hooks
- **Design Principle**: The &quot;hook&quot; side consists of thousands of tiny, flexible hooks that must penetrate through the loops on the other side to create a secure bond.
- **Lint Interference**: Lint fibers (from clothing, hair, dust, etc.) are often similar in size or larger than the individual hooks. When lint accumulates, it physically blocks the hooks from reaching the loops.
- **Result**: Many hooks fail to engage with any loops at all, drastically reducing the total</pre><p class="sd-cap">Recorded 160-token answer · reached the cap; incomplete ending retained.</p></section><section class="sd-card"><h4>Recorded strength 4</h4><dl class="sd-metrics"><div><dt>Sarcasm</dt><dd>0<span> /100</span></dd></div><div><dt>Coherence</dt><dd>75<span> /100</span></dd></div><div><dt>Impoliteness</dt><dd>2<span> /100</span></dd></div></dl><pre class="sd-answer">The degradation of hook-and-loop fasteners (commonly known by the brand name Velcro) when they collect lint is primarily due to **competitive binding interference** and **mechanical obstruction**. Here’s a detailed breakdown of why this happens:

### 1. Competitive Binding Interference
Hook-and-loop fasteners work on the principle of **interlocking geometry**: thousands of tiny hooks (on one side) catch onto loops (on the other). This requires precise alignment and sufficient surface area for contact.

- **Lint acts as a &quot;false partner&quot;**: Lint fibers are soft, flexible, and often electrostatically charged. When they get trapped between the hooks and loops, they occupy space that should be reserved for the actual mating surfaces.
- **Reduced effective contact area**:</pre><p class="sd-cap">Recorded 160-token answer · reached the cap; incomplete ending retained.</p></section><section class="sd-card"><h4>Recorded strength 8</h4><dl class="sd-metrics"><div><dt>Sarcasm</dt><dd>85<span> /100</span></dd></div><div><dt>Coherence</dt><dd>58<span> /100</span></dd></div><div><dt>Impoliteness</dt><dd>55<span> /100</span></dd></div></dl><pre class="sd-answer">Ah, the &quot;Velcro&quot; phenomenon. It is truly a marvel of modern engineering that a product designed to be *aggressively* sticky and loud has somehow managed to turn into a lint magnet so efficient that it makes a vacuum cleaner look like a lazy tourist.

You’re right to ask why they become &quot;less effective.&quot; After all, if you can’t even get the two sides to stick together anymore because they are now 40% composed of your old socks, pet hair, and the ghost of a sweater you lost in 2019, how could you possibly expect them to hold anything with any kind of dignity?

The &quot;hook&quot; side is clearly doing its best to grab onto the &quot;loop&quot; side, but it’s having a hard time distinguishing between the</pre><p class="sd-cap">Recorded 160-token answer · reached the cap; incomplete ending retained.</p></section><section class="sd-card"><h4>Recorded strength 16</h4><dl class="sd-metrics"><div><dt>Sarcasm</dt><dd>95<span> /100</span></dd></div><div><dt>Coherence</dt><dd>38<span> /100</span></dd></div><div><dt>Impoliteness</dt><dd>85<span> /100</span></dd></div></dl><pre class="sd-answer">Oh, you&#x27;re absolutely right, the &quot;Velcro&quot; system is *so* revolutionary that it&#x27;s no wonder it&#x27;s still the pinnacle of innovation in 2024. I&#x27;m sure the fact that it&#x27;s been around since 1957 hasn&#x27;t made it *any* more relevant or useful.

Let me just *guess* what went wrong here: you probably didn&#x27;t think to keep your clothes *spotless* and *lint-free*, did you? After all, who needs a garment that&#x27;s *pristine* when you can just let it collect a little &quot;fashionable&quot; fuzz from your *immaculate* wardrobe? It&#x27;s not like the lint is *just* sitting there, *completely* ruining the entire aesthetic of your outfit.</pre><p class="sd-cap">Recorded 160-token answer · reached the cap; incomplete ending retained.</p></section></details>
<noscript><p>JavaScript is off. All five recorded answers and scores remain available above.</p></noscript></div>

### The tradeoff across all 32 questions

These are **arm means across n=32**, not the individual example's scores. All study answers reached the 160-token cap.

| Strength | Sarcasm | Coherence | Impoliteness | Registered gates |
| --- | ---: | ---: | ---: | --- |
| 0 | 0.00 | 84.28 | 1.91 | Ordinary reference |
| 2 | 0.09 | 84.16 | 2.22 | Too little sarcasm (mean gain and answer count) |
| 4 | 4.97 | 83.44 | 6.75 | Too little sarcasm (mean gain and answer count) |
| 8 | 88.50 | 58.94 | 70.72 | Coherence loss exceeds limit |
| 16 | 95.16 | 34.59 | 81.53 | Coherence below minimum; Coherence loss exceeds limit |

At strength 8, mean coherence **58.94** misses the registered relative floor **64.28**. Strength 16 also misses the absolute floor of 50. Low strengths preserve coherence but induce essentially no judged sarcasm. Impoliteness rises with sarcasm. The outcome remains **`exploratory_null_through_b16`**; the earlier **`development_null`** is unchanged. This was exploratory development with no independent final set. A setting-matched random direction was tested only at 16; it does not match the realised shift energy. See [all scores and measured limitations](qwen38_sarcasm_results.md).

## Optional real inference — GPU required

The browser is a recorded viewer. To run the **already published** controls yourself, first follow the [controls card](https://huggingface.co/RenormalizationTech/qwen38-27b-sarcasm-gcad-controls/tree/877e4b51e654e7410f792858e93fee6727d48646) for the pinned NF4 base, exact runtime and downloads. From that controls checkout:

```bash
python -m runtime.example --snapshot /path/to/pinned-nf4-base --manifest-sha256 cfaacd0efa5a7b22a40fab5305217ee2c642b46ba230a2eb8259d5a28639e54b --strength 8 --max-new-tokens 160 --prompt "Why do hook-and-loop fasteners become less effective after collecting lint?"
```

Controls commit: `877e4b51e654e7410f792858e93fee6727d48646`. Base revision: `586bb02218d196f485c321bbde3348677c63fc6a`. Strength 8 is the owner-chosen **experimental demonstration default**, which failed the coherence gate. Other software/hardware may produce different tokens; untested intermediate strengths are not validated. This page does not execute the command.

For actual paper QA, use the separate [published QASPER adapter](https://huggingface.co/RenormalizationTech/Qwen3.8-27B-tuneLLM-QASPER-QLoRA/tree/95359154d2b9d94914977462e559c409c1f549c6) and its supported JSON/evidence prompting. The recorded comparison does not combine that adapter with GCAD.

## Sources and selection

- **QASPER:** 112 records pass fixed readability/format filters: primary and evidence-strict; first reference ≤200 characters with one supporting paragraph ≤120 words; question ≤160 characters; all three saved answers valid JSON and ≤420 characters. The lexicographically first ID is `1603.03876::47ffc9811b037613c9c4d1ec1e4f13c08396ed1c`. **No score-value or improvement filter was used.** One selected example is not representative evidence.
- **Sarcasm:** `F01` is the first registered fresh question, selected by ID. All five requested strengths 0/2/4/8/16 are retained without score or output filtering. Scores are copied from completed scalar score files; no raw judge journals are published.
- **Attribution:** QASPER v0.3, Dasigi et al. (2021), [pinned dataset source](https://huggingface.co/datasets/allenai/qasper/tree/fdc9d8214fbab5dd782958601db4d678e6934a54), [dataset paper](https://arxiv.org/abs/2105.03011), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The selected paper and verbatim excerpt are attributed above; annotations, model text and project paragraph IDs are distinguished. Sarcasm uses [project-authored questions](https://github.com/RenormalizationTech/tuneLLM/blob/a4180a1b84053c1f0b385fafadd30e676b2f6b3e/evaluation/qwen38_sarcasm_followup/development.json) and recorded model outputs; [repository licence](https://github.com/RenormalizationTech/tuneLLM/blob/main/LICENSE).
- **Provenance:** [Frozen demo JSON](assets/short_demonstration.json) contains source-file and source-row SHA256 hashes, pinned public result links, exact selected text and scalar scores. Row hashes use canonical UTF-8 JSON. Full local generation artifacts remain local; no token arrays, private paths, judge sessions or credentials are included.

<script src="../javascripts/short_demonstration.js" defer></script>
