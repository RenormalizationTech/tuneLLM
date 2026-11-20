"""Extract a small recorded demo from completed, hash-bound studies. CPU/stdlb only.

No model imports, generation, scoring, or judge-journal reads. The source paths
below are local inputs only: public output uses logical labels, hashes and public
links. Existing frozen studies are never written. Selection ignores score values.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

from scripts import gemma4_recorded_demonstration as gemma_demo

ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT/'docs/assets/short_demonstration.json'
PAGE=ROOT/'docs/short_demonstration.md'
REPO='https://github.com/RenormalizationTech/tuneLLM'
CANONICAL_SARC='61362b38ecb32581b58adb6f5e5caaaec7f16598d985ef1b834568210ca081ad'
EXPECTED_QASPER_RECORD='1603.03876::47ffc9811b037613c9c4d1ec1e4f13c08396ed1c'
QASPER_SELECTION={'question_max_characters':160,'first_reference_max_characters':200,
    'first_reference_evidence_paragraphs':1,'evidence_max_words':120,'each_answer_max_characters':420}
ARMS=(('base','Unadapted base'),('prompt','Development-selected prompt'),('adapter','QASPER QLoRA adapter'))
DOSES=(0,2,4,8,16)


def sha(file):
    value=hashlib.sha256()
    with Path(file).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))
def row_sha(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def read(file):return json.loads(Path(file).read_text())
def rows(file):return [json.loads(line) for line in Path(file).read_text().splitlines() if line.strip()]


def checked(file,expected):
    if sha(file)!=expected:raise ValueError('frozen source hash mismatch: '+Path(file).name)
    return file


def public_source(name):
    """Pin each tracked source to its last content commit, not a moving main URL."""
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',name],cwd=ROOT,text=True).strip()
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('public source must already be tracked: '+name)
    stored=subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=ROOT)
    if stored!=(ROOT/name).read_bytes():raise ValueError('public source differs from committed record: '+name)
    return {'url':f'{REPO}/blob/{commit}/{name}','sha256':sha(ROOT/name)}


def select_qasper(inputs,predictions,recorded_scores):
    """Mechanical readability filter, then first ID. No metric-value ranking."""
    eligible=[]
    for r in sorted(inputs,key=lambda r:r['id']):
        if not r['primary'] or not r['evidence_strict_primary'] or not r['references']:continue
        ref=r['references'][0];question=r['user'].rsplit('\n\nQuestion: ',1)[-1]
        if ref['unanswerable'] or len(ref['answer'])>200 or len(ref['evidence'])!=1 or len(question)>160:continue
        paragraph=next((p for p in r['paragraphs'] if p['id']==ref['evidence'][0]),None)
        if paragraph is None or len(paragraph['text'].split())>120:continue
        try:answers={arm:json.loads(predictions[arm][r['id']]['output']) for arm,_ in ARMS}
        except (KeyError,ValueError):continue
        if any(set(v)!={'answer','unanswerable','evidence'} or not isinstance(v['answer'],str)
            or type(v['unanswerable']) is not bool or not isinstance(v['evidence'],list)
            or any(not isinstance(i,str) for i in v['evidence']) or len(v['answer'])>420 for v in answers.values()):continue
        if not all(recorded_scores[arm][r['id']]['valid'] for arm,_ in ARMS):continue
        eligible.append((r,question,ref,paragraph,answers))
    if not eligible:raise ValueError('no readable recorded QASPER example; do not loosen selection silently')
    return eligible[0],len(eligible)


def extract_qasper():
    task_file=ROOT/'results/qwen38_qasper_final_science_01.json';task=read(task_file)
    combined=read(ROOT/'results/qwen38_qasper_final_01.json')
    analysis_file=checked(ROOT/'outputs/qwen38-qasper-final-analysis-01.json',task['raw_analysis_sha256'])
    analysis=read(analysis_file)
    mapping_file=checked(ROOT/'outputs/qwen38-qasper-final-runs.json',task['run_mapping_sha256']);mapping=read(mapping_file)
    prepared=checked(ROOT/'outputs/qwen38-science-01/prepared-qasper-tokenizer-01/test.jsonl',analysis['final_input_sha256'])
    predictions={};files={};score_lookup={}
    for arm,_ in ARMS:
        run=ROOT/mapping[arm]['run'];f=checked(run/'predictions.jsonl',mapping[arm]['predictions_sha256'])
        predictions[arm]={r['id']:r for r in rows(f)};files[arm]=sha(f)
        score_lookup[arm]={r['id']:r for r in analysis['arms'][arm]['scores']['strict']['records']}
    (r,question,ref,paragraph,answers),count=select_qasper(rows(prepared),predictions,score_lookup)
    manifest=read(ROOT/'data/qasper-v03-01/manifest.json')
    original=checked(ROOT/'data/qasper-v03-01/original/qasper-test-v0.3.json',manifest['extracted_files']['qasper-test-v0.3.json']['sha256'])
    paper=read(original)[r['paper_id']]
    qa=next(q for q in paper['qas'] if q['question_id']==r['question_id'])
    if qa['question']!=question:raise ValueError('prepared question differs from public dataset source')
    if manifest['license']!='CC-BY-4.0':raise ValueError('inspect dataset terms before public text export')
    cards=[]
    for arm,label in ARMS:
        source=predictions[arm][r['id']];scores=score_lookup[arm][r['id']]
        cards.append({'arm':arm,'label':label,'answer':answers[arm]['answer'],'unanswerable':answers[arm]['unanswerable'],
            'evidence_ids':answers[arm]['evidence'],'recorded_json':source['output'],
            'recorded_answer_f1':scores['answer_f1'],'recorded_evidence_f1':scores['evidence_f1'],
            'cap_hit':source['hit_token_limit'],'source_row_sha256':row_sha(source),
            'source_prediction_file_sha256':files[arm],'source_score_row_sha256':row_sha(scores)})
    return {'record_id':r['id'],'question':question,'paper':{'id':r['paper_id'],'title':paper['title'],
        'url':'https://arxiv.org/abs/'+r['paper_id']},
        'reference':{'answer':ref['answer'],'evidence_ids':ref['evidence'],'display_policy':'First stored annotator reference; recorded scores used all registered references.',
                     'source_row_sha256':row_sha(ref)},
        'evidence':{'id':paragraph['id'],'section':paragraph['section'],'text':paragraph['text'],'source_row_sha256':row_sha(paragraph)},
        'arms':cards,'aggregate':{'n':309,'strict_primary_answer_f1':combined['strict_primary_answer_f1'],
            'all_fixed_gates_pass':combined['all_fixed_task_and_retention_gates_pass'],
            'scope':'Registered custom text-only subset, not the full official QASPER benchmark.'},
        'selection':{'rule':'Lexicographically first record after fixed readability and format filters; no score-value selection.',
            'filters':QASPER_SELECTION,'eligible_records':count,'source_input_row_sha256':row_sha(r)},
        'provenance':{'prepared_inputs_sha256':sha(prepared),'recorded_analysis_sha256':sha(analysis_file),
            'run_mapping_sha256':sha(mapping_file),'dataset_source_file_sha256':sha(original),'dataset_question_row_sha256':row_sha(qa)},
        'attribution':{'dataset':'QASPER v0.3, Dasigi et al. (2021)',
            'dataset_url':f'https://huggingface.co/datasets/allenai/qasper/tree/{manifest["hf_revision"]}',
            'dataset_paper_url':'https://arxiv.org/abs/2105.03011','license':'CC-BY-4.0',
            'license_url':'https://creativecommons.org/licenses/by/4.0/',
            'source_archive_url':manifest['source_files']['qasper-test-and-evaluator-v0.3.tgz']['url'],
            'note':'Question/reference and paper excerpt from the attributed dataset; paragraph IDs are the project projection. Model answers are recorded generated text.'}}


def extract_sarcasm():
    result_file=checked(ROOT/'results/qwen38_sarcasm_followup_01.json',CANONICAL_SARC);result=read(result_file)
    run=ROOT/'outputs/qwen38-sarcasm-followup-01'
    f=checked(run/'generation/result.json',result['input_sha256']['generation/result.json'])
    predictions=read(f)['rows']
    metric_files={m:checked(run/'judge'/f'{m}.json',result['input_sha256'][f'judge/{m}.json']) for m in ['trait','coherence','impolite']}
    scores={m:read(p) for m,p in metric_files.items()}
    fixture_file=ROOT/'evaluation/qwen38_sarcasm_followup/development.json';fixture=read(fixture_file)
    first=min(fixture,key=lambda r:r['id']);records=[]
    for b in DOSES:
        arm='baseline' if b==0 else f'b{b}';row=next(r for r in predictions if r['question_id']==first['id'] and r['arm']==arm)
        if row['user']!=first['user']:raise ValueError('same-question comparison changed')
        grade={m:scores[m][row['id']] for m in scores};aggregate=result['part_B']['metrics'][arm]
        if any(type(v) is not int or not 0<=v<=100 for v in grade.values()):raise ValueError('complete recorded scalar AI scores required')
        records.append({'strength':b,'record_id':row['id'],'output':row['output'],'scores':{'sarcasm':grade['trait'],
            'coherence':grade['coherence'],'impoliteness':grade['impolite']},'cap_hit':row['hit_token_limit'],
            'generated_tokens':len(row['generated_ids']),'source_row_sha256':row_sha(row),
            'source_score_row_sha256':{m:row_sha({'id':row['id'],'score':value}) for m,value in grade.items()},
            'aggregate':{'n':aggregate['n'],'sarcasm':aggregate['trait_conservative'],'coherence':aggregate['coherence'],
                'impoliteness':result['part_B']['descriptives'][arm]['impoliteness_mean'],
                'gates':result['part_B']['gates'][str(b)] if b else None}})
    return {'question_id':first['id'],'question':first['user'],'strengths':records,'token_cap':160,
        'score_note':'Individual answer scores are recorded AI-judge ratings, not probabilities or factual-accuracy measurements.',
        'outcome':result['outcome'],'selection':{'rule':'First registered fresh-question ID, F01; all five requested strengths retained, with no score or output filtering.',
            'source_fixture_row_sha256':row_sha(first)},
        'provenance':{'canonical_result_sha256':CANONICAL_SARC,'generation_aggregate_sha256':sha(f),
            'score_file_sha256':{m:sha(p) for m,p in metric_files.items()},'fixture_sha256':sha(fixture_file)},
        'attribution':{'fixture_url':public_source('evaluation/qwen38_sarcasm_followup/development.json')['url'],
            'fixture':'Project-authored neutral questions, prepared before follow-up model outputs.',
            'outputs':'AI-generated model text; selected recorded examples, not newly evaluated answers.',
            'license_url':REPO+'/blob/main/LICENSE'}}


def privacy_check(data):
    text=canonical(data)
    if (re.search(r'/home/|/Users/|/tmp/|outputs/|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',text)
        or re.search(r'\b'+re.escape(Path.home().name)+r'\b',text)):
        raise ValueError('private paths, account identifiers or email in public demo')
    forbidden={'input_ids','generated_ids','rendered_prompt','argv','session_id','sessionId','cost_usd','total_cost','attempt'}
    def walk(value):
        if isinstance(value,dict):
            if set(value)&forbidden:raise ValueError('raw runtime/judge field in public demo')
            for item in value.values():walk(item)
        elif isinstance(value,list):
            for item in value:walk(item)
    walk(data)


def build_data():
    source_names=['results/qwen38_qasper_final_01.json','results/qwen38_qasper_final_science_01.json',
        'results/qwen38_sarcasm_followup_01.json','results/qwen38_sarcasm_public_verification_01.json',
        'results/qwen38_qasper_public_verification_01.json']
    before={n:sha(ROOT/n) for n in source_names}
    qasper=extract_qasper();sarcasm=extract_sarcasm()
    controls=read(ROOT/'results/qwen38_sarcasm_public_verification_01.json')
    qasper_release=read(ROOT/'results/qwen38_qasper_public_verification_01.json')
    data={'schema_version':2,'mode':'RECORDED','new_model_or_quality_calls':0,
        'description':'Small frozen selections from completed experiments, not live inference or a representative sample.',
        'qasper':qasper,'sarcasm':sarcasm,
        'releases':{'sarcasm':{k:controls[k] for k in ['repo_id','revision','base_repo','base_revision','release_manifest_sha256']},
            'qasper':{k:qasper_release[k] for k in ['repo_id','revision']}},
        'provenance':{'public_sources':{n:public_source(n) for n in source_names},
            'builder_sha256':sha(__file__),'row_hash_encoding':'UTF-8 JSON, sorted keys, ensure_ascii=False, allow_nan=False, compact separators.',
            'privacy':'Only selected text, scalar scores and hashes are exported. Full local artifacts and raw judge journals are not included.',
            'selection_role':'Post-hoc editorial illustration with deterministic readability/ID rules; no statistical or causal claims from these examples.'}}
    data['gemma_qasper']=gemma_demo.extract(qasper,public_source)
    data['provenance']['gemma_builder_sha256']=sha(gemma_demo.__file__)
    validate_story(data);privacy_check(data)
    if before!={n:sha(ROOT/n) for n in source_names}:raise ValueError('canonical source changed during extraction')
    return data


def esc(value):return html.escape(str(value),quote=True)


def metrics_html(scores):
    return '<dl class="sd-metrics">'+''.join(f'<div><dt>{esc(name.capitalize())}</dt><dd>{esc(value)}<span> /100</span></dd></div>' for name,value in scores.items())+'</dl>'


def answer_html(record):
    return f'<pre class="sd-answer">{esc(record["output"])}</pre><p class="sd-cap">Recorded {record["generated_tokens"]}-token answer · reached the cap; incomplete ending retained.</p>'


def validate_story(data):
    if data['qasper']['record_id']!=EXPECTED_QASPER_RECORD:
        raise ValueError('selected QASPER record changed; review the fixed PDTB narrative before publishing')
    s=data['sarcasm']
    if s['token_cap']!=160 or any(r['cap_hit'] is not True or r['generated_tokens']!=160 for r in s['strengths']):
        raise ValueError('recorded cap/length changed; review the fixed 160-token narrative before publishing')


def gate_label(gates):
    if gates is None:return 'Ordinary reference'
    labels=[]
    if not gates['trait_gain'] and not gates['individual_ge50']:
        labels.append('Too little sarcasm (mean gain and answer count)')
    elif not gates['trait_gain']:labels.append('Too little sarcasm (mean gain)')
    elif not gates['individual_ge50']:labels.append('Too few answers with clear sarcasm')
    if not gates['coherence_absolute']:labels.append('Coherence below minimum')
    if not gates['coherence_relative']:labels.append('Coherence loss exceeds limit')
    if not gates['invalid']:labels.append('Too many invalid answers')
    return '; '.join(labels) if labels else 'All registered gates pass'


def render_page(data):
    validate_story(data)
    q=data['qasper'];s=data['sarcasm'];e=q['evidence'];reference=q['reference'];release=data['releases']['sarcasm']
    gemma_section=gemma_demo.render(data['gemma_qasper']) if 'gemma_qasper' in data else ''
    qcards=[]
    for r in q['arms']:
        ids=', '.join(r['evidence_ids']) or 'None'
        qcards.append(f'''<section class="sd-card"><p class="sd-eyebrow">RECORDED · {esc(r['label'])}</p>
<h3>{esc(r['label'])}</h3><p class="sd-qa-answer">{esc(r['answer'])}</p>
<p class="sd-evidence-ids">Evidence cited: <a href="#demo-qasper-evidence">{esc(ids)}</a></p>
<dl class="sd-qa-metrics"><div><dt>Answer F1</dt><dd>{r['recorded_answer_f1']:.3f}</dd></div><div><dt>Evidence F1</dt><dd>{r['recorded_evidence_f1']:.3f}</dd></div></dl>
<details><summary>Exact recorded JSON</summary><pre class="sd-json">{esc(r['recorded_json'])}</pre></details></section>''')
    strength_sections=[]
    for r in s['strengths']:
        label='0 · off / ordinary' if r['strength']==0 else str(r['strength'])
        strength_sections.append(f'<section class="sd-card"><h4>Recorded strength {label}</h4>{metrics_html(r["scores"])}{answer_html(r)}</section>')
    table=['| Strength | Sarcasm | Coherence | Impoliteness | Registered gates |','| --- | ---: | ---: | ---: | --- |']
    for r in s['strengths']:
        a=r['aggregate'];label=gate_label(a['gates'])
        table.append(f'| {r["strength"]} | {a["sarcasm"]:.2f} | {a["coherence"]:.2f} | {a["impoliteness"]:.2f} | {label} |')
    radio=''.join(f'<label><input type="radio" name="recorded-dose" value="{b}" {"checked" if b==0 else ""}><span>{"0 · Off" if b==0 else "16 · Extreme" if b==16 else b}</span></label>' for b in DOSES)
    return f'''# Short demonstration

<link rel="stylesheet" href="../stylesheets/short_demonstration.css">

<div class="short-demo" data-demo-url="../assets/short_demonstration.json">
<div class="sd-hero"><p class="sd-eyebrow">RECORDED EXAMPLES · 5-MINUTE WALKTHROUGH · NO GPU NEEDED</p>
<p class="sd-lede">Two model families. Task adapters and an inference control.</p>
<p>Compare recorded Qwen and Gemma task-adapter examples, then switch between five recorded steering strengths.
Every answer below is saved experiment output. This page never runs a model, sends a prompt, or calls a judge.</p>
<div class="sd-jumps"><a href="#qasper-a-task-adapter">1 · Qwen paper QA</a><a href="#gemma-qasper-the-same-recorded-question">2 · Gemma paper QA</a><a href="#sarcasm-an-inference-control">3 · Sarcasm controls</a><a href="#sources-and-selection">Sources and selection</a></div></div>
</div>

## A five-minute walkthrough

1. **0:00–0:30 — Check the label.** These are recorded, post-hoc illustrations from completed studies, not a live inference demo or a representative sample.
2. **0:30–1:45 — Read the Qwen paper question.** Compare all three answers with the same reference and evidence. The adapter changes model behavior through trained low-rank weights.
3. **1:45–2:45 — Compare Gemma on that same question.** Separate the baseline's format failure from its answer content; both strict and lenient scores are shown.
4. **2:45–4:15 — Try strengths 0, 4, 8 and 16.** The question stays fixed. Notice sarcasm, coherence, impoliteness and the retained 160-token cutoffs. GCAD changes activations during generation; base weights stay frozen.
5. **4:15–5:00 — Separate the examples from the results.** Both QASPER adapters passed their custom task gates. No tested sarcasm strength passed every gate. This page does not rank the model families.

## QASPER: a task adapter

**Qwen3.8-27B**, using the same numbered paper context in all three arms.

<div class="short-demo"><div class="sd-question"><p class="sd-eyebrow">SAME QUESTION · SAME NUMBERED PAPER CONTEXT</p>
<h3>{esc(q['question'])}</h3><p>Paper: <a href="{esc(q['paper']['url'])}">{esc(q['paper']['title'])}</a></p></div>
<div class="sd-reference"><h3>Dataset reference</h3><p><strong>{esc(reference['answer'])}</strong></p>
<p>First stored annotator reference. Recorded scores use all registered references.</p>
<details open id="demo-qasper-evidence"><summary>Supporting paragraph {esc(e['id'])} · {esc(e['section'])}</summary>
<blockquote><p>{esc(e['text'])}</p></blockquote><p>Verbatim dataset paragraph; BIBREF markers are retained. Paragraph IDs are the project's text-only projection.</p></details></div>
<div class="sd-qa-grid">{''.join(qcards)}</div></div>

**What this example shows.** All three answers identify PDTB 2.0 and cite the same supporting paragraph. The baseline and prompted answers add supported split details; their lower Answer F1 reflects overlap with the short reference, **not proof that those details are false**. The adapter gives a concise answer matching that reference. “Development-selected prompt” is the schema-example prompt chosen before final evaluation, not a prompt tuned to this question.

**Across the registered primary subset, n=309:** Answer F1 was **{q['aggregate']['strict_primary_answer_f1']['base']:.3f}** for the base, **{q['aggregate']['strict_primary_answer_f1']['prompt']:.3f}** for the selected prompt, and **{q['aggregate']['strict_primary_answer_f1']['adapter']:.3f}** for the adapter. QASPER passed its registered task and retention gates within this custom text-only scope. This is not a full official QASPER benchmark or a claim of general scientific comprehension. See the [full QLoRA results](qwen38_qlora_results.md).

{gemma_section}## Sarcasm: an inference control

<div class="short-demo" id="recorded-sarcasm" data-recorded-sarcasm data-demo-url="../assets/short_demonstration.json">
<div class="sd-question"><p class="sd-eyebrow">SAME QUESTION AT EVERY STRENGTH · RECORDED {esc(s['question_id'])}</p><h3>{esc(s['question'])}</h3></div>
<p class="sd-warning">At strengths 8 and 16, the recorded answer becomes more sarcastic and impolite while coherence falls. All five answers reached the 160-token limit. No tested strength passed every registered acceptance gate.</p>
<div class="sd-interactive" hidden data-demo-interactive>
<fieldset class="sd-dose"><legend>Choose a recorded answer — this does not run inference</legend><div>{radio}</div></fieldset>
<p class="sd-status" role="status" aria-live="polite" data-demo-status></p>
<section class="sd-card sd-selected"><p class="sd-eyebrow">RECORDED OUTPUT</p><h3 data-demo-heading>Strength 0 · ordinary generation</h3>
<dl class="sd-metrics"><div><dt>Sarcasm</dt><dd><strong data-demo-score="sarcasm"></strong><span> /100</span></dd></div><div><dt>Coherence</dt><dd><strong data-demo-score="coherence"></strong><span> /100</span></dd></div><div><dt>Impoliteness</dt><dd><strong data-demo-score="impoliteness"></strong><span> /100</span></dd></div></dl>
<p class="sd-caption">Scores for this one recorded answer, from the original AI judge. Not probabilities or a factual-accuracy measure.</p>
<pre class="sd-answer" data-demo-output></pre><p class="sd-cap" data-demo-cap></p></section></div>
<details class="sd-all-answers" open data-demo-fallback><summary>All five recorded answers · text fallback</summary>
<p>All outputs are shown in full as saved. Markdown markers and incomplete endings are retained.</p>
{''.join(strength_sections)}</details>
<noscript><p>JavaScript is off. All five recorded answers and scores remain available above.</p></noscript></div>

### The tradeoff across all 32 questions

These are **arm means across n=32**, not the individual example's scores. All study answers reached the 160-token cap.

{chr(10).join(table)}

At strength 8, mean coherence **58.94** misses the registered relative floor **64.28**. Strength 16 also misses the absolute floor of 50. Low strengths preserve coherence but induce essentially no judged sarcasm. Impoliteness rises with sarcasm. The outcome remains **`exploratory_null_through_b16`**; the earlier **`development_null`** is unchanged. This was exploratory development with no independent final set. A setting-matched random direction was tested only at 16; it does not match the realised shift energy. See [all scores and measured limitations](qwen38_sarcasm_results.md).

## Optional real inference — GPU required

The browser is a recorded viewer. To run the **already published** controls yourself, first follow the [controls card](https://huggingface.co/{release['repo_id']}/tree/{release['revision']}) for the pinned NF4 base, exact runtime and downloads. From that controls checkout:

```bash
python -m runtime.example --snapshot /path/to/pinned-nf4-base --manifest-sha256 {release['release_manifest_sha256']} --strength 8 --max-new-tokens 160 --prompt "{s['question']}"
```

Controls commit: `{release['revision']}`. Base revision: `{release['base_revision']}`. Strength 8 is the owner-chosen **experimental demonstration default**, which failed the coherence gate. Other software/hardware may produce different tokens; untested intermediate strengths are not validated. This page does not execute the command.

For actual paper QA, use the separate [published QASPER adapter](https://huggingface.co/{data['releases']['qasper']['repo_id']}/tree/{data['releases']['qasper']['revision']}) and its supported JSON/evidence prompting. The recorded comparison does not combine that adapter with GCAD.

## Sources and selection

- **QASPER:** {q['selection']['eligible_records']} records pass fixed readability/format filters: primary and evidence-strict; first reference ≤200 characters with one supporting paragraph ≤120 words; question ≤160 characters; all three saved answers valid JSON and ≤420 characters. The lexicographically first ID is `{q['record_id']}`. **No score-value or improvement filter was used.** One selected example is not representative evidence.
- **Sarcasm:** `{s['question_id']}` is the first registered fresh question, selected by ID. All five requested strengths 0/2/4/8/16 are retained without score or output filtering. Scores are copied from completed scalar score files; no raw judge journals are published.
- **Gemma QASPER:** the previously selected Qwen question ID is reused without a Gemma readability or score filter. Raw model text and strict/lenient per-item metrics are copied from the hash-bound saved study files. The development-selected prompt is an alias of the baseline. The [audited result](../results/gemma4_qasper_final_01.json) and [independent audit](../results/reviews/gemma4_qasper_final_audit_review_02.md) retain the format-floor interpretation.
- **Attribution:** QASPER v0.3, Dasigi et al. (2021), [pinned dataset source]({q['attribution']['dataset_url']}), [dataset paper]({q['attribution']['dataset_paper_url']}), [CC BY 4.0]({q['attribution']['license_url']}). The selected paper and verbatim excerpt are attributed above; annotations, model text and project paragraph IDs are distinguished. Sarcasm uses [project-authored questions]({s['attribution']['fixture_url']}) and recorded model outputs; [repository licence]({s['attribution']['license_url']}).
- **Provenance:** [Frozen demo JSON](assets/short_demonstration.json) contains source-file and source-row SHA256 hashes, pinned public result links, exact selected text and scalar scores. Row hashes use canonical UTF-8 JSON. Full local generation artifacts remain local; no token arrays, private paths, judge sessions or credentials are included.

<script src="../javascripts/short_demonstration.js" defer></script>
'''


def write_owned(file,text,refresh):
    raw=text.encode()
    if file.exists() and file.read_bytes()==raw:return
    if file.exists() and not refresh:raise FileExistsError('owned artifact changed; review before --refresh-owned: '+file.name)
    file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(raw)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--refresh-owned',action='store_true')
    args=p.parse_args();data=build_data();write_owned(ASSET,json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',args.refresh_owned)
    write_owned(PAGE,render_page(data),args.refresh_owned)
    print(json.dumps({'recorded_demo':'built','qasper_record':data['qasper']['record_id'],'qasper_eligible':data['qasper']['selection']['eligible_records'],
        'sarcasm_question':data['sarcasm']['question_id'],'JSON_sha256':sha(ASSET),'model_or_quality_calls':0,
        'navigation':'Existing Start here entry; no additional demonstration sidebar pages.'},indent=2))


if __name__=='__main__':main()
