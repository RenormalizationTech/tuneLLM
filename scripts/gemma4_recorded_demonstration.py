"""Append Gemma's recorded answers to the previously selected Qwen question.

Only copies saved text and saved per-item scores. No model or scorer imports.
The example ID was fixed before the Gemma study; no Gemma answer is filtered.
"""
import hashlib
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RECORD="1603.03876::47ffc9811b037613c9c4d1ec1e4f13c08396ed1c"
CANONICAL="a149de809997eb8ec18b670a4c1d32eca13edf014bd07fc28333abad3d50b39f"


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):value.update(chunk)
    return value.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().split("\n") if s.strip()]
def row_sha(row):return hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()).hexdigest()


def extract(qwen_record,public_source):
    if qwen_record["record_id"]!=RECORD:raise ValueError("Gemma must use the pre-existing Qwen demo selection")
    canonical=ROOT/"results/gemma4_qasper_final_01.json"
    if sha(canonical)!=CANONICAL:raise ValueError("Gemma canonical result changed")
    result=read(canonical)
    r0=ROOT/"results/protocols/gemma4-qasper-01.json"
    if sha(r0)!=result["registration_sha256"]:raise ValueError("Gemma registration differs")
    study=read(r0)
    amendment=ROOT/result["analysis_amendment"]["registration"]
    if sha(amendment)!=result["analysis_amendment"]["sha256"]:raise ValueError("analysis amendment differs")
    seal=read(amendment)["fixed_inputs"]
    final=ROOT/study["data"]["final"]["path"]
    if sha(final)!=study["data"]["final"]["sha256"]:raise ValueError("Gemma final input changed")
    question=next(r for r in rows(final) if r["id"]==RECORD)
    if row_sha(question)!=qwen_record["selection"]["source_input_row_sha256"]:
        raise ValueError("Gemma and Qwen input records differ")
    arms=[]
    for arm,role,label in (("base","final_base","Unadapted Gemma base"),("adapter","final_adapter","Gemma QASPER QLoRA adapter")):
        run=ROOT/study["prospective_runs"][role]
        files=[run/"predictions.jsonl",run/"scores.json"]
        for file in files:
            if sha(file)!=seal[file.relative_to(ROOT).as_posix()]:raise ValueError("saved Gemma demo evidence changed")
        prediction=next(r for r in rows(files[0]) if r["id"]==RECORD)
        scores=read(files[1])["base"]
        modes={}
        for mode in ("strict","lenient"):
            score=next(r for r in scores[mode]["records"] if r["id"]==RECORD)
            modes[mode]={k:score[k] for k in ("valid","answer_f1","evidence_f1")}
            modes[mode]["source_score_row_sha256"]=row_sha(score)
        arms.append({"arm":arm,"label":label,"recorded_output":prediction["output"],"scores":modes,
            "cap_hit":prediction["hit_token_limit"],"generated_tokens":len(prediction["generated_ids"]),
            "source_row_sha256":row_sha(prediction),"source_prediction_file_sha256":sha(files[0]),
            "source_score_file_sha256":sha(files[1])})
    aggregates={arm:{mode:{"answer_f1":result["arms"][arm][mode]["primary"]["answer_f1"],
        "evidence_f1":result["arms"][arm][mode]["primary"]["evidence_f1"]} for mode in ("strict","lenient")} for arm in ("base","adapter")}
    source_names=["results/gemma4_qasper_final_01.json","results/protocols/gemma4-qasper-01.json",
        "results/protocols/gemma4-qasper-analysis-amendment-01.json",
        "results/reviews/gemma4_qasper_final_audit_review_02.md","results/short_demonstration_01.json"]
    publication=ROOT/"results/gemma4_qasper_public_verification_01.json"
    release=None
    if publication.exists():
        published=read(publication)
        if published.get("status")!="published_and_verified" or published.get("anonymous_pinned_download_verified")is not True:
            raise ValueError("public Gemma release must be verified before inclusion")
        release={k:published[k] for k in ("repo_id","revision","release_manifest_sha256")}
        source_names.append(publication.relative_to(ROOT).as_posix())
    return {"model":"Gemma 4 31B IT","record_id":RECORD,"arms":arms,"release":release,
        "prompt_alias":"Development selection tied at strict F1 zero and chose base; prompted is an alias, not an additional generation.",
        "aggregate":{"n":309,"scores":aggregates,"all_fixed_gates_pass":result["all_fixed_task_and_retention_gates_pass"],
            "lenient_gain_ci95":result["analysis"]["bootstrap"]["lenient"]["base"]["ci95"]},
        "selection":{"rule":"Same ID already fixed for the Qwen demo before Gemma evaluation; no Gemma score or output filtering.",
            "prior_selection_receipt_sha256":sha(ROOT/"results/short_demonstration_01.json"),"source_input_row_sha256":row_sha(question)},
        "provenance":{"canonical_result_sha256":CANONICAL,"public_sources":{name:public_source(name) for name in source_names}},
        "attribution":"Same attributed QASPER question, reference and numbered paper context displayed in the Qwen section above; Gemma answers are recorded model output."}


def render(record):
    if record["record_id"]!=RECORD:raise ValueError("fixed Gemma demonstration ID changed")
    cards=[]
    for arm in record["arms"]:
        strict,lenient=arm["scores"]["strict"],arm["scores"]["lenient"]
        cards.append(f'''<section class="sd-card"><p class="sd-eyebrow">RECORDED · GEMMA 4 31B</p>
<h3>{html.escape(arm['label'])}</h3><pre class="sd-json">{html.escape(arm['recorded_output'])}</pre>
<dl class="sd-qa-metrics"><div><dt>Strict Answer F1</dt><dd>{strict['answer_f1']:.3f}</dd></div>
<div><dt>Lenient Answer F1</dt><dd>{lenient['answer_f1']:.3f}</dd></div>
<div><dt>Lenient Evidence F1</dt><dd>{lenient['evidence_f1']:.3f}</dd></div></dl>
<p class="sd-cap">{arm['generated_tokens']} recorded tokens · {'reached the cap' if arm['cap_hit'] else 'original ending retained'}</p></section>''')
    aggregate=record["aggregate"];scores=aggregate["scores"];ci=aggregate["lenient_gain_ci95"]
    published=record.get("release")
    usage=(f"Use the [pinned Gemma adapter](https://huggingface.co/{published['repo_id']}/tree/{published['revision']}) "
        "with the [verified local loading instructions](gemma4_qasper_results.md#use-the-adapter).\n\n") if published else ""
    return f'''## Gemma QASPER: the same recorded question

The same question, reference and supporting paragraph above are reused for **Gemma 4 31B**.
The question ID was chosen for the Qwen demonstration before Gemma evaluation.
No Gemma score or output was used to select this example.

<div class="short-demo"><div class="sd-qa-grid sd-gemma-grid">{''.join(cards)}</div></div>

**Read format and content separately.** The baseline identifies PDTB 2.0 and cites
the supporting paragraph, but its Markdown code fence is retained exactly as
generated. The strict parser rejects that format; the
lenient diagnostic recovers its answer. Its lower overlap with the short
reference does not establish that the additional details are false. The adapter
gives the short answer in the required plain JSON format. The development-selected
Gemma prompt was the baseline itself, so there is no duplicate prompted panel.

**Across the custom primary subset, n={aggregate['n']}:** strict Answer F1 was
**{scores['base']['strict']['answer_f1']:.3f} → {scores['adapter']['strict']['answer_f1']:.3f}**;
lenient Answer F1 was **{scores['base']['lenient']['answer_f1']:.3f} → {scores['adapter']['lenient']['answer_f1']:.3f}**,
with a paired paper-bootstrap gain interval **[{ci[0]:.3f}, {ci[1]:.3f}]**.
All 403 baseline answers were fenced, so the strict gain includes a large format
compliance component. Evidence superiority was not established. The task and
retention gates passed. These reused project questions do not support a general
Gemma-versus-Qwen ranking. See [Gemma QASPER results and use](gemma4_qasper_results.md).

{usage}'''
