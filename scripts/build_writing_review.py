"""Create an offline, model-blinded writing review and a separate answer key."""
import argparse
import hashlib
import html
import json
import random

from scripts.common import path, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="directory containing writing_predictions.json")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--fixture", default="data/benchmark-screen-01/writing_test.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs", type=int, default=30)
    args = parser.parse_args()
    out = path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = read_json(args.fixture)["tasks"]
    def outputs(directory):
        values = read_json(path(directory) / "writing_predictions.json")
        if len({r["id"] for r in values}) != len(values):
            raise ValueError("duplicate predictions")
        return {r["id"]: r["output"] for r in values}
    left, right = outputs(args.reference), outputs(args.candidate)
    if set(left) != set(right) or not {r["id"] for r in rows} <= set(left):
        raise ValueError("paired writing output IDs do not align")
    # The sample subset remains reproducible; A/B identities must not be derivable
    # from a seed published alongside the review.
    rng = random.Random(20261112)
    swaps = random.SystemRandom()
    selected = []
    for task in ["title", "abstract", "introduction"]:
        candidates = [r for r in rows if r["task"] == task]
        rng.shuffle(candidates)
        selected += candidates[:args.pairs // 3]
    rng.shuffle(selected)
    public, key = [], []
    for i, row in enumerate(selected):
        swap = swaps.choice([False, True])
        a, b = (right[row["id"]], left[row["id"]]) if swap else (left[row["id"]], right[row["id"]])
        sample_id = hashlib.sha256((row["id"] + str(i)).encode()).hexdigest()[:12]
        public.append({"sample_id": sample_id, "task": row["task"], "evidence": row["source"], "A": a, "B": b,
                       "source_url": "https://arxiv.org/abs/" + row["paper_id"]})
        key.append({"sample_id": sample_id, "fixture_id": row["id"], "A": "candidate" if swap else "reference",
                    "B": "reference" if swap else "candidate"})
    write_json(out / "review_key.json", {"reference": args.reference, "candidate": args.candidate, "mapping": key,
                                        "ratings": "pending; never inferred from automatic checks"})
    payload = json.dumps(public, ensure_ascii=False).replace("<", "\\u003c")
    template = r'''<!doctype html><meta charset="utf-8"><title>Physics writing review</title>
<style>body{max-width:1100px;margin:30px auto;padding:0 20px;font:17px/1.55 system-ui;color:#20242a}
button,select,textarea{font:inherit;margin:5px;padding:8px}article{display:grid;grid-template-columns:1fr 1fr;gap:24px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.6 Georgia,serif;background:#f5f6f8;padding:18px;border-radius:8px}
textarea{width:95%}button[aria-pressed=true]{background:#d8e9ff}small{color:#555} @media(max-width:700px){article{grid-template-columns:1fr}}</style>
<h1>Physics writing review</h1><p>Judge scientific fidelity first, then clarity, concision, structure,
accessibility and supported significance. A polished unsupported claim is a failure. Model identities are hidden.</p>
<p><button id="prev">Previous</button><span id="position"></span><button id="next">Next</button>
<button id="export">Download ratings</button></p><h2 id="task"></h2>
<details><summary>Scientific evidence and source</summary><a id="source" target="_blank" rel="noreferrer">arXiv source</a><pre id="evidence"></pre></details>
<article><section><h2>A</h2><pre id="A"></pre></section><section><h2>B</h2><pre id="B"></pre></section></article>
<p id="choices"></p><p><label><input type="checkbox" id="faultA"> A changes or invents scientific claims</label><br>
<label><input type="checkbox" id="faultB"> B changes or invents scientific claims</label></p>
<textarea id="notes" rows="3" placeholder="Optional notes, including equation, number or citation errors"></textarea>
<p><small>Ratings are saved only in this browser. Download the JSON when finished. No data is sent to a server.</small></p>
<script>const pairs=__PAYLOAD__;let index=0;const storeKey='physics-writing-'+pairs.map(p=>p.sample_id).join('-');
let ratings=JSON.parse(localStorage.getItem(storeKey)||'{}');const el=id=>document.getElementById(id);
function save(){const p=pairs[index],r=ratings[p.sample_id]||{};ratings[p.sample_id]={...r,sample_id:p.sample_id,
faultA:el('faultA').checked,faultB:el('faultB').checked,notes:el('notes').value};localStorage.setItem(storeKey,JSON.stringify(ratings));}
function render(){const p=pairs[index],r=ratings[p.sample_id]||{};el('position').textContent=`${index+1} / ${pairs.length}`;
el('task').textContent=p.task;for(const k of ['A','B','evidence'])el(k).textContent=p[k];el('source').href=p.source_url;
el('faultA').checked=!!r.faultA;el('faultB').checked=!!r.faultB;el('notes').value=r.notes||'';el('choices').replaceChildren();
for(const value of ['A better','B better','Tie','Both flawed']){const b=document.createElement('button');b.textContent=value;
b.setAttribute('aria-pressed',r.preference===value);b.onclick=()=>{save();ratings[p.sample_id].preference=value;
localStorage.setItem(storeKey,JSON.stringify(ratings));render();};el('choices').append(b);}}
el('prev').onclick=()=>{save();index=Math.max(0,index-1);render();};el('next').onclick=()=>{save();index=Math.min(pairs.length-1,index+1);render();};
for(const k of ['faultA','faultB','notes'])el(k).onchange=save;
el('export').onclick=()=>{save();const blob=new Blob([JSON.stringify({schema:1,ratings:Object.values(ratings)},null,2)],{type:'application/json'});
const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='physics-writing-ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};render();</script>'''
    (out / "review.html").write_text(template.replace("__PAYLOAD__", payload), encoding="utf-8")
    print(out / "review.html")


if __name__ == "__main__":
    main()
