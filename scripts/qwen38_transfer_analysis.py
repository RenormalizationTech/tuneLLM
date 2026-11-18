"""Paired paper-bootstrap NLL transfer diagnostics; no selection or quality gate."""
import argparse
import json
import math
from pathlib import Path

from scripts.qwen38_science_runtime import sha,write

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/qwen38-transfer-01'


def compare(a,b,draws=10000):
    import numpy as np
    aa={r['id']:r for r in a};bb={r['id']:r for r in b}
    if not aa or len(aa)!=len(a) or len(bb)!=len(b) or set(aa)!=set(bb):raise ValueError('unique paired IDs required')
    papers=sorted({r['paper_id'] for r in a});sums={p:[0.,0.,0] for p in papers}
    for key,row in aa.items():
        other=bb[key];n=row['tokens']
        if row['paper_id']!=other['paper_id'] or n!=other['tokens'] or type(n) is not int or n<=0 or not all(math.isfinite(r['nll']) for r in [row,other]):raise ValueError('paired paper/token identity and finite NLL required')
        sums[row['paper_id']][0]+=row['nll']*n;sums[row['paper_id']][1]+=other['nll']*n;sums[row['paper_id']][2]+=n
    values=np.array([sums[p] for p in papers]);totals=values.sum(axis=0)
    indices=np.random.default_rng(20261117).integers(0,len(papers),(draws,len(papers)))
    sampled=values[indices].sum(axis=1)
    differences=(sampled[:,0]-sampled[:,1])/sampled[:,2]
    return {'difference_candidate_minus_base':float((totals[0]-totals[1])/totals[2]),
            'ci95':np.quantile(differences,[.025,.975]).tolist(),'candidate_nll':float(totals[0]/totals[2]),
            'base_nll':float(totals[1]/totals[2]),'tokens':int(totals[2]),'papers':len(papers),'items':len(a),
            'bootstrap':{'draws':draws,'seed':20261117,'unit':'paper','estimator':'token-weighted assistant/EOS NLL'}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-map',required=True,help='JSON qasper/prl completed training-run paths')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior analysis')
    protocol=json.loads((RUN/'registration.json').read_text())
    if (RUN/'registration.json').read_bytes()!=(ROOT/'results/protocols/qwen38-transfer-01.json').read_bytes():raise ValueError('protocol changed')
    results={};bindings={};training_map=json.loads(a.training_map)
    for arm in ['base','qasper','prl']:
        folder=RUN/arm;r=json.loads((folder/'result.json').read_text());binding=json.loads((folder/'binding.json').read_text())
        if not r.get('completed') or binding['registration_sha256']!=sha(RUN/'registration.json'):raise ValueError('complete registered probes required')
        if len(r['records'])!=sum(protocol['counts'].values()):raise ValueError('probe total count differs')
        if arm=='base':
            if binding['adapter_audit'] is not None:raise ValueError('base contains adapter')
        else:
            from scripts.qwen38_eval_adapter import validate
            training=ROOT/training_map[arm];selected=json.loads((training/'result.json').read_text())['selected']['path']
            if validate(selected,arm,'base')!=binding['adapter_audit']:raise ValueError('selected checkpoint differs')
        for task in ['qasper','prl']:
            file=RUN/f'{task}.jsonl'
            if sha(file)!=protocol['input_sha256'][task]:raise ValueError('encoded input changed')
            encoded=[json.loads(x) for x in file.read_text().splitlines() if x.strip()]
            expected={x['id']:x for x in encoded};records=[x for x in r['records'] if x['task']==task]
            if len(records)!=len(expected) or {x['id'] for x in records}!=set(expected):raise ValueError('probe row IDs/counts differ')
            for row in records:
                target=expected[row['id']]
                if row['paper_id']!=target['paper_id'] or row['tokens']!=sum(x!=-100 for x in target['labels'][1:]):raise ValueError('probe paper/target count differs')
        results[arm]=r
        bindings[arm]={'result_sha256':sha(folder/'result.json'),'binding_sha256':sha(folder/'binding.json')}
    comparisons={}
    for arm in ['qasper','prl']:
        for task in ['qasper','prl']:
            report=compare([r for r in results[arm]['records'] if r['task']==task],
                           [r for r in results['base']['records'] if r['task']==task])
            report['interpretation']='cross-task descriptive transfer' if arm!=task else 'checkpoint selection-set NLL; not independent in-task evidence'
            comparisons[f'{arm}_adapter_on_{task}']=report
    report={'scope':'Descriptive likelihood diagnostics; not answer/summary quality evidence',
            'registration_sha256':sha(RUN/'registration.json'),'interpretation_supplement_sha256':sha(ROOT/'results/protocols/qwen38-transfer-interpretation-01.json'),
            'source_sha256':sha(__file__),'input_result_bindings':bindings,'comparisons':comparisons,
            'limits':['Compare representations only within a task, not QASPER/PRL raw NLL magnitudes',
                      'Single-annotation QA JSON and human reference abstract likelihood; lower loss does not prove better or more faithful generation',
                      'Unadjusted descriptive intervals; no checkpoint/prompt/gate selection']}
    write(a.output,report);print(json.dumps({'output':str(a.output),'comparisons':list(comparisons)}))


if __name__=='__main__':main()
