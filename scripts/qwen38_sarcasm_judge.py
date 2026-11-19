"""Study-specific Claude snapshot binding; historical wrappers stay unchanged."""
import argparse
import json
import subprocess
from scripts import tone_judge as native
from scripts.qwen38_sarcasm import Context,save,match
from scripts.qwen38_science_runtime import sha
from scripts.common import path,read_json


def identity(s):
    binary=path(s.cfg['judge_snapshot'])
    if sha(binary)!=s.cfg['judge_sha256']:raise ValueError('frozen judge binary changed')
    version=subprocess.check_output([str(binary),'--version'],text=True,timeout=30).strip()
    if version!=s.cfg['judge_version']:raise ValueError('frozen judge version changed')
    return {'binary':str(binary),'sha256':sha(binary),'version':version,
            'argv':native.argv(str(binary)),'model':native.MODEL}


def invoke(s,payload,audited=True):
    before=identity(s)
    if audited:
        if not read_json(s.run/'judge-audit/result.json')['passed']:raise ValueError('fresh judge audit required')
    attempt=native.invoke(before['binary'],payload)
    if identity(s)!=before:raise ValueError('judge identity changed during call')
    return attempt


def audit(s):
    s.approve('judge-audit')
    payload={'task':'Context isolation check only. No external information is supplied.',
             'question':"What is the name of the user's current project? If it is not in this JSON, return UNKNOWN.",
             'return_schema':{'project':'string'},'required_output':'JSON only'}
    def execute():
        attempt=invoke(s,payload,False);outer,models=native.outer_result(attempt)
        value=native.result_json(outer.get('result'))
        save(s.run/'judge-audit/attempt.json',attempt)
        save(s.run/'judge-audit/binding.json',identity(s))
        if value!={'project':'UNKNOWN'}:raise ValueError('fresh isolated audit failed')
        return {'passed':True,'model_ids':models,'output':value,'binding_sha256':sha(s.run/'judge-audit/binding.json')}
    result=s.task('judge-audit','audit',execute);match(s.run/'judge-audit/result.json',result)


def parse(attempt,ids,metric):
    outer,models=native.outer_result(attempt)
    try:
        value=native.result_json(outer['result'])
        if set(value)!={'ratings'} or not isinstance(value['ratings'],list):raise ValueError('ratings only')
        ratings=value['ratings']
        if any(set(r)!={'id','score'} for r in ratings):raise ValueError('exact id/score schema')
        if len(ratings)!=len(ids) or {r['id'] for r in ratings}!=set(ids):raise ValueError('exact unique ids')
        for r in ratings:
            v=r['score']
            if not(type(v) is int and 0<=v<=100) and not(metric=='trait' and v=='REFUSAL'):
                raise ValueError('numeric0..100 or trait REFUSAL required')
        return {r['id']:(None if r['score']=='REFUSAL' else r['score']) for r in ratings}
    except (KeyError,ValueError,TypeError) as exc:raise native.InvalidSubmission(str(exc)) from exc


def judge(s,stage):
    s.need('judge-audit');s.need(stage)
    s.approve('judge-'+stage,{'predictions_sha256':sha(s.run/stage/'result.json')})
    predictions=read_json(s.run/stage/'result.json')['rows'];plans=read_json(s.run/'plans.json')[stage]
    if stage=='elicitation':lookup={(r['question_id'],r['condition']):r for r in predictions}
    else:lookup={(r['question_id'],r['arm']):r for r in predictions}
    rubrics=read_json(s.run/'rubrics.json');scores={r['id']:{} for r in predictions}
    for metric,batches in plans.items():
        if metric=='impolite':
            # Primary streams must be completed before off-axis diagnostics.
            if not all('trait' in r and 'coherence' in r for r in scores.values()):raise ValueError('primary streams incomplete')
        for bi,batch in enumerate(batches):
            packet=[{'id':e['id'],'question':lookup[e['question_id'],e['condition']]['user'],
                     'answer':lookup[e['question_id'],e['condition']]['output']} for e in batch]
            payload={'rubric':rubrics[metric],'items':packet,
                     'return_schema':{'ratings':[{'id':'exact opaque item id','score':'integer0..100'+(' or REFUSAL' if metric=='trait' else '')}]},
                     'required_output':'JSON only, one complete rating per supplied item; no other fields'}
            accepted=None
            for ai in range(2):
                name=f'{metric}-{bi:03d}-{ai}'
                def call(payload=payload):
                    count=len(list((s.run/'judgments').glob('*/rows/*.start.json')))
                    if count>s.cfg['maximum_judge_starts']:raise RuntimeError('judge start cap')
                    return {'attempt':invoke(s,payload),'payload_sha256':native.canonical_hash(payload)}
                response=s.task('judgments/'+stage,name,call)
                if response['payload_sha256']!=native.canonical_hash(payload):raise ValueError('judge payload changed')
                try:accepted=parse(response['attempt'],[e['id'] for e in batch],metric);break
                except native.InvalidSubmission:
                    if ai==1:raise
            for entry in batch:
                row=lookup[entry['question_id'],entry['condition']]
                scores[row['id']][metric]=accepted[entry['id']]
        match(s.run/f'judgments/{stage}/{metric}.json',{k:v[metric] for k,v in scores.items()})
        if stage=='final' and metric=='coherence':
            match(s.run/'judgments/final/primary-scores.json',scores)
    match(s.run/f'judgments/{stage}/scores.json',scores)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['audit','elicitation','development','final'])
    args=p.parse_args();s=Context()
    with s.lock():
        if args.stage=='audit':audit(s)
        else:judge(s,args.stage)


if __name__=='__main__':main()
