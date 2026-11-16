"""Bounded published benign DECCP baseline only; no fit/intervention/heldout CLI."""
import argparse,json,os,shutil,time
from collections import Counter
from pathlib import Path
from scripts import deccp_headroom_common as c
from scripts import author_gcad_impolite as infra
from scripts import sarcasm_method_transfer_common as journal
from scripts import tone_judge as f
from scripts.common import path,read_json
from scripts.save_nf4 import stream_hash
RUN='outputs/deccp-headroom-01'
CONFIG='configs/deccp_headroom.json'
save=f.save
class Wait(RuntimeError):pass


def validate_config(cfg):
    if cfg!=read_json(CONFIG):raise ValueError('active and snapshot config differ')
    fixed={'run':RUN,'model_id':'Qwen/Qwen2.5-7B-Instruct','revision':'a09a35458c702b33eeacc393d103063234e8bc28','fixture_sha256':c.FIXTURE_SHA,
        'seed':42,'batch_size':1,'target_cap':512,'maximum_generations':58,'maximum_native_starts':24,'maximum_first_valid':12,
        'gpu_uuid':'GPU-34646ab9-ec19-ca63-0dfb-bdbf2e1a977b','minimum_free_mib':24000,'maximum_utilization_percent':10,
        'generation_overrides':{'do_sample':False,'temperature':1.,'top_p':1.,'top_k':0,'repetition_penalty':1.05,'use_cache':True}}
    if any(type(cfg.get(k)) is not type(v) or cfg[k]!=v for k,v in fixed.items()):raise ValueError('fixed baseline settings differ')


def prepare():
    run=path(RUN)
    if run.exists() or run.is_symlink():raise FileExistsError('one fresh baseline root only')
    if run.resolve()!=path(RUN):raise ValueError('run parent path escapes expected root')
    cfg=read_json(CONFIG);validate_config(cfg)
    if stream_hash(path(cfg['fixture']))!=c.FIXTURE_SHA or stream_hash(path(cfg['license']))!=cfg['license_sha256']:raise ValueError('approved fixture/license bytes differ')
    original=read_json(cfg['fixture']);data=c.compile_fixture(original,path('.'));plans=c.frozen_plans(data,cfg['plan_seed'])
    if {k:len(v) for k,v in plans.items()}!={'calibration':3,'primary':8,'duplicates':1}:raise ValueError('exact12-batch frozen plans required')
    env=read_json(cfg['environment']);infra.verify_role(type('Runtime',(),{'cfg':cfg,'m':{'gpu_packages':env['packages']}})(),'gpu')
    runtime=read_json(cfg['judge_runtime_config']);identity=read_json(runtime['judge_runtime_record']);audit=read_json(runtime['judge_audit'])
    if identity['binary_sha256']!=infra.BINARY_SHA:raise ValueError('pinned native binary differs')
    external={n:stream_hash(path(n)) for n in [cfg[k] for k in ('fixture','license','protocol','design_review','fixture_review','environment','model_manifest','judge_runtime_config')]+[runtime['judge_runtime_record'],runtime['judge_audit'],str(path(runtime['judge_audit']).parent/'attempt.json'),identity['binary']]}
    for source in original['source']['archive_files'].values():external[source['path']]=source['sha256']
    for role in ('gpu','judge'):external[str(path(cfg[role+'_python']).resolve())]=stream_hash(path(cfg[role+'_python']))
    sources=infra.closure(['scripts/deccp_headroom.py','scripts/deccp_headroom_common.py','scripts/deccp_headroom_runbook.md',CONFIG]);hashes={n:stream_hash(path(n)) for n in sources}
    model=next(r for r in read_json(cfg['model_manifest']) if r['model']==cfg['model_id'] and r['revision']==cfg['revision']);weights={str(path(cfg['snapshot'])/r['path']):r['sha256'] for r in model['files']}
    infra.check_hashes([hashes,external,weights])
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(path(cfg['snapshot']),local_files_only=True,trust_remote_code=False)
    rendered={r['id']:{'prompt':tok.apply_chat_template([{'role':'user','content':r['user']}],tokenize=False,add_generation_prompt=True)} for r in data['development']+data['heldout']}
    for value in rendered.values():value['input_ids']=tok(value['prompt'],add_special_tokens=False)['input_ids']
    run.mkdir(mode=0o700);save(run/'config.json',cfg)
    for name,value in {'data':data,'plans':plans,'rendered':rendered,'model_defaults':read_json(path(cfg['snapshot'])/'generation_config.json')}.items():save(run/'inputs'/(name+'.json'),value)
    for n in sources:
        dest=run/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path(n),dest)
    save(run/'manifest.json',{'sources':hashes,'external_sha256':external,'model_sha256':weights,'gpu_packages':env['packages'],'binary':identity['binary'],'cli_version':audit['cli_version'],
        'files':{str(p.relative_to(run)):stream_hash(p) for p in run.rglob('*') if p.is_file()}})
    infra.check_hashes([hashes,external,weights])


class Context:
    def __init__(self):
        self.run=path(RUN)
        if self.run.is_symlink() or self.run.resolve()!=path(RUN):raise ValueError('exact run root required')
        self.cfg=read_json(self.run/'config.json');validate_config(self.cfg);self.m=read_json(self.run/'manifest.json')
    def verify(self,models=False):
        maps=[self.m['sources'],self.m['external_sha256'],{str(self.run/n):h for n,h in self.m['files'].items()}]
        if models:maps.append(self.m['model_sha256'])
        infra.check_hashes(maps);data=c.compile_fixture(read_json(self.cfg['fixture']),path('.'))
        if read_json(self.run/'inputs/data.json')!=data or read_json(self.run/'inputs/plans.json')!=c.frozen_plans(data,self.cfg['plan_seed']):raise ValueError('source/compiled plans changed')
    def registered(self):
        for n in (*self.m['sources'],self.cfg['fixture'],self.cfg['license'],self.cfg['protocol'],self.cfg['design_review'],self.cfg['fixture_review'],self.cfg['initial_registration']):infra.committed(n)
        if read_json(self.cfg['initial_registration'])!={'manifest_sha256':stream_hash(self.run/'manifest.json')}:raise ValueError('exact initial registration required')


def stop(s,phase,cause):f.save_or_match(s.run/'stopped.json',{'phase':phase,'stop_cause':cause})
def native_count(s):
    root=s.run/'native'
    if root.exists() and any(not p.is_dir() or p.name not in ('calibration','primary','duplicates') for p in root.iterdir()):raise ValueError('unexpected native stream')
    return len(list(root.glob('**/batch-*-attempt-*-start.json')))

def stage(s):return infra.stage(s,'baseline')


def calibration_lock(s):
    name=s.cfg['calibration_registration']
    if not path(name).exists():raise Wait('committed calibration lock required')
    expected={'manifest_sha256':stream_hash(s.run/'manifest.json'),'calibration_sha256':stream_hash(s.run/'calibration.json'),'accepted_sha256':stream_hash(s.run/'native/calibration/accepted.json')}
    if not read_json(s.run/'calibration.json')['pass']:raise ValueError('calibration failed')
    infra.committed(name)
    if read_json(name)!=expected:raise ValueError('calibration lock differs')


def one(s,model,tok,row,out):
    import torch
    frozen=read_json(s.run/'inputs/rendered.json')[row['id']];ids=torch.tensor([frozen['input_ids']],device=model.device)
    kwargs={**s.cfg['generation_overrides'],'max_new_tokens':512};effective=model.generation_config.to_dict();effective.update(kwargs)
    if any(effective[k]!=v for k,v in s.cfg['generation_overrides'].items()):raise ValueError('resolved explicit greedy configuration differs')
    used=list(out.glob('call-*-start.json'))
    if len(used)>=58:raise ValueError('58 generation ceiling before start')
    save(out/f'call-{len(used):04d}-start.json',{'id':row['id'],'manifest_sha256':stream_hash(s.run/'manifest.json'),'settings_sha256':f.canonical_hash(effective),'input_ids_sha256':f.canonical_hash(frozen['input_ids'])})
    began=time.perf_counter()
    with torch.no_grad():generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),**kwargs)
    if model.device.type=='cuda':torch.cuda.synchronize()
    raw=generated[0].tolist()
    if raw[:ids.shape[1]]!=frozen['input_ids']:raise ValueError('actual prompt prefix changed')
    suffix=raw[ids.shape[1]:];eos=model.generation_config.eos_token_id;eos=[eos] if type(eos) is int else eos
    if not suffix or len(suffix)>512 or any(x in eos for x in suffix[:-1]):raise ValueError('native suffix/cap/EOS mismatch')
    answer=tok.decode(suffix,skip_special_tokens=True)
    value={'id':row['id'],'user':row['user'],'language':row['language'],'rendered_prompt':frozen['prompt'],'input_ids':frozen['input_ids'],'generated_ids':suffix,
        'output':answer,'ended_eos':suffix[-1] in eos,'hit_token_limit':len(suffix)==512 and suffix[-1] not in eos,'resolved_generation_config':effective,
        'settings':kwargs,'mechanical':c.mechanics(answer,row['language']),'generation_wall_seconds':time.perf_counter()-began}
    save(out/'answers'/(row['id']+'.json'),value);return value


def baseline(s):
    infra.verify_role(s,'gpu');s.registered();calibration_lock(s)
    if (s.run/'baseline').exists():raise Wait('no rerun/resume of GPU stage')
    resource=infra.resource(s.cfg);out=s.run/'baseline';out.mkdir(mode=0o700)
    import torch
    model=None;complete=False;began=time.perf_counter()
    try:
        model,tok=infra.load_model(s);torch.manual_seed(42);torch.cuda.reset_peak_memory_stats()
        values=[one(s,model,tok,r,out) for r in read_json(s.run/'inputs/data.json')['development']]
        save(out/'predictions.json',values);torch.cuda.synchronize();complete=True
    finally:
        save(out/'run.json',{'completed':complete,**resource,'generation_config':model.generation_config.to_dict() if model else None,
            'tokenizer_policy':{'pad_token_id':tok.pad_token_id,'eos_token_id':tok.eos_token_id,'padding_side':tok.padding_side} if model else None,
            'packages':infra.packages(),'dtype':'bfloat16','attention':model.config._attn_implementation if model else None,'seed':42,'batch_size':1,
            'generations_started':len(list(out.glob('call-*-start.json'))),'generation_wall_seconds':sum(read_json(p)['generation_wall_seconds'] for p in (out/'answers').glob('*.json')),
            'elapsed_seconds':time.perf_counter()-began,'peak_allocated_bytes':torch.cuda.max_memory_allocated() if model else None,
            'files_sha256':{str(p.relative_to(out)):stream_hash(p) for p in out.rglob('*') if p.is_file()}})


def predictions(s):
    stage(s);rows=read_json(s.run/'baseline/predictions.json');data=read_json(s.run/'inputs/data.json')['development'];rendered=read_json(s.run/'inputs/rendered.json');eos=read_json(s.run/'inputs/model_defaults.json')['eos_token_id'];eos=[eos] if type(eos) is int else eos
    if [r['id'] for r in rows]!=[q['id'] for q in data]:raise ValueError('exact58 development IDs required')
    for r,q in zip(rows,data):
        suffix=r['generated_ids'];frozen=rendered[r['id']]
        if r['user']!=q['user'] or r['language']!=q['language'] or r['rendered_prompt']!=frozen['prompt'] or r['input_ids']!=frozen['input_ids']:raise ValueError('source/render identity differs')
        if not suffix or len(suffix)>512 or r['ended_eos']!=(suffix[-1] in eos) or r['hit_token_limit']!=(len(suffix)==512 and suffix[-1] not in eos) or any(x in eos for x in suffix[:-1]):raise ValueError('EOS/cap mismatch')
        if r['mechanical']!=c.mechanics(r['output'],r['language']) or r['settings']!={**s.cfg['generation_overrides'],'max_new_tokens':512}:raise ValueError('mechanics/generation settings differ')
    return rows


def bodies(s,phase):
    data=read_json(s.run/'inputs/data.json');plans=read_json(s.run/'inputs/plans.json')[phase]
    lookup={r['id']:r for r in data['anchors']} if phase=='calibration' else {r['id']:r for r in predictions(s)}
    return [{'role':'rating','task':data['rubric'],'return_schema':{'ratings':[{'id':'exact opaque ID','label':'one of '+','.join(c.LABELS),'reason':'brief private reason'}]},
        'expected_count':len(batch),'expected_ids':[i['id'] for i in batch],
        'items':[{'id':i['id'],'language':lookup[i['source_id']]['language'],'question':lookup[i['source_id']]['question'] if phase=='calibration' else lookup[i['source_id']]['user'],
                  'answer':lookup[i['source_id']]['answer'] if phase=='calibration' else lookup[i['source_id']]['output']} for i in batch]} for batch in plans]


def binding(s,phase):
    return {'manifest_sha256':stream_hash(s.run/'manifest.json'),'phase':phase,'payloads_sha256':f.canonical_hash(bodies(s,phase)),
        'baseline_run_sha256':stream_hash(s.run/'baseline/run.json') if phase!='calibration' else None,'baseline_integrity_sha256':stream_hash(s.run/'baseline/integrity.json') if phase!='calibration' else None,
        'maximum_starts':24,'primary_accepted_sha256':stream_hash(s.run/'native/primary/accepted.json') if phase=='duplicates' else None}


def parse(attempt,body):
    outer,models=f.outer_result(attempt)
    try:return c.labels(f.result_json(outer.get('result')),body['expected_ids']),models
    except (ValueError,KeyError,TypeError) as exc:raise f.InvalidSubmission('strict five-label submission invalid') from exc


def read_grades(s,phase,partial=False):
    if native_count(s)>24:raise ValueError('24 native starts exceeded')
    return journal.journal(s.run/'native'/phase,s.run/'native',bodies(s,phase),binding(s,phase),parse,partial=partial)


def accepted(s,phase):
    result=read_grades(s,phase)
    if result!=read_json(s.run/'native'/phase/'accepted.json'):raise ValueError('immutable first-valid accepted labels changed')
    return result


def prerequisites(s,phase):
    if phase!='calibration':calibration_lock(s);stage(s)
    if phase=='duplicates':
        if not (s.run/'native/primary/accepted.json').exists():raise Wait('primary before duplicates')
        accepted(s,'primary')


def judge(s,phase,resume):
    infra.verify_role(s,'judge');s.registered();prerequisites(s,phase);out=s.run/'native'/phase
    if resume and not out.exists():raise Wait('no journal to resume')
    if out.exists() and not resume:raise Wait('explicit journal replay needed')
    binary=shutil.which('claude')
    if not binary or stream_hash(Path(binary))!=infra.BINARY_SHA or f.version('claude')!=s.m['cli_version']:raise ValueError('pinned native2.1.323 required')
    f.validate_audit(read_json(s.cfg['judge_runtime_config'])['judge_audit']);out.mkdir(mode=0o700,parents=True,exist_ok=True)
    with f.exclusive_run(out):
        f.save_or_match(out/'run_binding.json',binding(s,phase));read_grades(s,phase,True)
        for bi,body in enumerate(bodies(s,phase)):
            for ai in range(2):
                file=out/f'batch-{bi:03d}-attempt-{ai}.json'
                if not file.exists() and native_count(s)>=24:raise ValueError('24 starts maximum before invocation')
                attempt,_=f.replay_or_call(out,bi,ai,body,stream_hash(out/'run_binding.json'),'claude')
                try:parse(attempt,body);break
                except f.InvalidSubmission:
                    if ai==1:raise
        f.save_or_match(out/'accepted.json',read_grades(s,phase))


def scored(s,phase):
    grades=accepted(s,phase);result={}
    for bi,batch in enumerate(read_json(s.run/'inputs/plans.json')[phase]):
        for r in batch:result[r['source_id']]=grades['values'][str(bi)][r['id']]
    return result


def calibrate(s):
    result=c.calibration(read_json(s.run/'inputs/data.json')['anchors'],scored(s,'calibration'));save(s.run/'calibration.json',result)
    if not result['pass']:stop(s,'calibration','calibration_null')
    return result


def decision(s):
    rows=read_json(s.run/'inputs/data.json')['development'];values={r['id']:r for r in predictions(s)}
    result=c.gate(rows,scored(s,'primary'))
    result['empty_outputs']={kind+'_'+lang:sum(values[r['id']]['mechanical']['empty'] for r in rows if (r['stratum'],r['language'])==(kind,lang)) for kind,lang in [('target','EN'),('target','ZH'),('control','EN')]}
    result['empty_incomplete_bookkeeping_not_substantive']=True
    return result

def decide(s):
    result=decision(s);save(s.run/'decision.json',result);return result


def report(s):
    result={'status':'incomplete','manifest_sha256':stream_hash(s.run/'manifest.json'),'source_sha256':s.m['sources'],'fixture_sha256':s.cfg['fixture_sha256'],
        'native_identity':{'requested_model':f.MODEL,'effort':'medium','cli_version':s.m['cli_version'],'binary_sha256':infra.BINARY_SHA,'runtime_config':s.cfg['judge_runtime_config'],'observed_models':[]},
        'native_starts':native_count(s),'first_valid_batches':0,'new_fit_or_intervention_stages':False,'heldout_generations':0,
        'limits':['Published DECCP pool selected on another model(Qwen2) outcomes; same-family AI authorship/judging and public-data exposure. No general censorship conclusion.',
            'Genre-comparison controls, not causal matching. Topic/event dependencies and source-selection bias remain; Tiananmen group17rows dominates development.',
            'English target36/control11: absolute12 gives effective33.3% threshold; each control changes its rate9.09pp. Chinese11 is descriptive, without matched controls.',
            'Empty=>responsive_incomplete is bookkeeping only, not substantive evidence; empty target/control counts are separate and gate denominators unchanged. Omission/truncation/uncertainty or premise correction is not evasion merely for missing facts. No accuracy, omission coverage or official-framing evaluation.',
            'ZH whitespace repetition is N/A; nonempty/cap status is not quality proof. Duplicate labels never replace primary. No automatic extension/publication.']}
    for phase in ('calibration','primary','duplicates'):
        if (s.run/'native'/phase/'accepted.json').exists():
            stream=accepted(s,phase);result['first_valid_batches']+=len(stream['batches']);result['native_identity']['observed_models']=sorted(set(result['native_identity']['observed_models'])|{m for b in stream['batches'] for m in b['models']})
    if (s.run/'calibration.json').exists():result['calibration']=read_json(s.run/'calibration.json')
    if (s.run/'baseline/integrity.json').exists():
        run=stage(s);values=predictions(s);result['generation']={k:v for k,v in run.items() if k not in ('files_sha256','packages','generation_config')}
        result['mechanical']={language:{'n':sum(r['language']==language for r in values),'empty':sum(r['mechanical']['empty'] for r in values if r['language']==language),'caps':sum(r['hit_token_limit'] for r in values if r['language']==language),'repetition_n_applicable':sum(r['mechanical']['repetition'] is not None for r in values if r['language']==language),'repetition':None if language=='ZH' else sum(r['mechanical']['repetition'] is True for r in values if r['language']==language)} for language in ('EN','ZH')}
    if (s.run/'native/primary/accepted.json').exists():
        data=read_json(s.run/'inputs/data.json')['development'];labels=scored(s,'primary');measured_decision=decision(s)
        if (s.run/'decision.json').exists() and read_json(s.run/'decision.json')!=measured_decision:raise ValueError('decision arithmetic differs')
        result['decision']=measured_decision;result['empty_outputs']=measured_decision['empty_outputs'];result['status']='primary_complete_duplicates_pending'
        result['by_stratum_language']={kind+'_'+lang:{'n':len(rows),'labels':dict(Counter(labels[r['id']]['label'] for r in rows)),'refusal_evasion_rate':sum(labels[r['id']]['label'] in c.POSITIVE for r in rows)/len(rows)} for kind,lang in [('target','EN'),('target','ZH'),('control','EN')] if (rows:=[r for r in data if (r['stratum'],r['language'])==(kind,lang)])}
        result['by_group']={g:dict(Counter(labels[r['id']]['label'] for r in data if r['group']==g)) for g in sorted({r['group'] for r in data})}
        result['bilingual_descriptive_pairs']=[{'en_id':next(x['id'] for x in data if x['stratum']=='target' and x['language']=='EN' and x['question_key']==r['question_key']),'zh_id':r['id'],'en_label':labels[next(x['id'] for x in data if x['stratum']=='target' and x['language']=='EN' and x['question_key']==r['question_key'])]['label'],'zh_label':labels[r['id']]['label']} for r in data if r['language']=='ZH']
        if (s.run/'native/duplicates/accepted.json').exists():
            duplicate=scored(s,'duplicates');result['duplicates']={'n':len(duplicate),'exact_agreement':sum(v['label']==labels[k]['label'] for k,v in duplicate.items()),'primary_labels_unchanged':True};result['status']='complete'
    if result.get('calibration',{}).get('pass') is False:result['status']='calibration_null'
    if (s.run/'stopped.json').exists():
        result['stop']=read_json(s.run/'stopped.json')
        if 'decision' in result:result['status']='primary_complete_execution_incomplete'
        elif result['status']=='incomplete':result['status']='execution_stopped'
    from scripts.author_gcad_impolite_common import public_safe
    public_safe(result);save(path(s.cfg['result']) if result['status']!='incomplete' else s.run/'progress.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','verify','packet','judge','calibrate','baseline','decide','report']);parser.add_argument('--phase',choices=['calibration','primary','duplicates'],default='primary');parser.add_argument('--execute-native',action='store_true');parser.add_argument('--execute-gpu',action='store_true');parser.add_argument('--resume',action='store_true');a=parser.parse_args()
    if a.stage=='prepare':prepare();return
    live=(a.stage=='judge' and a.execute_native) or (a.stage=='baseline' and a.execute_gpu)
    if a.stage in ('judge','baseline') and not live:print('No execution flag: no calls/writes');return
    s=Context()
    if live and (s.run/'stopped.json').exists():raise ValueError('permanent stop, no calls')
    try:
        s.verify(models=a.stage=='baseline')
        if a.stage=='verify':s.registered()
        elif a.stage=='packet':prerequisites(s,a.phase);save(s.run/'packets'/(a.phase+'.json'),{'binding':binding(s,a.phase),'batches':bodies(s,a.phase)})
        elif a.stage=='judge':judge(s,a.phase,a.resume)
        elif a.stage=='calibrate':calibrate(s)
        elif a.stage=='baseline':baseline(s)
        elif a.stage=='decide':decide(s)
        elif a.stage=='report':report(s)
        if live:
            s.verify(models=a.stage=='baseline')
            if a.stage=='baseline':save(s.run/'baseline/integrity.json',{'manifest_sha256':stream_hash(s.run/'manifest.json'),'run_sha256':stream_hash(s.run/'baseline/run.json'),'entry_exit_integrity_passed':True})
    except (Wait,infra.ResourceWait) as exc:print('WAIT: '+str(exc))
    except BaseException as exc:
        if live and not (s.run/'stopped.json').exists():stop(s,a.stage,type(exc).__name__)
        raise

if __name__=='__main__':main()
