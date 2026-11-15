"""Pure symmetric PV-grid reuse, selection and bounded journal operations."""
from scripts import persona_reference as pv
from scripts import persona_reference_common as reference
from scripts import tone_judge as f
from scripts.common import read_json
from scripts.save_nf4 import stream_hash

LAYERS=reference.LAYERS
COEFFICIENTS=(.5,1.,2.)
SETTINGS={f'l{layer}_c{coefficient:g}':{'author_layer':layer,'block':layer-1,'coefficient':coefficient}
          for layer in LAYERS for coefficient in COEFFICIENTS}
PHASES=('grid','final','custom')


def reuse_map(trait):
    if trait=='profanity':return {}
    if trait!='rudeness':raise ValueError('isolated trait required')
    return {'baseline':{'phase':'layer','condition':'baseline'},
        **{f'l{k}_c2':{'phase':'layer','condition':f'layer{k}'} for k in LAYERS},
        'l8_c0.5':{'phase':'strength','condition':'coef0.5'},'l8_c1':{'phase':'strength','condition':'coef1'}}


def missing(trait):return [n for n in ('baseline',*SETTINGS) if n not in reuse_map(trait)]


def plans(split,trait,seeds):
    groups=[r['id'] for r in split['selection']];report=[r['id'] for r in split['report']]
    return {'grid':reference.make_plan(groups,missing(trait),seeds['grid']),
        'final':reference.make_plan(report,('baseline','selected','random','prompt'),seeds['final']),
        'custom':reference.make_plan(report,('baseline','selected','random','prompt'),seeds['custom'])}


def choose(values):
    if set(values)!={'baseline',*SETTINGS}:raise ValueError('complete fixed27-condition grid required')
    base=values['baseline'];n=base['n']
    if n!=10 or any(v['n']!=n for v in values.values()):raise ValueError('fixed10-question denominator required')
    eligible=[name for name in SETTINGS if values[name]['coherence_sum']>=75*n and values[name]['coherence_sum']>=base['coherence_sum']-10*n
        and all(values[name][k]<=base[k]+1 for k in ('empty','degenerate','capped'))]
    best=min(eligible,key=lambda name:(-values[name]['score_sum'],SETTINGS[name]['coefficient'],SETTINGS[name]['author_layer'])) if eligible else None
    difference=values[best]['score_sum']-base['score_sum'] if best else None
    return {'selected':best,'setting':SETTINGS[best] if best else None,'gain':difference/n if difference is not None else None,
        'gain_sum':difference,'selection_null':best is None or difference<10*n,'eligible':eligible}


def measures(predictions,ratings):
    return {**reference.metrics(predictions,ratings),'score_sum':sum(r['trait'] or 0 for r in ratings),'coherence_sum':sum(r['coherence'] for r in ratings)}


def finish_judgments(run,directory,body,bound,maximum,custom=False):
    ratings=[];records=[]
    for bi,b in enumerate(body):
        record={'expected_ids':b['expected_ids'],'canonical_stdin_sha256':f.canonical_hash(b),'attempts':[]}
        f.save_or_match(directory/f'batch-{bi:03d}-input.json',record)
        for ai in range(2):
            if (run/'stopped.json').exists():raise RuntimeError('child permanently stopped')
            stem=f'batch-{bi:03d}-attempt-{ai}'
            if not list(directory.glob(stem+'*')) and len(list((run/'grades').glob('*/batch-*-attempt-*-start.json')))>=maximum:raise RuntimeError('child call budget exhausted BEFORE start')
            attempt,receipt=f.replay_or_call(directory,bi,ai,b,bound,'claude');outer,models=pv.checked_outer(attempt)
            entry={'receipt':receipt,'actual_model_usage_ids':models,'structural_valid':False};record['attempts'].append(entry)
            try:valid=pv.parse(outer,b['expected_ids'],custom)
            except f.InvalidSubmission:
                if ai==1:f.save_or_match(directory/f'batch-{bi:03d}-provenance.json',record);raise
                continue
            entry['structural_valid']=True;record['first_valid_attempt']=ai;ratings.extend(valid);break
        f.save_or_match(directory/f'batch-{bi:03d}-provenance.json',record);records.append(record)
    return {'ratings':ratings,'batches':records,'binding_sha256':bound,'first_valid_only':True}


def read_judgments(directory,body,binding,custom=False):
    if read_json(directory/'run_binding.json')!=binding:raise ValueError('child grade binding differs')
    bound=stream_hash(directory/'run_binding.json');ratings=[];records=[]
    for bi,b in enumerate(body):
        record=read_json(directory/f'batch-{bi:03d}-provenance.json');first=record['first_valid_attempt']
        base={'expected_ids':b['expected_ids'],'canonical_stdin_sha256':f.canonical_hash(b),'attempts':[]}
        if type(first) is not int or first not in (0,1) or len(record['attempts'])!=first+1 or read_json(directory/f'batch-{bi:03d}-input.json')!=base or any(record[k]!=v for k,v in base.items() if k!='attempts'):raise ValueError('first-valid/batch membership differs')
        for ai,entry in enumerate(record['attempts']):
            stem=f'batch-{bi:03d}-attempt-{ai}';file=directory/(stem+'.json');start=directory/(stem+'-start.json')
            receipt={'run_binding_sha256':bound,'canonical_stdin_sha256':f.canonical_hash(b),'file':file.name,'sha256':stream_hash(file),'start_sha256':stream_hash(start)}
            if entry['receipt']!=receipt or read_json(directory/(stem+'-receipt.json'))!=receipt or any(read_json(start).get(k)!=receipt[k] for k in ('run_binding_sha256','canonical_stdin_sha256')):raise ValueError('child receipt/start differs')
            attempt=read_json(file);outer,models=pv.checked_outer(attempt)
            if attempt.get('canonical_stdin_sha256')!=f.canonical_hash(b) or models!=entry['actual_model_usage_ids']:raise ValueError('child attempt/model/payload differs')
            try:valid=pv.parse(outer,b['expected_ids'],custom)
            except f.InvalidSubmission:
                if ai==first or entry['structural_valid'] is not False:raise
                continue
            if ai!=first or entry['structural_valid'] is not True:raise ValueError('earlier valid child grades not retained')
            ratings.extend(valid)
        if first==0 and list(directory.glob(f'batch-{bi:03d}-attempt-1*')):raise ValueError('post-valid child attempt')
        records.append(record)
    if read_json(directory/'accepted.json')!={'ratings':ratings,'batches':records,'binding_sha256':bound,'first_valid_only':True}:raise ValueError('child accepted grades differ')
    return ratings
