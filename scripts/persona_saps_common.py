"""Pure SAPS/native-HF operations. B1, all-prefix broadcast, no weight edits."""
from contextlib import contextmanager
import hashlib
import math
import random
import statistics
import unicodedata

from scripts.tone_common import degenerate
from scripts.persona_reference_common import validate_ratings as custom_ratings

BLOCKS=(11,15,19,23)
FIELDS=('sarcasm','entailment','grammaticality','clarity')
SETTINGS={f'{method}_b{block}_a{alpha:g}':{'method':method,'block':block,'alpha':alpha}
          for block in BLOCKS for method,alphas in [('fixed',(1.,)),('saps',(1.,1.4))] for alpha in alphas}
PHASES=('initial','validation','test','custom')


def task_user(sentence,instruction):
    return f'\n{instruction}\n\nSentence:\n"{sentence}"\n'


def validate_data(data):
    norm=lambda text:' '.join(unicodedata.normalize('NFKC',text).casefold().split())
    sets=[]
    for split,n in [('train',500),('val',50),('test',450)]:
        rows=data[split]
        if not isinstance(rows,list) or len(rows)!=n or any(not isinstance(r,str) or not r.strip() for r in rows):raise ValueError('published split shape differs')
        normalized=set(map(norm,rows))
        if len(normalized)!=n:raise ValueError('duplicate source strings')
        if any(normalized&old for old in sets):raise ValueError('split overlap')
        sets.append(normalized)
    prompts=data['prompts']
    if prompts['task']!='sarcasm' or not isinstance(prompts['plain'],str) or len(prompts['style'])!=5 or len(set(prompts['style']))!=5:raise ValueError('author prompt definitions differ')


def partition(data):
    validate_data(data);order=list(range(450));random.Random(702).shuffle(order)
    return {'train':[{'id':f'train-{i:03d}','source_row':i,'sentence':v} for i,v in enumerate(data['train'])],
        'validation':[{'id':f'val-{i:03d}','source_row':i,'sentence':v} for i,v in enumerate(data['val'])],
        'test':[{'id':f'test-{i:03d}','source_row':i,'sentence':data['test'][i]} for i in order[:100]],
        'custom_ids':[f'test-{i:03d}' for i in order[:20]]}


def instruction_draws():
    rng=random.Random(42);return [rng.randrange(5) for _ in range(1000)]


def make_plan(rows,arms,seed,width):
    rng=random.Random(seed);groups=[r['id'] for r in rows];rng.shuffle(groups);total=len(groups)*len(arms)
    capacity=[min(width,total-i) for i in range(0,total,width)];batches=[[] for _ in capacity]
    if len(arms)>len(batches):raise ValueError('insufficient distinct-source batches')
    for group in groups:
        indices=list(range(len(batches)));rng.shuffle(indices);indices.sort(key=lambda i:capacity[i],reverse=True)
        conditions=list(arms);rng.shuffle(conditions)
        for bi,condition in zip(indices,conditions):
            if capacity[bi]<=0:raise ValueError('constrained deal failed')
            alias=hashlib.sha256(f'{seed}|{group}|{condition}'.encode()).hexdigest()[:24]
            batches[bi].append({'id':alias,'question_id':group,'condition':condition});capacity[bi]-=1
    if any(capacity):raise ValueError('incomplete plan')
    for batch in batches:rng.shuffle(batch)
    full=[b for b in batches if len(b)==width];rng.shuffle(full)
    return full+[b for b in batches if len(b)!=width]


def plans(split,cfg):
    """All absent-method branches sealed before outputs; exactly one used."""
    result={'initial':make_plan(split['validation'],['baseline','prompt'],cfg['plan_seeds']['initial'],20),
        'validation':make_plan(split['validation'],list(SETTINGS),cfg['plan_seeds']['validation'],20)}
    custom_rows=[r for r in split['test'] if r['id'] in split['custom_ids']]
    for branch,methods in [('both',['fixed','saps']),('fixed',['fixed']),('saps',['saps'])]:
        arms=['baseline',*methods,'random','prompt']
        result['test_'+branch]=make_plan(split['test'],arms,cfg['plan_seeds']['test'],20)
        result['custom_'+branch]=make_plan(custom_rows,arms,cfg['plan_seeds']['custom'],8)
    return result


@contextmanager
def capture(model,blocks,collected,first_prefill=False):
    """Read-only block outputs; stack on device, exactly one CPU copy per forward."""
    state={'active':True};values={};handles=[]
    def collect(block):
        def hook(module,args,output):
            if state['active']:
                h=output[0] if isinstance(output,tuple) else output
                if h.shape[0]!=1:raise ValueError('source capture requires B1')
                values[block]=h[0,-1,:].detach()
        return hook
    def after(module,args,output):
        if state['active']:
            if set(values)!=set(blocks):raise ValueError('capture block coverage failed')
            import torch
            stacked=torch.stack([values[b] for b in blocks]).float().cpu()
            collected.update({b:stacked[i] for i,b in enumerate(blocks)})
            if first_prefill:state['active']=False
    try:
        for b in blocks:handles.append(model.model.layers[b].register_forward_hook(collect(b)))
        handles.append(model.register_forward_hook(after));yield
    finally:
        for handle in handles:handle.remove()


def fixed_scalar(target,h0,u,alpha=1.):
    if target.ndim!=0:raise ValueError('target must be a0D scalar')
    projection=(h0.detach().float().cpu()*u.detach().float().cpu()).sum().item()
    return (target.detach().float().cpu().item()-projection)*alpha


@contextmanager
def intervention(model,block,u,target,alpha,method,constant=None,random_unit=None,audit=None):
    """Exact B1 source hook ordering; adaptive scalar along u, optional random axis."""
    import torch
    if method not in ('fixed','saps'):raise ValueError('unknown author controller')
    if target.ndim!=0:raise ValueError('target must be a0D scalar to preserve native dtype arithmetic')
    if method=='fixed' and (type(constant) is not float or not math.isfinite(constant)):raise ValueError('fixed c must be a finite Python float')
    audit={} if audit is None else audit;stored={};handles=[]
    # Author generate.py subtracts separate .item() operands in Python; keep c
    # a Python scalar through model-dtype u*c (tensor promotion differs in BF16).
    constant_value=constant if method=='fixed' else None
    # Source SAPS renormalizes with epsilon; fixed uses the saved unit directly.
    direction=u/(u.norm()+1e-8) if method=='saps' else u
    def hook(module,args,output):
        h=output[0] if isinstance(output,tuple) else output
        if h.shape[0]!=1:raise ValueError('author broadcast is registered B1 only')
        key=(h.device,h.dtype)
        if key not in stored:
            stored[key]=(direction.to(device=h.device,dtype=h.dtype),target.to(h.device),
                         (random_unit if random_unit is not None else direction).to(device=h.device,dtype=h.dtype),
                         constant_value)
        native,t,axis,c=stored[key]
        scalar=c if method=='fixed' else alpha*(t-(h[:,-1,:]*native).sum(-1,keepdim=True))
        delta=scalar*axis
        # Fixed rule computes c in FP32; apply resulting shift in model dtype.
        delta=delta.to(h.dtype);changed=h+delta
        phase='prefill' if audit.get('calls',0)==0 else 'decode';audit['calls']=audit.get('calls',0)+1
        audit[phase+'_calls']=audit.get(phase+'_calls',0)+1
        # Keep scalar/norm tensors on device until context exit: no per-step item().
        audit.setdefault('_norms_'+phase,[]).append(delta.detach().float().norm().reshape(1))
        if audit[phase+'_calls']==1:audit['_delta_'+phase]=(changed-h).detach().abs().max()
        return (changed,*output[1:]) if isinstance(output,tuple) else changed
    try:
        handles.append(model.model.layers[block].register_forward_hook(hook));yield
    finally:
        for handle in handles:handle.remove()
        for phase in ('prefill','decode'):
            values=audit.pop('_norms_'+phase,[])
            if values:audit.setdefault('shift_norms',{})[phase]=torch.cat(values).cpu().tolist()
            delta=audit.pop('_delta_'+phase,None)
            if delta is not None:audit[phase+'_max_abs_hidden_delta']=delta.item()
        audit['hooks_removed']=True
        audit['shift_norm_scope']='L2 of additive vector PER POSITION; not full-prefix Frobenius norm (which scales sqrt(prefix width))'


def validate_ratings(ids,ratings,custom=False):
    if custom:return custom_ratings(ids,ratings,custom=True)
    if not isinstance(ratings,list) or any(not isinstance(r,dict) or set(r)!={'id',*FIELDS} for r in ratings):raise ValueError('exact four-score schema')
    received=[r['id'] for r in ratings]
    if len(received)!=len(ids) or len(set(received))!=len(received) or set(received)!=set(ids):raise ValueError('complete unique exact IDs')
    if any(type(r[k]) is not int or r[k] not in range(6) for r in ratings for k in FIELDS):raise ValueError('integer0..5 required, bool not score')
    return ratings


def metrics(predictions,ratings):
    validate_ratings([r['id'] for r in predictions],ratings);n=len(ratings)
    return {'n':n,**{k:sum(r[k] for r in ratings)/n for k in FIELDS},'style_at_least3':sum(r['sarcasm']>=3 for r in ratings),
        'empty':sum(not r['output'].strip() for r in predictions),'degenerate':sum(degenerate(r['output'])['degenerate'] for r in predictions),
        'capped':sum(r['hit_token_limit'] for r in predictions),'mean_tokens':sum(r['generated_tokens_including_eos'] for r in predictions)/n}


def elicitation_gate(values):
    control=values['prompt'];return control['style_at_least3']>=25 and control['sarcasm']>=3 and control['grammaticality']>=3.5 and control['clarity']>=3.5


def quality(base,value):
    return value['grammaticality']>=3.5 and value['clarity']>=3.5 and value['entailment']>=3 and all(value[k]<=base[k]+3 for k in ('empty','degenerate','capped'))


def select(values,baseline):
    eligible={name:record for name,record in SETTINGS.items() if quality(baseline,values[name])}
    key=lambda name:(-values[name]['sarcasm'],SETTINGS[name]['alpha'],SETTINGS[name]['block'],SETTINGS[name]['method']!='fixed')
    best={method:min([n for n in eligible if SETTINGS[n]['method']==method],key=key,default=None) for method in ('fixed','saps')}
    winner=min(eligible,key=key) if eligible else None
    gain=values[winner]['sarcasm']-baseline['sarcasm'] if winner else None
    return {'best':best,'winner':winner,'setting':SETTINGS[winner] if winner else None,'gain':gain,'selection_null':winner is None or gain<.5,
        'branch':'both' if all(best.values()) else ('fixed' if best['fixed'] else 'saps' if best['saps'] else None),
        'eligible':list(eligible)}


def reference_gates(values,winner):
    chosen=values[winner];base=values['baseline'];gates={'gain_baseline':chosen['sarcasm']>=base['sarcasm']+.5,
        'gain_random':chosen['sarcasm']>=values['random']['sarcasm']+.5,'quality':quality(base,chosen),
        'prompt_positive':2*values['prompt']['style_at_least3']>=values['prompt']['n']}
    return {'gates':gates,'all_criteria_pass':all(gates.values())}


def distribution(values):
    return {'count':len(values),'mean':statistics.mean(values) if values else None,'min':min(values) if values else None,
        'median':statistics.median(values) if values else None,'max':max(values) if values else None}


def norm_diagnostics(real,random):
    result={}
    for phase in ('prefill','decode'):
        a=[n for audit in real for n in audit.get('shift_norms',{}).get(phase,[])];b=[n for audit in random for n in audit.get('shift_norms',{}).get(phase,[])]
        ma=statistics.mean(a) if a else None;mb=statistics.mean(b) if b else None;ratio=mb/ma if ma is not None and ma>0 and mb is not None else None
        result[phase]={'real':distribution(a),'random':distribution(b),'random_over_real_mean':ratio,'norm_mismatched':ratio is None or not .8<=ratio<=1.25,'report_only':True,
            'scope':'Per-position additive-vector L2; prefix Frobenius L2 would scale sqrt(S), squared norm scales S'}
    return result
