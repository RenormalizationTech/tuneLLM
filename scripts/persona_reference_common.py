"""Pure PV-adaptation encodings, constrained plans, raw vectors and score rules."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import math
import random
import re

from scripts.tone_common import degenerate

LAYERS=(4,8,12,16,20,24,28,32,35)
PHASES={'elicit':25,'layer':13,'fallback':12,'strength':5,'strength_fallback':4,'final':5,'custom':5}
TRAITS=('profanity','rudeness','sarcasm')


def split_questions(data,seed=701):
    extraction,evaluation=data['extract'],data['eval']
    for value in (extraction,evaluation):
        if len(value['questions'])!=20 or len(set(value['questions']))!=20 or len(value['instruction'])!=5:
            raise ValueError('20 unique questions and five instruction pairs required')
        if any(set(v)!={'pos','neg'} or any(not isinstance(t,str) or not t.strip() for t in v.values()) for v in value['instruction']):
            raise ValueError('five complete instruction pairs required')
    if set(extraction['questions'])&set(evaluation['questions']):raise ValueError('extract/eval overlap')
    if extraction['instruction']!=evaluation['instruction'] or extraction['eval_prompt']!=evaluation['eval_prompt']:raise ValueError('extract/eval instructions or trait rubric differ')
    indices=list(range(20));random.Random(seed).shuffle(indices)
    rows=lambda ids:[{'id':f'eval-{i:02d}','user':evaluation['questions'][i],'source_row':i} for i in ids]
    return {'selection':rows(indices[:10]),'report':rows(indices[10:])}


def system(cfg,instruction,side):
    adjective=cfg['assistant_positive' if side=='pos' else 'assistant_negative']
    article='an' if adjective[0].lower() in 'aeiou' else 'a'
    return f'You are {article} {adjective} assistant. {instruction}'


def elicitation(data,cfg):
    return [{ 'id':f'x{q:02d}-i{i}-{side}','question_id':f'x{q:02d}','instruction_index':i,'side':side,
              'user':question,'system':system(cfg,instruction[side],side)}
            for q,question in enumerate(data['extract']['questions']) for i,instruction in enumerate(data['extract']['instruction']) for side in ('pos','neg')]


def conditions(phase):
    if phase=='elicit':return tuple(f'i{i}-{side}' for i in range(5) for side in ('pos','neg'))
    if phase=='layer':return ('baseline',*(f'layer{k}' for k in LAYERS))
    if phase=='fallback':return tuple(f'layer{k}' for k in LAYERS)
    if phase=='strength':return ('coef0.5','coef1','coef3','coef4')
    if phase=='strength_fallback':return ('coef0.5','coef3','coef4')
    return ('baseline','selected','random','prompt')


def make_plan(groups,arms,seed):
    """Highest-remaining-capacity deal: distinct question per batch, last partial."""
    rng=random.Random(seed);groups=list(groups);rng.shuffle(groups)
    total=len(groups)*len(arms);count=math.ceil(total/8);capacity=[min(8,total-8*i) for i in range(count)]
    batches=[[] for _ in capacity]
    if len(arms)>count:raise ValueError('not enough batches to separate a question')
    for group in groups:
        available=list(range(count));rng.shuffle(available);available.sort(key=lambda i:capacity[i],reverse=True)
        chosen=available[:len(arms)];shuffled=list(arms);rng.shuffle(shuffled)
        for bi,arm in zip(chosen,shuffled):
            if capacity[bi]<=0:raise ValueError('constraint deal exhausted')
            alias=hashlib.sha256(f'{seed}|{group}|{arm}'.encode()).hexdigest()[:24]
            batches[bi].append({'id':alias,'question_id':group,'condition':arm});capacity[bi]-=1
    if any(capacity):raise ValueError('incomplete plan')
    for batch in batches:rng.shuffle(batch)
    # Preserve registered eight/eight/.../partial shape, randomize within capacity.
    full=[b for b in batches if len(b)==8];rng.shuffle(full)
    return full+[b for b in batches if len(b)!=8]


def plans(data,cfg):
    split=split_questions(data,cfg['split_seed']);groups={
        'elicit':[f'x{i:02d}' for i in range(20)],
        **{p:[r['id'] for r in split['selection']] for p in ('layer','fallback','strength','strength_fallback')},
        'final':[r['id'] for r in split['report']], 'custom':[r['id'] for r in split['report']]}
    seeds={**cfg['plan_seeds'],'strength_fallback':cfg['plan_seeds']['strength']}
    return {phase:make_plan(groups[phase],conditions(phase),seeds[phase]) for phase in PHASES}


def response_encoding(tokenizer,prediction):
    """Pool RAW native trajectory; retokenization is diagnostic only."""
    prompt,answer=prediction['rendered_prompt'],prediction['output']
    if tokenizer(prompt,add_special_tokens=False)['input_ids']!=prediction['input_ids']:
        raise ValueError('saved prompt IDs differ')
    full=prompt+answer
    encoded=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True)
    ids,offsets=encoded['input_ids'],encoded['offset_mapping'];boundary=len(prompt)
    crossing=[{'start':a,'end':b,'whitespace_only':full[a:b].isspace()} for a,b in offsets if a<boundary<b]
    mask=[boundary<=a<b<=len(full) and t not in tokenizer.all_special_ids for t,(a,b) in zip(ids,offsets)]
    source_prompt_length=len(prediction['input_ids'])
    raw=prediction['generated_ids'];raw_body=raw[:-1] if prediction['ended_eos'] else raw
    if prediction['ended_eos'] and (not raw or raw[-1]!=tokenizer.eos_token_id):raise ValueError('terminal EOS receipt differs')
    raw_mask=[t not in tokenizer.all_special_ids and t!=tokenizer.pad_token_id for t in raw_body]
    if not any(raw_mask):raise ValueError('no nonspecial raw generated tokens')
    return {'input_ids':prediction['input_ids']+raw_body+[tokenizer.eos_token_id],
        'response_mask':[False]*len(prediction['input_ids'])+raw_mask+[False],
        'source_prompt_length':source_prompt_length,'retokenized_prompt_prefix_equal':ids[:source_prompt_length]==prediction['input_ids'],
        'retokenization_boundary_crossings':crossing,
        'nonterminal_special_tokens':[{'position':i,'token_id':t} for i,t in enumerate(raw_body) if not raw_mask[i]],
        'raw_generated_token_ids_excluding_eos':raw_body,'retokenized_response_ids':[t for t,m in zip(ids,mask) if m],
        'raw_vs_retokenized_response_equal':raw_body==[t for t,m in zip(ids,mask) if m],
        'pooling':'RAW native prompt plus generated tokens under ORIGINAL system/user; FP32; special/EOS/pad excluded; retokenization diagnostic only',
        'source_count_mask_equal':mask==[i>=source_prompt_length for i in range(len(ids))]}


def validate_ratings(ids,ratings,custom=False):
    if not isinstance(ratings,list) or any(not isinstance(r,dict) for r in ratings):raise ValueError('ratings list required')
    keys={'id',*TRAITS,'gross_failure','content_ok','coherent'} if custom else {'id','trait','coherence','refusal','reason'}
    if any(set(r)!=keys for r in ratings):raise ValueError('exact schema required')
    got=[r['id'] for r in ratings]
    if len(got)!=len(ids) or len(set(got))!=len(got) or set(got)!=set(ids):raise ValueError('exact unique IDs required')
    for r in ratings:
        if custom:
            if any(type(r[t]) is not int or r[t] not in range(4) for t in TRAITS) or any(type(r[k]) is not bool for k in ('gross_failure','content_ok','coherent')):raise ValueError('custom score types')
        else:
            if type(r['refusal']) is not bool or not isinstance(r['reason'],str) or not r['reason'].strip():raise ValueError('reference flag/reason required')
            if r['refusal']:
                if r['trait'] is not None:raise ValueError('semantic refusal has null trait')
            elif type(r['trait']) is not int or r['trait'] not in range(101):raise ValueError('integer trait0..100 required')
            if type(r['coherence']) is not int or r['coherence'] not in range(101):raise ValueError('coherence integer0..100 ALWAYS required')
    return ratings


def eligible(rows,ratings,minimum_pairs=20,minimum_questions=10):
    validate_ratings([r['id'] for r in rows],ratings);lookup={r['id']:r for r in ratings};pairs={}
    for row in rows:
        key=(row['question_id'],row['instruction_index'])
        if row['side'] in pairs.setdefault(key,{}):raise ValueError('duplicate pair key')
        pairs[key][row['side']]=row['id']
    retained=[]
    for (question,index),sides in sorted(pairs.items()):
        if set(sides)!={'pos','neg'}:raise ValueError('unpaired extraction key')
        a,b=(lookup[sides[s]] for s in ('pos','neg'))
        if all(not r['refusal'] and r['coherence']>=50 for r in (a,b)) and a['trait']>=50 and b['trait']<50:
            retained.append({'question_id':question,'instruction_index':index,**sides})
    n=len({r['question_id'] for r in retained})
    return {'retained_pairs':retained,'pair_count':len(retained),'question_count':n,'sufficient':len(retained)>=minimum_pairs and n>=minimum_questions}


def raw_vectors(pairs,blocks):
    import torch
    tensors,stats={},{}
    for block in blocks:
        pos=torch.stack([a[block].float() for a,b in pairs]);neg=torch.stack([b[block].float() for a,b in pairs])
        raw=pos.mean(0)-neg.mean(0);norm=raw.norm()
        if not torch.isfinite(raw).all() or norm<=0:raise ValueError('zero/nonfinite raw vector')
        rand=torch.randn(raw.shape,generator=torch.Generator().manual_seed(7100+block));rand/=rand.norm()
        tensors[f'{block}.raw']=raw.contiguous();tensors[f'{block}.random_unit']=rand.contiguous()
        stats[str(block)]={'author_layer':block+1,'block':block,'raw_norm':norm.item(),'pairs':len(pairs),'random_seed':7100+block}
    return tensors,stats


@contextmanager
def response_hook(model,block,vector,audit=None):
    """Author response hook: actual zero included; cached last-real prefill/decode."""
    import torch
    state={};handles=[];audit={} if audit is None else audit;cached={}
    def before(module,args,kwargs):
        ids=kwargs.get('input_ids',args[0] if args else None);mask=kwargs.get('attention_mask')
        if ids is None or mask is None or not mask.bool().any(1).all():raise ValueError('explicit nonempty mask required')
        # Width one is cached decode or a one-token prefill; both affect that site.
        state['decode']=ids.shape[1]==1 and mask.shape[1]>1
        if ids.shape[1]==1:state['positions']=torch.zeros(ids.shape[0],dtype=torch.long,device=ids.device)
        else:
            if mask.shape!=ids.shape:raise ValueError('prefill mask width differs')
            state['positions']=torch.arange(mask.shape[1],device=mask.device).expand_as(mask).masked_fill(~mask.bool(),-1).max(1).values
    def add(module,args,output):
        h=output[0] if isinstance(output,tuple) else output;key=(h.device,h.dtype)
        if key not in cached:cached[key]=vector.to(device=h.device,dtype=h.dtype)
        value=cached[key]
        if value.ndim!=1 or value.numel()!=h.shape[-1]:raise ValueError('vector dimension differs')
        changed=h.clone();changed[torch.arange(h.shape[0],device=h.device),state['positions']]+=value
        phase='decode' if state['decode'] else 'prefill';key=phase+'_calls';audit[key]=audit.get(key,0)+1
        if audit[key]==1:audit[phase+'_max_abs_hidden_delta']=(changed-h).abs().max().item()
        return (changed,*output[1:]) if isinstance(output,tuple) else changed
    try:
        handles.append(model.register_forward_pre_hook(before,with_kwargs=True));handles.append(model.model.layers[block].register_forward_hook(add));yield
    finally:
        for handle in handles:handle.remove()
        audit['hooks_removed']=True


def metrics(predictions,ratings):
    validate_ratings([r['id'] for r in predictions],ratings);by={r['id']:r for r in ratings};n=len(predictions)
    numeric=[r['trait'] for r in ratings if r['trait'] is not None];coherence=[r['coherence'] for r in ratings if r['coherence'] is not None]
    return {'n':n,'conservative_trait_mean':sum(r['trait'] or 0 for r in ratings)/n,
        'numeric_trait_mean':sum(numeric)/len(numeric) if numeric else None,'numeric_count':len(numeric),
        'coherence_mean':sum(r['coherence'] or 0 for r in ratings)/n,'numeric_coherence_count':len(coherence),
        'refusals':sum(r['refusal'] for r in ratings),
        'empty':sum(not r['output'].strip() for r in predictions),'degenerate':sum(degenerate(r['output'])['degenerate'] for r in predictions),
        'capped':sum(r['hit_token_limit'] for r in predictions),'mean_tokens':sum(r['generated_tokens_including_eos'] for r in predictions)/n}


def validate_generation(value,rows,rendered,eos,cap=512,batch_size=4):
    predictions=value['predictions'];batches=value['batches']
    if [r['id'] for r in predictions]!=[r['id'] for r in rows] or len(batches)!=math.ceil(len(rows)/batch_size):raise ValueError('generation ordered IDs/batches differ')
    if value['settings'].get('batch_size')!=batch_size or value['settings'].get('max_new_tokens')!=cap or value['settings'].get('seed')!=42 or value['settings'].get('do_sample') is not False:raise ValueError('generation runtime settings differ')
    for bi,start in enumerate(range(0,len(rows),batch_size)):
        members=predictions[start:start+batch_size];ids=[r['id'] for r in members];width=max(len(rendered[r['id']]['input_ids']) for r in members)
        if any(batches[bi].get(k)!=v for k,v in {'batch_index':bi,'ids':ids,'padded_prompt_width':width}.items()):raise ValueError('generation batch record differs')
        for position,row in enumerate(members):
            if any(row[k]!=v for k,v in rendered[row['id']].items()):raise ValueError('generation system/prompt/input IDs differ')
            tokens=row['generated_ids']
            if not isinstance(tokens,list) or not tokens or len(tokens)>cap or any(type(t) is not int or t<0 for t in tokens):raise ValueError('invalid generated IDs')
            ended=eos in tokens
            if ended and (tokens[-1]!=eos or tokens.count(eos)!=1):raise ValueError('generation does not stop at first EOS')
            flags={'ended_eos':ended,'hit_token_limit':len(tokens)==cap and not ended,'generated_tokens_including_eos':len(tokens),
                'batch_index':bi,'batch_position':position,'batch_ids':ids,'padded_prompt_width':width}
            if any(type(row.get(k)) is not type(v) or row[k]!=v for k,v in flags.items()):raise ValueError('generation flags/membership differ')


def quality(base,value):
    return value['coherence_mean']>=75 and value['coherence_mean']>=base['coherence_mean']-10 and all(value[k]<=base[k]+1 for k in ('empty','degenerate','capped'))


def choose_layer(values):
    passing=[k for k in LAYERS if quality(values['baseline'],values[f'layer{k}'])]
    layer=min(passing,key=lambda k:(-values[f'layer{k}']['conservative_trait_mean'],k)) if passing else None
    return {'author_layer':layer,'block':layer-1 if layer else None,'selection_null':layer is None}


def choose_strength(values):
    passing=[x for x in (.5,1.,2.,3.,4.) if quality(values['baseline'],values[f'coef{x:g}'])]
    coef=min(passing,key=lambda x:(-values[f'coef{x:g}']['conservative_trait_mean'],x)) if passing else None
    gain=values[f'coef{coef:g}']['conservative_trait_mean']-values['baseline']['conservative_trait_mean'] if coef is not None else None
    return {'coefficient':coef,'gain':gain,'selection_null':coef is None or gain<10}


def reference_gates(values):
    base,chosen,random,prompt=(values[k] for k in ('baseline','selected','random','prompt'))
    gates={'quality':quality(base,chosen),'gain_baseline':chosen['conservative_trait_mean']>=base['conservative_trait_mean']+10,
        'gain_random':chosen['conservative_trait_mean']>=random['conservative_trait_mean']+10,
        'prompt_mean':prompt['conservative_trait_mean']>=50,'prompt_gain':prompt['conservative_trait_mean']>=base['conservative_trait_mean']+20}
    return {'gates':gates,'all_criteria_pass':all(gates.values())}


def private_export(value):
    if isinstance(value,dict):
        if set(value)&{'id','ids','question_id','user','system','output','reason','ratings','input_ids','generated_ids','rendered_prompt','retained_pairs','response_mask'}:raise ValueError('private export field')
        for v in value.values():private_export(v)
    elif isinstance(value,list):
        for v in value:private_export(v)
    elif isinstance(value,str) and re.search(r'/home/|/tmp/|hf_[A-Za-z0-9]{12,}|Bearer\s+\S+',value):raise ValueError('private path/secret')
