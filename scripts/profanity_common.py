"""Small pure planning/scoring/vector helpers for the isolated profanity pilot."""
from collections import Counter
from contextlib import contextmanager
import math
import random
import statistics

from scripts import persona_rating_calibration as calibration
from scripts.tone_common import degenerate, token_length
from scripts.tone_common import steer as original_steer

TRAITS=calibration.TRAITS
LAYERS=(17,23)
BETAS=(0.05,0.10,0.20)
SETTINGS={f'l{l}_b{round(b*1000):03d}':(l,b) for l in LAYERS for b in BETAS}
CONDITIONS=('baseline',*SETTINGS)
PHASES={'eligibility':(2,8,501),'development':(7,14,502),'confirmation':(3,9,503)}
FAMILIES={'ass':'ass','bullshit':'bullshit','crap':'crap','damn':'damn','damned':'damn','fuck':'fuck','fucked':'fuck','fucking':'fuck','hell':'hell','shit':'shit','shitty':'shit'}


def validate_fixtures(fixtures,lexicon):
    topics=set();families=set();prompts=set()
    for phase,count,nfamilies in [('extraction',32,16),('development',16,8),('confirmation',24,12)]:
        data=fixtures[phase];rows=data['items']
        if data.get('schema_version')!=1 or data.get('split')!=phase or data.get('authored_synthetic') is not True or len(rows)!=count or len({r['id'] for r in rows})!=count:raise ValueError('fixed authored fixture/schema/count differs')
        counts=Counter(r['family'] for r in rows);local_topics={r['topic'] for r in rows}
        if len(counts)!=nfamilies or set(counts.values())!={2} or families&set(counts) or topics&local_topics:raise ValueError('two-per-family and disjoint topic/family splits required')
        families.update(counts);topics.update(local_topics)
        for row in rows:
            required={'id','family','topic','context','user','requirements','reference','factual'}
            if phase=='extraction':required|={'neutral','profane','word_family','profanity_phrase'}
            if set(row)!=required or type(row['factual']) is not bool or any(not isinstance(row[k],str) or not row[k].strip() for k in ('id','family','topic','context','user')):raise ValueError('exact task schema required')
            if any(not isinstance(row[k],list) or not row[k] or any(not isinstance(v,str) or not v.strip() for v in row[k]) for k in ('requirements','reference')):raise ValueError('task requirements/reference facts required')
            normalized=' '.join(row['user'].casefold().split())
            if normalized in prompts or calibration.lexical(row['user'],lexicon)['present']:raise ValueError('unique nonprofane user requests required')
            prompts.add(normalized)
            if phase=='extraction':
                a,b=(calibration.lexical(row[k],lexicon) for k in ('neutral','profane'))
                actual={FAMILIES[v] for v in b['variants']}
                if a['present'] or not b['present'] or actual!={row['word_family']}:raise ValueError('paired lexical contrast/family differs')
    if sum(r['factual'] for r in fixtures['confirmation']['items'])!=8:raise ValueError('eight factual confirmation items required')
    diversity=Counter(r['word_family'] for r in fixtures['extraction']['items'])
    if len(diversity)<4 or any(3*n>32 for n in diversity.values()):raise ValueError('authored word diversity fails')
    stats=position_phrase_stats(fixtures['extraction']['items'],lexicon)
    if stats['position_bins']['opening']>12 or stats['position_bins']['middle']<10 or stats['position_bins']['closing']<10:raise ValueError('authored position quotas fail')
    for family in diversity:
        if len({r['profanity_phrase'].casefold() for r in fixtures['extraction']['items'] if r['word_family']==family})<2:raise ValueError('two natural phrasings per family required')
    if any(n>3 for n in stats['phrase_counts'].values()):raise ValueError('phrase repeats exceed three')


def position_phrase_stats(rows,lexicon):
    positions=Counter({'opening':0,'middle':0,'closing':0});phrases=Counter();hit_counts=Counter()
    for row in rows:
        phrase=row['profanity_phrase']
        if not isinstance(phrase,str) or not phrase.strip() or phrase.casefold() not in row['profane'].casefold() or not calibration.lexical(phrase,lexicon)['present']:raise ValueError('phrase must be a real substring containing a lexical hit')
        words=row['profane'].split();first=next((i for i,w in enumerate(words) if calibration.lexical(w,lexicon)['present']),None)
        if first is None:raise ValueError('positive lexical hit missing')
        positions['opening' if 3*first<len(words) else 'middle' if 3*first<2*len(words) else 'closing']+=1
        phrases[phrase.casefold()]+=1
        hit_counts[calibration.lexical(row['profane'],lexicon)['count']]+=1
    return {'position_bins':dict(positions),'phrase_counts':dict(phrases),'position_definition':'zero-based whitespace-word index i / N; exact comparisons 3*i < N and 3*i < 2*N',
        'total_lexical_hits':sum(n*count for n,count in hit_counts.items()),'pairs_with_multiple_hits':sum(count for n,count in hit_counts.items() if n>1),
        'lexical_hits_per_pair_histogram':{str(n):count for n,count in sorted(hit_counts.items())}}


def constrained_plan(rows,conditions,batches,seed):
    if len(rows)*len(conditions)!=8*batches or len(conditions)>batches:raise ValueError('fixed balanced constrained plan shape required')
    rng=random.Random(seed);groups=list(rows);rng.shuffle(groups);dealt=[[] for _ in range(batches)]
    for gi,row in enumerate(groups):
        appearances=list(conditions);rng.shuffle(appearances)
        for ai,condition in enumerate(appearances):
            dealt[(gi*len(conditions)+ai)%batches].append({'id':format(rng.getrandbits(96),'024x'),'group_id':row['id'],'condition':condition})
    rng.shuffle(dealt)
    for batch in dealt:rng.shuffle(batch)
    if any(len(b)!=8 or len({r['group_id'] for r in b})!=8 for b in dealt):raise ValueError('same-prompt comparisons or wrong batch size')
    flat=[r for b in dealt for r in b]
    if len({r['id'] for r in flat})!=len(flat) or Counter((r['group_id'],r['condition']) for r in flat)!=Counter((r['id'],c) for r in rows for c in conditions):raise ValueError('complete unique condition coverage required')
    return dealt


def plans(fixtures):
    return {'eligibility':constrained_plan(fixtures['extraction']['items'],('neutral','profane'),8,501),
        'development':constrained_plan(fixtures['development']['items'],CONDITIONS,14,502),
        'confirmation':constrained_plan(fixtures['confirmation']['items'],('baseline','selected','random'),9,503)}


def validate_ratings(ids,values):
    keys={'id',*TRAITS,'gross_failure','content_ok','coherent','factual_correct'}
    if not isinstance(values,list) or any(not isinstance(r,dict) or set(r)!=keys for r in values):raise ValueError('exact extended schema required')
    calibration.validate_ratings(ids,[{k:r[k] for k in ('id',*TRAITS,'gross_failure')} for r in values])
    if any(type(r[k]) is not bool for r in values for k in ('content_ok','coherent','factual_correct')):raise ValueError('content/coherence/factual booleans required')
    return values


def eligibility(rows,reviews,ratios,lexicon):
    retained=[];audit=[]
    for row in rows:
        a,b=reviews[row['id']]['neutral'],reviews[row['id']]['profane'];ratio=ratios[row['id']]
        checks={'lexical':not calibration.lexical(row['neutral'],lexicon)['present'] and calibration.lexical(row['profane'],lexicon)['present'],
            'neutral_traits':all(a[t]<=1 for t in TRAITS),'profane_off_target':b['rudeness']<=1 and b['sarcasm']<=1,
            'content':all(v['content_ok'] and v['coherent'] and not v['gross_failure'] for v in (a,b)),
            'token_ratio':0.85<=ratio<=1.15}
        good=all(checks.values());audit.append({'id':row['id'],'checks':checks,'eligible':good,'token_ratio':ratio})
        if good:retained.append(row['id'])
    counts=Counter(r['word_family'] for r in rows if r['id'] in retained)
    return {'retained_ids':retained,'retained_count':len(retained),'word_family_counts':dict(counts),'audit':audit,
        'sufficient':len(retained)>=24 and len(counts)>=4 and all(3*n<=len(retained) for n in counts.values()),
        'retained_position_phrase_stats':position_phrase_stats([r for r in rows if r['id'] in retained],lexicon),
        'ambiguous_scores':{t:sum(reviews[r['id']][side][t]==1 for r in rows for side in ('neutral','profane')) for t in TRAITS}}


@contextmanager
def capture_response(model,layers,mask,collected):
    """FP32 response means plus token-weighted residual norms; never modify output."""
    import torch
    handles=[]
    try:
        for layer in layers:
            def hook(module,args,output,layer=layer):
                hidden=output[0] if isinstance(output,tuple) else output
                selected=mask.to(device=hidden.device,dtype=torch.bool)
                if tuple(selected.shape)!=tuple(hidden.shape[:2]) or not selected.any():raise ValueError('invalid response mask')
                values=hidden[selected].detach().float();norms=values.norm(dim=-1)
                collected[layer]={'mean':values.mean(0).cpu(),'token_norm_sum':norms.sum().item(),'tokens':values.shape[0]}
            handles.append(model.model.layers[layer].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:handle.remove()


@contextmanager
def audited_steer(model,layer,vector,audit):
    """Unchanged steering hook; count all calls, sample delta once per cache phase."""
    import torch
    state={};sampled=set();block=model.model.layers[layer]
    def before_model(module,args,kwargs):
        ids=kwargs.get('input_ids',args[0] if args else None);mask=kwargs.get('attention_mask');cache=kwargs.get('past_key_values')
        past=cache.get_seq_length() if hasattr(cache,'get_seq_length') else (cache[0][0].shape[-2] if isinstance(cache,(tuple,list)) and cache else 0)
        decode=past>0 or (ids is not None and ids.shape[1]==1 and mask is not None and mask.shape[1]>1)
        phase='decode' if decode else 'prefill';state['phase']=phase;audit[phase+'_calls']=audit.get(phase+'_calls',0)+1
        state['sample']=phase not in sampled
        if state['sample']:
            if ids is None or mask is None or mask.ndim!=2:raise ValueError('explicit hook audit IDs/mask required')
            state['positions']=None if decode else torch.arange(mask.shape[1],device=mask.device).expand_as(mask).masked_fill(~mask.bool(),-1).max(1).values
    def select(hidden):
        if state['positions'] is None:return hidden[:,-1]
        return hidden[torch.arange(hidden.shape[0],device=hidden.device),state['positions'].to(hidden.device)]
    def before_add(module,args,output):
        if state['sample']:
            hidden=output[0] if isinstance(output,tuple) else output;state['original']=select(hidden).detach().float()
    def after_add(module,args,output):
        if state['sample']:
            hidden=output[0] if isinstance(output,tuple) else output;delta=(select(hidden).detach().float()-state.pop('original')).abs().max().item()
            audit['max_abs_hidden_delta']=max(audit.get('max_abs_hidden_delta',0),delta);sampled.add(state['phase'])
            audit['sampled_phases']=sorted(sampled)
    root_handle=model.register_forward_pre_hook(before_model,with_kwargs=True);first=block.register_forward_hook(before_add)
    try:
        with original_steer(model,layer,vector,1):
            last=block.register_forward_hook(after_add)
            try:yield
            finally:last.remove()
    finally:
        first.remove();root_handle.remove();audit['hooks_removed']=True


def direction_statistics(pairs,layer):
    import torch
    if not pairs:raise ValueError('eligible pairs required')
    neutral=[p[0]['mean'].float() for p in pairs];profane=[p[1]['mean'].float() for p in pairs]
    raw=torch.stack([b-a for a,b in zip(neutral,profane)]).mean(0)
    norm=raw.norm().item();scale=statistics.median(v.norm().item() for v in neutral)
    if not math.isfinite(norm) or not math.isfinite(scale) or norm<=0 or scale<=0 or not torch.isfinite(raw).all():raise ValueError('zero/nonfinite direction or reference scale')
    unit=raw/norm;generator=torch.Generator(device='cpu').manual_seed(4242+layer)
    random_unit=torch.randn(raw.shape,generator=generator,dtype=torch.float32);random_unit/=random_unit.norm()
    count=sum(p[side]['tokens'] for p in pairs for side in (0,1));token_norm=sum(p[side]['token_norm_sum'] for p in pairs for side in (0,1))/count
    if not math.isfinite(token_norm) or token_norm<=0:raise ValueError('invalid token-norm diagnostic')
    tensors={f'{layer}.raw':raw.contiguous(),f'{layer}.unit':unit.contiguous(),f'{layer}.random':random_unit.contiguous(),f'{layer}.R':torch.tensor(scale,dtype=torch.float32)}
    magnitudes={str(b):{'beta_R':b*scale,'fraction_of_mean_per_token_norm':b*scale/token_norm,
        'effective_BF16_L2_norm':(unit*tensors[f'{layer}.R']*b).to(torch.bfloat16).float().norm().item(),
        'effective_random_BF16_L2_norm':(random_unit*tensors[f'{layer}.R']*b).to(torch.bfloat16).float().norm().item()} for b in BETAS}
    return tensors,{'pair_count':len(pairs),'raw_norm':norm,'R_median_neutral_response_mean_norm':scale,'serialized_R_fp32':tensors[f'{layer}.R'].item(),'raw_norm_over_R':norm/scale,
        'mean_per_token_residual_norm_both_sides':token_norm,'response_tokens_both_sides':count,'random_seed':4242+layer,
        'magnitudes':magnitudes,
        'weighting':'equal pair means; R median neutral response-mean norm; token diagnostic weighted by response-token counts'}


def delta(tensors,layer,beta,random=False):
    import torch
    unit=tensors[f'{layer}.random' if random else f'{layer}.unit'];R=tensors[f'{layer}.R']
    if beta not in BETAS or unit.ndim!=1 or not torch.isfinite(unit).all() or R.numel()!=1 or not torch.isfinite(R).all() or R.item()<=0:raise ValueError('invalid fixed intervention')
    return unit.float()*R.float()*beta


def measures(rows,predictions,reviews,lexicon):
    if [r['id'] for r in rows]!=[p['id'] for p in predictions]:raise ValueError('prediction order/coverage differs')
    n=len(rows);details=[]
    for row,pred in zip(rows,predictions):
        v=reviews[row['id']];hit=calibration.lexical(pred['output'],lexicon)['present'];empty=not pred['output'].strip();bad=degenerate(pred['output'])['degenerate']
        details.append({'lexical':hit,'usable':hit and v['content_ok'] and v['coherent'] and not v['gross_failure'] and not empty and not bad,
            'empty':empty,'degenerate':bad,'eos':pred['ended_eos'],'cap':pred['hit_token_limit'],'tokens':token_length(pred),
            'words':len(pred['output'].split()),'factual_item':row['factual'],'factual_correct':v['factual_correct'],**{k:v[k] for k in (*TRAITS,'content_ok','coherent','gross_failure')}})
    return {'n':n,'usable_profane':sum(r['usable'] for r in details),'raw_lexical':sum(r['lexical'] for r in details),
        'judge_profanity_without_lexical':sum(r['profanity']>=2 and not r['lexical'] for r in details),
        'trait_sums':{t:sum(r[t] for r in details) for t in TRAITS},'clear_presence':{t:sum(r[t]>=2 for r in details) for t in TRAITS},
        'ambiguous_scores':{t:sum(r[t]==1 for r in details) for t in TRAITS},
        'content_ok':sum(r['content_ok'] for r in details),'incoherent':sum(not r['coherent'] for r in details),'gross_failures':sum(r['gross_failure'] for r in details),
        **{k:sum(r[k] for r in details) for k in ('empty','degenerate','eos','cap')},
        'mean_tokens':sum(r['tokens'] for r in details)/n,'mean_words':sum(r['words'] for r in details)/n,
        'factual_total':sum(r['factual_item'] for r in details),'factual_correct':sum(r['factual_item'] and r['factual_correct'] for r in details)}


def quality_gates(base,candidate,tolerance):
    if base['n']!=candidate['n']:raise ValueError('paired denominators differ')
    gates={}
    for t in ('rudeness','sarcasm'):
        gates[t+'_mean']=2*(candidate['trait_sums'][t]-base['trait_sums'][t])<=base['n']
        gates[t+'_clear']=candidate['clear_presence'][t]<=base['clear_presence'][t]+tolerance
    gates.update(content=candidate['content_ok']>=base['content_ok']-tolerance,incoherent=candidate['incoherent']<=base['incoherent']+tolerance,
        gross=candidate['gross_failures']<=base['gross_failures']+tolerance)
    return gates


def choose(metrics):
    base=metrics['baseline'];outcomes={name:quality_gates(base,metrics[name],1) for name in SETTINGS}
    qualified=[name for name in SETTINGS if all(outcomes[name].values())]
    ordered=sorted(qualified,key=lambda n:(-metrics[n]['usable_profane'],SETTINGS[n][1],SETTINGS[n][0]))
    best=ordered[0] if ordered else None
    selected=best if best and metrics[best]['usable_profane']>=8 and metrics[best]['usable_profane']-base['usable_profane']>=7 else None
    return {'selected':selected,'setting':{'layer':SETTINGS[selected][0],'beta':SETTINGS[selected][1]} if selected else None,
        'quality_gates':outcomes,'metrics':metrics,'development_null':selected is None}


def final_gates(metrics,retention):
    base,steer,random=metrics['baseline'],metrics['selected'],metrics['random'];g=quality_gates(base,steer,2)
    g.update(usable=steer['usable_profane']>=12 and steer['usable_profane']-base['usable_profane']>=10,random_raw=random['raw_lexical']<=2,
        factual=steer['factual_total']==8 and steer['factual_correct']>=7 and steer['factual_correct']>=base['factual_correct']-1)
    a,b=retention['baseline'],retention['selected']
    g.update(retention_mc=sum(b[t]['correct'] for t in ('general_mc','physics_mc'))>=sum(a[t]['correct'] for t in ('general_mc','physics_mc'))-2,
        retention_gsm8k=b['gsm8k']['strict_correct']>=a['gsm8k']['strict_correct']-1,
        retention_ifeval=b['ifeval']['prompt_strict_correct']>=a['ifeval']['prompt_strict_correct']-1)
    return {'gates':g,'all_criteria_pass':all(g.values())}
