"""Pure span/plan/consensus/gate helpers for separate rudeness and sarcasm studies."""
from collections import Counter
import re

from scripts import persona_rating_calibration as calibration
from scripts import profanity_common as mechanics
from scripts.tone_common import degenerate, token_length

TRAITS=calibration.TRAITS
LAYERS,BETAS,SETTINGS,CONDITIONS=mechanics.LAYERS,mechanics.BETAS,mechanics.SETTINGS,mechanics.CONDITIONS
capture_response,direction_statistics,delta,audited_steer=mechanics.capture_response,mechanics.direction_statistics,mechanics.delta,mechanics.audited_steer
PHASES={'eligibility':(2,8,501),'development':(7,14,502),'confirmation':(3,9,503),'confirmation_repeat':(3,9,504)}


def validate_span(row):
    for side in ('neutral','target'):
        span=row[side+'_span'];text=row[side]
        if set(span)!={'start','end','text'} or any(type(span[k]) is not int for k in ('start','end')) or not 0<=span['start']<span['end']<=len(text) or text[span['start']:span['end']]!=span['text'] or not span['text'].strip():raise ValueError('invalid exact replacement span')
        if span['start'] not in {m.start() for m in re.finditer(r'\S+',text)}:raise ValueError('span must begin at a whitespace word boundary')
    a,b=row['neutral_span'],row['target_span']
    if row['neutral'][:a['start']]!=row['target'][:b['start']] or row['neutral'][a['end']:]!=row['target'][b['end']:]:raise ValueError('outside-span bytes differ')
    if a['text']==b['text']:raise ValueError('target span must differ')


def span_stats(rows):
    bins=Counter({'opening':0,'middle':0,'closing':0});phrases=Counter();families=Counter()
    for row in rows:
        validate_span(row);i=len(row['target'][:row['target_span']['start']].split());N=len(row['target'].split())
        bins['opening' if 3*i<N else 'middle' if 3*i<2*N else 'closing']+=1
        phrases[row['target_span']['text'].casefold()]+=1;families[row['style_family']]+=1
    return {'position_bins':dict(bins),'phrase_counts':dict(phrases),'style_family_counts':dict(families),
        'position_definition':'zero-based first annotated-span whitespace word i / N; exact integer thirds'}


def validate_fixtures(data,lexicon):
    extraction=data['extraction'];rows=extraction['items'];target=extraction['target_trait']
    if target not in ('rudeness','sarcasm') or len(rows)!=32 or extraction.get('schema_version')!=1 or extraction.get('authored_synthetic') is not True:raise ValueError('fixed trait/extraction fixture required')
    families=Counter(r['family'] for r in rows)
    if len(families)!=16 or set(families.values())!={2} or len({r['id'] for r in rows})!=32:raise ValueError('16 two-item families required')
    for row in rows:
        if set(row)!={'id','family','topic','context','user','requirements','reference','factual','neutral','target','style_family','neutral_span','target_span'} or row['factual'] is not False:raise ValueError('extraction schema differs')
        validate_span(row)
        if any(calibration.lexical(row[k],lexicon)['present'] for k in ('neutral','target','user')):raise ValueError('fixed lexical profanity must be absent on both sides')
        if not row['requirements'] or not row['reference'] or any(not isinstance(v,str) or not v.strip() for k in ('context','user') for v in [row[k]]):raise ValueError('task/context/reference required')
    stats=span_stats(rows);counts=stats['style_family_counts']
    if len(counts)<4 or any(3*n>32 for n in counts.values()) or max(stats['phrase_counts'].values())>3:raise ValueError('phrasing diversity fails')
    for family in counts:
        if len({r['target_span']['text'] for r in rows if r['style_family']==family})<2:raise ValueError('two phrases per style family required')
    bins=stats['position_bins']
    if bins['opening']>12 or bins['middle']<10 or bins['closing']<10:raise ValueError('position quota fails')
    for split,n in [('development',16),('confirmation',24)]:
        if len(data[split]['items'])!=n:raise ValueError('shared task count differs')
    if sum(r['factual'] for r in data['confirmation']['items'])!=8:raise ValueError('eight shared factual probes required')


def plans(data):
    result={'eligibility':mechanics.constrained_plan(data['extraction']['items'],('neutral','target'),8,501),
        'development':mechanics.constrained_plan(data['development']['items'],CONDITIONS,14,502),
        'confirmation':mechanics.constrained_plan(data['confirmation']['items'],('baseline','selected','random'),9,503)}
    repeat=mechanics.constrained_plan(data['confirmation']['items'],('baseline','selected','random'),9,504)
    aliases={(e['group_id'],e['condition']):e['id'] for b in result['confirmation'] for e in b}
    for batch in repeat:
        for e in batch:e['id']=aliases[e['group_id'],e['condition']]
    result['confirmation_repeat']=repeat
    return result


validate_ratings=mechanics.validate_ratings


def eligibility(rows,reviews,ratios,lexicon,target):
    retained=[];audit=[];other=[t for t in TRAITS if t!=target]
    for row in rows:
        a,b=reviews[row['id']]['neutral'],reviews[row['id']]['target']
        checks={'lexical_absent':not any(calibration.lexical(row[k],lexicon)['present'] for k in ('neutral','target')),
            'neutral_traits':all(a[t]<=1 for t in TRAITS),'target_present':b[target]>=2,'other_traits':all(b[t]<=1 for t in other),
            'quality':all(v['content_ok'] and v['coherent'] and not v['gross_failure'] for v in (a,b)),
            'token_ratio':.85<=ratios[row['id']]<=1.15}
        good=all(checks.values());audit.append({'id':row['id'],'checks':checks,'eligible':good,'token_ratio':ratios[row['id']]})
        if good:retained.append(row['id'])
    stats=span_stats([r for r in rows if r['id'] in retained]);counts=stats['style_family_counts']
    return {'target_trait':target,'retained_ids':retained,'retained_count':len(retained),'retained_span_stats':stats,'audit':audit,
        'sufficient':len(retained)>=24 and len(counts)>=4 and all(3*n<=len(retained) for n in counts.values()),
        'ambiguous_scores':{t:sum(reviews[r['id']][side][t]==1 for r in rows for side in ('neutral','target')) for t in TRAITS}}


def consensus(a,b,target):
    return {'id':a['id'],**{t:min(a[t],b[t]) if t==target else max(a[t],b[t]) for t in TRAITS},
        'content_ok':a['content_ok'] and b['content_ok'],'coherent':a['coherent'] and b['coherent'],
        'factual_correct':a['factual_correct'] and b['factual_correct'],'gross_failure':a['gross_failure'] or b['gross_failure']}


def measures(rows,predictions,reviews,lexicon,target,second=None):
    if [r['id'] for r in rows]!=[v['id'] for v in predictions]:raise ValueError('prediction order/coverage differs')
    details=[]
    for row,pred in zip(rows,predictions):
        a=reviews[row['id']];b=second[row['id']] if second else a;v=consensus(a,b,target) if second else a
        empty=not pred['output'].strip();bad=degenerate(pred['output'])['degenerate'];present=v[target]>=2
        details.append({**{t:v[t] for t in TRAITS},'present':present,'either':a[target]>=2 or b[target]>=2,
            'usable':present and v['content_ok'] and v['coherent'] and not v['gross_failure'] and not empty and not bad,
            'content_ok':v['content_ok'],'incoherent':not v['coherent'],'gross':v['gross_failure'],'lexical':calibration.lexical(pred['output'],lexicon)['present'],
            'empty':empty,'degenerate':bad,'eos':pred['ended_eos'],'cap':pred['hit_token_limit'],'tokens':token_length(pred),'words':len(pred['output'].split()),
            'factual_item':row['factual'],'factual_correct':v['factual_correct']})
    n=len(rows)
    return {'n':n,'target_trait':target,'consensus':second is not None,'usable_target':sum(v['usable'] for v in details),'raw_target':sum(v['present'] for v in details),
        'raw_target_either':sum(v['either'] for v in details),'lexical_profanity':sum(v['lexical'] for v in details),
        'trait_sums':{t:sum(v[t] for v in details) for t in TRAITS},'clear_presence':{t:sum(v[t]>=2 for v in details) for t in TRAITS},
        'ambiguous_scores':{t:sum(v[t]==1 for v in details) for t in TRAITS},
        'rudeness_only':sum(v['rudeness']>=2 and v['sarcasm']<2 for v in details),'sarcasm_only':sum(v['sarcasm']>=2 and v['rudeness']<2 for v in details),
        'rudeness_and_sarcasm':sum(v['rudeness']>=2 and v['sarcasm']>=2 for v in details),
        **{k:sum(v[k] for v in details) for k in ('content_ok','incoherent','empty','degenerate','eos','cap')},'gross_failures':sum(v['gross'] for v in details),
        'mean_tokens':sum(v['tokens'] for v in details)/n,'mean_words':sum(v['words'] for v in details)/n,
        'factual_total':sum(v['factual_item'] for v in details),'factual_correct':sum(v['factual_item'] and v['factual_correct'] for v in details)}


def quality_gates(base,candidate,tolerance):
    target=base['target_trait']
    if target!=candidate['target_trait'] or base['n']!=candidate['n']:raise ValueError('profile/denominator mismatch')
    g={}
    for t in TRAITS:
        if t==target:continue
        g[t+'_mean']=2*(candidate['trait_sums'][t]-base['trait_sums'][t])<=base['n']
        g[t+'_clear']=candidate['clear_presence'][t]<=base['clear_presence'][t]+tolerance
    g.update(content=candidate['content_ok']>=base['content_ok']-tolerance,incoherent=candidate['incoherent']<=base['incoherent']+tolerance,gross=candidate['gross_failures']<=base['gross_failures']+tolerance)
    return g


def choose(metrics):
    base=metrics['baseline'];gates={name:quality_gates(base,metrics[name],1) for name in SETTINGS}
    candidates=sorted([n for n in SETTINGS if all(gates[n].values())],key=lambda n:(-metrics[n]['usable_target'],SETTINGS[n][1],SETTINGS[n][0]))
    best=candidates[0] if candidates else None;selected=best if best and metrics[best]['usable_target']>=8 and metrics[best]['usable_target']-base['usable_target']>=7 else None
    return {'target_trait':base['target_trait'],'selected':selected,'setting':{'layer':SETTINGS[selected][0],'beta':SETTINGS[selected][1]} if selected else None,
        'quality_gates':gates,'metrics':metrics,'development_null':selected is None}


def final_gates(metrics,retention):
    base,selected,random=metrics['baseline'],metrics['selected'],metrics['random'];g=quality_gates(base,selected,2)
    g.update(usable=selected['usable_target']>=12 and selected['usable_target']-base['usable_target']>=10,random_raw=random['raw_target_either']<=2,
        factual=selected['factual_total']==8 and selected['factual_correct']>=7 and selected['factual_correct']>=base['factual_correct']-1)
    a,b=retention['baseline'],retention['selected']
    g.update(retention_mc=sum(b[t]['correct'] for t in ('general_mc','physics_mc'))>=sum(a[t]['correct'] for t in ('general_mc','physics_mc'))-2,
        retention_gsm8k=b['gsm8k']['strict_correct']>=a['gsm8k']['strict_correct']-1,retention_ifeval=b['ifeval']['prompt_strict_correct']>=a['ifeval']['prompt_strict_correct']-1)
    return {'gates':g,'all_criteria_pass':all(g.values())}


def repeat_diagnostics(rows,conditions,first,second):
    result={}
    for name in (*conditions,'all_conditions'):
        pairs=[(first[r['id']][condition],second[r['id']][condition]) for r in rows for condition in (conditions if name=='all_conditions' else [name])]
        result[name]={}
        for t in TRAITS:
            a=[x[t] for x,y in pairs];b=[y[t] for x,y in pairs];k=calibration.kappa_fraction(a,b)
            result[name][t]={'presence_disagreements':sum((x>=2)!=(y>=2) for x,y in zip(a,b)),
                'both_present':sum(x>=2 and y>=2 for x,y in zip(a,b)),'first_only':sum(x>=2 and y<2 for x,y in zip(a,b)),
                'second_only':sum(x<2 and y>=2 for x,y in zip(a,b)),'neither_present':sum(x<2 and y<2 for x,y in zip(a,b)),
                'quadratic_weighted_kappa':float(k) if k is not None else None,'kappa_is_gate':False}
    return result


def divergence(base,condition):
    if [r['id'] for r in base]!=[r['id'] for r in condition]:raise ValueError('divergence rows differ')
    items=[];positions=Counter();lengths=Counter({'shorter':0,'same':0,'longer':0})
    for a,b in zip(base,condition):
        left,right=a['generated_ids'],b['generated_ids'];position=next((i for i,(x,y) in enumerate(zip(left,right)) if x!=y),None)
        if position is None and len(left)!=len(right):position=min(len(left),len(right))
        change=len(right)-len(left);lengths['shorter' if change<0 else 'longer' if change>0 else 'same']+=1
        if position is not None:positions[str(position)]+=1
        items.append({'id':a['id'],'identical':position is None,'first_divergence_position':position,'generated_length_change':change})
    changed=sum(not r['identical'] for r in items)
    return {'changed':changed,'identical':len(items)-changed,'first_divergence_histogram':dict(positions),'length_changes':dict(lengths),'items':items,
        'interpretation':'posthoc mechanical diagnostic only; no setting or budget change'}
