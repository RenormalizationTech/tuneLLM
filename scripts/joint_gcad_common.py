"""Pure fixed joint-study arms, packed plans, scores and descriptive summaries."""
import itertools
import math
import random
from scripts import persona_reference_common as pv
from scripts import profanity_method_transfer_common as previous
from scripts import author_profanity_method_common as text

METRICS=('profanity','impolite','sarcasm','coherence')
TRAITS=METRICS[:3]
FULL={'profanity':5.,'impolite':3.5,'sarcasm':2.}


def arms():
    result={}
    for bits in itertools.product((0,1),repeat=3):
        name='baseline' if not any(bits) else 'full_'+''.join(str(x) for x in bits)
        result[name]={t:FULL[t]*b for t,b in zip(TRAITS,bits)}
    for i,t in enumerate(TRAITS):result['half_'+t]={k:FULL[k]/2 if k==t else 0. for k in TRAITS}
    result['half_triple']={t:v/2 for t,v in FULL.items()}
    return result


def plans(rows,anchors,seed):
    ids=[r['id'] for r in rows]
    if len(ids)!=24 or len(set(ids))!=24:raise ValueError('24 unique neutral question IDs required')
    primary={m:pv.make_plan(ids,list(arms()),seed+i) for i,m in enumerate(METRICS)}
    duplicates={};calibration={}
    for i,m in enumerate(METRICS):
        selected=[]
        for j,arm in enumerate(arms()):
            for k in (0,1):
                q=ids[2*j+k]
                alias=__import__('hashlib').sha256(f'{seed+100+i}|{q}|{arm}'.encode()).hexdigest()[:24]
                selected.append({'id':alias,'question_id':q,'condition':arm})
        random.Random(seed+100+i).shuffle(selected)
        duplicates[m]=[selected[j:j+8] for j in range(0,24,8)]
        anchor_ids=[a['id'] for a in anchors[m]]
        if len(anchor_ids)!=8 or len(set(anchor_ids))!=8:raise ValueError('eight unique anchors per metric required')
        calibration[m]=pv.make_plan(anchor_ids,['anchor'],seed+200+i)
    return {'calibration':calibration,'primary':primary,'duplicates':duplicates}


def calibration_checks(anchors,ratings):
    checks={}
    for m in METRICS:
        values={a['id']:ratings[m][a['id']]['score'] or 0 for a in anchors[m]}
        hit=sum(a['band'][0]<=values[a['id']]<=a['band'][1] for a in anchors[m])
        low=[values[a['id']] for a in anchors[m] if a['level']=='low']
        high=[values[a['id']] for a in anchors[m] if a['level']=='high']
        if len(low)!=4 or len(high)!=4:raise ValueError('four low/four high anchors required')
        difference=sum(high)/4-sum(low)/4
        checks[m]={'within_band':hit,'n':8,'high_minus_low_mean':difference,'pass':hit>=7 and difference>=30}
    return {'metrics':checks,'pass':all(v['pass'] for v in checks.values())}


def guide(metrics):
    base=metrics['baseline'];result={}
    for arm,matched in [('full_110',{'profanity':'full_100','impolite':'full_010'}),
                        ('full_101',{'profanity':'full_100','sarcasm':'full_001'}),
                        ('full_011',{'impolite':'full_010','sarcasm':'full_001'}),
                        ('full_111',{'profanity':'full_100','impolite':'full_010','sarcasm':'full_001'}),
                        ('half_triple',{t:'half_'+t for t in TRAITS})]:
        row=metrics[arm];targets={}
        for t,single in matched.items():
            gain=row[t]-base[t];sg=metrics[single][t]-base[t]
            targets[t]={'gain':gain,'matched_single_gain':sg,'pass':gain>=10 and gain>=sg/2 if sg>0 else None}
        quality={'coherence_absolute':row['coherence']>=50,'coherence_relative':row['coherence']>=base['coherence']-20,'invalid':row['invalid']<=2}
        result[arm]={'targets':targets,'quality':quality,'all_pass':all(v['pass'] is True for v in targets.values()) and all(quality.values()),'interpretation':'exploratory guide candidate, not a validated controller release'}
    return result


def rank(values):
    order=sorted(range(len(values)),key=values.__getitem__);out=[0.]*len(values);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        for k in range(i,j):out[order[k]]=(i+1+j)/2
        i=j
    return out

def spearman(x,y):
    a,b=rank(x),rank(y);ma=sum(a)/len(a);mb=sum(b)/len(b)
    vx=sum((v-ma)**2 for v in a);vy=sum((v-mb)**2 for v in b)
    return sum((v-ma)*(w-mb) for v,w in zip(a,b))/math.sqrt(vx*vy) if vx and vy else None


def summarize(predictions,grades,*,bootstrap_seed,bootstrap_repetitions):
    names=list(arms());n=24
    if set(predictions)!=set(names):raise ValueError('all twelve fixed arms required')
    if any(len(predictions[a])!=n or [r['id'] for r in predictions[a]]!=[r['id'] for r in predictions['baseline']] for a in names):raise ValueError('matched 24 questions required')
    if any(grades[a,r['id']]['coherence']['score'] is None for a in names for r in predictions[a]):raise ValueError('coherence must be numeric')
    vectors={a:{m:[grades[a,r['id']][m]['score'] or 0 for r in predictions[a]] for m in METRICS} for a in names}
    result={}
    for a in names:
        rows=predictions[a];checks=[text.text_checks(r['output']) for r in rows]
        result[a]={**{m:sum(vectors[a][m])/n for m in METRICS},'n':n,
            'invalid':sum(c['empty'] or c['distinct_bigram']<.6 for c in checks),
            'empty':sum(c['empty'] for c in checks),'caps':sum(r['hit_token_limit'] for r in rows),
            'lexicon_hits':sum(bool(text.hits(r['output'])) for r in rows),'mean_tokens':sum(len(r['generated_ids']) for r in rows)/n,
            'trait_refusals':{m:sum(grades[a,r['id']][m]['score'] is None for r in rows) for m in TRAITS}}
    factorial={}
    full=list(names[:8])
    for count in (1,2,3):
        for selected in itertools.combinations(range(3),count):
            contrast={}
            for a in full:
                bits=(0,0,0) if a=='baseline' else tuple(map(int,a[-3:]))
                # Average mixed finite difference over the unspecified factors.
                contrast[a]=math.prod(1 if bits[i] else -1 for i in selected)/(2**(3-count))
            factorial['x'.join(TRAITS[i] for i in selected)]=contrast
    for pair,sa,sb in [('full_110','full_100','full_010'),('full_101','full_100','full_001'),('full_011','full_010','full_001')]:
        factorial[pair+'_additivity_residual']={pair:1.,sa:-1.,sb:-1.,'baseline':1.}
    factorial['full_triple_additivity_residual']={'full_111':1.,'full_100':-1.,'full_010':-1.,'full_001':-1.,'baseline':2.}
    factorial['half_triple_additivity_residual']={'half_triple':1.,'half_profanity':-1.,'half_impolite':-1.,'half_sarcasm':-1.,'baseline':2.}
    contrasts={label:{m:sum(c*result[a][m] for a,c in weights.items()) for m in METRICS} for label,weights in factorial.items()}
    rng=random.Random(bootstrap_seed);samples={label:{m:[] for m in METRICS} for label in factorial}
    gains={a:{m:[] for m in METRICS} for a in names}
    for _ in range(bootstrap_repetitions):
        indices=[rng.randrange(n) for _ in range(n)]
        means={a:{m:sum(vectors[a][m][i] for i in indices)/n for m in METRICS} for a in names}
        for label,weights in factorial.items():
            for m in METRICS:samples[label][m].append(sum(c*means[a][m] for a,c in weights.items()))
        for a in names:
            for m in METRICS:gains[a][m].append(means[a][m]-means['baseline'][m])
    def interval(v):
        v=sorted(v);return [v[int(.025*(len(v)-1))],v[int(.975*(len(v)-1))]]
    correlations={}
    for m,k in itertools.combinations(METRICS,2):
        rawx=[x for a in names for x in vectors[a][m]];rawy=[x for a in names for x in vectors[a][k]]
        center=lambda key:[vectors[a][key][i]-sum(vectors[b][key][i] for b in names)/len(names) for a in names for i in range(n)]
        correlations[m+'__'+k]={'within_arm':{a:spearman(vectors[a][m],vectors[a][k]) for a in names},'raw_pooled':spearman(rawx,rawy),'prompt_centred_pooled':spearman(center(m),center(k))}
    return {'metrics':result,'guide':guide(result),'single_cross_effects':{t:{m:result[a][m]-result['baseline'][m] for m in METRICS} for t,a in zip(TRAITS,('full_100','full_010','full_001'))},
        'factorial_definition':'Average mixed finite differences over unspecified factors, weights product(2*bit-1)/2**(3-order); not standard effect-coded interaction scaling. Pair/triple residuals subtract sums of matched single changes.',
        'factorial_contrasts':contrasts,'factorial_question_bootstrap_ci95':{l:{m:interval(v) for m,v in scores.items()} for l,scores in samples.items()},
        'arm_gain_question_bootstrap_ci95':{a:{m:interval(v) for m,v in scores.items()} for a,scores in gains.items()},'spearman':correlations,
        'correlation_limit':'Within-arm correlations are primary descriptive geometry. Both raw and prompt-centred pooled correlations remain condition-coupled, not evidence of independent controls.',
        'bootstrap':{'unit':'matched question, all arms resampled together','n':n,'seed':bootstrap_seed,'repetitions':bootstrap_repetitions,'unadjusted':True}}
