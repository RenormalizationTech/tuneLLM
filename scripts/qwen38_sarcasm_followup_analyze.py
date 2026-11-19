"""Frozen-algorithm follow-up closeout; CPU-only question-cluster summaries.

Part A is post-hoc/observational. Part B is exploratory development. All
bootstrap resamples share the same 32-question weights across both contents.
Gram matrices avoid materializing 10,000 x 5120 bootstrap activation vectors.
"""
import json
import math
from pathlib import Path
import numpy as np
from scripts.common import ROOT,read_json
from scripts.qwen38_science_runtime import sha


def weights(n,samples,seed):
    return np.random.default_rng(seed).multinomial(n,np.full(n,1/n),size=samples).astype(np.float64)/n


def interval(values):
    values=np.asarray(values,dtype=np.float64);values=values[np.isfinite(values)]
    return np.quantile(values,[.025,.975]).tolist() if len(values) else None


def mean_ci(x,w):
    x=np.asarray(x,dtype=np.float64)
    return {'mean':float(x.mean()),'bootstrap95':interval(w@x),'n_questions':len(x)}


def cosine(x,direction):
    denom=np.linalg.norm(x)*np.linalg.norm(direction)
    return float(np.dot(x,direction)/denom) if denom>0 else None


def cosine_ci(x,direction,w):
    x=np.asarray(x,dtype=np.float64);direction=np.asarray(direction,dtype=np.float64)
    gram=x@x.T;projection=x@direction
    denominator=np.sqrt(np.maximum(0,((w@gram)*w).sum(1)))*np.linalg.norm(direction)
    samples=np.divide(w@projection,denominator,out=np.full(len(w),np.nan),where=denominator>0)
    ci=interval(samples)
    wording='positively_aligned' if ci is not None and ci[0]>0 else ('anti_aligned' if ci is not None and ci[1]<0 else 'no_clear_alignment')
    return {'cosine_of_mean':cosine(x.mean(0),direction),'bootstrap95':ci,'ci_wording':wording,
            'ci_role':'Prespecified descriptive wording, not a hypothesis test or causal claim.','n_questions':len(x),
            'undefined_bootstrap_samples':int((~np.isfinite(samples)).sum())}


def norm_mean_ci(x,w):
    x=np.asarray(x,dtype=np.float64);gram=x@x.T
    samples=np.sqrt(np.maximum(0,((w@gram)*w).sum(1)))
    return {'norm_of_mean':float(np.linalg.norm(x.mean(0))),'bootstrap95':interval(samples),'n_questions':len(x)}


def consistency(x):
    x=np.asarray(x,dtype=np.float64);total=x.sum(0)
    values=[cosine(v,(total-v)/(len(x)-1)) for v in x]
    numeric=[v for v in values if v is not None]
    return {'mean_question_to_leave_one_out_cosine':sum(numeric)/len(numeric) if numeric else None,
            'numeric_questions':len(numeric),'undefined_questions':len(x)-len(numeric)}


def paired_token_effect(s,l):
    """Match token positions, FP32 outputs, and per-token L2 denominators."""
    s=np.asarray(s,dtype=np.float32);l=np.asarray(l,dtype=np.float32)
    if s.shape!=l.shape or s.ndim!=2:raise ValueError('paired response-token arrays must match')
    norms=np.linalg.norm(s-l,axis=-1);native=np.linalg.norm(l,axis=-1);valid=native>0
    return {'mean_token_difference_norm':float(norms.mean()),
            'mean_token_difference_to_native_L_ratio':float((norms[valid]/native[valid]).mean()) if valid.any() else None,
            'norm_of_mean_difference':float(np.linalg.norm((s-l).mean(0))),
            'mean_native_L_token_norm':float(native.mean()),'tokens':len(norms),'zero_native_L_tokens':int((~valid).sum())}


def part_a(run,cfg,inputs):
    from safetensors import safe_open
    from safetensors.torch import load_file
    records=read_json(run/'diagnostic/result.json')['rows']
    if len(records)!=192 or len({r['id'] for r in records})!=192:raise ValueError('complete 32x2x3 diagnostics required')
    metadata={r['id']:r for r in records};cells={};tensor_hashes={}
    for r in records:
        file=run/r['tensor_file']
        if sha(file)!=r['tensor_sha256']:raise ValueError('activation tensor bytes differ')
        tensor_hashes[r['tensor_file']]=r['tensor_sha256']
        with safe_open(str(file),framework='pt',device='cpu') as f:
            cells[r['id']]={k:f.get_tensor(k).float().numpy() for k in f.keys() if not k.startswith('attention_tokens.')}
    keys=list(cells['C01_B_S']);w=weights(32,cfg['bootstrap_samples'],cfg['bootstrap_seed'])
    bank=load_file(str(run/'bank.safetensors'))
    random=[load_file(str(run/f'random-{seed}.safetensors')) for seed in [9831,9832,9833]]
    def vectors(key,context,content):return np.stack([cells[f'C{i:02d}_{content}_{context}'][key] for i in range(1,33)])
    def token_norm(key,context,content):return np.array([metadata[f'C{i:02d}_{content}_{context}']['summary'][key]['mean_token_norm'] for i in range(1,33)])
    residual={};alignment={};magnitudes={};gate={};nll={};supplemental={}
    for key in keys:
        sb,lb,sp,lp= [vectors(key,c,x) for c,x in [('S','B'),('L','B'),('S','P'),('L','P')]]
        has_N=not key.startswith('crop.')
        if has_N:nb,np_=[vectors(key,'N',content) for content in ['B','P']]
        ib,ip=sb-lb,sp-lp
        if key.startswith('block.'):
            li=int(key.split('.')[1]);residual[key]={'layer_type':inputs['model_geometry']['layer_types'][li]}
            for content,diff in [('B',ib),('P',ip)]:
                denominator=float(token_norm(key,'L',content).mean())
                residual[key]['instruction_'+content]={'norm_of_mean':float(np.linalg.norm(diff.mean(0))),
                    'mean_native_L_token_norm':denominator,'norm_of_mean_to_native_ratio':float(np.linalg.norm(diff.mean(0))/denominator) if denominator else None,
                    **consistency(diff)}
            denom=float(np.concatenate([token_norm(key,'S','B'),token_norm(key,'S','P')]).mean())
            residual[key]['content_at_S']={'norm_of_mean':float(np.linalg.norm((sp-sb).mean(0))),
                'mean_native_S_token_norm':denom,'norm_of_mean_to_native_ratio':float(np.linalg.norm((sp-sb).mean(0))/denom) if denom else None}
        if key.startswith(('attention.','crop.')):
            li=int(key.split('.')[1]);delta=bank[f'{li}.delta'].float().numpy()
            alignment[key]={}
            contrasts={'instruction_B':ib,'instruction_P':ip}
            if key.startswith('crop.'):contrasts['bank_analogue_SP_minus_LB']=sp-lb
            for name,diff in contrasts.items():
                alignment[key][name]={**cosine_ci(diff,delta,w),
                    'random_direction_cosines':[cosine(diff.mean(0),r[f'{li}.delta'].float().numpy()) for r in random],
                    'signed_random_cosine_expectation':0.,'random_cosine_SD_scale':1/math.sqrt(delta.size),
                    'reference_note':'Three descriptive random directions; neither a null interval nor a binary alignment test.'}
            if key.startswith('attention.'):
                magnitudes[key]={}
                for content,diff in [('B',ib),('P',ip)]:
                    denominator=float(token_norm(key,'L',content).mean());paired=[]
                    for qi in range(1,33):
                        def get(context):
                            record=metadata[f'C{qi:02d}_{content}_{context}']
                            with safe_open(str(run/record['tensor_file']),framework='pt',device='cpu') as f:
                                return f.get_tensor(f'attention_tokens.{li}').float().numpy()
                        paired.append(paired_token_effect(get('S'),get('L')))
                    ratios=[r['mean_token_difference_to_native_L_ratio'] for r in paired]
                    magnitudes[key][content]={'norm_of_mean_instruction_difference':float(np.linalg.norm(diff.mean(0))),
                        'norm_of_mean_instruction_difference_CI':norm_mean_ci(diff,w),
                        'mean_native_L_token_norm':denominator,'norm_of_mean_to_native_ratio':float(np.linalg.norm(diff.mean(0))/denominator) if denominator else None,
                        'paired_per_question':paired,
                        'mean_token_difference_norm':mean_ci([r['mean_token_difference_norm'] for r in paired],w),
                        'mean_token_difference_to_native_L_ratio':mean_ci(ratios,w) if all(r is not None for r in ratios) else None,
                        'old_requested_norm_bounds':{str(b):2*b*float(np.linalg.norm(delta)) for b in [1,2]},
                        'comparison_note':'Paired per-token actual differences, compared descriptively with Part B actual_ratio. Norm of mean can cancel; no causal dose equivalence.'}
        supplemental[key]={'interaction_norm_of_mean':float(np.linalg.norm((ip-ib).mean(0))),
            'content_effect_norm_of_mean':{c:float(np.linalg.norm((p-b).mean(0))) for c,p,b in [('S',sp,sb),('L',lp,lb)]}}
        if has_N:
            supplemental[key]['content_effect_norm_of_mean']['N']=float(np.linalg.norm((np_-nb).mean(0)))
            supplemental[key]['system_presence_position_confounded_norm_of_mean']={'B':float(np.linalg.norm((lb-nb).mean(0))),'P':float(np.linalg.norm((lp-np_).mean(0)))}
    for li in cfg['layers']:
        gate[str(li)]={}
        for content in ['B','P']:
            values={context:np.array([metadata[f'C{qi:02d}_{content}_{context}']['gate'][str(li)]['gate']['mean'] for qi in range(1,33)]) for context in ['S','L','N']}
            gate[str(li)][content]={'context_question_means':{c:float(v.mean()) for c,v in values.items()},'paired_S_minus_L':mean_ci(values['S']-values['L'],w)}
    for content in ['B','P']:
        s=np.array([metadata[f'C{qi:02d}_{content}_S']['nll'] for qi in range(1,33)])
        l=np.array([metadata[f'C{qi:02d}_{content}_L']['nll'] for qi in range(1,33)])
        nll[content]=mean_ci(s-l,w)
    return {'status':'complete','role':'post-hoc, observational, fixed raw continuations; no causal attribution',
        'question_clusters':32,'cells':192,'bootstrap':{'samples':cfg['bootstrap_samples'],'seed':cfg['bootstrap_seed'],
            'unit':'question; shared weights preserve both continuations and all contexts','method':'Gram-matrix cosine and paired question means'},
        'A1_alignment':alignment,'A2_magnitude':magnitudes,'A3_residual':residual,'A4_gate':gate,'A5_nll_S_minus_L':nll,
        'supplemental_contrasts':supplemental,'activation_sha256':tensor_hashes,
        'denominator_note':'Native per-token norm(dim=-1), never the old extraction whole-sequence Frobenius norm.'}


def telemetry_summary(rows):
    grouped={}
    for row in rows:
        if not row.get('telemetry'):continue
        arm=row['arm'];target=grouped.setdefault(arm,{})
        for site,phases in row['telemetry'].items():
            for phase,value in phases.items():
                summary=target.setdefault(site,{}).setdefault(phase,{'tokens':0,'calls':0,'zero_native_norm_tokens':0,'metrics':{}})
                for key in ['tokens','calls','zero_native_norm_tokens']:summary[key]+=value[key]
                for metric,v in value['metrics'].items():
                    r=summary['metrics'].setdefault(metric,{'sum':0.,'count':0,'min':None,'max':None})
                    r['sum']+=v['sum'];r['count']+=v['count']
                    if v['min'] is not None:r['min']=v['min'] if r['min'] is None else min(r['min'],v['min'])
                    if v['max'] is not None:r['max']=v['max'] if r['max'] is None else max(r['max'],v['max'])
    for sites in grouped.values():
        for phases in sites.values():
            for value in phases.values():
                for r in value['metrics'].values():r['mean']=r['sum']/r['count'] if r['count'] else None
    return {'weighting':'token positions within each arm/site/phase; no averaging call means',
            'actual_definition':'norm(steered.float() - original.float(),dim=-1)',
            'requested_definition':'norm(model-dtype coefficient*model-dtype delta addition operand,dim=-1)',
            'algebraic_requested_definition':'coefficient.float()*norm(model-dtype delta.float())',
            'arms':grouped}


def part_b(rows,scores,cfg):
    from scripts import sarcasm_method_transfer_common as scientific
    from scripts.author_profanity_method_common import text_checks
    from scripts.qwen38_sarcasm_analyze import pearson
    if len(rows)!=224 or set(scores)!={r['id'] for r in rows}:raise ValueError('complete fresh32x7 scores required')
    metrics={arm:scientific.summarize([r for r in rows if r['arm']==arm],scores) for arm in cfg['arms']}
    gates={str(b):scientific.gates(metrics['baseline'],metrics[f'b{b}']) for b in cfg['strengths']}
    passing=[b for b in cfg['strengths'] if gates[str(b)]['all_pass']]
    descriptives={}
    for arm in cfg['arms']:
        rr=[r for r in rows if r['arm']==arm];grade=[scores[r['id']] for r in rr]
        descriptives[arm]={'coherent_sarcastic':sum(g['trait'] is not None and g['trait']>=50 and g['coherence']>=50 for g in grade),
            'coherence_below50':sum(g['coherence']<50 for g in grade),
            'mostly_not_letters':sum(text_checks(r['output'])['mostly_not_letters'] for r in rr),
            'impoliteness_mean':sum(g['impolite'] for g in grade)/32 if all('impolite' in g for g in grade) else None,
            'sarcasm_impoliteness':pearson([(g['trait'],g['impolite']) for g in grade if g['trait'] is not None]) if all('impolite' in g for g in grade) else None}
    complete=all('impolite' in g for g in scores.values())
    return {'status':'complete' if complete else 'primary_complete_descriptive_incomplete',
        'role':'fresh exploratory development; no independent final or validated controller',
        'outcome':'exploratory_candidate' if passing else 'exploratory_null_through_b16',
        'exploratory_candidate':min(passing) if passing else None,'metrics':metrics,'gates':gates,
        'random16_descriptive_gates':scientific.gates(metrics['baseline'],metrics['random16']),
        'b16_random_margin20':metrics['b16']['trait_conservative']>=metrics['random16']['trait_conservative']+20,
        'b16_direction_check':{'sarcasm_margin':metrics['b16']['trait_conservative']-metrics['random16']['trait_conservative'],
            'margin_at_least20':metrics['b16']['trait_conservative']>=metrics['random16']['trait_conservative']+20,
            'role':'fixed exploratory direction reference at b16; no independent confirmation'},
        'prompted_validity_concern':metrics['prompted']['trait_conservative']<metrics['baseline']['trait_conservative']+20,
        'descriptives':descriptives,
        'pooled_sarcasm_impoliteness':pearson([(g['trait'],g['impolite']) for g in scores.values() if g['trait'] is not None]) if complete else None,
        'pooled_caveat':'Confounded by arm; not evidence for within-arm coupling.', 'telemetry':telemetry_summary(rows)}


def analyze(run,status,error):
    run=Path(run);cfg=read_json(run/'config.json');manifest=read_json(run/'manifest.json')
    inputs=read_json(run/'inputs.json');hashes={}
    for name in ['config.json','manifest.json','inputs.json','plans.json','rubrics.json','bank.safetensors',
                 'mechanics/result.json','diagnostic/result.json','generation/result.json','judge/trait.json','judge/coherence.json','judge/impolite.json']:
        if (run/name).is_file():hashes[name]=sha(run/name)
    counters={}
    for kind in ['generation','teacher_forced','logit','judge']:
        files=sorted((run/'counters'/kind).glob('*.json'));values=[read_json(p) for p in files]
        counters[kind]={'reserved_starts':len(values),'by_stage':{s:sum(v['stage']==s for v in values) for s in sorted({v['stage'] for v in values})}}
        if kind=='generation':counters[kind]['reserved_tokens']=sum(v['tokens'] for v in values)
        hashes.update({str(p.relative_to(run)):sha(p) for p in files})
    sessions={}
    for kind in ['gpu','audit','judge','launcher']:
        files=sorted((run/'sessions').glob(f'{kind}-*.done.json'))
        sessions[kind]={'seconds':sum(read_json(p)['seconds'] for p in files),'completed_sessions':len(files)}
        hashes.update({str(p.relative_to(run)):sha(p) for p in files})
    result={'outcome':status,'execution_error':error,'scope':cfg['scope'],'old_outcome_unchanged':'development_null',
        'manifest_sha256':sha(run/'manifest.json'),'analyzer_sha256':sha(__file__),
        'ledger':{'counters':counters,'sessions':sessions},'part_A':{'status':'incomplete'},'part_B':{'status':'incomplete'},
        'limitations':['Part A post-hoc and context-conditioned; no causal attribution.',
            'N context is position-confounded; S/L match token length and response positions.',
            'One instruction pair; old bank pooled five pairs.',
            'All study answers capped at160, n32, seven arms, no independent final.',
            'NF4 coordinates and known strict cache/full-prefix nonequivalence; identical cached batch1 arms.',
            'Fresh questions mean new project wording, not verified absence from pretraining.',
            'Random direction control exists only at b16; lower-strength passes are not direction-tested.',
            'S minus L changes whole fixed wording and does not isolate pure trait specificity.',
            'Norm units corrected: no old whole-sequence Frobenius denominator or closed-study per-token ratios are used.',
            'Part A fixes continuations; Part B free generation diverges. Their per-token ratios condition on different sequences.',
            'Dose is never concluded to have caused the old development null.']}
    if (run/'diagnostic/result.json').exists():result['part_A']=part_a(run,cfg,inputs)
    if (run/'generation/result.json').exists():
        rows=read_json(run/'generation/result.json')['rows'];streams={}
        for m in ['trait','coherence','impolite']:
            if (run/'judge'/f'{m}.json').exists():streams[m]=read_json(run/'judge'/f'{m}.json')
        if {'trait','coherence'}<=streams.keys():
            ids={r['id'] for r in rows}
            if any(set(values)!=ids for values in streams.values()):raise ValueError('judge stream IDs differ')
            scores={i:{m:v[i] for m,v in streams.items()} for i in ids};result['part_B']=part_b(rows,scores,cfg)
            result['outcome']=result['part_B']['status'] if result['part_B']['status']!='complete' else result['part_B']['outcome']
        else:result['part_B']['telemetry']=telemetry_summary(rows)
    result['input_sha256']=hashes
    return result
