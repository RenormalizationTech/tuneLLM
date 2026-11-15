"""Pure plans, gates, author extraction access and observed shift accounting."""
import ast
from contextlib import contextmanager
import math
import unicodedata

from scripts import persona_reference_common as pv
from scripts import author_profanity_method_common as text
from scripts import author_gcad_impolite_common as gcad


def normalized(value):return ' '.join(unicodedata.normalize('NFKC',value).casefold().split())


def development_settings(cfg,has_gcad):
    return {'baseline':{'method':'baseline'},
            **{f'{m}_{v:g}':{'method':m,'strength':v} for m in ('fixed','saps') for v in cfg['projection_alphas']},
            **({f'gcad_{v:g}':{'method':'gcad','strength':v} for v in cfg['gcad_base_coefs']} if has_gcad else {})}


def frozen_plans(data,cfg):
    groups={'elicitation':[r['id'] for r in data['train']], 'development':[r['id'] for r in data['calibration']], 'heldout':[r['id'] for r in data['heldout']]}
    arms={'elicitation':[f'i{i}_{side}' for i in range(5) for side in ('pos','neg')],
          'development':list(development_settings(cfg,True)),
          'heldout':['baseline',*['selected_'+m for m in ('fixed','saps','gcad')],*['random_'+m for m in ('fixed','saps','gcad')]]}
    return {phase:{metric:pv.make_plan(groups[phase],arms[phase],cfg['plan_seeds'][phase+'_'+metric]) for metric in ('trait','coherence')} for phase in groups}


def active_plan(plan,available):
    # A branch only removes predeclared conditions. Never regroup or rerandomize.
    return [[r for r in batch if r['condition'] in available] for batch in plan if any(r['condition'] in available for r in batch)]


def eligible_pairs(records,grades):
    by={(r['question_id'],r['instruction_index'],r['side']):r for r in records};result=[]
    if len(by)!=len(records):raise ValueError('duplicate elicitation identity')
    for q,i in sorted({(r['question_id'],r['instruction_index']) for r in records}):
        pos=by[q,i,'pos'];neg=by[q,i,'neg'];p=grades[pos['id']];n=grades[neg['id']]
        if p['trait'] is not None and n['trait'] is not None and p['trait']>=50 and n['trait']<50 and p['coherence']>=50 and n['coherence']>=50:
            result.append({'question_id':q,'instruction_index':i,'pos':pos['id'],'neg':neg['id']})
    return result


def summarize(rows,grades):
    if len(rows)!=32 or len({r['id'] for r in rows})!=32:raise ValueError('fixed32 condition required')
    scores=[grades[r['id']] for r in rows];checks=[text.text_checks(r['output']) for r in rows]
    trait=[r['trait'] for r in scores if r['trait'] is not None]
    return {'n':32,'trait_conservative':sum(trait)/32,'trait_numeric_mean':sum(trait)/len(trait) if trait else None,
            'refusals':32-len(trait),'coherence':sum(r['coherence'] for r in scores)/32,
            'lexicon_hits':sum(bool(text.hits(r['output'])) for r in rows),
            'invalid':sum(c['empty'] or c['distinct_bigram']<.6 for c in checks),'empty':sum(c['empty'] for c in checks),
            'caps':sum(r['hit_token_limit'] for r in rows),'eos':sum(r['ended_eos'] for r in rows),
            'mean_tokens':sum(len(r['generated_ids']) for r in rows)/32,
            'mean_distinct_bigram':sum(c['distinct_bigram'] for c in checks)/32}


def gates(base,value,random=None):
    result={'trait_gain':value['trait_conservative']>=base['trait_conservative']+20,
            'lexicon_absolute':value['lexicon_hits']>=8,'lexicon_gain':value['lexicon_hits']>=base['lexicon_hits']+8,
            'coherence_absolute':value['coherence']>=50,'coherence_relative':value['coherence']>=base['coherence']-20,
            'invalid':value['invalid']<=3}
    if random is not None:result['trait_random_gain']=value['trait_conservative']>=random['trait_conservative']+20
    return {**result,'all_pass':all(result.values())}


def choose(metrics,settings):
    details={name:gates(metrics['baseline'],value) for name,value in metrics.items() if name!='baseline'};selected={}
    for method in ('fixed','saps','gcad'):
        candidates=[name for name in details if settings[name]['method']==method and details[name]['all_pass']]
        if candidates:selected[method]=min(candidates,key=lambda name:settings[name]['strength'])
    return {'selected':selected,'gates':details,'selection_null':not selected}


def final_decision(metrics,methods):
    checks={m:gates(metrics['baseline'],metrics['selected_'+m],metrics['random_'+m]) for m in methods}
    passing=[m for m in methods if checks[m]['all_pass']];tie={'fixed':0,'saps':1,'gcad':2}
    rank=sorted(passing,key=lambda m:(-(metrics['selected_'+m]['trait_conservative']-metrics['random_'+m]['trait_conservative']),-metrics['selected_'+m]['coherence'],tie[m]))
    return {'criteria':checks,'passing':rank,'preferred':rank[0] if rank else None,'no_posttest_reselection':True}


def extraction_functions(delta_file,key_file):
    """AST-isolate the two pinned author definitions, never pipeline imports."""
    import torch
    from tqdm import tqdm
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    result={}
    for file,name in ((delta_file,'get_prompt_attn_delta_p_and_r'),(key_file,'compute_k_pos_and_avg_demand')):
        nodes=[n for n in ast.parse(file.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name]
        if len(nodes)!=1 or nodes[0].decorator_list:raise ValueError('exact author extraction definition required')
        namespace={'torch':torch,'tqdm':tqdm,'apply_rotary_pos_emb':apply_rotary_pos_emb,'SYS_END_MARKER':'<|im_end|>\n'}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),namespace);result[name]=namespace[name]
    return result


@contextmanager
def hook_cleanup(model):
    before={m:(set(m._forward_hooks),set(m._forward_pre_hooks)) for m in model.modules()}
    try:yield
    finally:
        for module,(post,pre) in before.items():
            for key in set(module._forward_hooks)-post:del module._forward_hooks[key]
            for key in set(module._forward_pre_hooks)-pre:del module._forward_pre_hooks[key]


@contextmanager
def observe_shift(model,sites,intervention,audit):
    """Measure actual BF16 after-minus-before norms over ALL B*T positions.

    No full-prefix Frobenius surrogate; only scalars retained on device per call.
    First-phase hidden maxima sampled, every position contributes to norm mean.
    """
    import torch
    handles=[];pending={};counts={};sums={};records={}
    def before(name):
        def hook(module,args,output):pending[name]=(output[0] if isinstance(output,(tuple,list)) else output).detach()
        return hook
    def after(name):
        def hook(module,args,output):
            old=pending.pop(name);new=(output[0] if isinstance(output,(tuple,list)) else output).detach();delta=new.float()-old.float()
            norms=delta.norm(dim=-1);sums[name]=sums.get(name,0)+norms.sum();counts[name]=counts.get(name,0)+norms.numel()
            row=records.setdefault(name,{'prefill_calls':0,'decode_calls':0});phase='prefill' if row['prefill_calls']+row['decode_calls']==0 else 'decode'
            row[phase+'_calls']+=1
            if row[phase+'_calls']==1:row[phase+'_max_abs_hidden_delta']=delta.abs().max()
        return hook
    try:
        for name,module in sites.items():handles.append(module.register_forward_hook(before(name)))
        with intervention:
            for name,module in sites.items():handles.append(module.register_forward_hook(after(name)))
            yield
    finally:
        for h in handles:h.remove()
        for name,row in records.items():
            audit[name]={k:v.item() if torch.is_tensor(v) else v for k,v in row.items()}
            audit[name].update(norm_sum=sums[name].item(),positions=counts[name],realized_position_mean_norm=sums[name].item()/counts[name])
        audit['hooks_removed']=True


def position_weighted_norm(records):
    sites=[site for r in records for key,site in r.get('shift_audit',{}).items() if key!='hooks_removed']
    n=sum(s['positions'] for s in sites)
    if not n:raise ValueError('missing measured development positions')
    return sum(s['norm_sum'] for s in sites)/n


@contextmanager
def constant_shift(model,block,unit,norm):
    """Random projection control: one constant displacement, all positions."""
    if type(norm) is not float or not math.isfinite(norm) or norm<0:raise ValueError('finite measured norm required')
    vector=(unit*norm).to(device=model.device,dtype=next(model.parameters()).dtype)
    def hook(module,args,output):
        hidden=output[0] if isinstance(output,(tuple,list)) else output;changed=hidden+vector
        return (changed,*output[1:]) if isinstance(output,(tuple,list)) else changed
    h=model.model.layers[block].register_forward_hook(hook)
    try:yield
    finally:h.remove()
