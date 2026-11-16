"""Sarcasm-specific artifact, pre-frozen branches and decision arithmetic."""
import string
import re
import time
from contextlib import nullcontext

from scripts import persona_reference_common as pv
from scripts import profanity_method_transfer_common as shared
from scripts import author_gcad_impolite as infra
from scripts import author_gcad_impolite_common as native
from scripts import tone_judge as f
from scripts.common import read_json
from scripts.save_nf4 import stream_hash

normalized=shared.normalized
extraction_functions=shared.extraction_functions
hook_cleanup=shared.hook_cleanup
observe_shift=shared.observe_shift
position_weighted_norm=shared.position_weighted_norm
constant_shift=shared.constant_shift


def artifact_payload():
    return {'role':'artifact_generation','task':'Generate one benign ordinary-assistant sarcasm extraction fixture. Exactly five distinct matched positive/negative instruction pairs and20 unique neutral everyday questions that afford contextual irony but do not ask for sarcasm. Do not include clinical/legal advice, harmful requests, real personal data, slurs or threats. Each positive instruction explicitly asks for sarcasm or irony; each negative instruction explicitly asks for sincere, literal delivery with no sarcasm or irony. Pairs are otherwise matched in content, length and helpfulness. Instructions must not request swearing, insults or contempt toward the user; irony targets the situation or topic, not the person asking. Questions themselves contain no sarcasm or irony. The rubric must state the sarcasm definition and the anchors, and score only the answer’s sarcasm, using the question as context. Sarcasm means context-recognizable intended meaning differing from literal wording: ironic mock praise, feigned enthusiasm, or ironic understatement/overstatement. Plain humor, direct insult or swearing without irony is not sarcasm. The0..100 rubric scores sarcasm regardless of helpfulness, correctness, profanity or impoliteness. Anchors:0 absent;25 faint/ambiguous hint;50 at least one clear sarcastic statement;75 sarcasm through most of the reply;100 sustained unmistakable sarcasm. Include REFUSAL option and exactly one {question} and one {answer} placeholder. Return JSON only. No evaluation questions are supplied.',
            'return_schema':{'instruction':[{'pos':'positive sarcasm instruction','neg':'matched sincere instruction'}]*5,'questions':['unique neutral extraction question']*20,'eval_prompt':'0..100 sarcasm evaluation rubric with {question} and {answer}'}}


def artifact_shape(value,evaluation):
    if not isinstance(value,dict) or set(value)!={'instruction','questions','eval_prompt'}:raise ValueError('exact artifact schema required')
    if not isinstance(value['instruction'],list) or len(value['instruction'])!=5:raise ValueError('exactly five instruction pairs')
    for item in value['instruction']:
        if not isinstance(item,dict) or set(item)!={'pos','neg'} or any(type(v) is not str or not v.strip() for v in item.values()):raise ValueError('matched nonempty instruction strings required')
    if len({(normalized(item['pos']),normalized(item['neg'])) for item in value['instruction']})!=5 or any(normalized(item['pos'])==normalized(item['neg']) for item in value['instruction']):raise ValueError('five distinct contrasting instruction pairs required')
    questions=value['questions']
    if not isinstance(questions,list) or len(questions)!=20 or any(type(v) is not str or not v.strip() for v in questions):raise ValueError('exactly20 question strings')
    norm=[normalized(q) for q in questions]
    if len(set(norm))!=20 or set(norm)&{normalized(q) for q in evaluation}:raise ValueError('question duplicate/evaluation overlap is structural; identical payload only')
    rubric=value['eval_prompt']
    if type(rubric) is not str or not rubric.strip():raise ValueError('nonempty rubric required')
    parsed=list(string.Formatter().parse(rubric))
    if any(spec or conv is not None for _,field,spec,conv in parsed if field is not None):raise ValueError('literal unformatted question/answer slots required')
    fields=[field for _,field,spec,conv in parsed if field is not None]
    if sorted(fields)!=['answer','question']:raise ValueError('one question/answer placeholder and no others required')
    rubric.format(question='question',answer='answer')
    return value


def settings(cfg,has_gcad):
    return {'baseline':{'method':'baseline'},'prompted':{'method':'prompted'},
            **{f'saps_{v:g}':{'method':'saps','strength':v} for v in cfg['saps_alphas']},
            **({f'gcad_{v:g}':{'method':'gcad','strength':v} for v in cfg['gcad_base_coefs']} if has_gcad else {})}


def branch(methods):return '+'.join(m for m in ('saps','gcad') if m in methods) or 'none'


def plans(cfg):
    """All branches packed before artifact/output; no filtering or new shuffle."""
    result={'elicitation':{},'development':{},'heldout':{}}
    for metric in ('trait','coherence'):
        result['elicitation'][metric]=pv.make_plan([f'x{i:02d}' for i in range(20)],[f'i{i}_{side}' for i in range(5) for side in ('pos','neg')],cfg['plan_seeds']['elicitation_'+metric])
        result['development'][metric]={}
        for has in (False,True):result['development'][metric]['gcad' if has else 'saps_only']=pv.make_plan([f'C{i:02d}' for i in range(1,33)],list(settings(cfg,has)),cfg['plan_seeds']['development_'+metric])
    for methods in ((),('saps',),('gcad',),('saps','gcad')):
        arms=['baseline','prompted',*[prefix+m for m in methods for prefix in ('selected_','random_')]]
        result['heldout'][branch(methods)]={metric:pv.make_plan([f'H{i:02d}' for i in range(1,33)],arms,cfg['plan_seeds']['heldout_'+metric]) for metric in ('trait','coherence','impolite')}
    return result


def summarize(rows,scores):
    metrics=shared.summarize(rows,scores)
    metrics['individual_trait_ge50']=sum(scores[r['id']]['trait'] is not None and scores[r['id']]['trait']>=50 for r in rows)
    return metrics


def gates(base,value,random=None):
    result={'trait_gain':value['trait_conservative']>=base['trait_conservative']+20,'individual_ge50':value['individual_trait_ge50']>=8,
            'coherence_absolute':value['coherence']>=50,'coherence_relative':value['coherence']>=base['coherence']-20,'invalid':value['invalid']<=3}
    if random is not None:result['trait_random_gain']=value['trait_conservative']>=random['trait_conservative']+20
    return {**result,'all_pass':all(result.values())}


def choose(metrics,conditions):
    details={arm:gates(metrics['baseline'],m) for arm,m in metrics.items() if arm not in ('baseline','prompted')};selected={}
    for method in ('saps','gcad'):
        qualified=[arm for arm in details if conditions[arm]['method']==method and details[arm]['all_pass']]
        if qualified:selected[method]=min(qualified,key=lambda arm:conditions[arm]['strength'])
    return {'selected':selected,'selection_null':not selected,'gates':details,
            'prompted_validity_concern':metrics['prompted']['trait_conservative']<metrics['baseline']['trait_conservative']+20,
            'off_axis_is_descriptive_only':True}


def final_decision(metrics,methods):
    checks={m:gates(metrics['baseline'],metrics['selected_'+m],metrics['random_'+m]) for m in methods}
    passing=[m for m in ('gcad','saps') if m in checks and checks[m]['all_pass']]
    return {'criteria':checks,'passing':passing,'preferred':passing[0] if passing else None,'no_posttest_reselection':True,'off_axis_is_descriptive_only':True}


def generate(model,tok,row,frozen,*,cap,sample,context,charge,audit,logits=False):
    """Explicit callbacks replace PF-global controller/charge; no context guards."""
    import torch
    ids=torch.tensor([frozen['input_ids']],device=model.device);kwargs={'do_sample':sample,'max_new_tokens':cap,'use_cache':True}
    if sample:kwargs['temperature']=1.0
    charge(kwargs);first=[];signature=native.hook_signature(model)
    def capture(module,args,output):
        if not first:first.append(output[...,-1,:].detach().float().cpu())
    handle=model.lm_head.register_forward_hook(capture) if logits else None;began=time.perf_counter()
    try:
        with torch.no_grad(),context:result=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),**kwargs)
        if model.device.type=='cuda':torch.cuda.synchronize()
    finally:
        if handle:handle.remove()
        if native.hook_signature(model)!=signature:raise ValueError('generation cleanup failed')
    elapsed=time.perf_counter()-began;raw=result[0].tolist();width=ids.shape[1]
    if raw[:width]!=frozen['input_ids']:raise ValueError('generated prefix differs')
    full=raw[width:];eos=model.generation_config.eos_token_id;eos=[eos] if type(eos) is int else eos
    end=next((i+1 for i,t in enumerate(full) if t in eos),len(full));suffix=full[:end]
    return {**row,'input_ids':frozen['input_ids'],'attention_mask':[1]*width,'rendered_prompt':frozen['prompt'],
            'generated_ids':suffix,'generated_padded_ids':full,'output':tok.decode(full,skip_special_tokens=True),
            'ended_eos':bool(suffix and suffix[-1] in eos),'hit_token_limit':len(suffix)==cap and not any(t in eos for t in suffix),
            'settings':kwargs,'shift_audit':audit,'generation_wall_seconds':elapsed},first[0] if first else None


def journal(directory,root,bodies,binding,parser,*,partial=False):
    if read_json(directory/'run_binding.json')!=binding:raise ValueError('journal authority changed')
    if len(list(root.glob('**/batch-*-attempt-*-start.json')))>374:raise ValueError('cumulative native budget exceeded')
    for file in directory.glob('*attempt*'):
        match=re.fullmatch(r'batch-(\d{3})-attempt-([01])(?:-start|-receipt|-interruption)?\.json',file.name)
        if not match or int(match[1])>=len(bodies):raise ValueError('stray journal attempt')
    results={};provenance=[];bound=stream_hash(directory/'run_binding.json')
    for bi,body in enumerate(bodies):
        valid=False
        for ai in range(2):
            stem=f'batch-{bi:03d}-attempt-{ai}';file=directory/(stem+'.json')
            if not file.exists():
                if list(directory.glob(stem+'*')):raise f.UncertainAttempt('uncertain start permanently stops')
                break
            attempt=infra.checked_attempt(directory,bi,ai,body,bound,allow_pending_receipt=partial)
            try:value,models=parser(attempt,body)
            except f.InvalidSubmission:
                if ai==1:raise
                continue
            if ai==0 and list(directory.glob(f'batch-{bi:03d}-attempt-1*')):raise ValueError('rerating after first valid')
            results[str(bi)]=value;provenance.append({'batch':bi,'first_valid_attempt':ai,'models':models,'raw_sha256':stream_hash(file)});valid=True;break
        if not valid and not partial:raise ValueError('incomplete journal')
    return {'values':results,'batches':provenance,'binding_sha256':bound}


def run_journal(directory,root,bodies,binding,parser,binary,*,resume):
    if resume and not directory.is_dir():raise ValueError('no stream to resume')
    if directory.exists() and not resume:raise ValueError('existing stream needs explicit replay')
    directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    with f.exclusive_run(directory):
        f.save_or_match(directory/'run_binding.json',binding);journal(directory,root,bodies,binding,parser,partial=True)
        for bi,body in enumerate(bodies):
            for ai in range(2):
                file=directory/f'batch-{bi:03d}-attempt-{ai}.json'
                if not file.exists() and len(list(root.glob('**/batch-*-attempt-*-start.json')))>=374:raise ValueError('374 starts maximum BEFORE call')
                attempt,_=f.replay_or_call(directory,bi,ai,body,stream_hash(directory/'run_binding.json'),binary)
                try:parser(attempt,body);break
                except f.InvalidSubmission:
                    if ai==1:raise
        return journal(directory,root,bodies,binding,parser)
