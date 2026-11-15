"""Source-native TL math access, observational recording and aggregate rules."""
from collections import Counter
from contextlib import contextmanager
import importlib.util
import math
from pathlib import Path
import random

from scripts.author_gcad_impolite_common import canonical

ARMS={'saps0':('saps',0.),'fixed1':('fixed_projection',1.),'saps1':('saps',1.),'saps1p4':('saps',1.4)}
METRICS=('sarcasm','entailment','grammaticality','clarity')


def author_module(file):
    expected=Path(__file__).resolve().parents[1]/'outputs/persona-literature-audit-01/saps-steering/src/steering.py'
    if Path(file).resolve()!=expected:raise ValueError('pinned standalone author helper only')
    spec=importlib.util.spec_from_file_location('_author_saps_original',file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fixed_scalar(h0,unit,target,alpha):
    if target.ndim!=0:raise ValueError('source target must be scalar')
    # Exact generate.py arithmetic: FP32 CPU dot, separate item(), Python c.
    projection=(h0*unit).sum().item()
    c=(target.item()-projection)*alpha
    if type(c) is not float or not math.isfinite(c):raise ValueError('finite source Python scalar required')
    return c


def plans():
    rng=random.Random(8501);ids=list(range(450));rng.shuffle(ids);variants={i:list(ARMS) for i in ids}
    for values in variants.values():rng.shuffle(values)
    flat=[{'sentence_id':i,'arm':variants[i][round_],'id':canonical([8501,i,variants[i][round_]])[:24]} for round_ in range(4) for i in ids]
    # Repeated same 450-ID order and 20<450 preserve distinct IDs at round joins.
    batches=[flat[i:i+20] for i in range(0,1800,20)];rng.shuffle(batches)
    for batch in batches:rng.shuffle(batch)
    if len(batches)!=90 or any(len(b)!=20 or len({x['sentence_id'] for x in b})!=20 for b in batches):raise ValueError('frozen distinct-sentence plan failed')
    return batches


class CaptureProxy:
    """Run the unchanged author learner, observing calls without patching globals."""
    def __init__(self,model,author,train,prompts,before):
        self.model=model;self.author=author;self.train=train;self.prompts=prompts;self.before=before;self.pending=None;self.ledger=[]
    @property
    def cfg(self):return self.model.cfg
    def to_tokens(self,text):
        ids=self.model.to_tokens(text);self.pending=(text,ids.detach().cpu().tolist());return ids
    def run_with_cache(self,tokens,**kwargs):
        index=len(self.ledger)
        if index>=1500:raise ValueError('1500 source extraction forwards exceeded')
        row=index//2 if index<1000 else index-1000;kind='plain' if index<1000 and index%2==0 else 'style'
        text,ids=self.pending;sentence=self.train[row]
        candidates=[self.prompts['plain']] if kind=='plain' else self.prompts['style']
        matched=[i for i,instruction in enumerate(candidates) if text==self.author.build_chat_prompt(self.model.tokenizer,sentence,instruction)]
        if len(matched)!=1:raise ValueError('actual author instruction/prompt not identifiable')
        self.before(index,{'source_row':row,'kind':kind,'tokens':ids})
        result=self.model.run_with_cache(tokens,**kwargs)
        self.ledger.append({'call':index,'source_row':row,'pass':'delta' if index<1000 else 'target','kind':kind,
                            'instruction_index':None if kind=='plain' else matched[0],'rendered_prompt':text,'input_ids':ids[0]})
        return result


@contextmanager
def record_hook(source_hook,audit):
    """Delegate the actual author hook unchanged, sampling diagnostics twice."""
    name,fn=source_hook;audit.update(prefill_calls=0,decode_calls=0,finite=True,scope='all-prefix B1 source broadcast; sampled per-position norms, not total Frobenius')
    def wrapped(resid,hook):
        phase='prefill' if audit['prefill_calls']==0 else 'decode';first=audit[phase+'_calls']==0;audit[phase+'_calls']+=1
        before=resid.detach().clone() if first else None;result=fn(resid,hook)
        if first:
            delta=result.float()-before.float();norm=delta.norm(dim=-1)
            audit[phase]={'positions':norm.numel(),'mean_shift_norm_per_position':norm.mean().item(),'max_abs_hidden_delta':delta.abs().max().item(),
                          'prefix_frobenius_norm':delta.norm().item(),'finite':bool(result.isfinite().all().item())}
            audit['finite'] &= audit[phase]['finite']
        return result
    yield name,wrapped


def hook_state(model):
    return {name:(len(point.fwd_hooks),len(point.bwd_hooks)) for name,point in model.hook_dict.items()}


def generate(model,prompt,author,source_hook,cap):
    """Exact author str-input/str-output call; observe actual embedded token IDs."""
    import torch,time
    embedded=[];decoded_tokens=[];audit={};before=hook_state(model)
    def capture(module,args):embedded.append(args[0].detach().clone())
    handle=model.embed.register_forward_pre_hook(capture)
    # Capture the complete sequence at the inspected source's final decode,
    # including the final sampled token, also embedded by pinned TL2.17.
    tokenizer=model.tokenizer;local_decode='decode' in tokenizer.__dict__;previous_local=tokenizer.__dict__.get('decode');original_decode=tokenizer.decode
    def decode(tokens,*args,**kwargs):
        values=tokens.detach().cpu().tolist() if isinstance(tokens,torch.Tensor) else list(tokens)
        decoded_tokens.append(values)
        return original_decode(tokens,*args,**kwargs)
    tokenizer.decode=decode
    context=record_hook(source_hook,audit) if source_hook is not None else None
    torch.cuda.synchronize() if str(model.cfg.device).startswith('cuda') else None;began=time.perf_counter()
    try:
        if context is None:output=model.generate(prompt,max_new_tokens=cap,do_sample=False,verbose=False)
        else:
            with context as hook,model.hooks(fwd_hooks=[hook]):output=model.generate(prompt,max_new_tokens=cap,do_sample=False,verbose=False)
        torch.cuda.synchronize() if str(model.cfg.device).startswith('cuda') else None
    finally:
        handle.remove()
        if local_decode:tokenizer.decode=previous_local
        else:del tokenizer.decode
        if hook_state(model)!=before:raise ValueError('source TL hook cleanup failed')
    if not embedded or embedded[0].ndim!=2 or embedded[0].shape[0]!=1 or any(x.shape!=(1,1) for x in embedded[1:]):raise ValueError('TL embedding trace layout differs from inspected generate source')
    input_ids=embedded[0][0].cpu().tolist()
    if len(decoded_tokens)!=1 or decoded_tokens[0][:len(input_ids)]!=input_ids:raise ValueError('one final B1 source decode with unchanged prefix required')
    full_ids=decoded_tokens[0];suffix=full_ids[len(input_ids):]
    fed=torch.cat(embedded[1:],dim=1)[0].cpu().tolist() if len(embedded)>1 else []
    if fed!=suffix:raise ValueError('pinned TL2.17 complete embedding trajectory differs from final source decode')
    expected=model.to_tokens(prompt)[0].cpu().tolist()
    if input_ids!=expected or not isinstance(output,str):raise ValueError('source default BOS or return-type differs')
    decoded=model.tokenizer.decode(input_ids+suffix,skip_special_tokens=True)
    if decoded!=output:raise ValueError('actual embed trace cannot reconstruct original str return')
    eos=model.tokenizer.eos_token_id
    if len(suffix)>cap or any(x==eos for x in suffix[:-1]):raise ValueError('B1 source EOS/cap trace invalid')
    return {'rendered_prompt':prompt,'input_ids':input_ids,'generated_ids':suffix,'full_token_ids':input_ids+suffix,
            'source_full_decoded_output':output,'output':author.strip_chat_template(output),'ended_eos':bool(suffix and suffix[-1]==eos),
            'hit_token_limit':len(suffix)==cap and not (suffix and suffix[-1]==eos),'cap':cap,
            'leading_bos_count':next((i for i,x in enumerate(input_ids) if x!=model.tokenizer.bos_token_id),len(input_ids)),
            'generation_wall_seconds':time.perf_counter()-began,'hook_audit':audit,'hooks_removed':True,
            'final_sampled_token_observed_in_embedding':len(fed)==len(suffix),'token_trace_authority':'Actual final tokenizer.decode input; delegate restored on every exit',
            'source_generate_kwargs':{'max_new_tokens':cap,'do_sample':False,'verbose':False}}


def scores(value,expected):
    if not isinstance(value,dict) or set(value)!={'ratings'} or not isinstance(value['ratings'],list):raise ValueError('exact ratings-only object')
    found={}
    for row in value['ratings']:
        if not isinstance(row,dict) or set(row)!={'id',*METRICS} or not isinstance(row['id'],str) or row['id'] in found or row['id'] not in expected:raise ValueError('exact unique IDs/fields required')
        if any(type(row[m]) is not int or not 0<=row[m]<=5 for m in METRICS):raise ValueError('integer0..5 metrics required')
        found[row['id']]=row
    if set(found)!=set(expected):raise ValueError('complete expected IDs required')
    return found


def aggregate(predictions,ratings):
    from scripts.tone_common import degenerate
    results={}
    for arm,rows in predictions.items():
        values=[ratings[(arm,r['sentence_id'])] for r in rows]
        results[arm]={'n':450,'means':{m:sum(x[m] for x in values)/450 for m in METRICS},
                      'distributions':{m:{str(k):sum(x[m]==k for x in values) for k in range(6)} for m in METRICS},
                      'sarcasm_ge3':sum(x['sarcasm']>=3 for x in values),'sarcasm_ge4':sum(x['sarcasm']>=4 for x in values),
                      'caps':sum(x['hit_token_limit'] for x in rows),'eos':sum(x['ended_eos'] for x in rows),'empty':sum(not x['output'].strip() for x in rows),
                      'degenerate':sum(degenerate(x['output'])['degenerate'] for x in rows),'mean_generated_tokens':sum(len(x['generated_ids']) for x in rows)/450,
                      'cap_and_sarcasm_gt3':sum(r['hit_token_limit'] and score['sarcasm']>3 for r,score in zip(rows,values))}
    contrasts={};pairs={}
    for a,b in [('saps1','saps0'),('saps1','fixed1'),('saps1p4','saps1')]:
        pairs[a+'_minus_'+b]={m:{'mean':sum(ratings[(a,i)][m]-ratings[(b,i)][m] for i in range(450))/450,
                                  'higher':sum(ratings[(a,i)][m]>ratings[(b,i)][m] for i in range(450)),
                                  'lower':sum(ratings[(a,i)][m]<ratings[(b,i)][m] for i in range(450)),
                                  'equal':sum(ratings[(a,i)][m]==ratings[(b,i)][m] for i in range(450))} for m in METRICS}
    contrasts['adaptive1_sarcasm_gain0_5']=sum(ratings[('saps1',i)]['sarcasm']-ratings[('saps0',i)]['sarcasm'] for i in range(450))>=225
    contrasts['adaptive1_grammar_ge_fixed1']=sum(ratings[('saps1',i)]['grammaticality']-ratings[('fixed1',i)]['grammaticality'] for i in range(450))>=0
    contrasts['adaptive1p4_sarcasm_gt_adaptive1']=sum(ratings[('saps1p4',i)]['sarcasm']-ratings[('saps1',i)]['sarcasm'] for i in range(450))>0
    return {'conditions':results,'paired_sentence_differences':pairs,'descriptive_directional_checks':contrasts,'not_a_selection_gate':True}
