"""Observational hooks around the immutable Qwen3.8 GCAD port.

Derived-work attribution: qwen38_gcad_port_NOTICE and qwen3_gcad_port_LICENSE.
The frozen steerer alone computes/returns the intervention. Observer hooks
never replace its output. All means below weight token positions, not calls.
"""
from contextlib import AbstractContextManager
import torch
from scripts import qwen38_gcad_port as frozen


def signature(model):
    return {n:(tuple(m._forward_hooks),tuple(m._forward_pre_hooks)) for n,m in model.named_modules()}


class Moments:
    def __init__(self):
        self.items = {}
        self.tokens = 0
        self.calls = 0
        self.zero_native = 0
        self.finite = None

    def add(self, values, native_zero):
        n = native_zero.numel()
        self.tokens += n; self.calls += 1
        zero = native_zero.detach().sum()
        self.zero_native = self.zero_native + zero
        for name,x in values.items():
            x=x.detach().float().flatten()
            valid = ~native_zero.flatten() if name.endswith('_ratio') else torch.ones_like(x,dtype=torch.bool)
            finite=torch.isfinite(x).all()
            self.finite = finite if self.finite is None else self.finite & finite
            selected=x[valid]
            if name not in self.items:
                self.items[name]={'sum':x.new_zeros(()),'count':0,'min':x.new_tensor(float('inf')),'max':x.new_tensor(-float('inf'))}
            r=self.items[name];r['sum']+=selected.sum();r['count']+=selected.numel()
            if selected.numel():
                r['min']=torch.minimum(r['min'],selected.min());r['max']=torch.maximum(r['max'],selected.max())

    def summary(self):
        if self.finite is not None and not bool(self.finite):
            raise ValueError('nonfinite steering telemetry')
        return {'calls':self.calls,'tokens':self.tokens,'zero_native_norm_tokens':int(self.zero_native),
                'finite':True,'metrics':{k:{'sum':float(v['sum']),'count':v['count'],
                    'mean':float(v['sum'])/v['count'] if v['count'] else None,
                    'min':float(v['min']) if v['count'] else None,'max':float(v['max']) if v['count'] else None}
                    for k,v in self.items.items()}}


def token_metrics(original, steered, requested, demand, center, gate, coefficient):
    """Actual difference is computed in FP32 from stored model-dtype outputs.

    Requested is the model-dtype addition operand. Ratios exclude zero native
    output norms and report their count, rather than dividing by an epsilon.
    """
    native=original.float().norm(dim=-1)
    requested_norm=requested.float().norm(dim=-1)
    actual=(steered.float()-original.float()).norm(dim=-1)
    valid=native>0
    denom=torch.where(valid,native,torch.ones_like(native))
    return {'demand':demand,'centered_demand':demand-center,'gate':gate,'coefficient':coefficient,
            'native_norm':native,'requested_norm':requested_norm,'actual_norm':actual,
            'requested_ratio':requested_norm/denom,'actual_ratio':actual/denom},~valid


class InstrumentedSteerer(frozen.Steerer):
    """Observe the frozen shift without rewriting or mutating its arithmetic."""
    def __init__(self,*args,**kwargs):
        self.retain_last=kwargs.pop('retain_last',False)
        super().__init__(*args,**kwargs)
        self.observers=[];self.pending={};self.before_signature=None
        self.stats={i:{p:Moments() for p in ('prefill','decode')} for i in self.layers}
        self.trace={i:{p:[] for p in ('prefill','decode')} for i in self.layers}
        self.last={}

    def __enter__(self):
        if self.observers or self.handles:
            raise ValueError('instrumented steerer cannot be nested')
        self.before_signature=signature(self.model)
        try:
            for i in self.layers:
                def before(m,args,output,i=i):
                    q,_,_=self.qk_gate(i)
                    key=self.tensors[f'{i}.key'].to(q).repeat_interleave(m.num_key_value_groups,0)
                    demand=(q.transpose(1,2)*key).sum(-1).mean(-1)/self.cfg.head_dim**.5
                    center=self.tensors[f'{i}.demand'].to(q)
                    gate=torch.sigmoid(self.scale*(demand-center))
                    coefficient=2*self.base_coef*gate
                    original=output[0].detach()
                    phase='prefill' if original.shape[1]>1 else 'decode'
                    if phase=='decode' and self.stats[i]['prefill'].calls==0:
                        raise ValueError('fixed prompt must have more than one token')
                    requested=coefficient.unsqueeze(-1)*self.tensors[f'{i}.delta'].to(original)
                    self.pending[i]=(original,requested,demand,center,gate,coefficient,phase)
                self.observers.append(self.model.model.layers[i].self_attn.register_forward_hook(before))
            # The frozen hook is now between the observer pair.
            super().__enter__()
            for i in self.layers:
                def after(m,args,output,i=i):
                    if i not in self.pending:
                        raise RuntimeError('telemetry capture missing; no silent skip')
                    original,requested,demand,center,gate,coef,phase=self.pending.pop(i)
                    values,zero=token_metrics(original,output[0].detach(),requested,demand,center,gate,coef)
                    delta_norm=self.tensors[f'{i}.delta'].to(original).float().norm()
                    algebraic=coef.float()*delta_norm
                    bound=torch.ones_like(algebraic)*(2*self.base_coef)*delta_norm
                    native=values['native_norm'];denom=torch.where(native>0,native,torch.ones_like(native))
                    values.update(algebraic_requested_norm=algebraic,requested_bound_norm=bound,bound_ratio=bound/denom)
                    self.stats[i][phase].add(values,zero)
                    self.trace[i][phase].append({k:v.detach().float().flatten() for k,v in values.items()})
                    if self.retain_last:
                        self.last[i]={'original':original.detach(),'requested':requested.detach(),
                            'steered':output[0].detach(),'phase':phase}
                self.observers.append(self.model.model.layers[i].self_attn.register_forward_hook(after))
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def __exit__(self,*exc):
        super().__exit__(*exc)
        for h in self.observers:h.remove()
        self.observers.clear();self.pending.clear()
        if self.before_signature is not None and signature(self.model)!=self.before_signature:
            raise RuntimeError('original hook signatures not restored')

    def summary(self):
        out={str(i):{phase:m.summary() for phase,m in stats.items()} for i,stats in self.stats.items()}
        for i,phases in self.trace.items():
            for phase,steps in phases.items():
                if not steps:continue
                merged={k:torch.cat([s[k] for s in steps]) for k in steps[0]}
                for k,x in merged.items():
                    valid=x[merged['native_norm']>0] if k.endswith('_ratio') else x
                    out[str(i)][phase]['metrics'][k]['percentiles_5_50_95']=torch.quantile(valid,valid.new_tensor([.05,.5,.95])).tolist() if valid.numel() else None
                if phase=='decode':
                    out[str(i)][phase]['steps']=[{'step':j,**{k:float(v[0]) for k,v in s.items()}} for j,s in enumerate(steps)]
        return out


class ActivationCapture(AbstractContextManager):
    """No interventions: response-position means at all blocks / selected attention.

    Stores FP32 token sums and token-norm sums only. No full trajectories are
    retained. Positional masks exclude special tokens and are supplied by the
    frozen raw-ID replay plan.
    """
    def __init__(self,model,response_mask,attention_layers,retain_attention_tokens=False):
        self.model=model;self.mask=response_mask;self.layers=list(attention_layers)
        self.handles=[];self.values={};self.before=None
        self.retain_attention_tokens=retain_attention_tokens

    def __enter__(self):
        self.before=signature(self.model)
        def hook(key):
            def collect(m,args,output):
                x=output[0] if isinstance(output,(tuple,list)) else output
                mask=torch.as_tensor(self.mask,dtype=torch.bool,device=x.device)
                if x.shape[0]!=1 or mask.shape!=(x.shape[1],) or not mask.any():
                    raise ValueError('nonempty unpadded response mask required')
                selected=x[0,mask].detach().float()
                if key in self.values:raise ValueError('diagnostic capture called twice at one site')
                self.values[key]={'sum':selected.sum(0).cpu(),'tokens':selected.shape[0],
                                  'norm_sum':float(selected.norm(dim=-1).sum()),
                                  'finite':bool(torch.isfinite(selected).all())}
                if self.retain_attention_tokens and key.startswith('attention.'):
                    self.values[key]['response_native']=x[0,mask].detach().cpu()
            return collect
        for i,block in enumerate(self.model.model.layers):
            self.handles.append(block.register_forward_hook(hook(f'block.{i}')))
            if i in self.layers:self.handles.append(block.self_attn.register_forward_hook(hook(f'attention.{i}')))
        return self

    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        self.handles.clear()
        if signature(self.model)!=self.before:raise RuntimeError('activation hooks not restored')
        if not exc[0] and (len(self.values)!=len(self.model.model.layers)+len(self.layers) or not all(v['finite'] for v in self.values.values())):
            raise ValueError('incomplete or nonfinite activation capture')


def replay_ids(tokenizer,prompt_ids,continuation_ids,judged_output,eos_ids):
    """Append identical saved continuation IDs to either context, without retokenization."""
    if tokenizer.decode(continuation_ids,skip_special_tokens=True)!=judged_output:
        raise ValueError('saved continuation decode differs from judged output')
    suffix=list(continuation_ids)
    if suffix and suffix[-1] in eos_ids:suffix.pop()
    mask=[False]*len(prompt_ids)+[i not in tokenizer.all_special_ids for i in suffix]
    if not any(mask):raise ValueError('empty raw continuation response mask')
    return list(prompt_ids)+suffix,mask


@torch.no_grad()
def diagnostic_forward(model,ids,mask,system_length,bank,layers):
    """One no-GCAD pass: all block means, native/cropped attention, gates, NLL."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
    selected=torch.as_tensor(mask,dtype=torch.bool,device=ids.device)
    positions=torch.where(selected)[0]
    if not positions.numel() or positions.min()<1:raise ValueError('nonempty predicted continuation required')
    original_signature=signature(model)
    with ActivationCapture(model,mask,layers,system_length is not None) as activation,frozen.Capture(model,layers) as cap:
        for i in layers:
            def native(m,args,output,i=i):cap.state[i]['native_output']=output[0].detach()
            cap.handles.append(model.model.layers[i].self_attn.register_forward_hook(native))
        output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=positions-1)
        logits=output.logits
        nll=torch.nn.functional.cross_entropy(logits[0].float(),ids[0,positions])
        if not torch.isfinite(nll):raise ValueError('nonfinite continuation likelihood')
        vectors={k:v['sum']/v['tokens'] for k,v in activation.values.items()}
        vectors.update({k.replace('attention.','attention_tokens.'):v['response_native'] for k,v in activation.values.items() if 'response_native' in v})
        summaries={k:{'tokens':v['tokens'],'mean_token_norm':v['norm_sum']/v['tokens']} for k,v in activation.values.items()}
        gates={};decomposition={}
        c=model.config;t=ids.shape[1]
        for i in layers:
            q,k,gate=cap.qk_gate(i);a=model.model.layers[i].self_attn
            key=bank[f'{i}.key'].to(q).repeat_interleave(a.num_key_value_groups,0)
            demand=(q.transpose(1,2)*key).sum(-1).mean(-1)/c.head_dim**.5
            g=torch.sigmoid(1.5*(demand-bank[f'{i}.demand'].to(q)))
            gates[str(i)]={}
            for name,value in [('demand',demand[0,selected]),('gate',g[0,selected])]:
                value=value.float();gates[str(i)][name]={'mean':float(value.mean()),'std_population':float(value.std(unbiased=False)),
                    'percentiles_5_50_95':torch.quantile(value,value.new_tensor([.05,.5,.95])).tolist(),'tokens':value.numel()}
            if system_length is None:continue
            v=cap.state[i]['v_proj'].view(1,t,c.num_key_value_heads,c.head_dim).transpose(1,2)
            v=repeat_kv(v,a.num_key_value_groups)
            scores=(q@repeat_kv(k,a.num_key_value_groups).transpose(-2,-1))*a.scaling
            scores=scores.masked_fill(~torch.ones(t,t,dtype=torch.bool,device=q.device).tril(),torch.finfo(q.dtype).min)
            native_mask=cap.state[i]['mask']
            if native_mask is not None:
                scores=scores.masked_fill(~native_mask,torch.finfo(scores.dtype).min) if native_mask.dtype==torch.bool else scores+native_mask
            weights=scores.softmax(-1,dtype=torch.float32).to(q.dtype)
            def project(left,right,gated=True):
                x=(weights[...,left:right]@v[...,left:right,:]).transpose(1,2).reshape(1,t,-1)
                return a.o_proj(x*gate if gated else x)
            full=project(0,t);crop=project(0,system_length);other=project(system_length,t)
            native=cap.state[i]['native_output'].float();denom=native.norm().clamp_min(1e-30)
            decomposition[str(i)]={'gated_relative_error':float((full.float()-native).norm()/denom),
                'omitted_gate_relative_error':float((project(0,t,False).float()-native).norm()/denom),
                'split_relative_error':float((crop.float()+other.float()-full.float()).norm()/full.float().norm().clamp_min(1e-30))}
            vectors[f'crop.{i}']=crop[0,selected].float().mean(0).cpu()
        result={'vectors':vectors,'summary':summaries,'gate':gates,'nll':float(nll),
                'response_tokens':positions.numel(),'decomposition':decomposition}
    if signature(model)!=original_signature:raise RuntimeError('diagnostic hooks changed')
    if not all(torch.isfinite(v).all() for v in vectors.values()):raise ValueError('nonfinite diagnostic mean')
    return result,logits
