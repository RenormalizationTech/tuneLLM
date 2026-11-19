"""Full-attention-only GCAD for the pinned Qwen3.8 text runtime.

The demand/shift arithmetic follows qwen3_gcad_port; see
qwen38_gcad_port_NOTICE and qwen3_gcad_port_LICENSE (Apache-2.0).
Linear attention is never hooked. Native attention/cache implementations run
unchanged; extraction reconstructs only bounded, teacher-forced attention.
"""
from contextlib import AbstractContextManager
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv


def validate_geometry(model, layers):
    c = model.config
    if c.model_type != 'qwen3_5_text' or model.training:
        raise ValueError('Qwen3_5 text model in eval mode required')
    if not layers or len(set(layers)) != len(layers):
        raise ValueError('nonempty unique full-attention indices required')
    if c.num_attention_heads % c.num_key_value_heads:
        raise ValueError('integer GQA grouping required')
    for i in layers:
        if i < 0 or i >= c.num_hidden_layers or c.layer_types[i] != 'full_attention':
            raise ValueError('no direct linear-attention interventions')
        a = model.model.layers[i].self_attn
        h, k, d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        for name, width in [('q_proj', 2*h*d), ('k_proj', k*d), ('v_proj', k*d)]:
            p = getattr(a, name)
            if p.in_features != c.hidden_size or p.out_features != width or p.bias is not None:
                raise ValueError('pinned native projection geometry required')
        if (a.o_proj.in_features, a.o_proj.out_features, a.o_proj.bias) != (h*d, c.hidden_size, None):
            raise ValueError('bias-free native output projection required')
        if any(tuple(getattr(a, n).weight.shape) != (d,) for n in ('q_norm', 'k_norm')):
            raise ValueError('native per-head Q/K norms required')
        if a.attention_dropout != 0 or a.scaling != d**-.5:
            raise ValueError('zero dropout and native head scaling required')
    return c


class Capture(AbstractContextManager):
    """Capture exact native norms and RoPE, including cached position offsets."""
    def __init__(self, model, layers):
        self.model, self.layers = model, list(layers)
        self.cfg = validate_geometry(model, self.layers)
        self.state, self.handles = {}, []

    def __enter__(self):
        if self.handles:
            raise ValueError('capture cannot be nested')
        for i in self.layers:
            a = self.model.model.layers[i].self_attn
            def pre(m, args, kw, i=i):
                pe = kw.get('position_embeddings', args[1] if len(args) > 1 else None)
                if pe is None:
                    raise ValueError('exact native position embeddings missing')
                self.state[i] = {'pe': pe, 'mask': kw.get('attention_mask', args[2] if len(args) > 2 else None)}
            self.handles.append(a.register_forward_pre_hook(pre, with_kwargs=True))
            for name in ('q_norm', 'k_norm', 'v_proj', 'q_proj'):
                def capture(m, args, output, i=i, name=name):
                    self.state[i][name] = output.detach()
                self.handles.append(getattr(a, name).register_forward_hook(capture))
        return self

    def qk_gate(self, i):
        s, c = self.state[i], self.cfg
        q, k = apply_rotary_pos_emb(s['q_norm'].transpose(1, 2), s['k_norm'].transpose(1, 2), *s['pe'])
        # Gate is interleaved WITHIN each head; splitting the global half is wrong.
        gate = s['q_proj'].reshape(*s['q_proj'].shape[:-1], c.num_attention_heads, 2*c.head_dim).chunk(2, -1)[1]
        return q, k, gate.flatten(-2).sigmoid()

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.state.clear()


@torch.no_grad()
def extract(model, input_ids, system_length, prompt_length, response_mask, layers):
    """One raw trajectory: sample-mean cropped output, system K and response Q.

    No retokenization, truncation, padding, cache, or alternate eager model path.
    The native sigmoid gate multiplies each crop before the bias-free o_proj.
    """
    c = validate_geometry(model, layers)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError('unpadded batch one required')
    t = input_ids.shape[1]
    mask = torch.as_tensor(response_mask, dtype=torch.bool, device=input_ids.device)
    if mask.shape != (t,) or not 0 < system_length <= prompt_length < t or mask[:prompt_length].any() or not mask.any():
        raise ValueError('nonempty raw response mask and exact boundaries required')
    result = {}
    with Capture(model, layers) as cap:
        for i in layers:
            def native(m, args, output, i=i):
                cap.state[i]['native_output'] = output[0].detach()
            cap.handles.append(model.model.layers[i].self_attn.register_forward_hook(native))
        model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False, logits_to_keep=1)
        for i in layers:
            a = model.model.layers[i].self_attn
            q, k, gate = cap.qk_gate(i)
            v = cap.state[i]['v_proj'].view(1, t, c.num_key_value_heads, c.head_dim).transpose(1, 2)
            scores = (q @ repeat_kv(k, a.num_key_value_groups).transpose(-2, -1)) * a.scaling
            # Native SDPA receives None for unpadded causal input in some versions.
            allowed = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            native_mask = cap.state[i]['mask']
            if native_mask is not None:
                if native_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~native_mask, torch.finfo(scores.dtype).min)
                else:
                    scores = scores + native_mask
            w = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
            crop = w[..., :system_length] @ repeat_kv(v, a.num_key_value_groups)[..., :system_length, :]
            crop = crop.transpose(1, 2).reshape(1, t, -1) * gate
            projected = a.o_proj(crop)
            full = (w @ repeat_kv(v, a.num_key_value_groups)).transpose(1, 2).reshape(1,t,-1)
            reconstructed = a.o_proj(full * gate)
            omitted = a.o_proj(full)
            other = w[..., system_length:] @ repeat_kv(v, a.num_key_value_groups)[..., system_length:, :]
            other = a.o_proj(other.transpose(1,2).reshape(1,t,-1) * gate)
            norm = cap.state[i]['native_output'].float().norm().clamp_min(1e-30)
            gated_error = (reconstructed.float()-cap.state[i]['native_output'].float()).norm()/norm
            omitted_error = (omitted.float()-cap.state[i]['native_output'].float()).norm()/norm
            split_error = (projected.float()+other.float()-reconstructed.float()).norm()/reconstructed.float().norm().clamp_min(1e-30)
            result[i] = {'delta': projected[0, mask].float().mean(0).cpu(),
                         'key_sum': k[0, :, :system_length].float().sum(1).cpu(),
                         'key_count': system_length,
                         'query_sum': q[0, :, mask].float().sum(1).cpu(),
                         'query_count': int(mask.sum()),
                         'diagnostics': {'gated_relative_error': float(gated_error),
                             'omitted_gate_relative_error': float(omitted_error),
                             'split_relative_error': float(split_error),
                             'attention_output_norm': float(norm)}}
    return result


def bank(positive, negative, layers):
    """Raw paired delta; token-weighted positive K/Q, author demand centering."""
    if not positive or len(positive) != len(negative):
        raise ValueError('fresh matched extraction pairs required')
    out = {}
    for i in layers:
        delta = torch.stack([p[i]['delta'] - n[i]['delta'] for p, n in zip(positive, negative)]).mean(0)
        key = sum(p[i]['key_sum'] for p in positive) / sum(p[i]['key_count'] for p in positive)
        query = sum(p[i]['query_sum'] for p in positive) / sum(p[i]['query_count'] for p in positive)
        groups = query.shape[0] // key.shape[0]
        demand = (query * key.repeat_interleave(groups, 0)).sum(-1).mean() / key.shape[-1]**.5
        out.update({f'{i}.delta': delta, f'{i}.key': key, f'{i}.demand': demand})
    if not all(torch.isfinite(v).all() for v in out.values()):
        raise ValueError('nonfinite fresh bank')
    return out


class Steerer(Capture):
    def __init__(self, model, tensors, layers, base_coef, scale=1.5):
        super().__init__(model, layers)
        self.tensors, self.base_coef, self.scale = tensors, float(base_coef), float(scale)
        self.audit = {i: {'prefill': 0, 'decode': 0, 'errors': 0} for i in layers}
        for i in layers:
            for name, shape in [('delta', (self.cfg.hidden_size,)), ('key', (self.cfg.num_key_value_heads, self.cfg.head_dim)), ('demand', ())]:
                value = tensors[f'{i}.{name}']
                if value.shape != shape or not torch.isfinite(value).all():
                    raise ValueError('invalid fresh bank shape/value')

    def __enter__(self):
        super().__enter__()
        for i in self.layers:
            def shift(m, args, output, i=i):
                if i not in self.state or any(k not in self.state[i] for k in ('pe','q_norm','k_norm','q_proj')):
                    self.audit[i]['errors'] += 1
                    raise RuntimeError(f'GCAD layer {i}: missing native Q/K/RoPE capture; no silent skip')
                q, _, _ = self.qk_gate(i)
                key = self.tensors[f'{i}.key'].to(q).repeat_interleave(m.num_key_value_groups, 0)
                demand = (q.transpose(1, 2)*key).sum(-1).mean(-1) / self.cfg.head_dim**.5
                coef = 2*self.base_coef*torch.sigmoid(self.scale*(demand-self.tensors[f'{i}.demand'].to(q)))
                original = output[0]
                self.audit[i]['prefill' if original.shape[1] > 1 else 'decode'] += 1
                self.state.pop(i)
                return (original + coef.unsqueeze(-1)*self.tensors[f'{i}.delta'].to(original), *output[1:])
            self.handles.append(self.model.model.layers[i].self_attn.register_forward_hook(shift))
        return self
