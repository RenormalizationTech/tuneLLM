"""CPU-tested joint GCAD hook; no runner, bank loading, defaults or model I/O.

Apache-2.0 derivative of scripts/qwen3_gcad_port.py (see its LICENSE/NOTICE),
which preserves NihiI-obstat GCAD 6c879f4d12e2b01274fc96c9afdb0f4f3a26d4de.
Qwen2-only adapter: capture one shared native query, compute independent gates,
sum raw displacements in canonical trait-name order, add to attention output exactly once.
Caller must verify checkpoint, tokenizer, bank hashes and registered settings.
"""
import math
import torch
from scripts import qwen3_gcad_port as core


class JointGCAD(core.MultiLayerGatedSteerer):
    def __init__(self, model, components, *, layer_indices):
        if model.config.model_type != 'qwen2':
            raise ValueError('joint adapter is restricted to native Qwen2')
        if not isinstance(components, dict) or not components:
            raise ValueError('explicit nonempty ordered trait mapping required')
        layers = list(layer_indices)
        if not layers or len(set(layers)) != len(layers) or any(type(l) is not int or not 0 <= l < model.config.num_hidden_layers for l in layers):
            raise ValueError('unique valid explicit blocks required')
        self.components = {}
        cfg = model.config
        shapes = {'delta': (cfg.num_hidden_layers, cfg.hidden_size),
                  'key': (cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads),
                  'demand': (cfg.num_hidden_layers,)}
        for trait in sorted(components):
            c = components[trait]
            if type(trait) is not str or not trait or trait == 'sum' or set(c) != {*shapes, 'base_coef', 'scale', 'inverse'}:
                raise ValueError('exact component tensor/settings fields required')
            if any(type(c[k]) not in (int, float) or not math.isfinite(c[k]) or c[k] < 0 for k in ('base_coef', 'scale')) or type(c['inverse']) is not bool:
                raise ValueError('finite nonnegative strength/scale and explicit inverse required')
            for k, shape in shapes.items():
                t = c[k]
                if not isinstance(t, torch.Tensor) or t.dtype != torch.float32 or t.device.type != 'cpu' or tuple(t.shape) != shape or not torch.isfinite(t).all():
                    raise ValueError('finite native-shape FP32 CPU tensor required: ' + k)
            # Frozen constructor supplies author dtype conversion/GQA key expansion.
            self.components[trait] = core.MultiLayerGatedSteerer(model,
                {l: c['delta'][l] for l in layers}, {l: c['key'][l] for l in layers},
                {l: c['demand'][l].item() for l in layers}, layer_indices=layers,
                base_coef=c['base_coef'], scale=c['scale'], inverse=c['inverse'])
        first = components[sorted(components)[0]]
        super().__init__(model, {l: first['delta'][l] for l in layers},
            {l: first['key'][l] for l in layers}, {l: first['demand'][l].item() for l in layers},
            layer_indices=layers, base_coef=first['base_coef'], scale=first['scale'], inverse=first['inverse'])
        self._used = False
        self._device_vectors = {}
        self.measurements = {l: {p: {} for p in ('prefill', 'decode')} for l in layers}

    def _make_attn_hook(self, l):
        def hook(module, inputs, output):
            attn_out = output[0] if isinstance(output, (tuple, list)) else output
            phase = 'prefill' if attn_out.shape[1] > 1 or not any(v['applied'] for v in self.audit[l].values()) else 'decode'
            q = self._q_captured.pop(l, None); pe = self._pos_emb.pop(l, None)
            if q is None or pe is None:
                self.audit[l][phase]['skipped'] += 1
                raise ValueError('missing shared native query/position embeddings')
            self.audit[l][phase]['applied'] += 1
            qr = q.transpose(1, 2)
            qr, _ = core.apply_rotary_pos_emb(qr, qr, *(x.to(device=q.device, dtype=q.dtype) for x in pe))
            qr = qr.transpose(1, 2)
            total = None; stats = self.measurements[l][phase]
            for trait, c in self.components.items():
                key = (l, trait, qr.device, qr.dtype)
                if key not in self._device_vectors:
                    self._device_vectors[key] = (c._k[l].to(device=qr.device, dtype=qr.dtype), c._v[l].to(device=qr.device, dtype=qr.dtype))
                k, v = self._device_vectors[key]
                demand = (qr * k).sum(-1).mean(-1) / (self.head_dim ** .5)
                coef = 2.0 * c.base_coef * torch.sigmoid(c._sign * c.scale * (demand - c._avg_d[l]))
                shift = coef.unsqueeze(-1) * v
                total = shift if total is None else total + shift
                self._record(stats, trait, coef, shift)
            self._record(stats, 'sum', None, total)
            steered = attn_out + total
            return (steered,) + tuple(output[1:]) if isinstance(output, (tuple, list)) else steered
        return hook

    @staticmethod
    def _record(stats, name, coef, shift):
        # Device scalars only; no per-token host synchronizations or raw traces.
        fresh = name not in stats
        d = stats.setdefault(name, {'calls': 0, 'positions': 0})
        d['calls'] += 1; d['positions'] += shift.shape[0] * shift.shape[1]
        finite = torch.isfinite(shift).all()
        norm_sum = shift.detach().float().norm(dim=-1).sum()
        d['finite'] = finite if fresh else d['finite'] & finite
        d['norm_sum'] = norm_sum if fresh else d['norm_sum'] + norm_sum
        if coef is not None:
            c = coef.detach().float()
            d['coef_sum'] = c.sum() if fresh else d['coef_sum'] + c.sum()
            d['coef_min'] = c.min() if fresh else torch.minimum(d['coef_min'], c.min())
            d['coef_max'] = c.max() if fresh else torch.maximum(d['coef_max'], c.max())

    def audit_snapshot(self):
        result = {}
        for l, phases in self.measurements.items():
            result[l] = {}
            for phase, values in phases.items():
                result[l][phase] = {}
                for name, d in values.items():
                    row = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
                    row['position_weighted_norm'] = row['norm_sum'] / row['positions']
                    if 'coef_sum' in row: row['coef_mean'] = row['coef_sum'] / row['positions']
                    result[l][phase][name] = row
        return {'coverage': self.audit, 'measurements': result, 'hooks_removed': not self._handles}

    def __enter__(self):
        if self._handles: raise ValueError('one context cannot nest')
        if self._used: raise ValueError('fresh context required for each generation')
        self._used = True
        try: return super().__enter__()
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        super().__exit__(*exc)
        self._device_vectors.clear()


def controller(model, components, *, layer_indices):
    """Study factory: baseline has no hooks; singles omit inactive banks."""
    from contextlib import nullcontext
    active = {name: c for name, c in components.items() if c['base_coef'] != 0}
    return JointGCAD(model, active, layer_indices=layer_indices) if active else nullcontext()
