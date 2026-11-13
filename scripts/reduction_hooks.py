"""All-position Qwen3 runtime interventions; no parameter or cache mutation."""
from contextlib import contextmanager

METHODS = ("subtract23", "ablate_all")


def projection_remove(hidden, unit):
    """FP32 dot/subtraction, then restore activation dtype (out of place)."""
    values = hidden.float()
    direction = unit.to(device=hidden.device, dtype=values.dtype)
    return (values - (values @ direction).unsqueeze(-1) * direction).to(hidden.dtype)


def hook_sites(model, method):
    if method not in METHODS:
        raise ValueError("unregistered intervention")
    layers = model.model.layers
    if method == "subtract23":
        if len(layers) <= 23:
            raise ValueError("subtraction requires block23")
        return [("23.output", layers[23], "output")]
    return [(f"{i}.{kind}", module, kind) for i, block in enumerate(layers)
            for kind, module in (("input", block), ("attention", block.self_attn.o_proj),
                                 ("mlp", block.mlp.down_proj))]


@contextmanager
def intervene(model, method, vector=None, *, identity=False, audit=None):
    """Every position, including pads and cached decoding; fresh cache per call.

    Identity uses actual installed hooks and a zero direction, never a bypass.
    Optional audit accumulates device scalars without a sync at every hook.
    """
    import torch
    if method == "baseline":
        yield
        return
    sites = hook_sites(model, method)
    if vector is None or vector.ndim != 1 or not torch.isfinite(vector).all() or vector.norm().item() == 0:
        raise ValueError("finite nonzero raw direction required")
    if vector.numel() != model.config.hidden_size:
        raise ValueError("direction/hidden dimension mismatch")
    raw = vector.detach().float()
    unit = raw / raw.norm()
    effective = torch.zeros_like(raw) if identity else (unit if method == "ablate_all" else raw)
    handles, records, device_vectors = [], {}, {}

    def vectors(hidden):
        if hidden.device not in device_vectors:
            device_vectors[hidden.device] = (effective.to(hidden.device), unit.to(hidden.device))
        return device_vectors[hidden.device]

    def record(name, before, after):
        if audit is None:
            return
        row = records.setdefault(name, {"calls": 0, "prefill_calls": 0, "decode_calls": 0})
        row["calls"] += 1
        phase = "prefill_calls" if before.shape[-2] > 1 else "decode_calls"
        row[phase] += 1
        if row[phase] != 1:
            return
        _, local_unit = vectors(after)
        diagnostics = {"max_abs_hidden_delta": (after.float() - before.float()).abs().max(),
                       "max_abs_projection": (after.float() @ local_unit).abs().max(),
                       "nonfinite": (~torch.isfinite(after)).any()}
        for key, value in diagnostics.items():
            row[key] = value if key not in row else torch.maximum(row[key], value)

    def change(name, hidden):
        local_effective, _ = vectors(hidden)
        if method == "subtract23":
            after = (hidden.float() - 0.5 * local_effective).to(hidden.dtype)
        else:
            after = projection_remove(hidden, local_effective)
        record(name, hidden, after)
        return after

    try:
        for name, module, kind in sites:
            if kind == "input":
                def pre(module, args, kwargs, name=name):
                    if args:
                        return (change(name, args[0]), *args[1:]), kwargs
                    return args, {**kwargs, "hidden_states": change(name, kwargs["hidden_states"])}
                handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
            else:
                def post(module, args, output, name=name):
                    if isinstance(output, tuple):
                        return (change(name, output[0]), *output[1:])
                    return change(name, output)
                handles.append(module.register_forward_hook(post))
        if method == "ablate_all" and audit is not None:
            # Observe mid/post residuals, without modifying their skip branches.
            for i, block in enumerate(model.model.layers):
                def mid(module, args, i=i):
                    record(f"{i}.mid_residual", args[0], args[0])
                def end(module, args, output, i=i):
                    hidden = output[0] if isinstance(output, tuple) else output
                    record(f"{i}.post_residual", hidden, hidden)
                handles.append(block.post_attention_layernorm.register_forward_pre_hook(mid))
                handles.append(block.register_forward_hook(end))
        yield
    finally:
        for handle in handles:
            handle.remove()
        if audit is not None:
            audit.update(method=method, identity=identity, all_positions=True,
                         numerical_sampling="first prefill and first decode at each node; counters include all calls",
                         intervention_sites=[name for name, _, _ in sites],
                         nodes={name: {k: v.item() if torch.is_tensor(v) else v for k, v in row.items()}
                                for name, row in records.items()}, hooks_removed=True)
