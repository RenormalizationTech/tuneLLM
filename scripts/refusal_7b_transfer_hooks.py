"""Minimal refusal03 port: block18 output subtraction; unchanged all-block ablation.

Not the cancelled canonical block-input method. Operations, FP32 arithmetic and
sampled diagnostics follow scripts/reduction_hooks.py; old files stay unchanged.
"""
from contextlib import contextmanager
from scripts.reduction_hooks import intervene as original_intervene, hook_sites as original_sites

METHODS = ("subtract18", "ablate_all")


def hook_sites(model, method):
    if method == "ablate_all":
        return original_sites(model, method)
    if method != "subtract18" or len(model.model.layers) <= 18:
        raise ValueError("fixed block18 output or all-block ablation required")
    return [("18.output", model.model.layers[18], "output")]


@contextmanager
def intervene(model, method, vector=None, *, identity=False, audit=None):
    if method in ("baseline", "ablate_all"):
        with original_intervene(model, method, vector, identity=identity, audit=audit):
            yield
        return
    import torch
    sites = hook_sites(model, method)
    if (vector is None or vector.ndim != 1 or vector.numel() != model.config.hidden_size
            or not torch.isfinite(vector).all() or vector.float().norm().item() == 0
            or not torch.isfinite(vector.float().norm())):
        raise ValueError("finite nonzero matching raw direction required")
    raw = vector.detach().float().clone()
    unit = raw / raw.norm()
    effective = torch.zeros_like(raw) if identity else raw
    local, records, handles = {}, {}, []

    def change(hidden):
        if hidden.device not in local:
            local[hidden.device] = effective.to(hidden.device), unit.to(hidden.device)
        direction, local_unit = local[hidden.device]
        after = (hidden.float() - .5 * direction).to(hidden.dtype)
        if audit is not None:
            row = records.setdefault("18.output", {"calls": 0, "prefill_calls": 0, "decode_calls": 0})
            row["calls"] += 1
            phase = "prefill_calls" if hidden.shape[-2] > 1 else "decode_calls"
            row[phase] += 1
            if row[phase] == 1:
                for key, value in {
                    "max_abs_hidden_delta": (after.float()-hidden.float()).abs().max(),
                    "max_abs_projection": (after.float() @ local_unit).abs().max(),
                    "nonfinite": (~torch.isfinite(after)).any(),
                }.items():
                    row[key] = value if key not in row else torch.maximum(row[key], value)
        return after

    def post(module, args, output):
        return (change(output[0]), *output[1:]) if isinstance(output, tuple) else change(output)

    try:
        handles.append(sites[0][1].register_forward_hook(post))
        yield
    finally:
        for handle in handles:
            handle.remove()
        if audit is not None:
            audit.update(method=method, identity=identity, all_positions=True,
                         numerical_sampling="first prefill and first decode per node; counters include all calls",
                         intervention_sites=[s[0] for s in sites], hooks_removed=True,
                         nodes={n: {k: v.item() if torch.is_tensor(v) else v for k, v in r.items()}
                                for n, r in records.items()})
