"""Qwen3 GCAD port; Apache-2.0 derived work, author arithmetic retained.

NOTICE: Derived from NihiI-obstat/Gated-Cropped-Attention-Delta commit
6c879f4d12e2b01274fc96c9afdb0f4f3a26d4de, core/generate_vec.py,
scripts/generate_vec_prompt_attn_k.py and core/gated_steerer.py.
Original Apache-2.0 license: outputs/persona-author-replication-audit-01/gcad/LICENSE.
Changes: explicit config.head_dim, native per-head Q/K norms before RoPE,
Qwen3 rotary/repeat imports. Optional norms preserve the Qwen2 regression path.
No parameter edits, new gate, normalization of delta/key, or extraction cutoff.
"""
import torch
from tqdm import tqdm
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

SYS_END_MARKER = '<|im_end|>\n'


def validate_geometry(model):
    cfg=model.config;kind=cfg.model_type
    if kind not in ('qwen2','qwen3'):raise ValueError('only reviewed Qwen2 regression or Qwen3 geometry')
    hd=getattr(cfg,'head_dim',cfg.hidden_size//cfg.num_attention_heads)
    if cfg.rope_scaling is not None or getattr(cfg,'use_sliding_window',False):raise ValueError('unscaled global RoPE required')
    for block in model.model.layers:
        a=block.self_attn
        if (a.q_proj.out_features!=cfg.num_attention_heads*hd or a.k_proj.out_features!=cfg.num_key_value_heads*hd
            or a.v_proj.out_features!=cfg.num_key_value_heads*hd or a.o_proj.in_features!=cfg.num_attention_heads*hd
            or a.o_proj.out_features!=cfg.hidden_size or a.o_proj.bias is not None):raise ValueError('native projection geometry differs')
        for name in ('q_norm','k_norm'):
            norm=getattr(a,name,None)
            if kind=='qwen3' and (norm is None or tuple(norm.weight.shape)!=(hd,)):raise ValueError('Qwen3 per-head norms required')
            if kind=='qwen2' and norm is not None:raise ValueError('Qwen2 regression must have no Q/K norm')
    return hd


def _norm(attention, name, values):
    module = getattr(attention, name, None)
    return module(values) if module is not None else values


def get_prompt_attn_delta_p_and_r(model, tokenizer, prompts, responses, layer_list=None):
    """Extract the attention output contributed ONLY by system prompt tokens.

    For each response token i at layer l, computes:
        sys_contribution_i = o_proj( concat_heads( sum_{j in sys_tokens} alpha_ij * V_j ) )
    where alpha_ij are post-softmax attention weights (already normalized over
    the full sequence) and V_j are value states.

    This isolates the direct influence of the system prompt on attention output,
    excluding the effect of divergent generated content.

    Requires output_attentions=True (falls back to eager attention, slower but exact).
    Uses two hooks per layer:
      1. v_proj hook — captures raw value states
      2. self_attn hook — grabs attn_weights from output, combines with captured V,
         computes sys_contribution, applies o_proj, discards intermediates
    """
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv

    validate_geometry(model)
    max_layer = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(0, max_layer))

    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // num_heads)

    prompt_avg = [[] for _ in range(max_layer + 1)]
    response_avg = [[] for _ in range(max_layer + 1)]
    prompt_last = [[] for _ in range(max_layer + 1)]
    texts = [p + a for p, a in zip(prompts, responses)]

    for text, prompt in tqdm(zip(texts, prompts), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

        # Find system prompt token boundary:
        # prompt = "<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n..."
        sys_end_marker = "<|im_end|>\n"
        sys_end_pos = prompt.index(sys_end_marker) + len(sys_end_marker)
        sys_text = prompt[:sys_end_pos]
        sys_token_len = len(tokenizer.encode(sys_text, add_special_tokens=False))

        # Per-sample hook state (captured_v is consumed by attn_hook via pop)
        captured_v = {}
        sys_contributions = {}
        handles = []

        def make_v_hook(layer_idx):
            def hook(module, input, output):
                captured_v[layer_idx] = output.detach()
            return hook

        def make_attn_hook(layer_idx, sys_len):
            def hook(module, input, output):
                attn_weights = output[1]  # (batch, num_heads, seq_len, seq_len)
                v_raw = captured_v.pop(layer_idx)  # (batch, seq_len, num_kv_heads * head_dim)

                bsz, seq_len_v = v_raw.shape[0], v_raw.shape[1]
                # Reshape to (batch, num_kv_heads, seq_len, head_dim) and GQA-repeat
                v = v_raw.view(bsz, seq_len_v, num_kv_heads, head_dim).transpose(1, 2)
                v = repeat_kv(v, num_kv_groups)  # (batch, num_heads, seq_len, head_dim)

                # Attention output from system prompt tokens only
                sys_w = attn_weights[:, :, :, :sys_len]   # (batch, num_heads, seq_len, sys_len)
                sys_v = v[:, :, :sys_len, :]               # (batch, num_heads, sys_len, head_dim)
                sys_out = sys_w @ sys_v                     # (batch, num_heads, seq_len, head_dim)

                # Reshape back and apply o_proj (linear, so decomposition is exact)
                sys_out = sys_out.transpose(1, 2).reshape(bsz, -1, num_heads * head_dim)
                sys_out = module.o_proj(sys_out)

                sys_contributions[layer_idx] = sys_out.detach().cpu()
            return hook

        for layer in layer_list:
            handles.append(
                model.model.layers[layer].self_attn.v_proj.register_forward_hook(make_v_hook(layer)))
            handles.append(
                model.model.layers[layer].self_attn.register_forward_hook(make_attn_hook(layer, sys_token_len)))

        with torch.no_grad():
            model(**inputs, output_attentions=True, output_hidden_states=False)

        for handle in handles:
            handle.remove()

        for layer in layer_list:
            sys_out = sys_contributions[layer]
            # Average over response token positions
            response_avg[layer].append(sys_out[:, prompt_len:, :].mean(dim=1))

    for layer in layer_list:
        if len(response_avg[layer]) > 0:
            response_avg[layer] = torch.cat(response_avg[layer], dim=0)
    return prompt_avg, prompt_last, response_avg


def compute_k_pos_and_avg_demand(model, tokenizer, prompts, responses, layer_list=None):
    """Single pass: accumulate post-RoPE K over sys positions and post-RoPE Q over response positions.

    Returns:
        k_pos: (n_layers, n_kv_heads, head_dim) — mean post-RoPE K over sys tokens, averaged over samples.
        avg_demand: (n_layers,) — mean per-token demand over response tokens (pos prompts).
    """
    validate_geometry(model)
    n_layers = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(n_layers))

    nh = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, "head_dim", model.config.hidden_size // nh)
    ng = nh // nkv  # query heads per kv head

    texts = [p + a for p, a in zip(prompts, responses)]

    captured_q = {l: None for l in layer_list}
    captured_k = {l: None for l in layer_list}

    def make_q_hook(l):
        def hook(m, inp, out):
            captured_q[l] = out.detach()
        return hook

    def make_k_hook(l):
        def hook(m, inp, out):
            captured_k[l] = out.detach()
        return hook

    handles = []
    for l in layer_list:
        attn = model.model.layers[l].self_attn
        handles.append(attn.q_proj.register_forward_hook(make_q_hook(l)))
        handles.append(attn.k_proj.register_forward_hook(make_k_hook(l)))

    k_sums = {l: None for l in layer_list}
    k_counts = {l: 0 for l in layer_list}
    q_sums = {l: None for l in layer_list}
    q_counts = {l: 0 for l in layer_list}

    try:
        for text, prompt in tqdm(zip(texts, prompts), total=len(texts)):
            if SYS_END_MARKER not in prompt:
                raise ValueError("Could not find system prompt end marker in prompt text")
            sys_end_pos = prompt.index(SYS_END_MARKER) + len(SYS_END_MARKER)
            sys_text = prompt[:sys_end_pos]
            sys_token_len = len(tokenizer.encode(sys_text, add_special_tokens=False))
            prompt_token_len = len(tokenizer.encode(prompt, add_special_tokens=False))

            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            T = inputs["input_ids"].shape[1]
            position_ids = torch.arange(T, device=model.device).unsqueeze(0)

            for l in layer_list:
                captured_q[l] = None
                captured_k[l] = None

            with torch.no_grad():
                model(**inputs, output_hidden_states=False, output_attentions=False, use_cache=False)

            for l in layer_list:
                q_raw = captured_q[l]  # (1, T, nh*hd) on layer device
                k_raw = captured_k[l]  # (1, T, nkv*hd)
                dev = q_raw.device
                # (1, H, T, D)
                q = _norm(model.model.layers[l].self_attn, "q_norm", q_raw.view(1, T, nh, hd)).transpose(1, 2)
                k = _norm(model.model.layers[l].self_attn, "k_norm", k_raw.view(1, T, nkv, hd)).transpose(1, 2)
                cos, sin = model.model.rotary_emb(k, position_ids.to(dev))
                cos = cos.to(dev)
                sin = sin.to(dev)
                q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
                q_rope = q_rope.float().cpu()
                k_rope = k_rope.float().cpu()

                # K_pos: sum over sys positions
                sys_k_sum = k_rope[0, :, :sys_token_len, :].sum(dim=1)  # (nkv, hd)
                if k_sums[l] is None:
                    k_sums[l] = sys_k_sum
                else:
                    k_sums[l] += sys_k_sum
                k_counts[l] += sys_token_len

                # Q_resp: sum over response positions (prompt_token_len:)
                if prompt_token_len < T:
                    q_resp_sum = q_rope[0, :, prompt_token_len:, :].sum(dim=1)  # (nh, hd)
                    n_resp = T - prompt_token_len
                    if q_sums[l] is None:
                        q_sums[l] = q_resp_sum
                    else:
                        q_sums[l] += q_resp_sum
                    q_counts[l] += n_resp
    finally:
        for h in handles:
            h.remove()

    k_pos = torch.stack(
        [k_sums[l] / max(k_counts[l], 1) for l in range(n_layers)], dim=0
    )  # (n_layers, nkv, hd)

    # avg_demand per layer: (1/nh) sum_h (Q_mean[h] . K_pos[h//ng]) / sqrt(hd)
    avg_demand_list = []
    for l in range(n_layers):
        if q_sums[l] is None or q_counts[l] == 0:
            avg_demand_list.append(torch.tensor(0.0))
            continue
        q_mean = q_sums[l] / q_counts[l]  # (nh, hd)
        k_per_q = k_pos[l].repeat_interleave(ng, dim=0)  # (nh, hd)
        d = (q_mean * k_per_q).sum(dim=-1).mean() / (hd ** 0.5)
        avg_demand_list.append(d)
    avg_demand = torch.stack(avg_demand_list, dim=0)  # (n_layers,)

    return k_pos, avg_demand


class MultiLayerGatedSteerer:
    """Gated steering across multiple layers. Each layer has its own V, K_pos, avg_demand."""

    def __init__(
        self,
        model: torch.nn.Module,
        v_vectors: dict,        # {layer_idx: (hidden,)}
        k_pos_vectors: dict,    # {layer_idx: (nkv, hd)}
        avg_demands: dict,      # {layer_idx: float}
        *,
        layer_indices: list,
        base_coef: float,
        scale: float,
        inverse: bool = False,
        debug: bool = False,
    ):
        validate_geometry(model)
        self.model = model
        self.layer_indices = sorted(layer_indices)
        self.base_coef = float(base_coef)
        self.scale = float(scale)
        self.inverse = bool(inverse)
        self.debug = debug

        cfg = model.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // self.num_heads)
        self._sign = -1.0 if inverse else 1.0

        # Per-layer tensors (kept on CPU, moved lazily in hooks)
        self._v = {}
        self._k = {}
        self._avg_d = {}
        for l in self.layer_indices:
            target_dtype = model.model.layers[l].self_attn.q_proj.weight.dtype
            v = v_vectors[l]
            k = k_pos_vectors[l]
            if v.ndim != 1 or v.numel() != cfg.hidden_size:
                raise ValueError(f"v_vectors[{l}] must be 1-D length {cfg.hidden_size}, got {v.shape}")
            if k.shape != (self.num_kv_heads, self.head_dim):
                raise ValueError(f"k_pos_vectors[{l}] must be ({self.num_kv_heads}, {self.head_dim}), got {k.shape}")
            self._v[l] = v.to(dtype=target_dtype)
            self._k[l] = k.repeat_interleave(self.num_kv_groups, dim=0).to(dtype=target_dtype)
            self._avg_d[l] = float(avg_demands[l])

        # Per-layer captured state
        self._q_captured = {}
        self._pos_emb = {}
        self._handles = []
        self.audit = {l: {p: {"applied": 0, "skipped": 0} for p in ("prefill", "decode")} for l in self.layer_indices}

    # ── Hook factories (closure over layer index) ──
    def _make_pre_hook(self, l):
        def hook(module, args, kwargs):
            pe = kwargs.get("position_embeddings", None)
            if pe is None and len(args) >= 2:
                pe = args[1]
            self._pos_emb[l] = pe
        return hook

    def _make_q_hook(self, l):
        def hook(module, inputs, output):
            bsz, seq_len, _ = output.shape
            self._q_captured[l] = _norm(self.model.model.layers[l].self_attn, "q_norm", output.view(bsz, seq_len, self.num_heads, self.head_dim))
        return hook

    def _make_attn_hook(self, l):
        def hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                attn_out = output[0]; rest = output[1:]
            else:
                attn_out = output; rest = None

            q = self._q_captured.get(l)
            pe = self._pos_emb.get(l)
            phase = "prefill" if attn_out.shape[1] > 1 or sum(sum(x.values()) for x in self.audit[l].values()) == 0 else "decode"
            self.audit[l][phase]["skipped" if q is None or pe is None else "applied"] += 1
            if q is None or pe is None:
                return output

            cos, sin = pe
            q_bhtd = q.transpose(1, 2)
            cos_d = cos.to(device=q.device, dtype=q.dtype)
            sin_d = sin.to(device=q.device, dtype=q.dtype)
            q_rope, _ = apply_rotary_pos_emb(q_bhtd, q_bhtd, cos_d, sin_d)
            q_rope = q_rope.transpose(1, 2)

            k_ph = self._k[l].to(device=q_rope.device, dtype=q_rope.dtype)
            v_vec = self._v[l].to(device=q_rope.device, dtype=q_rope.dtype)

            dot_per_head = (q_rope * k_ph).sum(dim=-1)
            demand = dot_per_head.mean(dim=-1) / (self.head_dim ** 0.5)
            centered = demand - self._avg_d[l]
            coef = 2.0 * self.base_coef * torch.sigmoid(self._sign * self.scale * centered)

            steered = attn_out + coef.unsqueeze(-1) * v_vec

            if self.debug:
                with torch.no_grad():
                    print(f"[gated L{l}] demand: mean={demand.mean():.3f} "
                          f"std={demand.std():.3f} (avg_d={self._avg_d[l]:.3f}) | "
                          f"coef: mean={coef.mean():.3f} range=[{coef.min():.3f}, {coef.max():.3f}]")

            self._q_captured.pop(l, None)
            self._pos_emb.pop(l, None)

            if rest is not None:
                return (steered,) + tuple(rest)
            return steered
        return hook

    # ── Context manager ──
    def __enter__(self):
        if self._handles: raise ValueError("one steerer instance cannot be nested")
        self._q_captured.clear()
        self._pos_emb.clear()
        for l in self.layer_indices:
            attn = self.model.model.layers[l].self_attn
            self._handles.append(attn.register_forward_pre_hook(self._make_pre_hook(l), with_kwargs=True))
            self._handles.append(attn.q_proj.register_forward_hook(self._make_q_hook(l)))
            self._handles.append(attn.register_forward_hook(self._make_attn_hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._q_captured.clear()
        self._pos_emb.clear()

    def remove(self):
        self.__exit__(None, None, None)
