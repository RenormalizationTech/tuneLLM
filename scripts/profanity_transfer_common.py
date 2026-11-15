"""Explicit-parameter PF method primitives; no runner, data choices or model I/O."""
from collections.abc import Mapping
import math
import re

from scripts import persona_saps_common as saps
from scripts.save_nf4 import tensor_hash

# Pure existing HF counterparts of the pinned source math. No frozen globals
# or guards are changed. Capture is B1, last prompt token, raw post-block basis.
capture_last_prompt = saps.capture
fixed_scalar = saps.fixed_scalar


def _sha(value):
    return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None


def _finite_float(value):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError('explicit finite Python float required')


def validate_bank(model, bank, *, expected_basis, expected_data_sha256,
                  representation, forbidden_tensor_hashes):
    """Verify an externally approved PF provenance/basis, not semantic quality.

    Caller supplies the approved basis/data hash and independently obtained
    forbidden tensor hashes (including the downloaded impolite bank). Metadata
    alone cannot prove how an artifact was built; its extraction must be audited.
    No implicit checkpoint, tokenizer, layer, corpus or normalization choice.
    """
    import torch
    basis_keys={'model_id','revision','model_source','tokenizer_sha256','rendering_sha256','coordinate_system'}
    if not isinstance(expected_basis,Mapping) or set(expected_basis)!=basis_keys:
        raise ValueError('explicit complete expected model/token basis required')
    if not all(type(expected_basis[k]) is str and expected_basis[k] for k in basis_keys):
        raise ValueError('nonempty model/token basis required')
    if expected_basis['coordinate_system']!='native_hf' or not all(_sha(expected_basis[k]) for k in ('tokenizer_sha256','rendering_sha256')):
        raise ValueError('native HF tokenizer/rendering fingerprint required')
    if not _sha(expected_data_sha256):raise ValueError('approved PF training data hash required')
    if not isinstance(forbidden_tensor_hashes,(set,frozenset)) or not all(_sha(h) for h in forbidden_tensor_hashes):
        raise ValueError('explicit forbidden artifact tensor-hash set required')
    required={'trait','source_trait','basis','training_data_sha256','representation','tensors','tensor_sha256'}
    if not isinstance(bank,Mapping) or set(bank)!=required:raise ValueError('exact bank schema required')
    if bank['trait']!='profanity' or bank['source_trait']!='profanity':raise ValueError('no impolite/sarcasm bank substitution')
    if bank['basis']!=expected_basis or bank['training_data_sha256']!=expected_data_sha256:
        raise ValueError('approved model/token/data basis differs')
    if bank['representation']!=representation:raise ValueError('prompt-state and attention representations are distinct')
    cfg=model.config;layers=cfg.num_hidden_layers;hidden=cfg.hidden_size
    if getattr(cfg,'_name_or_path',None)!=expected_basis['model_source']:
        raise ValueError('actual model configuration source differs from approved checkpoint')
    if len(model.model.layers)!=layers:raise ValueError('native decoder depth mismatch')
    if representation=='last_prompt_token_contrast':shapes={'units':(layers,hidden),'targets':(layers,)}
    elif representation=='system_attention_delta':
        if hidden%cfg.num_attention_heads or cfg.num_attention_heads%cfg.num_key_value_heads:
            raise ValueError('invalid GQA dimensions')
        shapes={'delta':(layers,hidden),'key':(layers,cfg.num_key_value_heads,hidden//cfg.num_attention_heads),'demand':(layers,)}
    else:raise ValueError('unknown explicit representation')
    tensors=bank['tensors']
    if not isinstance(tensors,Mapping) or set(tensors)!=set(shapes) or set(bank['tensor_sha256'])!=set(shapes):
        raise ValueError('exact representation tensor set required')
    for name,shape in shapes.items():
        value=tensors[name]
        if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or value.device.type!='cpu' or tuple(value.shape)!=shape or not torch.isfinite(value).all():
            raise ValueError('finite FP32 CPU bank with native model dimensions required: '+name)
        digest=tensor_hash(value)
        if digest in forbidden_tensor_hashes:raise ValueError('known foreign/impolite tensor cannot be relabelled')
        if digest!=bank['tensor_sha256'][name]:raise ValueError('tensor identity differs: '+name)
    if representation=='last_prompt_token_contrast' and not torch.allclose(tensors['units'].norm(dim=-1),torch.ones(layers),rtol=1e-5,atol=1e-6):
        raise ValueError('saved projection axes must already be unit vectors')
    return tensors


def mean_prompt_direction(plain, styled, target_styled):
    """Author sequential FP32 mean(style-plain), unit axes, second-style target.

    Inputs are captured last-real-prompt-token raw post-block states [N,L,H].
    This does not accept response-pooled vectors as a source-faithful substitute.
    The caller chooses and freezes prompts, systems, draws and sample counts.
    """
    import torch
    for value in (plain,styled,target_styled):
        if not isinstance(value,torch.Tensor) or value.ndim!=3 or value.dtype!=torch.float32 or value.device.type!='cpu' or not torch.isfinite(value).all():
            raise ValueError('finite FP32 CPU [samples,blocks,hidden] captures required')
    if plain.shape!=styled.shape or plain.shape!=target_styled.shape or min(plain.shape)<1:
        raise ValueError('aligned nonempty source-style capture shapes required')
    raw=torch.zeros_like(plain[0])
    for neutral,positive in zip(plain,styled):raw+=positive-neutral
    raw/=len(plain)
    if (raw.norm(dim=-1)==0).any():raise ValueError('zero contrast has no unit axis')
    units=torch.nn.functional.normalize(raw,dim=1);targets=torch.zeros(raw.shape[0])
    for positive in target_styled:targets+=(positive*units).sum(dim=1)
    targets/=len(target_styled)
    return {'raw':raw,'units':units,'targets':targets}


def projection_context(model, bank, *, expected_basis, expected_data_sha256,
                       forbidden_tensor_hashes, block, alpha, method, h0, audit):
    """Fixed: c=alpha*(T-dot(h0,u)); adaptive: source last-token/B1 broadcast.

    The unchanged HF helper preserves model-dtype vector multiplication and
    source adaptive epsilon. Both change every prefix position during prefill
    and the cached token during decoding. Alpha0 installs actual zero hooks.
    """
    import torch
    values=validate_bank(model,bank,expected_basis=expected_basis,expected_data_sha256=expected_data_sha256,
                         representation='last_prompt_token_contrast',forbidden_tensor_hashes=forbidden_tensor_hashes)
    _finite_float(alpha)
    if type(block) is not int or not 0<=block<model.config.num_hidden_layers:raise ValueError('explicit zero-based block required')
    if method not in ('fixed','saps'):raise ValueError('fixed or saps required')
    target=values['targets'][block];unit=values['units'][block]
    if method=='fixed':
        if not isinstance(h0,torch.Tensor) or h0.dtype!=torch.float32 or h0.device.type!='cpu' or h0.shape!=unit.shape or not torch.isfinite(h0).all():
            raise ValueError('fixed controller needs same-basis FP32 CPU per-prompt h0')
        constant=fixed_scalar(target,h0,unit,alpha)
    else:
        if h0 is not None:raise ValueError('adaptive does not use a cached fixed h0')
        constant=None
    return saps.intervention(model,block,unit,target,alpha,method,constant=constant,audit=audit)


def gcad_context(model, bank, *, expected_basis, expected_data_sha256,
                 forbidden_tensor_hashes, author_factory, layers, base_coef,
                 scale, inverse, debug):
    """Delegate unchanged pinned author GCAD hook, with explicit PF-only tensors.

    Caller resolves and hash-verifies the reviewed author_factory; this helper
    never imports a pipeline, credential setup, checkpoint or tensor file.
    Delta is passed RAW. No residual-vector substitute or implicit layer range.
    """
    values=validate_bank(model,bank,expected_basis=expected_basis,expected_data_sha256=expected_data_sha256,
                         representation='system_attention_delta',forbidden_tensor_hashes=forbidden_tensor_hashes)
    if not isinstance(layers,(list,tuple)) or not layers or len(set(layers))!=len(layers) or any(type(i) is not int or not 0<=i<model.config.num_hidden_layers for i in layers):
        raise ValueError('explicit unique native GCAD block indices required')
    _finite_float(base_coef);_finite_float(scale)
    if type(inverse) is not bool or type(debug) is not bool or not callable(author_factory):raise ValueError('explicit author factory and boolean flags required')
    return author_factory(model,v_vectors={i:values['delta'][i] for i in layers},
        k_pos_vectors={i:values['key'][i] for i in layers},avg_demands={i:values['demand'][i].item() for i in layers},
        layer_indices=list(layers),base_coef=base_coef,scale=scale,inverse=inverse,debug=debug)
