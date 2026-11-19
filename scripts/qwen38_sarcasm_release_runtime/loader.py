"""Local-only loader for an externally hash-pinned experimental controls bundle."""
from contextlib import contextmanager,nullcontext
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path,PurePosixPath
import platform
import re
import stat
import threading
import weakref

SITES=(23,27,31,35,39,43)
SCALE=1.5
BANK_SHA='4fd9f86bac07cd858c8f4177ce0e9f1095d30d105f0179713c78da671ab545de'
BASE_REPO='RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4'
BASE_REVISION='586bb02218d196f485c321bbde3348677c63fc6a'
BASE_FILES={'model.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'}
_MODEL_LOCKS=weakref.WeakKeyDictionary()
_LOCK=threading.Lock()


def sha(file):
    h=hashlib.sha256()
    with Path(file).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def validate_strength(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=16:
        raise ValueError('strength must be a finite number in [0,16]; booleans are refused')
    return float(value)


def _regular(path,kind):
    try:return kind(path.lstat().st_mode)
    except OSError:return False


def _link_target(file):
    target=Path(os.readlink(file))
    return Path(os.path.abspath(target if target.is_absolute() else file.parent/target))


def _cache_file_link(root,file):
    """Validate exactly the legacy or pinned Hub2.4 shared-store link graph.

    Inspect each hop before resolving: resolve() alone hides third links and
    directory escapes. Shared Xet hashes need not equal content SHA256; the
    caller separately checks the final bytes against the release manifest.
    """
    repo=root.parent.parent;cache=repo.parent;repo_blobs=repo/'blobs'
    if (root.parent.name!='snapshots' or not re.fullmatch('[0-9a-f]{40}',root.name)
        or not re.fullmatch('(models|datasets|spaces)--.+',repo.name)
        or any(not _regular(p,stat.S_ISDIR) for p in [root,root.parent,repo,cache,repo_blobs])):
        raise ValueError('standard HF snapshot/cache directories required')
    first=_link_target(file)
    if first.parent!=repo_blobs or not re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}',first.name):
        raise ValueError('first link must target the repository blob directory')
    if not first.is_symlink():
        if not _regular(first,stat.S_ISREG):raise ValueError('repository blob must be a regular file')
        return
    final=_link_target(first);shared=cache/'blobs'
    h=final.name;prefix=shared/h[:2];marker=shared/'.huggingface-shared-blobs'
    if (not re.fullmatch('[0-9a-f]{64}',h) or final!=prefix/h
        or not _regular(shared,stat.S_ISDIR) or not _regular(prefix,stat.S_ISDIR)
        or not _regular(marker,stat.S_ISREG) or marker.read_text()!='1\n'
        or not _regular(final,stat.S_ISREG) or not final.resolve().is_relative_to(cache.resolve())):
        raise ValueError('invalid shared blob layout, marker, directory link, escape, or third link')


def safe_file(root,name):
    """Allow regular files or one/two-hop standard HF cache file links.

    Directory links and all other file-link targets are refused. Manifest paths
    are normalized POSIX relative paths. Real local_dir downloads are preferred.
    """
    root=Path(os.path.abspath(root));rel=PurePosixPath(name)
    if not name or rel.is_absolute() or str(rel)!=name or any(p in ('..','.') for p in rel.parts) or '\\' in name or ':' in name or '\x00' in name:
        raise ValueError('unsafe manifest path')
    if root.is_symlink() or not root.is_dir():raise ValueError('snapshot/bundle root must be a real local directory')
    f=root.joinpath(*rel.parts)
    for parent in f.parents:
        if parent==root:break
        if parent.is_symlink():raise ValueError('directory symlinks are refused')
    if f.is_symlink():
        _cache_file_link(root,f)
    if not f.is_file():raise ValueError(f'missing required local file: {name}')
    return f


def check_file(root,name,record):
    f=safe_file(root,name)
    if f.stat().st_size!=record['bytes'] or sha(f)!=record['sha256']:
        raise ValueError(f'file bytes/hash differ: {name}')
    return f


def _consumer_metadata(rel):
    parts=rel.parts
    return (parts[0] in ('.git','.cache') or str(rel)=='.gitattributes'
        or len(parts)>=2 and parts[0]=='runtime' and '__pycache__' in parts and rel.suffix=='.pyc')


def verify_bundle(bundle,manifest_sha256):
    if not isinstance(manifest_sha256,str) or not re.fullmatch('[0-9a-f]{64}',manifest_sha256):
        raise ValueError('an externally supplied release manifest SHA256 is required')
    bundle=Path(bundle).absolute();mf=safe_file(bundle,'release_manifest.json')
    if sha(mf)!=manifest_sha256:raise ValueError('release manifest hash differs from supplied trust anchor')
    manifest=json.loads(mf.read_text())
    if (manifest['repo_id']!='RenormalizationTech/qwen38-27b-sarcasm-gcad-controls' or manifest['private'] is not False
        or manifest['base']['repo_id']!=BASE_REPO or manifest['base']['revision']!=BASE_REVISION
        or manifest['bank_sha256']!=BANK_SHA or manifest['sites']!=list(SITES) or manifest['scale']!=SCALE
        or set(manifest['base']['files'])!=BASE_FILES):
        raise ValueError('unsupported controls/base identity or fixed geometry')
    names=set(manifest['files'])
    if 'release_manifest.json' in names:raise ValueError('manifest cannot hash itself')
    required={'bank.safetensors','runtime/__init__.py','runtime/gcad.py','runtime/loader.py','runtime/example.py'}
    if not required<=names:raise ValueError('runtime/bank missing from payload inventory')
    observed=set()
    for f in bundle.rglob('*'):
        rel=f.relative_to(bundle)
        if _consumer_metadata(rel):continue
        if f.is_symlink() or f.is_file():
            safe_file(bundle,rel.as_posix());observed.add(rel.as_posix())
        elif f.is_dir() and f.is_symlink():raise ValueError('directory link in payload')
    if observed!=names|{'release_manifest.json'}:raise ValueError('payload inventory has extra or missing files')
    for name,record in manifest['files'].items():check_file(bundle,name,record)
    if manifest['files']['bank.safetensors']['sha256']!=BANK_SHA:raise ValueError('bank inventory hash differs')
    return manifest


def validate_environment(manifest):
    for name,expected in manifest['environment']['packages'].items():
        if importlib.metadata.version(name)!=expected:raise ValueError(f'pinned package version required: {name}=={expected}')
    if platform.python_version()!=manifest['environment']['python']:raise ValueError('pinned Python version required')
    import torch
    if str(torch.__version__)!=manifest['environment']['torch_build'] or torch.version.cuda!=manifest['environment']['cuda_build']:
        raise ValueError('pinned torch/CUDA build required')
    for name,expected in manifest['environment']['optional_kernels'].items():
        if (importlib.util.find_spec(name) is not None)!=expected:raise ValueError(f'optional kernel availability differs: {name}')
    import transformers
    source=Path(transformers.__file__).parent/'models/qwen3_5/modeling_qwen3_5.py'
    if sha(source)!=manifest['environment']['qwen35_modeling_sha256']:raise ValueError('installed native Qwen3.5 source differs')


def validate_snapshot(snapshot,manifest):
    snapshot=Path(snapshot).absolute()
    for name,record in manifest['base']['files'].items():check_file(snapshot,name,record)
    unexpected=[p.name for p in snapshot.iterdir() if p.name!='model.safetensors' and
        (p.suffix in ('.safetensors','.bin','.pt','.pth') or p.name.endswith('.index.json'))]
    if unexpected:raise ValueError('unexpected alternate base weight files')
    return snapshot


def validate_bank(bank):
    import torch
    expected={f'{i}.{name}':shape for i in SITES for name,shape in [('delta',(5120,)),('key',(4,256)),('demand',())]}
    if set(bank)!=set(expected):raise ValueError('bank tensor keys differ')
    for name,shape in expected.items():
        t=bank[name]
        if t.dtype!=torch.float32 or tuple(t.shape)!=shape or not torch.isfinite(t).all():raise ValueError(f'invalid bank dtype/shape/value: {name}')


def validate_model(model):
    import torch
    import bitsandbytes as bnb
    from .gcad import validate_geometry
    c=model.config
    if (c.model_type,c.num_hidden_layers,c.hidden_size,c.num_attention_heads,c.num_key_value_heads,c.head_dim)!=('qwen3_5_text',64,5120,24,4,256):
        raise ValueError('only the studied text hybrid geometry is supported')
    types=['full_attention' if i%4==3 else 'linear_attention' for i in range(64)]
    if list(c.layer_types)!=types or model.__class__.__name__!='Qwen3_5ForCausalLM':raise ValueError('native hybrid text model required')
    validate_geometry(model,[i for i,t in enumerate(types) if t=='full_attention'])
    quant=c.quantization_config;quant=quant.to_dict() if hasattr(quant,'to_dict') else quant
    if not quant.get('load_in_4bit') or quant.get('bnb_4bit_quant_type')!='nf4' or not quant.get('bnb_4bit_use_double_quant'):
        raise ValueError('saved NF4 double quantization required')
    for i,t in enumerate(types):
        if t!='full_attention':continue
        for name in ('q_proj','k_proj','v_proj','o_proj'):
            p=getattr(model.model.layers[i].self_attn,name)
            if not isinstance(p,bnb.nn.Linear4bit) or p.weight.quant_state is None or p.compute_dtype!=torch.bfloat16:
                raise ValueError('native Linear4bit projections with BF16 compute required')
    if any('visual' in n or 'mtp' in n for n,_ in model.named_modules()):raise ValueError('vision/MTP modules must be absent')


def validate_inputs(input_ids,attention_mask):
    import torch
    if not isinstance(input_ids,torch.Tensor) or not isinstance(attention_mask,torch.Tensor):raise ValueError('input_ids and attention_mask must be tensors')
    if input_ids.ndim!=2 or input_ids.shape[0]!=1 or input_ids.shape[1]<2 or input_ids.dtype!=torch.long:
        raise ValueError('unpadded batch1 with at least two integer tokens required')
    if attention_mask.shape!=input_ids.shape or not torch.all(attention_mask==1):
        raise ValueError('padding/masked positions are unsupported; attention_mask must be all ones')


class Runtime:
    __slots__=('model','tokenizer','bank','snapshot','manifest')
    def __init__(self,model,tokenizer,bank,snapshot,manifest):
        self.model=model;self.tokenizer=tokenizer;self.bank=bank;self.snapshot=Path(snapshot);self.manifest=manifest

    @contextmanager
    def steering(self,input_ids,attention_mask,strength):
        b=validate_strength(strength);validate_inputs(input_ids,attention_mask)
        with _LOCK:lock=_MODEL_LOCKS.setdefault(self.model,threading.Lock())
        if not lock.acquire(blocking=False):raise RuntimeError('nested/concurrent steering or strength changes during generation are refused')
        try:
            if self.model.training:raise ValueError('model must remain in eval mode')
            from .gcad import Steerer
            context=Steerer(self.model,self.bank,list(SITES),b,SCALE) if b else nullcontext()
            with context:yield context
        finally:lock.release()

    def forward(self,input_ids,attention_mask,strength=8):
        import torch
        with torch.inference_mode(),self.steering(input_ids,attention_mask,strength):
            return self.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=True,logits_to_keep=1)

    def generate_ids(self,input_ids,attention_mask,strength=8,max_new_tokens=160):
        import torch
        from transformers import GenerationConfig,set_seed
        validate_strength(strength);validate_inputs(input_ids,attention_mask)
        if type(max_new_tokens) is not int or not 1<=max_new_tokens<=160:raise ValueError('token cap must be an integer in1..160')
        cfg=GenerationConfig.from_pretrained(self.snapshot,local_files_only=True)
        cfg.update(do_sample=False,repetition_penalty=1.05,max_new_tokens=max_new_tokens,use_cache=True,num_beams=1,
            temperature=None,top_p=None,top_k=None,pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        with torch.inference_mode(),self.steering(input_ids,attention_mask,strength) as context:
            set_seed(42)
            output=self.model.generate(input_ids=input_ids,attention_mask=attention_mask,generation_config=cfg,logits_to_keep=1)
        ids=output[0,input_ids.shape[1]:].tolist()
        return {'generated_ids':ids,'text':self.tokenizer.decode(ids,skip_special_tokens=True),
                'audit':context.audit if strength else {}}

    def generate(self,prompt,strength=8,max_new_tokens=160):
        import torch
        validate_strength(strength)
        if not isinstance(prompt,str) or not prompt.strip():raise ValueError('one nonempty English user prompt required')
        text=self.tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=self.tokenizer(text,add_special_tokens=False)['input_ids']
        if len(ids)+max_new_tokens>2048:raise ValueError('demonstration sequence guard2048; no truncation')
        batch=torch.tensor([ids],device=self.model.device)
        return self.generate_ids(batch,torch.ones_like(batch),strength,max_new_tokens)


def load_runtime(snapshot,manifest_sha256,*,bundle=None,device='cuda:0'):
    if not isinstance(device,str) or not re.fullmatch(r'cuda:[0-9]+',device):raise ValueError('one explicit CUDA device required; no CPU/offload fallback')
    # Keep lexical __file__ parents: HF snapshot files may themselves be links.
    bundle=Path(bundle).absolute() if bundle is not None else Path(__file__).absolute().parent.parent
    manifest=verify_bundle(bundle,manifest_sha256);validate_environment(manifest);snapshot=validate_snapshot(snapshot,manifest)
    import torch
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM
    from safetensors.torch import load_file
    bank=load_file(str(safe_file(bundle,'bank.safetensors')),device='cpu');validate_bank(bank)
    model,info=Qwen3_5ForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,
        device_map={'':device},attn_implementation='sdpa',output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','mismatched_keys','error_msgs','unexpected_keys')):raise ValueError('saved text checkpoint did not load exactly')
    model.eval();model.config.use_cache=True;validate_model(model)
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    return Runtime(model,tok,bank,snapshot,manifest)
