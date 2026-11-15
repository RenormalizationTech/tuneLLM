"""R1 pure plans/ratings plus thin, source-native GCAD instrumentation."""
import ast
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import random
from pathlib import Path

ARMS = ('baseline', 'residual', 'gcad')
AUTHOR_FILES = ('eval/eval_multiturn_pipeline.py', 'core/gated_steerer.py', 'core/activation_steer.py')
VECTOR_FILES = {
    'residual': 'vectors/original/Qwen2.5-7B-Instruct/impolite_response_avg_diff.pt',
    'delta': 'vectors/prompt_attn_delta/Qwen2.5-7B-Instruct/impolite_response_avg_diff_prompt_attn_delta.pt',
    'key': 'vectors/prompt_attn_k_pos/Qwen2.5-7B-Instruct/impolite_prompt_key_pos.pt',
    'demand': 'vectors/prompt_attn_k_pos/Qwen2.5-7B-Instruct/impolite_avg_demand.pt'}


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def author_functions(file, torch=None):
    """Execute only these two reviewed definitions; never pipeline imports/setup."""
    syntax = ast.parse(Path(file).read_text())
    wanted = ('sample_question_groups', 'generate_batched_multiturn')
    nodes = [n for n in syntax.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if len(nodes) != 2 or any(n.decorator_list for n in nodes):
        raise ValueError('expected exactly two undecorated author definitions')
    namespace = {'random': random, 'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(file), 'exec'), namespace)
    return {name: namespace[name] for name in wanted}, {n.name: canonical(ast.dump(n, include_attributes=False)) for n in nodes}


def load_author_class(file, name):
    """Standalone reviewed torch-only modules; no author sys.path/pipeline import."""
    allowed={'gated_steerer.py':'MultiLayerGatedSteerer','activation_steer.py':'ActivationSteerer'}
    root=Path(__file__).resolve().parents[1]/'outputs/persona-author-replication-audit-01/gcad/core'
    if Path(file).name not in allowed or allowed[Path(file).name]!=name or Path(file).resolve().parent!=root.resolve():
        raise ValueError('restricted author module path')
    spec = importlib.util.spec_from_file_location('_gcad_r1_' + Path(file).stem, file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return getattr(module, name)


def question_plan(questions, sampler):
    if len(questions) != 20 or len(set(questions)) != 20:
        raise ValueError('twenty exact author questions required')
    full = sampler(questions, 20, 10, 42)
    return {'full_groups': full, 'first_turn_groups': [[group[0]] for group in full],
            'conversations': [{'id': f'g{gi:02d}-s{si}', 'group_idx': gi, 'sample_idx': si,
                               'question': group[0]} for gi, group in enumerate(full) for si in range(3)]}


def grading_plan(conversations, seed):
    if len(conversations) != 60 or len({r['id'] for r in conversations}) != 60:
        raise ValueError('exact sixty source conversations required')
    rng = random.Random(seed); groups = list(conversations); rng.shuffle(groups)
    variants = {}
    for row in groups:
        variants[row['id']] = list(ARMS); rng.shuffle(variants[row['id']])
    # Same group order each round: a batch boundary crosses disjoint end/start
    # subsets of sixty. Thus each eight-item batch has distinct conversations.
    flat = [{'source_id': r['id'], 'arm': variants[r['id']][round_],
             'id': canonical([seed, r['id'], variants[r['id']][round_]])[:24]}
            for round_ in range(3) for r in groups]
    batches = [flat[i:i+8] for i in range(0, 180, 8)]
    rng.shuffle(batches)
    for batch in batches: rng.shuffle(batch)
    if len(batches) != 23 or sorted(map(len, batches)) != [4] + [8]*22 or any(len({x['source_id'] for x in b}) != len(b) for b in batches):
        raise ValueError('blinded plan constraint failed')
    return batches


def coherence_definition(file):
    nodes = ast.parse(Path(file).read_text()).body
    values = [ast.literal_eval(n.value) for n in nodes if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Subscript) and ast.literal_eval(n.targets[0].slice) == 'coherence_0_100']
    if len(values) != 1: raise ValueError('exact author coherence definition required')
    return values[0]


def tensor_bank(files, config):
    import torch
    values = {k: torch.load(v, map_location='cpu', weights_only=True) for k, v in files.items()}
    h, layers = config['hidden_size'], config['num_hidden_layers']; hd = h // config['num_attention_heads']; kv = config['num_key_value_heads']
    shapes = {'residual': (layers+1, h), 'delta': (layers, h), 'key': (layers, kv, hd), 'demand': (layers,)}
    for k, value in values.items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or tuple(value.shape) != shapes[k] or not torch.isfinite(value).all():
            raise ValueError('tensor-only FP32 full-dimensional author bank required: ' + k)
    return values


def steerer(model, values, author, arm, zero=False):
    import torch
    if arm == 'baseline': return torch.no_grad()
    if arm == 'residual':
        vector = values['residual'][20]; vector = torch.zeros_like(vector) if zero else vector
        return author['residual'](model, vector, coeff=2., layer_idx=19, positions='all')
    if arm != 'gcad': raise ValueError('fixed arms only')
    return author['gcad'](model, v_vectors={i: torch.zeros_like(values['delta'][i]) if zero else values['delta'][i] for i in range(9,20)},
                          k_pos_vectors={i: values['key'][i] for i in range(9,20)}, avg_demands={i: values['demand'][i].item() for i in range(9,20)},
                          layer_indices=list(range(9,20)), base_coef=3.5, scale=1.5, inverse=False, debug=False)


def hook_signature(model):
    return {name: (len(m._forward_hooks), len(m._forward_pre_hooks)) for name, m in model.named_modules()}


@contextmanager
def observe(model, arm, audit):
    """Observe actual author-hook outputs; never return replacement activations."""
    import torch
    handles = []; snapshots = {}; audit['sites'] = {}
    sites = {str(i): (model.model.layers[i] if arm == 'residual' else model.model.layers[i].self_attn)
             for i in ([19] if arm == 'residual' else range(9,20))} if arm != 'baseline' else {}
    for key, module in sites.items():
        record = audit['sites'][key] = {'prefill_calls': 0, 'decode_calls': 0, 'prefill_max_abs_delta': 0., 'decode_max_abs_delta': 0.}
        def before(m, ins, out, key=key):
            value = out[0] if isinstance(out, (tuple,list)) else out
            phase = 'prefill' if audit['sites'][key]['prefill_calls'] == 0 else 'decode'
            if audit['sites'][key][phase+'_calls'] == 0: snapshots[key] = value.detach().clone()
        def after(m, ins, out, key=key):
            value = out[0] if isinstance(out, (tuple,list)) else out; rec = audit['sites'][key]
            phase = 'prefill' if rec['prefill_calls'] == 0 else 'decode'; rec[phase+'_calls'] += 1
            if key in snapshots: rec[phase+'_max_abs_delta'] = (value.float()-snapshots.pop(key).float()).abs().max().item()
        handles.extend([module.register_forward_hook(before, prepend=True), module.register_forward_hook(after)])
    try: yield
    finally:
        for h in handles: h.remove()
        snapshots.clear(); audit['observer_handles_removed'] = True


class GenerationProxy:
    """Delegate the unchanged author call, recording actual tensors and logits."""
    def __init__(self, model, tokenizer, arm, *, maximum, before_call=None, capture_logits=False):
        self.model = model; self.tokenizer = tokenizer; self.arm = arm; self.maximum = maximum
        self.before_call = before_call; self.records = []; self.batches = []; self.first_logits = None; self.capture_logits = capture_logits

    @property
    def device(self): return self.model.device

    def generate(self, **kwargs):
        import torch,time
        ids = kwargs['input_ids']; mask = kwargs['attention_mask']; n = len(ids)
        if len(self.records)+n > self.maximum: raise ValueError('generation budget before author call')
        if self.before_call: self.before_call(len(self.batches), n, kwargs)
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos,int) else list(eos)
        audit = {}; logits = []
        def capture(m, ins, out):
            if not logits: logits.append(out[..., -1, :].detach().float().cpu())
        handle = self.model.lm_head.register_forward_hook(capture) if self.capture_logits else None
        torch.cuda.synchronize() if ids.is_cuda else None; began = time.perf_counter()
        try:
            with observe(self.model, self.arm, audit): output = self.model.generate(**kwargs)
            torch.cuda.synchronize() if ids.is_cuda else None
        finally:
            if handle: handle.remove()
        elapsed = time.perf_counter()-began
        if logits: self.first_logits = logits[0]
        width = ids.shape[1]; batch_idx = len(self.batches)
        for i, (input_, mask_, row) in enumerate(zip(ids.tolist(),mask.tolist(),output.tolist())):
            if row[:width] != input_: raise ValueError('author generated prefix changed')
            raw = row[width:]; stop = next((j+1 for j,x in enumerate(raw) if x in eos),len(raw)); suffix = raw[:stop]
            self.records.append({'actual_input_ids': input_, 'attention_mask': mask_, 'generated_padded_ids': raw, 'generated_ids': suffix,
                                 'ended_eos': bool(suffix and suffix[-1] in eos), 'hit_token_limit': len(suffix)==kwargs['max_new_tokens'] and not any(x in eos for x in suffix),
                                 'decoded_source_answer': self.tokenizer.decode(raw,skip_special_tokens=True), 'batch_index': batch_idx, 'batch_position': i,
                                 'padded_width': width, 'eos_ids': eos})
        self.batches.append({'count': n, 'settings': {k:v for k,v in kwargs.items() if k not in ('input_ids','attention_mask')},
                             'generation_wall_seconds': elapsed, 'hook_audit': audit})
        return output


def ratings(value, expected, metric):
    if not isinstance(value,dict) or set(value) != {'ratings'} or not isinstance(value['ratings'],list): raise ValueError('ratings-only JSON required')
    seen = set(); result = {}
    for row in value['ratings']:
        fields = {'id','score','reason'} | ({'refusal'} if metric == 'trait' else set())
        if not isinstance(row,dict) or set(row) != fields or not isinstance(row['id'],str) or row['id'] in seen or row['id'] not in expected:
            raise ValueError('unexpected/duplicate/missing IDs or schema')
        if not isinstance(row['reason'],str) or not row['reason'].strip(): raise ValueError('brief private reason required')
        refusal = row.get('refusal',False)
        if type(refusal) is not bool or (refusal and row['score'] is not None) or (not refusal and (type(row['score']) is not int or not 0 <= row['score'] <= 100)):
            raise ValueError('integer score or explicit trait REFUSAL required')
        seen.add(row['id']); result[row['id']] = row
    if seen != set(expected): raise ValueError('incomplete IDs')
    return result


def metrics(predictions, grades):
    from scripts.tone_common import degenerate
    result = {}
    for arm in ARMS:
        records = predictions[arm]['records']; trait = [grades['trait'][(arm,x['id'])]['score'] for x in records]
        coherence = [grades['coherence'][(arm,x['id'])]['score'] for x in records]; numeric = [x for x in trait if x is not None]
        result[arm] = {'n': len(records), 'trait_numeric_count': len(numeric), 'trait_refusals': len(trait)-len(numeric),
                       'trait_numeric_sum': sum(numeric), 'coherence_sum': sum(coherence),
                       'trait_numeric_mean': sum(numeric)/len(numeric) if numeric else None, 'trait_conservative_mean': sum(numeric)/len(records),
                       'coherence_mean': sum(coherence)/len(coherence), 'caps': sum(x['hit_token_limit'] for x in records),
                       'eos': sum(x['ended_eos'] for x in records), 'empty': sum(not x['answer'].strip() for x in records),
                       'degenerate': sum(degenerate(x['answer'])['degenerate'] for x in records),
                       'mean_generated_tokens': sum(len(x['generated_ids']) for x in records)/len(records),
                       'mean_author_retokenized_tokens': sum(x['n_tokens'] for x in records)/len(records)}
    b,r,g = (result[k] for k in ARMS)
    def contrast(x,y):
        nx,ny=x['trait_numeric_count'],y['trait_numeric_count']
        return nx>0 and ny>0 and x['trait_numeric_sum']*ny-y['trait_numeric_sum']*nx>=30*nx*ny
    gates = {'gcad_trait_gain30': contrast(g,b), 'residual_trait_gain30': contrast(r,b),
             'gcad_coherence_over_residual30': g['coherence_sum']*r['n']-r['coherence_sum']*g['n']>=30*g['n']*r['n']}
    return {'conditions':result,'directional_contrasts':gates,'all_contrasts_pass':all(gates.values()),
            'conservative_trait_differences':{k:result[k]['trait_conservative_mean']-b['trait_conservative_mean'] for k in ('residual','gcad')},
            'gcad_coherence_vs_baseline':g['coherence_mean']-b['coherence_mean']}


def public_safe(value):
    import re
    def walk(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in {'question','answer','reason','rendered_prompt','input_ids','actual_input_ids','generated_ids','generated_padded_ids','attention_mask'}: raise ValueError('private export field')
                walk(v)
        elif isinstance(x,list):
            for v in x: walk(v)
        elif isinstance(x,str) and re.search(r'/home/|Bearer\s|hf_[A-Za-z0-9]{15,}|sk-[A-Za-z0-9]{15,}',x): raise ValueError('private path/secret export')
    walk(value); return value
