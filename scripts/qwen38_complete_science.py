"""Explicit, once-only QASPER395+8 completion; CPU prepare/assemble by default.

Requires the prospective amendment; preserves the failed original and delegates
GPU generation solely to the unchanged evaluator. No repeated IDs or quality
selection. Assemblies explicitly disclose the original phase-cap failure.
"""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

from scripts import qwen38_eval_science as runner
from scripts.qwen38_science_runtime import sha, versions

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'qwen38-science-completion-v1'
AMENDMENT = 'results/protocols/qwen38-qasper-baseline-completion-amendment-01.json'
ERROR = 'TimeoutError: registered generation time cap reached'
UNCHANGED = ['original403questions/order/targets', 'model/checkpoint selection', 'baseprep/dtypes/software/sourceversion',
    'baseprompt/template andgreedybatch1max512decoding', 'statisticalmetrics/bootstrap/qualitygates', 'originaltotalstudyGPUhourpools']
NOTICE = '403 first-time question coverage in two sequential GPU0 executions (395+8); original phase cap exceeded, explicitly amended, not an uninterrupted original-cap run.'


def load(path): return json.loads(Path(path).read_text())
def write(path, value):
    with Path(path).open('x') as f: f.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
def checked(path, expected):
    if sha(path) != expected: raise ValueError('bound hash differs: ' + Path(path).name)
def jsonl(path):
    raw = Path(path).read_bytes().splitlines(keepends=True)
    if not raw or any(not line.endswith(b'\n') or not line.strip() for line in raw): raise ValueError('nonempty newline-terminated JSONL required')
    return raw, [json.loads(line) for line in raw]
def local(root, path):
    path = (root / path).resolve()
    if not path.is_relative_to(root / 'outputs'): raise ValueError('run paths must stay inside project outputs')
    return path

def tokenize(root, supplied=None):
    if supplied is not None: return supplied
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(root / 'artifacts/qwen38-27b-nf4-01', local_files_only=True, trust_remote_code=False)


def check_predictions(predicted, records, cfg, tokenizer):
    if [r['id'] for r in predicted] != [r['id'] for r in records]: raise ValueError('predictions must be exact ordered prefix/suffix; no repeats or holes')
    for row, p in zip(records, predicted):
        prompt = runner.render(tokenizer, row, cfg['template'])
        ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        if p['paper_id'] != row['paper_id'] or p['rendered_prompt'] != prompt or p['input_ids'] != ids or len(ids) + 512 > 8192:
            raise ValueError('prediction paper/prompt/input IDs differ')
        tokens = p['generated_ids']; ended = bool(tokens and tokens[-1] == tokenizer.eos_token_id)
        if (not tokens or len(tokens) > 512 or any(type(t) is not int or t < 0 for t in tokens)
                or tokenizer.eos_token_id in tokens[:-1] or p['ended_with_eos'] is not ended
                or p['hit_token_limit'] is not (len(tokens) == 512 and not ended)
                or tokenizer.decode(tokens, skip_special_tokens=True) != p['output']
                or not isinstance(p['seconds'], (int, float)) or not math.isfinite(p['seconds']) or p['seconds'] < 0):
            raise ValueError('prediction suffix/EOS/timing differs')


def original_proof(original, amendment, root, tokenizer=None):
    original = local(root, original); amendment = Path(amendment).resolve()
    if amendment != root / AMENDMENT: raise ValueError('exact prospective completion amendment required')
    a, cfg, result = load(amendment), load(original / 'registration.json'), load(original / 'result.json')
    if (local(root, a['original_run']) != original or a['one_attempt_only'] is not True
            or a['quality_outputs_or_scores_used_for_amendment'] is not False or a['unchanged'] != UNCHANGED
            or a['full_denominator'] != 403 or a['completed_prefix'] != 395 or a['additional_phase_cap_seconds'] != 900
            or a['original_phase_cap_seconds'] != 10000 or a['original_completed'] is not False or a['original_error'] != ERROR
            or a['additional_parent_reservation'] != {'name': 'qasper-final-base-completion', 'gpu_hours': .25, 'pool': 'qasper_eval'}
            or a['parent_pool_limit_gpu_hours'] != 12 or not 0 < a['parent_accounted_including_remaining_reservations_gpu_hours'] <= 12):
        raise ValueError('prospective no-repeat/phase-cap/pool/gate amendment differs')
    stamp = datetime.fromisoformat(a['registered_at_utc'])
    if stamp.tzinfo is None or stamp > datetime.now(timezone.utc): raise ValueError('prospective amendment date invalid')
    for name, field in [('registration.json','original_registration_sha256'), ('result.json','original_result_sha256'), ('predictions.jsonl','original_predictions_sha256')]:
        checked(original / name, a[field])
    expected = {'task':'qasper','template':'base','template_text':'','purpose':'final','max_new_tokens':512,'max_context':8192,
        'max_seconds':10000,'seed':42,'batch_size':1,'cached_generation':True,'enable_thinking':False,'do_sample':False,
        'adapter':None,'adapter_files':{},'gpu':'0',
        'base_preparation':'same PEFT floating casts and rank16 wrapper; disabled adapter for unadapted arm'}
    if any(cfg.get(k) != v or type(cfg.get(k)) is not type(v) for k,v in expected.items()) or cfg['versions'] != versions():
        raise ValueError('original generation settings/software differ')
    if (result.get('completed') is not False or result.get('generations') != 395 or result.get('error') != ERROR
            or result['registration_sha256'] != sha(original / 'registration.json') or not result.get('runtime')
            or result['elapsed_seconds'] != a['original_measured_seconds'] or not math.isfinite(result['elapsed_seconds'])
            or result['elapsed_seconds'] < 10000): raise ValueError('exact capped395 failure required')
    source_names = {'scripts/qwen38_eval_science.py','scripts/qwen38_science_runtime.py','scripts/qwen38_nf4_smoke.py'}
    if set(cfg['source_sha256']) != source_names: raise ValueError('original source set differs')
    for name, digest in cfg['source_sha256'].items(): checked(root / name, digest)
    checked(runner.__file__, cfg['source_sha256']['scripts/qwen38_eval_science.py'])
    full = local(root, cfg['input_path']); checked(full, cfg['input_sha256']); checked(full, a['original_full_input_sha256'])
    evaluation = root / 'results/protocols/qwen38-qasper-evaluation-01.json'
    if load(evaluation)['final_input_sha256'] != sha(full): raise ValueError('original frozen final input differs')
    manifest = root / 'results/qwen38_27b_nf4_artifact_01.json'; checked(manifest, cfg['base_manifest_sha256'])
    token_files = [root / 'artifacts/qwen38-27b-nf4-01' / n for n,v in load(manifest)['files'].items() if n.startswith('tokenizer') or n == 'chat_template.jinja']
    for path in token_files: checked(path, load(manifest)['files'][path.name]['sha256'])
    raw, records = jsonl(full); _, prefix = jsonl(original / 'predictions.jsonl')
    if len(records) != 403 or len({r['id'] for r in records}) != 403 or cfg['ids'] != [r['id'] for r in records] or len(prefix) != 395:
        raise ValueError('fixed403 inputs and395 prefix required')
    if [r['id'] for r in records[395:]] != a['missing_suffix_ids']: raise ValueError('amended eight suffix IDs differ')
    tokenizer = tokenize(root, tokenizer)
    check_predictions(prefix, records[:395], cfg, tokenizer)
    for row in records[395:]:
        ids = tokenizer(runner.render(tokenizer, row, 'base'), add_special_tokens=False)['input_ids']
        if not ids or len(ids) + 512 > 8192: raise ValueError('remaining full prompt exceeds cap')
    paths = [original / n for n in ('registration.json','result.json','predictions.jsonl')] + [full, amendment, evaluation, manifest] + token_files + [root / n for n in source_names]
    return cfg, result, raw, records, {str(p): sha(p) for p in paths}, tokenizer


def prepare(original, output, amendment, gpu, max_seconds=900, root=ROOT, tokenizer=None):
    root = Path(root).resolve(); output = local(root, output)
    if output.exists(): raise FileExistsError('preserve existing completion')
    cfg, failure, raw, records, bound, _ = original_proof(original, amendment, root, tokenizer)
    if gpu != cfg['gpu'] or max_seconds != 900: raise ValueError('completion must use original GPU0 and900second phase cap')
    remaining = b''.join(raw[395:])
    args = {'input':str(output / 'remaining.jsonl'),'output':str(output / 'generation'),'task':'qasper','template':'base',
            'adapter':None,'gpu':gpu,'purpose':'final','max_new_tokens':512,'max_seconds':900}
    registration = {'version':VERSION,'original':str(local(root, original)),'amendment':str(Path(amendment).resolve()),
        'bound_files_sha256':bound,'full_input':cfg['input_path'],'suffix_ids':[r['id'] for r in records[395:]],
        'remaining_sha256':hashlib.sha256(remaining).hexdigest(),'generator_args':args,
        'source_sha256':sha(__file__),'python_executable':sys.executable,'versions':versions(),'interpretation':NOTICE}
    for name, digest in bound.items(): checked(name, digest)
    output.mkdir(parents=True); (output / 'remaining.jsonl').write_bytes(remaining); write(output / 'registration.json', registration)
    return {'prepared':str(output),'remaining':8,'GPU_started':False,'registration_sha256':sha(output / 'registration.json')}


def validate_run(run, root=ROOT, tokenizer=None):
    root = Path(root).resolve(); run = local(root, run); c = load(run / 'registration.json')
    if c['version'] != VERSION or c['source_sha256'] != sha(__file__) or c['python_executable'] != sys.executable or c['versions'] != versions():
        raise ValueError('completion source/environment differs')
    for name, digest in c['bound_files_sha256'].items(): checked(name, digest)
    cfg, failure, raw, records, bound, tokenizer = original_proof(c['original'], c['amendment'], root, tokenizer)
    args = {'input':str(run / 'remaining.jsonl'),'output':str(run / 'generation'),'task':'qasper','template':'base',
        'adapter':None,'gpu':cfg['gpu'],'purpose':'final','max_new_tokens':512,'max_seconds':900}
    if (c['generator_args'] != args or c['bound_files_sha256'] != bound or c['suffix_ids'] != [r['id'] for r in records[395:]]
            or c['full_input'] != cfg['input_path'] or c['interpretation'] != NOTICE or (run / 'remaining.jsonl').read_bytes() != b''.join(raw[395:])):
        raise ValueError('completion suffix/settings/bindings differ')
    checked(run / 'remaining.jsonl', c['remaining_sha256'])
    return c, cfg, failure, records, tokenizer


def execute(run, root=ROOT, tokenizer=None):
    run = local(Path(root).resolve(), run)
    if (run / 'attempt.json').exists() or (run / 'generation').exists(): raise FileExistsError('one explicit completion attempt only; no retry')
    c, *_ = validate_run(run, root, tokenizer)
    write(run / 'attempt.json', {'registration_sha256':sha(run / 'registration.json'),'started_at_utc':datetime.now(timezone.utc).isoformat()})
    return runner.execute(argparse.Namespace(**c['generator_args']))


def assembly_proof(run, root=ROOT, tokenizer=None):
    """CPU helper for consumers: validate segments before trusting/copying assembly."""
    run = local(Path(root).resolve(), run); c, cfg, failure, records, tokenizer = validate_run(run, root, tokenizer)
    if load(run / 'attempt.json')['registration_sha256'] != sha(run / 'registration.json'): raise ValueError('completion attempt registration differs')
    generation = run / 'generation'; gc, result = load(generation / 'registration.json'), load(generation / 'result.json')
    expected = {**cfg,'input_path':str(run / 'remaining.jsonl'),'input_sha256':sha(run / 'remaining.jsonl'),'ids':c['suffix_ids'],'max_seconds':900}
    if gc != expected or result.get('completed') is not True or result.get('error') or result.get('generations') != 8 or result['registration_sha256'] != sha(generation / 'registration.json'):
        raise ValueError('complete eight-item matching completion required')
    if result['runtime'] != failure['runtime'] or not math.isfinite(result['elapsed_seconds']) or not 0 <= result['elapsed_seconds'] <= 900: raise ValueError('runtime/timing differ across segments or completion exceeds900seconds')
    suffix_raw, predicted = jsonl(generation / 'predictions.jsonl'); check_predictions(predicted, records[395:], cfg, tokenizer)
    if 'predictions_sha256' in result: checked(generation / 'predictions.jsonl', result['predictions_sha256'])
    proof = segment_proof(run, c, cfg, failure, result)
    return cfg, failure, result, Path(c['original']).joinpath('predictions.jsonl').read_bytes() + b''.join(suffix_raw), proof


def segment_proof(run, c, cfg, failure, result):
    generation = run / 'generation'
    paths = [run / 'registration.json',run / 'attempt.json'] + [generation / n for n in ('registration.json','result.json','predictions.jsonl')]
    proof = {'version':VERSION,'assembler_source_sha256':sha(__file__),'completion_run':str(run),'amendment_sha256':sha(c['amendment']),
        'bound_files_sha256':{**c['bound_files_sha256'],**{str(p):sha(p) for p in paths}},'prefix_count':395,'suffix_count':8,
        'suffix_ids':c['suffix_ids'],'original_failed_result_sha256':sha(Path(c['original']) / 'result.json'),
        'segments':[{'run':c['original'],'gpu':cfg['gpu'],'elapsed_seconds':failure['elapsed_seconds'],'completed':False},
                    {'run':str(generation),'gpu':cfg['gpu'],'elapsed_seconds':result['elapsed_seconds'],'completed':True}],
        'original_phase_cap_met':False,'interpretation':NOTICE}
    return proof


def assemble(run, output, root=ROOT, tokenizer=None):
    root = Path(root).resolve(); output = local(root, output)
    if output.exists(): raise FileExistsError('preserve existing assembly')
    cfg, failure, suffix, raw, proof = assembly_proof(run, root, tokenizer)
    reg = {**cfg,'max_seconds':10900,'completion':proof,'assembler_source_sha256':sha(__file__)}
    result = {'completed':True,'generations':403,'runtime':failure['runtime'],'elapsed_seconds':failure['elapsed_seconds']+suffix['elapsed_seconds'],
        'gpu_count':1,'gpu':'0','completion':proof,'assembler_source_sha256':sha(__file__)}
    for name, digest in proof['bound_files_sha256'].items(): checked(name, digest)
    output.mkdir(parents=True); write(output / 'registration.json', reg); (output / 'predictions.jsonl').write_bytes(raw)
    result.update(registration_sha256=sha(output / 'registration.json'),predictions_sha256=sha(output / 'predictions.jsonl'))
    write(output / 'result.json', result)
    return {'assembled':str(output),'generations':403,'GPU_started':False,'interpretation':NOTICE}


def validate_assembly(output, root=ROOT, tokenizer=None):
    validate_assembly_hashes(output, root)
    output = local(Path(root).resolve(), output); reg, result = load(output / 'registration.json'), load(output / 'result.json')
    cfg, failure, suffix, raw, proof = assembly_proof(reg['completion']['completion_run'], root, tokenizer)
    if (reg != {**cfg,'max_seconds':10900,'completion':proof,'assembler_source_sha256':sha(__file__)}
            or result != {'completed':True,'generations':403,'runtime':failure['runtime'],'elapsed_seconds':failure['elapsed_seconds']+suffix['elapsed_seconds'],
                'gpu_count':1,'gpu':'0','completion':proof,'assembler_source_sha256':sha(__file__),
                'registration_sha256':sha(output / 'registration.json'),'predictions_sha256':sha(output / 'predictions.jsonl')}
            or (output / 'predictions.jsonl').read_bytes() != raw): raise ValueError('assembled coverage/provenance bytes differ')
    return proof


def validate_assembly_hashes(output, root=ROOT):
    """No tokenizer/ML imports: custody check before copying an existing assembly.

    Full prompt/token replay is performed by prepare, assemble and
    validate_assembly; this helper verifies their bound artifacts and exact join.
    """
    root = Path(root).resolve(); output = local(root, output)
    reg, result = load(output / 'registration.json'), load(output / 'result.json'); proof = reg['completion']
    if (reg['assembler_source_sha256'] != sha(__file__) or result['assembler_source_sha256'] != sha(__file__)
            or proof['assembler_source_sha256'] != sha(__file__) or result['completion'] != proof
            or proof['original_phase_cap_met'] is not False or proof['prefix_count'] != 395 or proof['suffix_count'] != 8
            or proof['interpretation'] != NOTICE): raise ValueError('assembly source/custody disclosure differs')
    for path, digest in proof['bound_files_sha256'].items(): checked(path, digest)
    checked(root / AMENDMENT, proof['amendment_sha256'])
    amendment = load(root / AMENDMENT); run = local(root, proof['completion_run']); wrapper = load(run / 'registration.json')
    if wrapper['source_sha256'] != sha(__file__) or Path(wrapper['amendment']) != root / AMENDMENT: raise ValueError('completion helper/amendment differs')
    original = local(root, amendment['original_run']); gc = load(run / 'generation/registration.json')
    cfg, failed, completed = load(original / 'registration.json'), load(original / 'result.json'), load(run / 'generation/result.json')
    required = {str(root / n) for n in cfg['source_sha256']} | {str(root / AMENDMENT), str(Path(cfg['input_path']))}
    if not required <= set(wrapper['bound_files_sha256']) or wrapper['versions'] != cfg['versions']:
        raise ValueError('completion source/input custody bindings missing')
    for name, digest in cfg['source_sha256'].items(): checked(root / name, digest)
    for name, key in [('registration.json','original_registration_sha256'),('result.json','original_result_sha256'),('predictions.jsonl','original_predictions_sha256')]: checked(original / name, amendment[key])
    checked(cfg['input_path'], amendment['original_full_input_sha256'])
    prefix, p = jsonl(original / 'predictions.jsonl'); suffix, s = jsonl(run / 'generation/predictions.jsonl')
    if (len(p) != 395 or len(s) != 8 or [r['id'] for r in p+s] != cfg['ids'] or len(set(cfg['ids'])) != 403
            or wrapper['original'] != str(original) or wrapper['suffix_ids'] != amendment['missing_suffix_ids']
            or proof != segment_proof(run,wrapper,cfg,failed,completed)
            or (run / 'remaining.jsonl').read_bytes() != b''.join(jsonl(cfg['input_path'])[0][395:])
            or load(run / 'attempt.json')['registration_sha256'] != sha(run / 'registration.json')
            or gc != {**cfg,'input_path':str(run / 'remaining.jsonl'),'input_sha256':sha(run / 'remaining.jsonl'),'ids':amendment['missing_suffix_ids'],'max_seconds':900}
            or failed['completed'] is not False or failed['error'] != ERROR or completed['completed'] is not True
            or completed.get('error') or completed['generations'] != 8 or completed['registration_sha256'] != sha(run / 'generation/registration.json')
            or not math.isfinite(completed['elapsed_seconds']) or not 0 <= completed['elapsed_seconds'] <= 900
            or failed['runtime'] != completed['runtime'] or cfg['gpu'] != '0'
            or reg != {**cfg,'max_seconds':10900,'completion':proof,'assembler_source_sha256':sha(__file__)}
            or result != {'completed':True,'generations':403,'runtime':failed['runtime'],'elapsed_seconds':failed['elapsed_seconds']+completed['elapsed_seconds'],
                'gpu_count':1,'gpu':'0','completion':proof,'assembler_source_sha256':sha(__file__),
                'registration_sha256':sha(output / 'registration.json'),'predictions_sha256':sha(output / 'predictions.jsonl')}
            or (output / 'predictions.jsonl').read_bytes() != b''.join(prefix+suffix)):
        raise ValueError('assembled segment hashes/settings/ordered coverage differ')
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__); mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true'); mode.add_argument('--execute-gpu',action='store_true'); mode.add_argument('--assemble',action='store_true')
    p.add_argument('--original',default='outputs/qwen38-qasper-final-base-01'); p.add_argument('--output'); p.add_argument('--run'); p.add_argument('--amendment')
    p.add_argument('--gpu',default='0'); p.add_argument('--max-seconds',type=int,default=900); a = p.parse_args()
    if a.prepare:
        if not a.amendment or not a.output: p.error('--prepare requires --amendment and --output')
        result = prepare(ROOT/a.original, ROOT/a.output, ROOT/a.amendment, a.gpu, a.max_seconds)
    elif a.execute_gpu:
        if not a.run: p.error('--run required')
        result = execute(ROOT/a.run)
    else:
        if not a.run or not a.output: p.error('--assemble requires --run and fresh --output')
        result = assemble(ROOT/a.run, ROOT/a.output)
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
