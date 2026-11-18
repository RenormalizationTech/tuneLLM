"""CPU registration by default; explicit bounded GPU diagnostic, never quality evaluation.

Four invented prompts cover the longest registered retention prompt, including MC.
Compare disabled/zero-active rank16 PEFT at batch1/batch4, <=32 tokens per item.
No optimization, checkpoint export, quality answers, prompt selection or retry.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts import qwen38_retention as retention

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'qwen38-retention-batch-preflight-v1'
CONDITIONS = [('disabled', 1), ('disabled', 4), ('zero_active', 1), ('zero_active', 4)]
SOURCE = 'scripts/qwen38_retention_batch_preflight.py'


def synthetic_prompts(tokenizer, minimum_long_tokens):
    """Fixed content/algorithm; no question text or reference answers consumed.

    Long prompts are identical in both adapter conditions, and reach/exceed the
    maximum real frozen generation-prompt token length without exceeding8192.
    """
    if type(minimum_long_tokens) is not int or not 1 <= minimum_long_tokens <= 8160:
        raise ValueError('retention prompt length must leave room for32 generated tokens')
    def rendered(user):
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False,
            return_dict=False, add_generation_prompt=True, enable_thinking=False)
        return prompt, tokenizer(prompt, add_special_tokens=False)['input_ids']
    prefix = 'This is an invented timing example. Continue with a short neutral description.\n'
    unit = 'An imaginary blue marker rests beside an imaginary green cube. '
    rows = []
    for i, target in enumerate((1, 64, minimum_long_tokens, minimum_long_tokens)):
        case = f'Timing case {i}. '
        repeats = 0
        prompt, ids = rendered(prefix + case + ('Short timing case.' if i < 2 else 'Long timing case.'))
        # Fixed linear growth avoids assumptions about tokenizer merge boundaries.
        while len(ids) < target:
            repeats += 1
            prompt, ids = rendered(prefix + case + unit * repeats)
            if repeats > 8192:
                raise ValueError('synthetic length construction did not converge')
        if len(ids) + 32 > 8192:
            raise ValueError('synthetic whole-text prompt would exceed8192; no silent truncation')
        rows.append({'id': f'synthetic-{i}', 'task': 'synthetic', 'prompt': prompt, 'input_ids': ids, 'max_new_tokens': 32})
    if max(len(r['input_ids']) for r in rows) < minimum_long_tokens:
        raise ValueError('synthetic prompts fail to cover retention length')
    return rows


def plans(rows, batch_size):
    """Use retention planning/padding, changing only the synthetic diagnostic cap."""
    # The planner's task cap validation stays unchanged for real retention.
    from scripts.batched_generation import left_pad
    groups = []
    for start in range(0, len(rows), batch_size):
        members = rows[start:start + batch_size]
        lengths = [len(r['input_ids']) for r in members]
        padded = left_pad([r['input_ids'] for r in members], 0)
        width = len(padded['input_ids'][0])
        groups.append({'batch_index': len(groups), 'group_id': f'synthetic:{len(groups):04d}', 'task': 'synthetic',
            'ids': [r['id'] for r in members], 'prompt_lengths': lengths, 'padding_lengths': [width - n for n in lengths],
            'padded_prompt_width': width, 'max_new_tokens': 32})
    return groups


def prepare(args, tokenizer=None):
    from scripts.qwen38_science_runtime import versions
    out, reference, base = map(lambda p: Path(p).resolve(), (args.output, args.retention_run, args.base))
    if out.exists(): raise FileExistsError('fresh preflight output required')
    if not args.gpu.isdecimal() or type(args.max_seconds) is not int or not 0 < args.max_seconds <= 600:
        raise ValueError('single GPU and maximum600seconds required')
    registration = json.loads((reference / 'registration.json').read_text())
    prepared = json.loads((reference / 'prepared.json').read_text())
    binding = registration['comparison_binding']
    if (retention.sha(reference / 'prepared.json') != registration['prepared_copy_sha256']
            or retention.digest(prepared) != binding['prepared_sha256']):
        raise ValueError('reference retention prepared token binding differs')
    lengths = [len(r['input_ids']) for r in prepared if r['task'] in retention.CAPS]
    if not lengths: raise ValueError('frozen retention generation prompts required')
    maximum = max(len(r['input_ids']) for r in prepared)
    manifest = ROOT / 'results/qwen38_27b_nf4_artifact_01.json'
    if retention.sha(manifest) != binding['base_manifest_sha256']:
        raise ValueError('reference retention base differs')
    if tokenizer is None:
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    pad, eos = tokenizer.pad_token_id, tokenizer.eos_token_id
    if type(pad) is not int or type(eos) is not int or min(pad, eos) < 0 or pad == eos:
        raise ValueError('single integer EOS and distinct PAD required')
    rows = synthetic_prompts(tokenizer, maximum)
    sources = {name: retention.sha(ROOT / name) for name in [*retention.SOURCES, SOURCE]}
    tokenizer_hashes = {p.name: retention.sha(p) for p in base.iterdir() if p.is_file() and (p.name.startswith('tokenizer') or p.name == 'chat_template.jinja')}
    cfg = {'version': VERSION, 'gpu': args.gpu, 'max_seconds': args.max_seconds, 'base': str(base),
        'base_manifest': str(manifest), 'base_manifest_sha256': retention.sha(manifest), 'source_sha256': sources,
        'python_executable': sys.executable, 'versions': versions(), 'tokenizer_sha256': tokenizer_hashes,
        'reference_registration': str(reference / 'registration.json'), 'reference_registration_sha256': retention.sha(reference / 'registration.json'),
        'reference_prepared': str(reference / 'prepared.json'), 'reference_prepared_sha256': retention.sha(reference / 'prepared.json'),
        'maximum_frozen_generation_prompt_tokens': max(lengths), 'maximum_frozen_prompt_tokens': maximum,
        'pad_token_id': pad, 'eos_token_id': eos,
        'prompts_sha256': retention.digest(rows), 'conditions': [{'adapter': mode, 'batch_size': size, 'groups': plans(rows, size)} for mode, size in CONDITIONS],
        'generation': {'max_new_tokens': 32, 'greedy': True, 'cached': True, 'thinking': False, 'dtype': 'BF16 autocast', 'logits_to_keep': 1, 'seed': 42},
        'scope': 'synthetic mechanics/timing only; token agreement descriptive; no quality gate or training/prompt selection',
        'zero_adapter': 'All rank16 LoRA tensors set to zero in memory; active forwards only, no optimization or saved tensors.'}
    out.mkdir(parents=True)
    retention.write_new(out / 'prompts.json', rows)
    cfg['prompts_file_sha256'] = retention.sha(out / 'prompts.json')
    for name in sources:
        destination = out / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    (out / 'source/scripts/__init__.py').write_text('')
    retention.write_new(out / 'registration.json', cfg)
    return {'prepared': str(out), 'gpu_launched': False, 'conditions': 4, 'prompts': 4,
            'maximum_frozen_prompt_tokens': maximum, 'registration_sha256': retention.sha(out / 'registration.json')}


def validate(run):
    from scripts.qwen38_science_runtime import versions
    run = Path(run).resolve()
    cfg = json.loads((run / 'registration.json').read_text())
    rows = json.loads((run / 'prompts.json').read_text())
    expected_conditions = [{'adapter': mode, 'batch_size': size, 'groups': plans(rows, size)} for mode, size in CONDITIONS]
    if (cfg['version'] != VERSION or cfg['conditions'] != expected_conditions or len(rows) != 4
            or cfg['prompts_sha256'] != retention.digest(rows) or cfg['prompts_file_sha256'] != retention.sha(run / 'prompts.json')
            or cfg['python_executable'] != sys.executable or cfg['versions'] != versions()
            or not cfg['gpu'].isdecimal() or not 0 < cfg['max_seconds'] <= 600):
        raise ValueError('preflight registration/prompts/software/conditions differ')
    if cfg['generation'] != {'max_new_tokens': 32, 'greedy': True, 'cached': True, 'thinking': False, 'dtype': 'BF16 autocast', 'logits_to_keep': 1, 'seed': 42}:
        raise ValueError('preflight fixed generation settings differ')
    synthetic_prompts_from_ids_check(rows, cfg)
    for name, expected in cfg['source_sha256'].items():
        if retention.sha(ROOT / name) != expected or retention.sha(run / 'source' / name) != expected:
            raise ValueError('preflight source changed')
    for path, expected in [(cfg['base_manifest'], cfg['base_manifest_sha256']), (cfg['reference_registration'], cfg['reference_registration_sha256']),
                           (cfg['reference_prepared'], cfg['reference_prepared_sha256'])]:
        if retention.sha(path) != expected: raise ValueError('preflight bound input changed')
    for name, expected in cfg['tokenizer_sha256'].items():
        if retention.sha(Path(cfg['base']) / name) != expected: raise ValueError('preflight tokenizer changed')
    return cfg, rows


def synthetic_prompts_from_ids_check(rows, cfg):
    if ([r['id'] for r in rows] != [f'synthetic-{i}' for i in range(4)]
            or any(r['task'] != 'synthetic' or r['max_new_tokens'] != 32 or not r['input_ids']
                   or len(r['input_ids']) + 32 > 8192 or any(type(t) is not int or t < 0 for t in r['input_ids']) for r in rows)
            or max(len(r['input_ids']) for r in rows) < cfg['maximum_frozen_prompt_tokens']):
        raise ValueError('invalid synthetic token IDs/length coverage')
    return rows


def worker(run):
    """Only the explicit watchdog-launched GPU child calls this."""
    cfg, rows = validate(run)
    os.environ.update(CUDA_VISIBLE_DEVICES=cfg['gpu'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    artifact = json.loads(Path(cfg['base_manifest']).read_text())
    for name, info in artifact['files'].items():
        path = Path(cfg['base']) / name
        if path.stat().st_size != info['bytes'] or retention.sha(path) != info['sha256']:
            raise ValueError('preflight base artifact changed')
    import torch
    from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, set_seed
    from scripts.qwen38_science_runtime import load_runtime
    started = time.monotonic()
    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return time.monotonic() - started >= cfg['max_seconds']
    def deadline():
        if time.monotonic() - started >= cfg['max_seconds']: raise TimeoutError('preflight600second cap')
    set_seed(42)
    model, runtime = load_runtime(rank=16, alpha=32, dropout=.05, base=cfg['base'])
    load_seconds = time.monotonic() - started
    tokenizer = AutoTokenizer.from_pretrained(cfg['base'], local_files_only=True, trust_remote_code=False)
    if (tokenizer.pad_token_id, tokenizer.eos_token_id) != (cfg['pad_token_id'], cfg['eos_token_id']) or synthetic_prompts(tokenizer, cfg['maximum_frozen_prompt_tokens']) != rows:
        raise ValueError('loaded tokenizer differs from synthetic prompt registration')
    zero_tensors = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.zero_(); zero_tensors += 1
    if not zero_tensors: raise ValueError('rank16 PEFT wrapper absent')
    model.eval()
    result = {'completed': False, 'registration_sha256': retention.sha(run / 'registration.json'), 'load_seconds': load_seconds,
              'runtime': {k: v for k, v in runtime.items() if k != 'targets'}, 'zero_adapter_tensors': zero_tensors,
              'conditions': [], 'device_name': torch.cuda.get_device_name(0), 'agreement_scope': 'descriptive only; no quality/gate/prompt selection'}
    retention.write_status(run / 'result.json', result)
    lookup = {r['id']: r for r in rows}
    for condition in cfg['conditions']:
        deadline()
        predictions, batches = [], []
        wall = time.monotonic()
        current = {**condition, 'completed': False, 'predictions': predictions, 'batches': batches}
        result['conditions'].append(current)
        context = model.disable_adapter() if condition['adapter'] == 'disabled' else nullcontext()
        with context, torch.inference_mode():
            for plan in condition['groups']:
                deadline()
                predicted, batch = retention.generate_group(model, tokenizer, [lookup[k] for k in plan['ids']], plan, StoppingCriteriaList([Deadline()]))
                predictions.extend(predicted); batches.append(batch)
                retention.write_status(run / 'result.json', result)
                deadline()
        seconds = sum(b['batch_seconds'] for b in batches)
        count = sum(len(p['generated_ids']) for p in predictions)
        current.update({'completed': True,
            'generation_seconds': seconds, 'pipeline_seconds': time.monotonic() - wall, 'generated_tokens': count,
            'tokens_per_second': count / seconds, 'seconds_per_token': seconds / count,
            'peak_allocated_bytes': max(b['peak_allocated_bytes'] for b in batches),
            'peak_reserved_bytes': max(b['peak_reserved_bytes'] for b in batches)})
        retention.write_status(run / 'result.json', result)
    baseline = {p['id']: p['generated_ids'] for p in result['conditions'][0]['predictions']}
    result['token_agreement_vs_disabled_batch1'] = [{
        'adapter': c['adapter'], 'batch_size': c['batch_size'], 'exact_items': sum(p['generated_ids'] == baseline[p['id']] for p in c['predictions']),
        'items': 4, 'by_id': {p['id']: p['generated_ids'] == baseline[p['id']] for p in c['predictions']}} for c in result['conditions']]
    validate(run); deadline()
    result['completed'] = True
    result['worker_elapsed_seconds'] = time.monotonic() - started
    retention.write_status(run / 'result.json', result)
    return result


def execute(run):
    """Wall-clock watchdog includes load; at most one attempt, never automatic retry."""
    run = Path(run).resolve()
    if (run / 'result.json').exists() or (run / 'resource_usage.json').exists():
        raise FileExistsError('preflight already attempted; no retry')
    cfg, _ = validate(run)
    retention.write_new(run / 'result.json', {'completed': False, 'registration_sha256': retention.sha(run / 'registration.json'), 'started': True})
    started = time.monotonic()
    process, error = None, None
    try:
        command = [sys.executable, '-B', '-m', 'scripts.qwen38_retention_batch_preflight', '--execute-gpu', '--execute-worker', '--run', str(run)]
        process = subprocess.Popen(command, cwd=ROOT, start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=cfg['gpu'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                     QWEN38_PREFLIGHT_CHILD=retention.sha(run / 'registration.json')))
        code = process.wait(timeout=cfg['max_seconds'])
        if code: raise RuntimeError(f'preflight child exit{code}; no retry')
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL); process.wait()
        raise
    finally:
        elapsed = time.monotonic() - started
        result = json.loads((run / 'result.json').read_text())
        if error: result.update(completed=False, error=error)
        result['elapsed_seconds'] = elapsed
        retention.write_status(run / 'result.json', result)
        retention.write_new(run / 'resource_usage.json', {'gpu_count': 1, 'elapsed_seconds': elapsed, 'gpu_hours': elapsed / 3600,
            'cap_seconds': cfg['max_seconds'], 'cap_exceeded': elapsed > cfg['max_seconds'], 'completed': result.get('completed') is True,
            'parent_ledger_must_debit': True, 'registration_sha256': retention.sha(run / 'registration.json'),
            'result_sha256': retention.sha(run / 'result.json'), 'scope': 'synthetic batch preflight; not a quality arm'})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--execute-gpu', action='store_true')
    parser.add_argument('--execute-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run')
    parser.add_argument('--output')
    parser.add_argument('--retention-run', default=str(ROOT / 'outputs/qwen38-retention-base-03'))
    parser.add_argument('--base', default=str(ROOT / 'artifacts/qwen38-27b-nf4-01'))
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--max-seconds', type=int, default=600)
    args = parser.parse_args()
    if args.execute_gpu or args.execute_worker:
        if not args.run: parser.error('--run required')
        if args.execute_worker and (not args.execute_gpu or os.environ.get('QWEN38_PREFLIGHT_CHILD') != retention.sha(Path(args.run) / 'registration.json')):
            parser.error('GPU worker requires explicit GPU mode and the parent watchdog binding')
        result = worker(Path(args.run).resolve()) if args.execute_worker else execute(args.run)
    else:
        if not args.output: parser.error('--output required for CPU preparation')
        result = prepare(args)
    print(retention.canonical(result))


if __name__ == '__main__': main()
