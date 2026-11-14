"""One exact procedural replay. No grading, repairs, alternate inputs or retries."""
import argparse
from contextlib import contextmanager
import hashlib
import os
import platform
import shutil
import time

from scripts import xstest_baseline as parent
from scripts.batched_generation import generate_batched, left_pad, render_prompt
from scripts.common import path, read_json
from scripts.save_nf4 import stream_hash
from scripts.tone_steering import committed, load_model

SOURCES = ('scripts/xstest_replay.py', 'configs/xstest_replay.json', 'scripts/xstest_replay_runbook.md')
RUN = 'outputs/xstest-replay-01'
save = parent.save


def utf8_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def guard(run):
    if run.resolve() != path(RUN).resolve():
        raise ValueError('only the new replay directory may be written')


def settings(cfg):
    expected = read_json('configs/xstest_replay.json')
    if cfg != expected or cfg['run'] != RUN or cfg['parent'] != 'outputs/xstest-baseline-01':
        raise ValueError('fixed replay config differs')
    return path(cfg['parent'])


def derived_batches(values, pad_id):
    """Parent intent reconstructed from saved unpadded inputs, not old tensors."""
    result = []
    for start in range(0, len(values), 8):
        members = values[start:start + 8]
        padded = left_pad([r['input_ids'] for r in members], pad_id)
        ids = [r['id'] for r in members]
        width = len(padded['input_ids'][0])
        for position, row in enumerate(members):
            if (row['batch_index'], row['batch_position'], row['batch_ids'], row['padded_prompt_width']) != (len(result), position, ids, width):
                raise ValueError('parent batch layout differs from derived intent')
        result.append({'batch_index': len(result), 'ids': ids, 'padded_prompt_width': width, **padded})
    return result


@contextmanager
def capture_prefills(model, expected):
    """Observe actual forward inputs only; never replace an activation or argument."""
    captured = []

    def hook(module, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        if ids is None:
            raise ValueError('input_ids missing from model forward')
        if ids.shape[1] == 1:  # Cached decoding; every registered prompt is longer.
            return
        index = len(captured)
        if index >= len(expected):
            raise ValueError('extra prefill call')
        mask = kwargs.get('attention_mask')
        actual = {'input_ids': ids.detach().cpu().tolist(),
                  'attention_mask': mask.detach().cpu().tolist() if mask is not None else None}
        captured.append({'batch_index': index, **actual})
        if any(actual[k] != expected[index][k] for k in actual):
            raise ValueError('actual left padding/attention mask differs from derived parent intent')

    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield captured
        if len(captured) != len(expected):
            raise ValueError('missing prefill calls')
    finally:
        handle.remove()


def first_difference(a, b):
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b)) if len(a) != len(b) else None


def compare(rows, old, new, refusal_ids, reviewed_ids):
    if len(rows) != len(old) or len(old) != len(new):
        raise ValueError('incomplete replay; no full comparison possible')
    differences = []
    fields = ('id', 'input_ids', 'generated_ids', 'batch_index', 'batch_position', 'batch_ids',
              'padded_prompt_width', 'ended_eos', 'hit_token_limit')
    for row, left, right in zip(rows, old, new):
        changed = [k for k in fields if left[k] != right[k]]
        if row['id'] != left['id']:
            raise ValueError('parent source order differs')
        # New generation uses this exact immutable user field; it is never rewritten.
        if right.get('user_utf8_sha256') != utf8_hash(row['user']):
            changed.append('user_utf8')
        for k in ('rendered_prompt', 'output'):
            if left[k].encode('utf-8') != right[k].encode('utf-8'):
                changed.append(k + '_utf8')
        if changed:
            differences.append({'id': row['id'], 'fields': changed,
                'first_differing_generated_token_position': first_difference(left['generated_ids'], right['generated_ids']),
                'parent_generated_length': len(left['generated_ids']), 'replay_generated_length': len(right['generated_ids'])})
    mismatch_ids = {r['id'] for r in differences}
    return {'all_match': not differences, 'compared_items': len(rows), 'mismatched_items': len(differences),
        'mismatches_among_parent_17_refusals': len(mismatch_ids & set(refusal_ids)),
        'mismatches_among_reviewed_19': len(mismatch_ids & set(reviewed_ids)),
        'differences': differences, 'parent_labels_reused_by_reference_only': not differences,
        'new_label_evidence': False, 'new_refusal_rate_computed': False,
        'stop_rule': '4B closed; no rerun, regrading, higher cap or follow-up sweep'}


def environment_comparison(actual, original):
    return {k: {'parent': original.get(k), 'replay': v,
                'comparison': 'unavailable' if k not in original else ('match' if original[k] == v else 'different')}
            for k, v in actual.items()}


def prepare(args):
    cfg = read_json(args.config); old_run = settings(cfg); run = path(RUN); guard(run)
    if run.exists():
        raise FileExistsError('fresh replay directory required')
    old_cfg, manifest = parent.verify(old_run, models=True)
    if stream_hash(old_run/'manifest.json') != cfg['parent_manifest_sha256'] or stream_hash(old_run/'generation/predictions.json') != cfg['parent_predictions_sha256']:
        raise ValueError('parent pins differ')
    rows, values = parent.predictions(old_run); ratings, _ = parent.accepted_grades(old_run)
    measured = parent.metrics(rows, values, ratings)
    refused = [r['id'] for r in measured['items'] if r['grade'] in ('P', 'I')]
    audit = path(cfg['audit_directory']); packet = read_json(audit/'packet.json'); review = read_json(audit/'review.json')
    if review['source_packet_sha256'] != stream_hash(audit/'packet.json') or review['source_manifest_sha256'] != stream_hash(audit/'manifest.json'):
        raise ValueError('audit bindings differ')
    reviewed = [r['id'] for r in packet['items']]
    if len(rows) != 250 or len(refused) != 17 or len(set(reviewed)) != 19:
        raise ValueError('fixed parent/intersection counts differ')
    external = dict(manifest['external_sha256']); external.update(manifest['sources'])
    for folder in (old_run, audit):
        external.update({str(p): stream_hash(p) for p in folder.rglob('*') if p.is_file()})
    for name in (cfg['protocol'], cfg['design_review'], cfg['parent_result'], cfg['audit_result']):
        external[str(path(name))] = stream_hash(path(name))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(manifest['snapshot'], local_files_only=True, trust_remote_code=False)
    frozen = read_json(old_run/'inputs/rendered_inputs.json')
    for row, value, expected in zip(rows, values, frozen):
        prompt = render_prompt(tokenizer, {'user': row['user'], 'system': old_cfg['system']})
        if prompt.encode('utf-8') != expected['rendered_prompt'].encode('utf-8') or prompt != value['rendered_prompt'] or tokenizer(prompt, add_special_tokens=False)['input_ids'] != value['input_ids'] or expected['input_ids'] != value['input_ids']:
            raise ValueError('CPU tokenizer replay differs')
    batches = derived_batches(values, tokenizer.pad_token_id)
    if len(batches) != 32 or any(len(r['input_ids']) <= 1 for r in values):
        raise ValueError('fixed prefill layout differs')
    parent.check_hashes(external); run.mkdir(mode=0o700)
    save(run/'config.json', cfg)
    save(run/'inputs/derived_batches.json', batches)
    save(run/'inputs/intersections.json', {'parent_refusal_ids': refused, 'reviewed_ids': reviewed})
    for name in (*SOURCES, cfg['protocol'], cfg['design_review']):
        target = run/'source'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path(name), target)
    save(run/'manifest.json', {'files': {str(p.relative_to(run)): stream_hash(p) for p in run.rglob('*') if p.is_file()},
        'sources': {n: stream_hash(path(n)) for n in SOURCES}, 'external_sha256': external,
        'model_sha256': manifest['model_sha256'], 'packages': manifest['packages'], 'snapshot': manifest['snapshot'],
        'parent_protected_external_count': len(manifest['external_sha256']),
        'mask_reference': 'derived parent intent; old actual tensors not stored', 'no_model_or_grader_calls': True})
    parent.check_hashes(external)
    print('CPU replay snapshot ready; no target model or grader calls.')


def verify(run, models=False):
    guard(run); cfg = read_json(run/'config.json'); settings(cfg); m = read_json(run/'manifest.json')
    parent.check_hashes({str(run/n): h for n, h in m['files'].items()})
    parent.check_hashes(m['sources']); parent.check_hashes(m['external_sha256'])
    if models:
        parent.check_hashes(m['model_sha256'])
    old_cfg, old_m = parent.verify(path(cfg['parent']))
    if m['packages'] != old_m['packages']:
        raise ValueError('package pins differ')
    return cfg, m, old_cfg


def closed(run, cause, exc=None):
    if not (run/'closed.json').exists():
        save(run/'closed.json', {'stop_cause': cause, 'error_type': type(exc).__name__ if exc else None})


def generate(args):
    if not args.execute_gpu:
        raise ValueError('separate root GPU release and --execute-gpu required')
    run = path(RUN); guard(run)
    if (run/'closed.json').exists() or (run/'generation').exists():
        raise RuntimeError('one attempt only; replay already started or closed')
    try:
        cfg, m, old_cfg = verify(run, models=True)
        for name in (*SOURCES, cfg['protocol'], cfg['design_review'], args.registration):
            committed(name)
        if read_json(args.registration)['manifest_sha256'] != stream_hash(run/'manifest.json'):
            raise ValueError('registration differs')
    except BaseException as exc:
        closed(run, 'preflight_fatal', exc); raise
    # A busy GPU is a wait condition, not a launched attempt. Never disturb jobs.
    resource = parent.b.resource_check(old_cfg)
    out = run/'generation'; out.mkdir(mode=0o700, exist_ok=False)
    save(out/'load_started.json', {'manifest_sha256': stream_hash(run/'manifest.json'), **resource})
    try:
        started = time.perf_counter(); model, tokenizer = load_model(old_cfg, m, 'bf16')
        import torch
        rows, old = parent.predictions(path(cfg['parent']))
        expected = read_json(run/'inputs/derived_batches.json')
        save(out/'attempt_started.json', {'single_attempt_consumed': True,
            'boundary': 'immediately before first generate_batched call', 'seed': 42})
        with capture_prefills(model, expected) as captured:
            generated = generate_batched(model, tokenizer, [{'id': r['id'], 'user': r['user'], 'system': old_cfg['system']} for r in rows],
                batch_size=8, max_new_tokens=256, seed=42, phase='single XSTest-safe replay')
        for row, value in zip(rows, generated['predictions']):
            value['output'] = parent.decode_body(tokenizer, value)
            value['user_utf8_sha256'] = utf8_hash(row['user'])
        generated['settings'].update(requested_temperature=0, temperature_argument=None,
            text_decode='generated body unchanged; terminal EOS omitted; no whitespace/marker stripping')
        save(out/'actual_prefills.json', captured); save(out/'predictions.json', generated)
        original = read_json(path(cfg['parent'])/'generation/run.json')
        original.update(read_json(path(cfg['parent'])/'manifest.json')['packages'])
        original.update({k: old_cfg[k] for k in ('dtype', 'attn_implementation', 'enable_thinking', 'seed', 'batch_size', 'max_new_tokens')})
        actual = {**resource, 'python': platform.python_version(), **m['packages'], 'cuda_runtime': torch.version.cuda,
            'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'dtype': old_cfg['dtype'],
            **{k: old_cfg[k] for k in ('attn_implementation', 'enable_thinking', 'seed', 'batch_size', 'max_new_tokens')},
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32}
        save(out/'run.json', {'completed': True, 'manifest_sha256': stream_hash(run/'manifest.json'),
            'prediction_sha256': stream_hash(out/'predictions.json'), 'actual_prefill_sha256': stream_hash(out/'actual_prefills.json'),
            'environment': actual, 'parent_environment_comparison': environment_comparison(actual, original),
            'elapsed_wall_seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'memory_scope': 'PyTorch allocator including model, not total VRAM'})
        verify(run)
        save(run/'post_integrity.json', {'original_hashes_unchanged': True,
            'verified_before_equality_result': True})
        ids = read_json(run/'inputs/intersections.json')
        comparison = compare(rows, old, generated['predictions'], ids['parent_refusal_ids'], ids['reviewed_ids'])
        save(run/'comparison.json', comparison)
        closed(run, 'exact_match' if comparison['all_match'] else 'mismatch_stop')
        print({k: v for k, v in comparison.items() if k != 'differences'})
    except BaseException as exc:
        try:
            verify(run)
        except BaseException as integrity_exc:
            closed(run, 'integrity_failure', integrity_exc)
            save(run/'post_hash_failure.json', {'error_type': type(integrity_exc).__name__}); raise
        closed(run, 'execution_fatal', exc); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'verify', 'generate'))
    parser.add_argument('--config', default='configs/xstest_replay.json')
    parser.add_argument('--registration', default='results/protocols/xstest-replay-01.json')
    parser.add_argument('--execute-gpu', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare': prepare(args)
    elif args.stage == 'verify': verify(path(RUN), models=True); print('Replay bindings verified.')
    else: generate(args)


if __name__ == '__main__':
    main()
