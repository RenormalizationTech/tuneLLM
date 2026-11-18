"""Synthetic CPU custody checks; the evaluator is stubbed, never GPU-executed."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import qwen38_complete_science as c

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + '\n')


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b''.join((json.dumps(row, ensure_ascii=False) + '\n').encode() for row in rows))


class Tokenizer:
    eos_token_id, pad_token_id = 1, 0
    def __call__(self, text, **kwargs): return {'input_ids': [ord(x) + 10 for x in text]}
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False and kwargs['add_generation_prompt'] is True
        return 'user:' + messages[0]['content'] + '\nassistant:'
    def decode(self, tokens, **kwargs): return ''.join(chr(x - 10) for x in tokens if x != 1)


def prediction(row, tokenizer):
    prompt = c.runner.render(tokenizer, row, 'base')
    text = '{"answer":"synthetic","unanswerable":false,"evidence":[]}'
    return {'id': row['id'], 'paper_id': row['paper_id'], 'input_ids': tokenizer(prompt)['input_ids'],
        'rendered_prompt': prompt, 'generated_ids': tokenizer(text)['input_ids'] + [1], 'output': text,
        'ended_with_eos': True, 'hit_token_limit': False, 'seconds': .5}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'versions', lambda: {'synthetic': 'CPU-only'})
    original = tmp_path / 'outputs/qwen38-qasper-final-base-01'
    full = tmp_path / 'outputs/prepared/test.jsonl'
    records = [{'id': f'q{i}', 'paper_id': f'paper{i // 5}', 'user': f'Invented example {i}: α'} for i in range(403)]
    lines(full, records)
    tokenizer = Tokenizer()
    cfg = {'task':'qasper','template':'base','template_text':'','purpose':'final','max_new_tokens':512,'max_context':8192,
        'max_seconds':10000,'seed':42,'batch_size':1,'cached_generation':True,'enable_thinking':False,'do_sample':False,
        'adapter':None,'adapter_files':{},'gpu':'0','base_preparation':'same PEFT floating casts and rank16 wrapper; disabled adapter for unadapted arm',
        'versions':c.versions(),'input_path':str(full),'input_sha256':c.sha(full),'ids':[r['id'] for r in records]}
    cfg['source_sha256'] = {}
    for name in ['scripts/qwen38_eval_science.py','scripts/qwen38_science_runtime.py','scripts/qwen38_nf4_smoke.py']:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((ROOT / name).read_bytes())
        cfg['source_sha256'][name] = c.sha(path)
    token_file = tmp_path / 'artifacts/qwen38-27b-nf4-01/tokenizer.json'
    token_file.parent.mkdir(parents=True); token_file.write_text('{}')
    manifest = tmp_path / 'results/qwen38_27b_nf4_artifact_01.json'
    write(manifest, {'files': {'tokenizer.json': {'bytes':2, 'sha256':c.sha(token_file)}}})
    cfg['base_manifest_sha256'] = c.sha(manifest)
    write(tmp_path / 'results/protocols/qwen38-qasper-evaluation-01.json', {'final_input_sha256': c.sha(full)})
    write(original / 'registration.json', cfg)
    result = {'completed':False,'generations':395,'error':c.ERROR,'elapsed_seconds':10034.74296683073,
              'registration_sha256':c.sha(original / 'registration.json'),'runtime':{'casts':'identical synthetic'}}
    write(original / 'result.json', result)
    lines(original / 'predictions.jsonl', [prediction(r, tokenizer) for r in records[:395]])
    amendment = tmp_path / c.AMENDMENT
    write(amendment, {'registered_at_utc':'2000-01-01T00:00:00+00:00','original_run':str(original),
        'original_phase_cap_seconds':10000,'original_measured_seconds':result['elapsed_seconds'],'original_completed':False,'original_error':c.ERROR,
        'original_registration_sha256':c.sha(original / 'registration.json'),'original_result_sha256':c.sha(original / 'result.json'),
        'original_predictions_sha256':c.sha(original / 'predictions.jsonl'),'original_full_input_sha256':c.sha(full),
        'full_denominator':403,'completed_prefix':395,'missing_suffix_ids':[r['id'] for r in records[395:]],
        'additional_phase_cap_seconds':900,'additional_parent_reservation':{'name':'qasper-final-base-completion','gpu_hours':.25,'pool':'qasper_eval'},
        'parent_pool_limit_gpu_hours':12,'parent_accounted_including_remaining_reservations_gpu_hours':11.775156228233504,
        'one_attempt_only':True,'unchanged':c.UNCHANGED,'quality_outputs_or_scores_used_for_amendment':False})
    return {'root':tmp_path,'original':original,'output':tmp_path / 'outputs/completion','amendment':amendment,
            'tokenizer':tokenizer,'gpu':'0'}


def prepare(f): return c.prepare(**f)


def complete(f):
    prepare(f)
    run = f['output']; cfg = c.load(f['original'] / 'registration.json'); reg = c.load(run / 'registration.json')
    write(run / 'attempt.json', {'registration_sha256':c.sha(run / 'registration.json'),'started_at_utc':'2026-11-18T03:00:00+00:00'})
    generation = run / 'generation'
    gc = {**cfg,'input_path':str(run / 'remaining.jsonl'),'input_sha256':c.sha(run / 'remaining.jsonl'),'ids':reg['suffix_ids'],'max_seconds':900}
    write(generation / 'registration.json', gc)
    records = c.jsonl(run / 'remaining.jsonl')[1]
    lines(generation / 'predictions.jsonl', [prediction(r, f['tokenizer']) for r in records])
    write(generation / 'result.json', {'completed':True,'generations':8,'registration_sha256':c.sha(generation / 'registration.json'),
        'elapsed_seconds':200.,'runtime':c.load(f['original'] / 'result.json')['runtime']})
    return run


def test_exact_suffix_and_byte_preserving_assembly_with_failed_receipt(fixture):
    f = fixture; before = {p.name:p.read_bytes() for p in f['original'].iterdir()}
    run = complete(f)
    assert (run / 'remaining.jsonl').read_bytes() == b''.join(c.jsonl(c.load(f['original'] / 'registration.json')['input_path'])[0][395:])
    out = f['root'] / 'outputs/assembled'
    result = c.assemble(run, out, root=f['root'], tokenizer=f['tokenizer'])
    proof = c.validate_assembly(out, root=f['root'], tokenizer=f['tokenizer'])
    assert c.validate_assembly_hashes(out, root=f['root']) == proof
    assert result['generations'] == 403 and not result['GPU_started']
    assert proof['prefix_count'] == 395 and proof['suffix_count'] == 8 and not proof['original_phase_cap_met']
    assert proof['original_failed_result_sha256'] == c.sha(f['original'] / 'result.json')
    assert proof['assembler_source_sha256'] == c.sha(c.__file__)
    assert all(p.read_bytes() == before[p.name] for p in f['original'].iterdir())
    expected = before['predictions.jsonl'] + (run / 'generation/predictions.jsonl').read_bytes()
    assert (out / 'predictions.jsonl').read_bytes() == expected
    reg, status = c.load(out / 'registration.json'), c.load(out / 'result.json')
    assert reg['max_seconds'] == 10900 and reg['input_sha256'] == c.load(f['amendment'])['original_full_input_sha256']
    assert status['elapsed_seconds'] == 10234.74296683073 and status['gpu_count'] == 1 and status['gpu'] == '0'
    assert 'not an uninterrupted original-cap run' in status['completion']['interpretation']
    with pytest.raises(FileExistsError): c.assemble(run, out, root=f['root'], tokenizer=f['tokenizer'])
    with pytest.raises(FileExistsError): prepare(f)


@pytest.mark.parametrize('kind',['duplicate','hole','prompt','paper','tokens','source','settings','hash','amendment'])
def test_original_prefix_or_source_settings_amendment_tamper_rejected(fixture, kind):
    f = fixture; original = f['original']; a = c.load(f['amendment'])
    if kind in ('duplicate','hole','prompt','paper','tokens'):
        rows = c.jsonl(original / 'predictions.jsonl')[1]
        if kind == 'duplicate': rows[-1] = rows[0]
        elif kind == 'hole': rows[100],rows[101] = rows[101],rows[100]
        elif kind == 'prompt': rows[0]['rendered_prompt'] += ' changed'
        elif kind == 'paper': rows[0]['paper_id'] = 'other'
        else: rows[0]['input_ids'][0] += 1
        lines(original / 'predictions.jsonl', rows)
        a['original_predictions_sha256'] = c.sha(original / 'predictions.jsonl')
    elif kind == 'source': (f['root'] / 'scripts/qwen38_science_runtime.py').write_text('# changed')
    elif kind == 'settings':
        cfg = c.load(original / 'registration.json'); cfg['do_sample'] = True; write(original / 'registration.json', cfg)
        a['original_registration_sha256'] = c.sha(original / 'registration.json')
    elif kind == 'hash': (original / 'result.json').write_text('{}')
    else: a['unchanged'] = []
    write(f['amendment'], a)
    with pytest.raises(ValueError): prepare(f)
    assert not f['output'].exists()


@pytest.mark.parametrize('gpu,cap',[('1',900),('0',901)])
def test_prepare_requires_original_gpu0_and_exact900_cap(fixture,gpu,cap):
    fixture['gpu'] = gpu
    with pytest.raises(ValueError,match='GPU0'): c.prepare(**fixture,max_seconds=cap)


@pytest.mark.parametrize('kind',['extra','wrong','incomplete','duplicate','settings','runtime','original','source','overcap'])
def test_completion_incomplete_or_mismatched_segments_rejected(fixture,kind):
    f = fixture; run = complete(f); gen = run / 'generation'
    if kind in ('extra','wrong','duplicate'):
        rows = c.jsonl(gen / 'predictions.jsonl')[1]
        if kind == 'extra': rows.append(rows[0])
        elif kind == 'wrong': rows[0]['id'] = 'q0'
        else: rows[-1] = rows[0]
        lines(gen / 'predictions.jsonl',rows)
    elif kind == 'overcap':
        result = c.load(gen / 'result.json'); result['elapsed_seconds'] = 901.; write(gen / 'result.json',result)
    elif kind in ('incomplete','runtime'):
        result = c.load(gen / 'result.json'); result['completed' if kind == 'incomplete' else 'runtime'] = False if kind == 'incomplete' else {'casts':'changed'}
        write(gen / 'result.json',result)
    elif kind == 'settings':
        cfg = c.load(gen / 'registration.json'); cfg['max_new_tokens'] = 256; write(gen / 'registration.json',cfg)
    elif kind == 'original':
        with (f['original'] / 'predictions.jsonl').open('ab') as stream: stream.write(b'\n')
    else: (f['root'] / 'scripts/qwen38_eval_science.py').write_text('# changed')
    with pytest.raises(ValueError): c.assemble(run,f['root'] / 'outputs/assembled',root=f['root'],tokenizer=f['tokenizer'])


def test_delegate_unchanged_runner_once_and_block_retry_after_failure(fixture,monkeypatch):
    f=fixture; prepare(f); calls=[]
    def stub(args):
        calls.append(vars(args))
        assert args.max_seconds==900 and args.max_new_tokens==512 and args.gpu=='0'
        assert args.output==str(f['output'] / 'generation') and args.input==str(f['output'] / 'remaining.jsonl')
        raise TimeoutError('synthetic runner failure; no GPU')
    monkeypatch.setattr(c.runner,'execute',stub)
    with pytest.raises(TimeoutError): c.execute(f['output'],root=f['root'],tokenizer=f['tokenizer'])
    with pytest.raises(FileExistsError): c.execute(f['output'],root=f['root'],tokenizer=f['tokenizer'])
    assert len(calls)==1


@pytest.mark.parametrize('change',['remaining-extra','remaining-order','helper-source'])
def test_completion_registration_guards_before_execution(fixture, change):
    f=fixture; prepare(f); run=f['output']
    if change=='helper-source':
        reg=c.load(run/'registration.json'); reg['source_sha256']='0'*64; write(run/'registration.json',reg)
    else:
        records=c.jsonl(run/'remaining.jsonl')[1]
        if change=='remaining-extra': records.append(records[0])
        else: records.reverse()
        lines(run/'remaining.jsonl',records)
    with pytest.raises(ValueError): c.validate_run(run,root=f['root'],tokenizer=f['tokenizer'])


def test_hash_only_helper_rejects_forged_assembly_proof(fixture):
    f=fixture; run=complete(f); out=f['root']/'outputs/assembled'
    c.assemble(run,out,root=f['root'],tokenizer=f['tokenizer'])
    reg=c.load(out/'registration.json'); reg['completion']['segments'][0]['completed']=True
    write(out/'registration.json',reg)
    result=c.load(out/'result.json'); result['completion']=reg['completion']; result['registration_sha256']=c.sha(out/'registration.json')
    write(out/'result.json',result)
    with pytest.raises(ValueError): c.validate_assembly_hashes(out,root=f['root'])
