"""Offline C6 closeout, separately registered from the frozen R0 runner.

Reads local JSON only; never imports the runner, loads a model, or calls a
judge. Output is a fresh immutable JSON file. Examples follow fixed diagnostic
rules and are descriptive; no analysis changes a selected strength or gate.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('baseline', 'prompted', 'selected', 'random')
TEST = ROOT / 'tests/test_qwen38_sarcasm_analyze.py'


def sha(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def finite_nonnegative(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('finite nonnegative ledger value required')
    return value


class Evidence:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.files = {}

    def read(self, name):
        file = self.run / name
        raw = file.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if name in self.files and self.files[name] != digest:
            raise ValueError(f'input changed during analysis: {name}')
        self.files[name] = digest
        return json.loads(raw)

    def exists(self, name):
        return (self.run / name).is_file()

    def verify(self):
        if any(sha(self.run/name) != digest for name, digest in self.files.items()):
            raise ValueError('input changed during analysis')


def pearson(pairs):
    n = len(pairs)
    if n < 2:
        return {'n': n, 'pearson': None, 'undefined_reason': 'fewer than two numeric pairs'}
    x, y = zip(*pairs)
    mx, my = sum(x)/n, sum(y)/n
    denominator = math.sqrt(sum((v-mx)**2 for v in x)*sum((v-my)**2 for v in y))
    if not denominator:
        return {'n': n, 'pearson': None, 'undefined_reason': 'zero variance'}
    return {'n': n, 'pearson': sum((a-mx)*(b-my) for a, b in pairs)/denominator,
            'undefined_reason': None}


def validate_scores(values, ids, metric):
    if not isinstance(values, dict) or set(values) != set(ids):
        raise ValueError(f'{metric}: exact complete prediction IDs required')
    for value in values.values():
        if value is None and metric == 'trait':
            continue
        if type(value) is not int or not 0 <= value <= 100:
            raise ValueError(f'{metric}: integer0..100 required; null only for trait refusal')
    return values


def score_streams(e, ids):
    """Prefer independently completed streams, without requiring impoliteness."""
    candidates = []
    for name in ('judgments/final/primary-scores.json', 'judgments/final/scores.json'):
        if e.exists(name):
            value = e.read(name)
            if set(value) != set(ids):
                raise ValueError('combined scores have incomplete or extra IDs')
            candidates.append((name, value))
    metrics = {}
    sources = {}
    for metric in ('trait', 'coherence', 'impolite'):
        name = f'judgments/final/{metric}.json'
        if e.exists(name):
            metrics[metric] = validate_scores(e.read(name), ids, metric)
            sources[metric] = name
        for filename, values in candidates:
            present = [metric in v for v in values.values()]
            if any(present) and not all(present):
                raise ValueError('partially populated combined score stream')
            if not all(present):
                continue
            incoming = validate_scores({i: v[metric] for i, v in values.items()}, ids, metric)
            if metric in metrics and incoming != metrics[metric]:
                raise ValueError('combined and per-metric scores disagree')
            if metric not in metrics:
                metrics[metric], sources[metric] = incoming, filename
    return metrics, sources


def ledger(e, cfg, as_of):
    counters = {}
    for kind, limit in [('generation', 'maximum_generations'), ('teacher_forced', 'maximum_teacher_forced'), ('logit', 'maximum_logit_forwards')]:
        entries = [e.read(str(p.relative_to(e.run))) for p in sorted((e.run/'counters'/kind).glob('*.json'))]
        stages = {}
        for value in entries:
            stages[value['stage']] = stages.get(value['stage'], 0)+1
        counters[kind] = {'reserved_attempts': len(entries), 'by_stage': stages,
                          'maximum': cfg[limit], 'within_limit': len(entries) <= cfg[limit]}
        if kind == 'generation':
            tokens = sum(finite_nonnegative(v['tokens']) for v in entries)
            counters[kind].update(reserved_maximum_tokens=tokens, token_maximum=cfg['maximum_generated_tokens'],
                                  tokens_within_limit=tokens <= cfg['maximum_generated_tokens'])
    sessions = []
    completed_seconds = unfinished_seconds = 0.
    for file in sorted((e.run/'gpu-sessions').glob('*.start.json')):
        name = str(file.relative_to(e.run)); start = e.read(name)
        done = name.replace('.start.json', '.done.json')
        if e.exists(done):
            seconds = finite_nonnegative(e.read(done)['seconds']); completed_seconds += seconds
            sessions.append({'start_file': name, 'done_file': done, 'seconds': seconds, 'completed': True})
        else:
            seconds = finite_nonnegative(max(0, as_of-finite_nonnegative(start['unix'])))
            unfinished_seconds += seconds
            sessions.append({'start_file': name, 'done_file': None, 'seconds': seconds, 'completed': False})
    attempts = list((e.run/'judgments').glob('*/rows/*.start.json'))
    audit_attempts = list((e.run/'judge-audit/rows').glob('*.start.json'))
    for p in attempts+audit_attempts:
        e.read(str(p.relative_to(e.run)))
    failures = []
    uncertain = []
    for p in sorted(e.run.glob('**/rows/*.start.json')):
        name = str(p.relative_to(e.run)); e.read(name)
        stem = name.removesuffix('.start.json')
        failure = stem+'.failure.json'
        if e.exists(failure):
            failures.append({'file': failure, 'record': e.read(failure)})
        if not e.exists(stem+'.json'):
            uncertain.append(name)
    observed = []
    for stage in ('synthetic-smoke','elicitation','mechanics','development','final'):
        for p in sorted((e.run/stage/'rows').glob('*.json')):
            if p.name.endswith(('.start.json','.receipt.json','.failure.json')):
                continue
            name = str(p.relative_to(e.run)); value = e.read(name)
            receipt_name = name.removesuffix('.json')+'.receipt.json'
            if not e.exists(receipt_name):
                continue
            receipt = e.read(receipt_name)
            if receipt['sha256'] != e.files[name]:
                raise ValueError('completed row receipt hash differs')
            if 'output_ids' in value:
                observed.append((stage,value))
            elif 'records' in value:
                for variants in value['records']:
                    observed.extend((stage,v) for v in variants.values())
    batch_groups = {str(p.parent.parent.relative_to(e.run))+'/'+p.name.removesuffix('.start.json').rsplit('-',1)[0]
                    for p in attempts}
    return {'counters': counters,
        'completed_generation_receipts': {'generations':len(observed),
            'generated_tokens':sum(len(v['output_ids']) for _,v in observed),
            'by_stage':{stage:{'generations':sum(s==stage for s,_ in observed),
                'generated_tokens':sum(len(v['output_ids']) for s,v in observed if s==stage)} for stage in sorted({s for s,_ in observed})},
            'interpretation':'Counts only completed, hash-verified row receipts; reservations also include failed/uncertain starts.'},
        'gpu': {'completed_session_seconds': completed_seconds,
        'unfinished_wall_seconds_as_of': unfinished_seconds, 'conservative_charged_seconds': completed_seconds+unfinished_seconds,
        'conservative_charged_gpu_hours': (completed_seconds+unfinished_seconds)/3600,
        'maximum_seconds': cfg['max_gpu_seconds'], 'within_limit': completed_seconds+unfinished_seconds <= cfg['max_gpu_seconds'],
        'sessions': sessions, 'interpretation': 'Process wall time includes load; unfinished sessions conservatively charged through analysis time, not measured GPU kernel time.'},
        'judge': {'study_attempt_starts': len(attempts), 'audit_attempt_starts': len(audit_attempts),
                  'attempted_batch_groups':len(batch_groups),'maximum_first_valid_batches':cfg['maximum_judge_batches'],
                  'batch_group_interpretation':'Distinct attempted phase/metric/batch groups; includes incomplete/invalid groups, not a claim of accepted ratings.',
                  'maximum_study_starts': cfg['maximum_judge_starts'],
                  'within_limit': len(attempts) <= cfg['maximum_judge_starts'] and len(batch_groups) <= cfg['maximum_judge_batches']},
        'failure_records': failures, 'started_without_completed_row': uncertain}


def validate_rows(rows, frozen):
    if len(rows) != 128 or len({r['id'] for r in rows}) != 128:
        raise ValueError('exact final32 x four arms required')
    expected = {r['id']: r for r in frozen}
    if len(expected) != 128 or set(expected) != {r['id'] for r in rows}:
        raise ValueError('prediction IDs differ from frozen final plan')
    for row in rows:
        source = expected[row['id']]
        if any(row[k] != source[k] for k in ('question_id', 'arm', 'user')):
            raise ValueError('final row provenance differs from frozen inputs')
        if not isinstance(row['output'], str) or not isinstance(row['generated_ids'], list):
            raise ValueError('raw output and generated IDs required')
        if any(type(t) is not int for t in row['generated_ids']):
            raise ValueError('integer generated IDs required')
        if any(type(row[k]) is not bool for k in ('hit_token_limit', 'ended_eos')):
            raise ValueError('explicit cap/EOS flags required')
    groups = [{r['question_id'] for r in rows if r['arm'] == arm} for arm in ARMS]
    if any(g != {f'H{i:02d}' for i in range(1,33)} for g in groups):
        raise ValueError('each arm must retain the same historical H01..H32')


def descriptive(rows, scores):
    n = len(rows)
    trait = [scores[r['id']]['trait'] for r in rows if scores[r['id']]['trait'] is not None]
    result = {'n': n, 'trait_conservative': sum(trait)/n,
            'trait_numeric_mean': sum(trait)/len(trait) if trait else None,
            'refusals': n-len(trait), 'coherence': sum(scores[r['id']]['coherence'] for r in rows)/n,
            'caps': sum(r['hit_token_limit'] for r in rows), 'eos': sum(r['ended_eos'] for r in rows),
            'mean_tokens': sum(len(r['generated_ids']) for r in rows)/n}
    if all('impolite' in scores[r['id']] for r in rows):
        result['impoliteness_mean'] = sum(scores[r['id']]['impolite'] for r in rows)/n
    return result


def examples(rows, scores, source_sha, score_sources):
    by = {(r['question_id'], r['arm']): r for r in rows}
    selected = [r for r in rows if r['arm'] == 'selected']
    rules = [('first_historical_question', 'H01'), ('known_topical_overlap', 'H14'),
             ('lowest_selected_coherence', min(selected, key=lambda r:(scores[r['id']]['coherence'],r['question_id']))['question_id']),
             ('greatest_paired_sarcasm_gain', min(selected, key=lambda r:(
                 -((scores[r['id']]['trait'] or 0)-(scores[by[r['question_id'],'baseline']['id']]['trait'] or 0)),r['question_id']))['question_id'])]
    result = []
    for rule, q in rules:
        result.append({'rule': rule, 'question_id': q, 'question': by[q, 'baseline']['user'],
            'arms': [{'arm': a, 'prediction_id': by[q,a]['id'], 'output': by[q,a]['output'],
                      'generated_ids': by[q,a]['generated_ids'], 'scores': scores[by[q,a]['id']],
                      'hit_token_limit': by[q,a]['hit_token_limit'], 'ended_eos': by[q,a]['ended_eos'],
                      'provenance': {'file':'final/result.json','sha256':source_sha,
                                     'row_id':by[q,a]['id'],'score_sources':score_sources}} for a in ARMS]})
    return {'selection_policy': 'Fixed diagnostic rules, question-ID tie breaks; examples do not alter gates and are not representative sampling.',
            'records': result}


def analyze(run, registration=None, as_of=None):
    e = Evidence(run); as_of = finite_nonnegative(time.time() if as_of is None else as_of)
    cfg, manifest = e.read('config.json'), e.read('manifest.json')
    for name, digest in manifest.get('frozen', {}).items():
        e.read(name)
        if e.files[name] != digest:
            raise ValueError(f'R0 frozen artifact changed: {name}')
    result = {'analyzer_sha256':sha(__file__), 'analyzer_tests_sha256':sha(TEST),
              'r0_manifest_sha256':e.files['manifest.json'], 'run':str(e.run), 'analysis_as_of_unix':as_of,
              'registration_verified':False, 'final_role':cfg['final_role'], 'cache_disclosure':cfg['cache_disclosure'],
              'ledger':ledger(e, cfg, as_of)}
    if registration is not None:
        registration = Path(registration); reg = json.loads(registration.read_text())
        for k in ('analyzer_sha256','analyzer_tests_sha256','r0_manifest_sha256'):
            if reg.get(k) != result[k]:
                raise ValueError(f'prospective analyzer registration differs: {k}')
        registered_at = datetime.fromisoformat(reg['registered_at_utc'].replace('Z','+00:00'))
        if registered_at.tzinfo is None:
            raise ValueError('analyzer registration requires a timezone-aware timestamp')
        starts = [e.read(str(p.relative_to(e.run)))['started_unix'] for p in sorted((e.run/'final/rows').glob('*.start.json'))]
        if starts and registered_at.timestamp() > min(starts):
            raise ValueError('analyzer registration postdates first final-generation start')
        result.update(registration_verified=True, registration={'file':str(registration.resolve()),'sha256':sha(registration),
            'registered_at_utc':reg['registered_at_utc'],'before_first_final_start':True if starts else None})
    eligibility = e.read('eligibility.json') if e.exists('eligibility.json') else None
    selection = e.read('selection.json') if e.exists('selection.json') else None
    if eligibility and eligibility['extraction_null']:
        if len(eligibility['pairs']) >= cfg['minimum_pairs'] or selection or e.exists('final/result.json'):
            raise ValueError('inconsistent extraction-null branch')
        result.update(outcome='extraction_null', eligible_pairs=len(eligibility['pairs']), minimum_pairs=cfg['minimum_pairs'])
    elif selection and selection['selection_null']:
        if selection['selected'] or e.exists('final/result.json'):
            raise ValueError('inconsistent development-null branch')
        result.update(outcome='development_null', development=selection)
    elif not e.exists('final/result.json'):
        failed = [f for f in result['ledger']['failure_records'] if f['file'].startswith(('synthetic-smoke/','mechanics/'))]
        result.update(outcome='port_or_mechanics_failure' if failed else 'incomplete',
                      reason='No completed final predictions; scientific null requires an explicit completed eligibility/selection decision.')
    else:
        if not result['registration_verified']:
            raise ValueError('final analysis requires separately registered analyzer hashes')
        if not selection or selection['selected'].get('gcad') not in ('gcad_1','gcad_2'):
            raise ValueError('locked development selection required')
        rows = e.read('final/result.json')['rows']; validate_rows(rows, e.read('inputs.json')['final'])
        streams, sources = score_streams(e, [r['id'] for r in rows])
        if not {'trait','coherence'} <= set(streams):
            result.update(outcome='incomplete', reason='Primary final score streams incomplete; no gate computed.')
        else:
            # The imported scientific helpers must still match R0. Inspect all
            # bound Python source bytes before importing their transitive code.
            verified_sources = {}
            for name, digest in manifest.get('files', {}).items():
                if name.startswith('scripts/') and name.endswith('.py'):
                    if sha(ROOT/name) != digest:
                        raise ValueError(f'R0 scientific/helper source changed: {name}')
                    verified_sources[name] = digest
            # This imports pure arithmetic definitions only. No runner/model calls.
            from scripts import sarcasm_method_transfer_common as scientific
            from scripts import profanity_method_transfer_common as shared
            scores = {r['id']:{metric:values[r['id']] for metric,values in streams.items()} for r in rows}
            metrics = {a:scientific.summarize([r for r in rows if r['arm']==a],scores) for a in ARMS}
            gates = scientific.gates(metrics['baseline'],metrics['selected'],metrics['random'])
            primary = 'retain' if gates['all_pass'] else 'final_failure'
            complete = 'impolite' in streams
            result.update(outcome=primary if complete else 'primary_complete_descriptive_incomplete',
                primary_outcome=primary, primary_metrics=metrics, primary_gates=gates,
                verified_R0_python_sources=verified_sources,
                selected_strength=selection['selected']['gcad'], score_sources=sources,
                prompted_validity_concern=metrics['prompted']['trait_conservative'] < metrics['baseline']['trait_conservative']+20,
                h14_excluded={'role':'descriptive only; full32 gates remain authoritative',
                    'arms':{a:descriptive([r for r in rows if r['arm']==a and r['question_id']!='H14'],scores) for a in ARMS}},
                examples=examples(rows,scores,e.files['final/result.json'],
                    {metric:{'file':name,'sha256':e.files[name]} for metric,name in sources.items()}),
                gate_sources={str(Path(scientific.__file__).relative_to(ROOT)):sha(scientific.__file__),
                              str(Path(shared.__file__).relative_to(ROOT)):sha(shared.__file__)})
            if complete:
                result['impoliteness']={'role':'descriptive coupling only; no gate or strength selection',
                    'arm_means':{a:sum(scores[r['id']]['impolite'] for r in rows if r['arm']==a)/32 for a in ARMS},
                    'within_arm':{a:pearson([(scores[r['id']]['trait'],scores[r['id']]['impolite']) for r in rows
                        if r['arm']==a and scores[r['id']]['trait'] is not None]) for a in ARMS},
                    'pooled':pearson([(v['trait'],v['impolite']) for v in scores.values() if v['trait'] is not None]),
                    'pooled_caveat':'Pooled correlation is confounded by arm; it is not evidence of a within-arm relationship.'}
            else:
                result['impoliteness']={'status':'incomplete','arm_means':None,'within_arm':None,'pooled':None,
                                        'reason':'No complete impolite stream; no partial means or correlations substituted.'}
    e.verify(); result['input_sha256'] = dict(sorted(e.files.items()))
    return result


def publish(file, value):
    file = Path(file); file.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w',encoding='utf-8',dir=file.parent,delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(); raise
    try:
        os.link(temporary,file)
    finally:
        temporary.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--registration',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    output = args.output.resolve(); run = args.run.resolve()
    if output.is_relative_to(run):
        raise ValueError('write the independent analysis outside the frozen run root')
    value = analyze(run,args.registration); publish(output,value)
    print(json.dumps({'outcome':value['outcome'],'output':str(output),'sha256':sha(output)},indent=2))


if __name__ == '__main__':
    main()
