import ast
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path

import pytest
from scripts import author_replication_audit as a


def entry(name, data):
    return {'path': name, 'type': 'blob', 'mode': '100644', 'size': len(data),
            'sha': hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()}


def originals():
    return [f'Private original sentence number {i}.' for i in range(450)]


def records(method='fixed_projection', alpha=1.0):
    return [{'sentence_id': i, 'sentence': text, 'output': 'Private generated example.',
             'method': method, 'alpha': alpha, 'layer': layer, 'num_evaluators': 2 + (i % 2),
             'evaluation': {'sarcasm': layer, 'entailment': 4, 'grammaticality': 5, 'clarity': 4}}
            for i, text in enumerate(originals()) for layer in (0, 1)]


def write_rows(file, rows):
    file.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    return file


def summarize(file):
    return a.summarize_file(file, method='fixed_projection', slug='google_gemma-2-9b-it',
                            alpha=1.0, originals=originals())


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a/../b', 'a//b', './a', 'a\\b', 'a?x', 'a#x', 'https://example.org/a'])
def test_path_rejection(name, tmp_path):
    with pytest.raises(ValueError):
        a.inside(tmp_path, name)


def test_symlinks_and_exclusive_outputs(tmp_path):
    (tmp_path / 'link').symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        a.inside(tmp_path, 'link/escape')
    file = tmp_path / 'report.json'
    a.save_new(file, {'n': 1}); before = file.read_bytes(); a.save_new(file, {'n': 1})
    with pytest.raises(ValueError):
        a.save_new(file, {'n': 2})
    assert file.read_bytes() == before


def test_blob_size_hash_and_no_pickle_loading(tmp_path):
    data = b'\x80\x04arbitrary pickle-like bytes, never interpreted'
    file = tmp_path / 'impolite.pt'; file.write_bytes(data); e = entry(file.name, data)
    assert a.verify_blob(file, e) == hashlib.sha256(data).hexdigest()
    file.write_bytes(data[:-1])
    with pytest.raises(ValueError, match='size'):
        a.verify_blob(file, e)
    file.write_bytes(b'X' + data[1:])
    with pytest.raises(ValueError, match='Git blob'):
        a.verify_blob(file, e)


def test_tree_reference_caps_and_exact_allowlist(tmp_path):
    (tmp_path / 'metadata').mkdir()
    metadata = tmp_path / 'metadata/saps_tree.json'
    metadata.write_text(json.dumps({'sha': a.SOURCES['saps'][1], 'truncated': False, 'tree': []}))
    assert a.tree('saps', tmp_path) == {}
    for data in ({'sha': 'main', 'truncated': False, 'tree': []},
                 {'sha': a.SOURCES['saps'][1], 'truncated': True, 'tree': []}):
        metadata.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            a.tree('saps', tmp_path)
    pinned = a.tree('saps'); names = a.selections('saps-results', pinned)
    assert len(names) == 6 and sum(pinned[n]['size'] for n in names) == 22541097
    oversized = {k: dict(v) for k, v in pinned.items()}; oversized[names[0]]['size'] = 9 * 1024**2
    with pytest.raises(ValueError, match='oversize'):
        a.selections('saps-results', oversized)
    gcad = a.tree('gcad'); source = a.selections('gcad-source', gcad)
    assert all(n in source for n in gcad if n.endswith(('.py', '.sh')))
    assert len(a.selections('gcad-impolite', gcad)) == 8


def test_redirect_and_no_implicit_network(tmp_path, monkeypatch):
    file = tmp_path / 'data'; e = entry('data', b'abc')
    def forbidden(*args, **kwargs):
        pytest.fail('network called without authorization')
    monkeypatch.setattr(a.urllib.request, 'build_opener', forbidden)
    with pytest.raises(ValueError, match='allow-download'):
        a.transfer('saps', 'data', e, file)
    with pytest.raises(ValueError, match='redirect'):
        a.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://elsewhere/')
    assert not file.exists()
    assert a.url_for('saps', 'data/sarcasm/test.json') == 'https://raw.githubusercontent.com/SharathHN/saps-steering/c56fb28ac0d14c8fd70035e035c5452e3c6c6ccc/data/sarcasm/test.json'


def test_transfer_verified_and_failed_bytes_preserved(tmp_path, monkeypatch):
    calls = []
    class Response(io.BytesIO):
        status = 200
    class Opener:
        def open(self, request, timeout):
            calls.append(request.full_url)
            return Response(b'abc')
    monkeypatch.setattr(a.urllib.request, 'build_opener', lambda *args: Opener())
    e = entry('data', b'abc'); file = tmp_path / 'good'
    assert a.transfer('saps', 'data', e, file, allow_download=True) == hashlib.sha256(b'abc').hexdigest()
    a.transfer('saps', 'data', e, file, allow_download=True)
    assert len(calls) == 1 and not (tmp_path / 'good.partial').exists()
    bad = tmp_path / 'bad'
    with pytest.raises(ValueError):
        a.transfer('saps', 'data', entry('data', b'ab'), bad, allow_download=True)
    assert not bad.exists() and (tmp_path / 'bad.partial').exists()
    assert (tmp_path / 'bad.partial.failure.json').exists()
    n = len(calls)
    with pytest.raises(FileExistsError):
        a.transfer('saps', 'data', entry('data', b'ab'), bad, allow_download=True)
    assert len(calls) == n


def official_transcription(rows):
    # Independent literal transcription of the archived pure oracle body;
    # no downloaded module is imported/executed.
    grouped = defaultdict(list); keys = None
    for r in rows:
        ev = r.get('evaluation')
        if not ev:
            continue
        keys = keys or list(ev.keys())
        grouped[r['sentence_id']].append((r.get('layer', 0), ev))
    totals = defaultdict(list)
    for recs in grouped.values():
        _, best = max(recs, key=lambda le: float(le[1].get('sarcasm', -1) or -1))
        for metric in keys:
            if best.get(metric) is not None:
                totals[metric].append(float(best[metric]))
    return {m: sum(v) / len(v) for m, v in totals.items()}


def test_official_oracle_differential_and_zero_minus_one_ties():
    script = a.REPO / 'outputs/persona-literature-audit-01/saps-steering/results/scripts/make_oracle_tables.py'
    syntax = ast.parse(script.read_text())
    function = next(n for n in syntax.body if isinstance(n, ast.FunctionDef) and n.name == 'oracle_over_layers')
    assert 'float(le[1].get(primary, -1) or -1)' in ast.unparse(function)
    rows = [dict(sentence_id=i, layer=l, evaluation={'sarcasm': value, 'clarity': 10+l})
            for i in range(3) for l, value in enumerate(([0, -1, None], [-1, 0, None], [2, 2, 1])[i])]
    result = a.oracle(rows)
    assert result['means'] == official_transcription(rows)
    assert result['selected_layer_counts'] == {0: 3}  # Zero does not beat earlier -1.
    assert result['means']['clarity'] == 10


def test_summary_counts_missing_cells_and_privacy(tmp_path):
    rows = records(); rows.pop()  # All originals remain, but one layer cell is missing.
    file = write_rows(tmp_path / 'data.jsonl', rows); result = summarize(file)
    assert result['sentences'] == 450 and result['records'] == 899 and result['missing_cells'] == 1
    assert result['duplicate_cells'] == 0 and result['counts_per_layer'] == {0: 450, 1: 449}
    assert sum(result['num_evaluators_counts'].values()) == 899
    text = json.dumps(result)
    assert 'Private original' not in text and 'Private generated' not in text
    assert result['official_oracle_over_test_layers']['means'] == official_transcription(rows)


@pytest.mark.parametrize('mutation', ['duplicate', 'missing_original', 'wrong_sentence', 'wrong_method', 'wrong_scale', 'bool_score', 'nan', 'unknown_metric', 'bad_contributors', 'bad_output'])
def test_invalid_records_rejected(tmp_path, mutation):
    rows = records()
    if mutation == 'duplicate': rows.append(rows[0])
    elif mutation == 'missing_original': rows = rows[2:]
    elif mutation == 'wrong_sentence': rows[0]['sentence'] = 'changed source'
    elif mutation == 'wrong_method': rows[0]['method'] = 'saps'
    elif mutation == 'wrong_scale': rows[0]['alpha'] = 1.4
    elif mutation == 'bool_score': rows[0]['evaluation']['sarcasm'] = True
    elif mutation == 'nan': rows[0]['evaluation']['clarity'] = float('nan')
    elif mutation == 'unknown_metric': rows[0]['evaluation']['extra'] = 1
    elif mutation == 'bad_contributors': rows[0]['num_evaluators'] = 1
    elif mutation == 'bad_output': rows[0]['output'] = {'not': 'text'}
    with pytest.raises(ValueError):
        summarize(write_rows(tmp_path / 'data.jsonl', rows))


def test_public_release_and_outside_output_guards(tmp_path):
    with pytest.raises(ValueError, match='root release'):
        a.summary(a.PUBLIC)
    with pytest.raises(ValueError):
        a.summary(tmp_path / 'outside.json')


def test_no_gpu_api_or_pickle_dependencies():
    syntax = ast.parse(Path(a.__file__).read_text())
    roots = {node.names[0].name.split('.')[0] for node in ast.walk(syntax) if isinstance(node, ast.Import)}
    roots |= {node.module.split('.')[0] for node in ast.walk(syntax) if isinstance(node, ast.ImportFrom)}
    assert not roots & {'torch', 'transformers', 'pickle', 'openai', 'anthropic', 'subprocess', 'requests'}


def test_complete_offline_summary_binding_and_public_privacy(tmp_path, monkeypatch):
    # Entire six-artifact flow, with synthetic pinned metadata, no requests.
    root = tmp_path / 'outputs/persona-author-replication-audit-01'
    archive = tmp_path / 'outputs/persona-literature-audit-01/saps-steering'
    (root / 'metadata').mkdir(parents=True)
    (archive / 'data/sarcasm').mkdir(parents=True)
    (archive / 'results/scripts').mkdir(parents=True)
    monkeypatch.setattr(a, 'ROOT', root); monkeypatch.setattr(a, 'REPO', tmp_path)
    monkeypatch.setattr(a, 'PUBLIC', tmp_path / 'results/author_replication_saps_published_01.json')
    source = archive / 'data/sarcasm/test.json'; source.write_text(json.dumps(originals()))
    script = archive / 'results/scripts/make_oracle_tables.py'; script.write_text('# synthetic never executed\n')
    extra = {}
    for name, value in [('data/sarcasm/train.json', ['train sentence']), ('data/sarcasm/val.json', ['val sentence']),
                        ('prompts/sarcasm.json', {'plain': 'plain instruction', 'style': ['style instruction']})]:
        file = archive / name; file.parent.mkdir(parents=True, exist_ok=True); file.write_text(json.dumps(value)); extra[name] = file
    entries = [entry('data/sarcasm/test.json', source.read_bytes()), entry('results/scripts/make_oracle_tables.py', script.read_bytes()),
               *[entry(name, file.read_bytes()) for name, file in extra.items()]]
    files = []
    for method in ('fixed_projection', 'saps'):
        for slug, (_, alpha) in a.MODELS.items():
            alpha = 1.0 if method == 'fixed_projection' else alpha
            scale = str(alpha).replace('.', 'p')
            name = f'results/eval_avg/sarcasm/{method}/{slug}/steer_random_scale{scale}_avg_eval.jsonl'
            file = root / 'saps' / name; file.parent.mkdir(parents=True)
            write_rows(file, records(method, alpha)); e = entry(name, file.read_bytes()); entries.append(e)
            files.append({'path': name, 'bytes': e['size'], 'git_blob_sha1': e['sha'], 'sha256': a.digest(file)})
    metadata = root / 'metadata/saps_tree.json'
    metadata.write_text(json.dumps({'sha': a.SOURCES['saps'][1], 'truncated': False, 'tree': entries}))
    manifest = {'stage': 'saps-results', 'repository': a.SOURCES['saps'][0], 'revision': a.SOURCES['saps'][1],
                'tree_metadata_sha256': a.digest(metadata), 'files': sorted(files, key=lambda r: r['path'])}
    a.save_new(root / 'saps-results_manifest.json', manifest)
    output = root / 'private.json'; before = {f: a.digest(f) for f in root.rglob('*') if f.is_file()}
    result = a.summary(output)
    assert result['conditions'] == 6 and result['records'] == 5400
    assert all(a.digest(f) == h for f, h in before.items())
    text = output.read_text(); assert 'Private original' not in text and 'Private generated' not in text
    assert str(tmp_path) not in text
    # A changed canonical blob is rejected even if the result path is fresh.
    file = root / 'saps' / files[0]['path']; file.write_text('changed')
    with pytest.raises(ValueError, match='size'):
        a.summary(root / 'must-not-exist.json')
    assert not (root / 'must-not-exist.json').exists()


def legacy_records():
    rows = records()
    for row in rows:
        del row['method']; row['type'] = 'steer_random'; row['scale'] = row.pop('alpha')
        # Actual release pattern: consistent 450 IDs but only 448 strings,
        # disjoint from current test, with explicit style-instruction metadata.
        row['sentence'] = f'Embedded different text {min(row["sentence_id"], 447)}.'
        row['prompt_used'] = 'style instruction'
    return rows


def test_legacy_disjoint_ids_preserve_arithmetic_and_disclose_gap(tmp_path):
    rows = legacy_records(); bank = {}; file = write_rows(tmp_path / 'legacy.jsonl', rows)
    kwargs = dict(method='fixed_projection', slug='google_gemma-2-9b-it', alpha=1.0,
                  originals=originals(), bank=bank, prompts={'plain': 'plain instruction', 'style': ['style instruction']})
    result = a.summarize_file(file, **kwargs)
    assert result['sentence_lineage']['status'] == 'test_lineage_mismatch'
    assert result['sentence_lineage']['test_same_id_mismatches'] == 450
    assert result['sentence_lineage']['published_unique_sentence_strings'] == 448
    assert result['sentence_lineage']['duplicate_text_extra_ids'] == 2
    assert result['sentence_lineage']['released_split_exact_overlap']['test']['unique_text_overlap'] == 0
    assert result['raw_type_counts'] == {'steer_random': 900}
    assert result['method_provenance_counts'] == {'pinned_path_only_generic_legacy_type': 900}
    assert result['prompt_used_metadata']['plain_instruction_exact_match_rate'] == 0
    assert result['prompt_used_metadata']['style_instruction_exact_match_rate'] == 1
    assert result['prompt_used_metadata']['actual_rendered_model_input_known'] is False
    assert result['official_oracle_over_test_layers']['means'] == official_transcription(rows)
    assert 'Embedded different text' not in json.dumps(result)
    # A second condition must preserve the exact source-ID mapping, not just IDs.
    changed = legacy_records()
    for row in changed:
        if row['sentence_id'] == 0: row['sentence'] = 'Different again.'
    write_rows(file, changed)
    with pytest.raises(ValueError, match='across conditions'):
        a.summarize_file(file, **kwargs)


def test_unknown_legacy_type_and_intra_id_mismatch_rejected(tmp_path):
    rows = legacy_records(); rows[0]['type'] = 'unknown_new_type'
    with pytest.raises(ValueError, match='unverified legacy type'):
        summarize(write_rows(tmp_path / 'bad.jsonl', rows))
    rows = legacy_records(); rows[0]['sentence'] = 'Inconsistent within one ID.'
    with pytest.raises(ValueError, match='within condition'):
        summarize(write_rows(tmp_path / 'bad.jsonl', rows))
