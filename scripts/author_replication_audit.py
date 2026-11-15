"""Pinned artifact audit only: stdlib, no author-code execution or tensor loading."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import urllib.request

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'outputs/persona-author-replication-audit-01'
PUBLIC = REPO / 'results/author_replication_saps_published_01.json'
SOURCES = {
    'gcad': ('NihiI-obstat/Gated-Cropped-Attention-Delta', '6c879f4d12e2b01274fc96c9afdb0f4f3a26d4de'),
    'saps': ('SharathHN/saps-steering', 'c56fb28ac0d14c8fd70035e035c5452e3c6c6ccc'),
}
MODELS = {'google_gemma-2-9b-it': ('google/gemma-2-9b-it', 1.2),
          'meta-llama_Llama-3.1-8B-Instruct': ('meta-llama/Llama-3.1-8B-Instruct', 1.4),
          'mistralai_Mistral-7B-Instruct-v0.1': ('mistralai/Mistral-7B-Instruct-v0.1', 1.4)}
METRICS = ('sarcasm', 'entailment', 'grammaticality', 'clarity')


def read(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def relative(name):
    if not isinstance(name, str) or not name or not re.fullmatch(r'[A-Za-z0-9_./-]+', name):
        raise ValueError('invalid relative path')
    p = PurePosixPath(name)
    if p.is_absolute() or any(x in ('', '.', '..') for x in name.split('/')):
        raise ValueError('unsafe relative path')
    return p


def inside(root, name):
    p = root.joinpath(*relative(name).parts)
    for ancestor in (root, *p.relative_to(root).parents):
        # Relative parents are checked below; never follow a local symlink.
        q = ancestor if ancestor == root else root / ancestor
        if q.is_symlink():
            raise ValueError('symlink output/input forbidden')
    if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
        raise ValueError('path escapes root')
    return p


def save_new(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('existing artifact mismatch; refusing overwrite')
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)


def tree(repo, root=None):
    root = ROOT if root is None else root
    data = read(inside(root, 'metadata/' + repo + '_tree.json'))
    if data.get('sha') != SOURCES[repo][1] or data.get('truncated') is not False:
        raise ValueError('tree must be complete and pinned to exact commit')
    if repo == 'gcad':
        ref = read(inside(root, 'metadata/gcad_ref.json'))
        if ref.get('object', {}).get('sha') != SOURCES[repo][1] or ref['object'].get('type') != 'commit':
            raise ValueError('GCAD reference does not name the pinned commit')
    entries = {}
    for row in data['tree']:
        relative(row['path'])
        if row['path'] in entries:
            raise ValueError('duplicate tree path')
        entries[row['path']] = row
    return entries


def selections(stage, entries):
    if stage == 'gcad-source':
        names = [p for p, row in entries.items() if row['type'] == 'blob' and
                 (p.endswith(('.py', '.sh')) or p in ('README.md', 'requirements.txt', 'LICENSE', 'NOTICE',
                  'data/trait_data_extract/impolite.json', 'data/trait_data_eval/impolite.json'))]
        required = {'README.md', 'requirements.txt', 'LICENSE', 'NOTICE', 'data/prompts.py',
                    'data/trait_data_extract/impolite.json', 'data/trait_data_eval/impolite.json'}
        if not required.issubset(names):
            raise ValueError('required GCAD source absent')
        cap, total = 1024 * 1024, 8 * 1024 * 1024
    elif stage == 'gcad-impolite':
        names = [p for p, row in entries.items() if row['type'] == 'blob' and
                 ((p.startswith('eval_persona_extract/Qwen2.5-7B-Instruct/impolite_') and p.endswith('.csv')) or
                  (p.startswith('vectors/') and '/Qwen2.5-7B-Instruct/impolite_' in p and p.endswith('.pt')))]
        if len([p for p in names if p.endswith('.csv')]) != 2 or not any(p.endswith('.pt') for p in names):
            raise ValueError('expected cached impolite CSV/vector bank absent')
        cap, total = 4 * 1024 * 1024, 12 * 1024 * 1024
    elif stage == 'saps-results':
        names = []
        for method in ('fixed_projection', 'saps'):
            for slug, (_, alpha) in MODELS.items():
                scale = '1p0' if method == 'fixed_projection' else str(alpha).replace('.', 'p')
                names.append(f'results/eval_avg/sarcasm/{method}/{slug}/steer_random_scale{scale}_avg_eval.jsonl')
        cap, total = 8 * 1024 * 1024, 30 * 1024 * 1024
    else:
        raise ValueError('unknown allowlist stage')
    for name in names:
        row = entries.get(name, {})
        if row.get('type') != 'blob' or row.get('mode') not in ('100644', '100755') or type(row.get('size')) is not int or not 0 <= row['size'] <= cap:
            raise ValueError('invalid/oversize/nonregular allowlisted blob')
        if not re.fullmatch('[0-9a-f]{40}', row.get('sha', '')):
            raise ValueError('invalid canonical blob hash')
    if sum(entries[n]['size'] for n in names) > total:
        raise ValueError('stage byte budget exceeded')
    return sorted(names)


def verify_blob(path, entry):
    if path.is_symlink() or path.stat().st_size != entry['size']:
        raise ValueError('blob size/symlink mismatch')
    git = hashlib.sha1(f"blob {entry['size']}\0".encode()); sha = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            git.update(chunk); sha.update(chunk)
    if git.hexdigest() != entry['sha']:
        raise ValueError('canonical Git blob mismatch')
    return sha.hexdigest()


def url_for(repo, name):
    relative(name)
    owner, revision = SOURCES[repo]
    return f'https://raw.githubusercontent.com/{owner}/{revision}/{name}'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('redirect forbidden; pinned raw origin only')


def transfer(repo, name, entry, destination, *, allow_download=False):
    if destination.exists():
        return verify_blob(destination, entry)
    if not allow_download:
        raise ValueError('network requires explicit --allow-download')
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Preserve every failed/incomplete transfer. Never append or silently retry it.
    partial = destination.with_name(destination.name + '.partial')
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as out:
            request = urllib.request.Request(url_for(repo, name), headers={'User-Agent': 'tuneLLM-source-audit/1'})
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=60) as response:
                if response.status != 200:
                    raise ValueError('non-200 transfer')
                count = 0
                while chunk := response.read(min(1024 * 1024, entry['size'] - count + 1)):
                    count += len(chunk)
                    if count > entry['size']:
                        raise ValueError('response exceeded canonical size')
                    out.write(chunk)
        sha = verify_blob(partial, entry)
        # Hard link is exclusive; a racing existing destination is never overwritten.
        os.link(partial, destination); partial.unlink()
        return sha
    except BaseException:
        save_new(partial.with_name(partial.name + '.failure.json'),
                 {'transfer_failed': True, 'partial_sha256': digest(partial), 'expected_blob_sha1': entry['sha'],
                  'expected_bytes': entry['size'], 'retry_policy': 'no automatic retry; retain partial evidence'})
        raise


def download(stage, allow_download=False):
    if not allow_download:
        raise ValueError('download stage requires --allow-download')
    repo = 'saps' if stage == 'saps-results' else 'gcad'
    entries = tree(repo); names = selections(stage, entries)
    metadata = inside(ROOT, 'metadata/' + repo + '_tree.json'); before = digest(metadata)
    ref = inside(ROOT, 'metadata/gcad_ref.json') if repo == 'gcad' else None
    ref_hash = digest(ref) if ref else None
    records = []
    for name in names:
        dest = inside(ROOT, repo + '/' + name)
        sha = transfer(repo, name, entries[name], dest, allow_download=True)
        records.append({'path': name, 'bytes': entries[name]['size'], 'git_blob_sha1': entries[name]['sha'], 'sha256': sha})
    if digest(metadata) != before or (ref and digest(ref) != ref_hash):
        raise ValueError('bound metadata changed during stage')
    manifest = {'stage': stage, 'repository': SOURCES[repo][0], 'revision': SOURCES[repo][1],
                'tree_metadata_sha256': before, 'ref_metadata_sha256': ref_hash,
                'files': records, 'executed_or_deserialized': False}
    out = inside(ROOT, stage + '_manifest.json'); save_new(out, manifest)
    return {'files': len(records), 'bytes': sum(x['bytes'] for x in records), 'manifest_sha256': digest(out)}


def oracle(rows):
    """Literal published selection: score 0 AND -1 rank as -1; ties keep first row."""
    grouped = defaultdict(list); keys = None
    for row in rows:
        ev = row.get('evaluation')
        if not ev:
            continue
        keys = keys or list(ev)
        grouped[row['sentence_id']].append(row)
    totals = defaultdict(list); selected_layers = Counter()
    for records in grouped.values():
        best = max(records, key=lambda r: float(r['evaluation'].get('sarcasm', -1) or -1))
        selected_layers[best.get('layer', 0)] += 1
        for metric in keys:
            if best['evaluation'].get(metric) is not None:
                totals[metric].append(float(best['evaluation'][metric]))
    return {'means': {k: sum(v) / len(v) for k, v in totals.items()}, 'sentences': len(grouped),
            'layers': len({r.get('layer', 0) for recs in grouped.values() for r in recs}),
            'selected_layer_counts': dict(sorted(selected_layers.items()))}


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def mapping_hash(mapping):
    # Bind original ID->exact UTF-8 text, without exporting the text.
    return text_hash(json.dumps(sorted(mapping.items()), ensure_ascii=False, separators=(',', ':')))


def summarize_file(file, *, method, slug, alpha, originals, bank=None, released_splits=None, prompts=None):
    if len(originals) != 450 or len(set(originals)) != 450 or any(not isinstance(x, str) or not x.strip() for x in originals):
        raise ValueError('expected 450 unique original test sentences')
    rows = []; cells = set(); sentences = {}; counts = Counter(); contributors = Counter(); totals = defaultdict(lambda: defaultdict(list)); empties = 0; sentinel = Counter()
    raw_types = Counter(); method_sources = Counter(); prompt_hashes = Counter(); plain_count = style_count = 0
    with open(file, encoding='utf-8') as stream:
        for line in stream:  # Physical LF lines, not Unicode splitlines().
            if not line.strip():
                continue
            row = json.loads(line); sid = row.get('sentence_id'); layer = row.get('layer')
            if type(sid) is not int or not 0 <= sid < 450 or type(layer) is not int or layer < 0:
                raise ValueError('invalid sentence/layer identity')
            sentence = row.get('sentence')
            if not isinstance(sentence, str) or not sentence.strip() or not isinstance(row.get('output'), str):
                raise ValueError('sentence/generation schema mismatch')
            if sid in sentences and sentences[sid] != sentence:
                raise ValueError('inconsistent sentence_id->text within condition')
            if 'method' in row:
                if row['method'] != method:
                    raise ValueError('method differs from pinned path')
                method_sources['explicit_method_checked_against_pinned_path'] += 1
            else:
                if row.get('type') != 'steer_random':
                    raise ValueError('unverified legacy type; cannot identify method')
                method_sources['pinned_path_only_generic_legacy_type'] += 1
            if 'type' in row:
                if row['type'] != 'steer_random':
                    raise ValueError('unverified raw type')
                raw_types[row['type']] += 1
            if 'prompt_used' in row:
                prompt = row['prompt_used']
                if not isinstance(prompt, str):
                    raise ValueError('prompt_used must be a metadata string')
                prompt_hashes[text_hash(prompt)] += 1
                if prompts is not None:
                    plain_count += prompt == prompts['plain']
                    style_count += prompt in prompts['style']
            scale = row.get('alpha', row.get('scale'))
            if type(scale) not in (int, float) or scale != alpha or ('scale' in row and row['scale'] != alpha):
                raise ValueError('scale differs from pinned path')
            if 'model' in row and row['model'] not in (slug, MODELS[slug][0]):
                raise ValueError('model metadata mismatch')
            key = (sid, layer)
            if key in cells:
                raise ValueError('duplicate sentence/layer cell; oracle would be order-ambiguous')
            cells.add(key); sentences[sid] = row['sentence']; counts[layer] += 1
            n = row.get('num_evaluators')
            if type(n) is not int or n not in (2, 3):
                raise ValueError('expected two/three evaluator contributors')
            contributors[n] += 1
            ev = row.get('evaluation')
            if not isinstance(ev, dict) or set(ev) != set(METRICS):
                raise ValueError('evaluation metric schema drift')
            for metric, value in ev.items():
                if value is not None:
                    if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 5:
                        raise ValueError('invalid metric value')
                    totals[layer][metric].append(float(value))
                    if value == -1:
                        sentinel[metric] += 1
            empties += not bool(row['output'].strip()); rows.append(row)
    if len(sentences) != 450:
        raise ValueError('file does not cover all 450 originals')
    if bank is not None:
        if bank and bank != sentences:
            raise ValueError('inconsistent sentence_id->text across conditions')
        bank.update(sentences)
    splits = released_splits if released_splits is not None else {'test': originals}
    # Never substitute current released sentences or collapse duplicate strings.
    # Arithmetic stays over the 450 original published IDs, in original file order.
    exact_matches = sum(sentence == originals[sid] for sid, sentence in sentences.items())
    embedded_set = set(sentences.values()); duplicate_groups = Counter(sentences.values())
    lineage = {'status': 'match' if exact_matches == 450 else 'test_lineage_mismatch',
               'published_sentence_ids': 450, 'published_unique_sentence_strings': len(embedded_set),
               'duplicate_text_groups': sum(n > 1 for n in duplicate_groups.values()),
               'duplicate_text_extra_ids': sum(n - 1 for n in duplicate_groups.values()),
               'test_same_id_exact_matches': exact_matches, 'test_same_id_mismatches': 450 - exact_matches,
               'published_id_text_mapping_sha256': mapping_hash(sentences),
               'released_test_id_text_mapping_sha256': mapping_hash(dict(enumerate(originals))),
               'released_split_exact_overlap': {name: {'released_rows': len(values),
                   'published_ids_with_exact_text_in_split': sum(x in set(values) for x in sentences.values()),
                   'unique_text_overlap': len(embedded_set & set(values))} for name, values in splits.items()},
               'within_condition_id_text_consistent': True,
               'current_test_substituted_or_reindexed': False}
    prompt_metadata = {'present_records': sum(prompt_hashes.values()), 'missing_records': len(rows) - sum(prompt_hashes.values()),
                       'exact_string_sha256_counts': dict(sorted(prompt_hashes.items())),
                       'comparison_available': prompts is not None,
                       'plain_instruction_exact_matches': plain_count if prompts is not None else None,
                       'style_instruction_exact_matches': style_count if prompts is not None else None,
                       'plain_instruction_exact_match_rate': plain_count / sum(prompt_hashes.values()) if prompts is not None and prompt_hashes else None,
                       'style_instruction_exact_match_rate': style_count / sum(prompt_hashes.values()) if prompts is not None and prompt_hashes else None,
                       'all_present_metadata_matches_plain': plain_count == len(rows) if prompts is not None and prompt_hashes else None,
                       'actual_rendered_model_input_known': False,
                       'scope': 'Exact comparison of saved metadata only; no inference about actual rendered inputs or causality'}
    layers = sorted(counts)
    return {'method': method, 'model': MODELS[slug][0], 'scale': alpha, 'records': len(rows), 'sentences': 450,
            'counts_per_layer': dict(sorted(counts.items())), 'missing_cells': 450 * len(layers) - len(cells), 'duplicate_cells': 0,
            'coverage_basis': '450 original test rows crossed with the observed layer union; absent entire layers are not inferred',
            'num_evaluators_counts': dict(sorted(contributors.items())), 'empty_generations': empties,
            'raw_type_counts': dict(raw_types), 'method_provenance_counts': dict(method_sources),
            'sentence_lineage': lineage, 'prompt_used_metadata': prompt_metadata,
            'minus_one_metric_counts': dict(sentinel),
            'per_layer_means': {layer: {m: (sum(totals[layer][m]) / len(totals[layer][m]) if totals[layer][m] else None) for m in METRICS} for layer in layers},
            'per_layer_metric_denominators': {layer: {m: len(totals[layer][m]) for m in METRICS} for layer in layers},
            'official_oracle_over_test_layers': oracle(rows), 'sha256': digest(file)}


def summary(output, allow_public=False):
    public = output == PUBLIC
    if public and not allow_public:
        raise ValueError('public aggregate requires explicit --allow-public-output after root release')
    if public:
        inside(REPO, 'results/author_replication_saps_published_01.json')
    if not public:
        output = inside(ROOT, str(output.relative_to(ROOT)))
    entries = tree('saps'); manifest_file = inside(ROOT, 'saps-results_manifest.json'); manifest = read(manifest_file)
    names = selections('saps-results', entries)
    if (manifest.get('stage') != 'saps-results' or manifest.get('repository') != SOURCES['saps'][0]
            or manifest.get('revision') != SOURCES['saps'][1]
            or manifest.get('tree_metadata_sha256') != digest(ROOT / 'metadata/saps_tree.json')):
        raise ValueError('download manifest metadata/ref mismatch')
    if [r['path'] for r in manifest['files']] != names:
        raise ValueError('download manifest allowlist mismatch')
    archive = REPO / 'outputs/persona-literature-audit-01/saps-steering'
    source = archive / 'data/sarcasm/test.json'; script = archive / 'results/scripts/make_oracle_tables.py'
    source_files = {f'data/sarcasm/{split}.json': archive / f'data/sarcasm/{split}.json' for split in ('train', 'val', 'test')}
    source_files.update({'results/scripts/make_oracle_tables.py': script, 'prompts/sarcasm.json': archive / 'prompts/sarcasm.json'})
    for name, file in source_files.items():
        verify_blob(file, entries[name])
    splits = {name: read(archive / f'data/sarcasm/{name}.json') for name in ('train', 'val', 'test')}
    if any(not isinstance(values, list) or any(not isinstance(x, str) for x in values) for values in splits.values()):
        raise ValueError('released split schema mismatch')
    prompts = read(archive / 'prompts/sarcasm.json')
    if not isinstance(prompts.get('plain'), str) or not isinstance(prompts.get('style'), list) or any(not isinstance(x, str) for x in prompts['style']):
        raise ValueError('declared instruction schema mismatch')
    originals = splits['test']; records = []; bank = {}
    before = {str(p): digest(p) for p in (manifest_file, *source_files.values(), ROOT / 'metadata/saps_tree.json')}
    for record in manifest['files']:
        name = record['path']; file = inside(ROOT, 'saps/' + name); sha = verify_blob(file, entries[name])
        if sha != record['sha256'] or record['bytes'] != entries[name]['size'] or record['git_blob_sha1'] != entries[name]['sha']:
            raise ValueError('download manifest hash/size mismatch')
        before[str(file)] = sha
        parts = PurePosixPath(name).parts; method, slug = parts[3:5]; alpha = 1.0 if method == 'fixed_projection' else MODELS[slug][1]
        records.append(summarize_file(file, method=method, slug=slug, alpha=alpha, originals=originals,
                                      bank=bank, released_splits=splits, prompts=prompts))
    if any(digest(Path(p)) != h for p, h in before.items()):
        raise ValueError('input changed during summary')
    result = {'status': 'test_lineage_mismatch' if any(r['sentence_lineage']['status'] != 'match' for r in records) else 'released_test_lineage_matches',
              'scope': 'CPU arithmetic reconstruction over embedded published IDs/text, not target-model or released-test-input reproduction',
              'revision': SOURCES['saps'][1], 'manifest_sha256': digest(manifest_file),
              'original_test_sha256': digest(source), 'official_oracle_script_sha256': digest(script), 'conditions': records,
              'declared_instruction_sha256': digest(archive / 'prompts/sarcasm.json'),
              'released_split_sha256': {name: digest(archive / f'data/sarcasm/{name}.json') for name in splits},
              'all_conditions_id_text_consistent': True, 'input_sha256': {str(p.relative_to(REPO)): h for p, h in ((Path(k), v) for k, v in before.items())},
              'limits': ['Oracle maximizes style per TEST sentence across layers, never fixed validation-layer performance.',
                         'Official max key maps both score 0 and -1 to -1; stable ties select first input record.',
                         'Only averaged scores are public; 2/3 contributor counts do not recover raw evaluator judgments.',
                         'Embedded published text is compared, never normalized/reindexed/replaced by current released test inputs; any mismatch is a material release-lineage gap.',
                         'Legacy steer_random labels do not identify the method: when absent, method is inferred only from the pinned artifact path.',
                         'Saved prompt_used metadata is compared exactly with released plain/style instructions; actual model input and causes of any discrepancy are unknown.',
                         'No author code executed, no tensors deserialized, no model or judge called.']}
    save_new(output, result)
    return {'conditions': len(records), 'records': sum(r['records'] for r in records), 'output_sha256': digest(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['gcad-source', 'gcad-impolite', 'saps-results', 'summary'])
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--allow-public-output', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'saps_summary.json')
    args = parser.parse_args()
    if ROOT.is_symlink() or ROOT.resolve() != REPO / 'outputs/persona-author-replication-audit-01':
        raise ValueError('fixed audit root only')
    if args.stage == 'summary':
        if args.allow_download:
            raise ValueError('summary is offline only')
        result = summary(args.output.absolute(), args.allow_public_output)
    else:
        if args.allow_public_output or args.output != ROOT / 'saps_summary.json':
            raise ValueError('output flags only apply to offline summary')
        result = download(args.stage, args.allow_download)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
