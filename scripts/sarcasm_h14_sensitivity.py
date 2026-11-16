"""Post-run descriptive H14 exclusion only; never modifies the primary study."""
import argparse
from pathlib import Path

from scripts import sarcasm_method_transfer as study
from scripts.common import path,read_json
from scripts.save_nf4 import stream_hash

REGISTRATION='results/protocols/sarcasm-method-transfer-sensitivity-01.json'
OUTPUT='results/diagnostics/sarcasm_h14_sensitivity_01.json'


def summarize(rows,scores,impolite=None):
    """Variable denominator; REFUSAL->0, matching primary conservative means."""
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('nonempty unique matched source IDs required')
    n=len(rows);traits=[];coherence=[];offaxis=[]
    for row in rows:
        grade=scores[row['id']];t=grade['trait'];c=grade['coherence']
        if (t is not None and (type(t) is not int or not 0<=t<=100)) or type(c) is not int or not 0<=c<=100:raise ValueError('valid accepted primary scores required')
        traits.append(t);coherence.append(c)
        if impolite is not None:
            value=impolite[row['id']]
            if value is not None and (type(value) is not int or not 0<=value<=100):raise ValueError('valid accepted descriptive scores required')
            offaxis.append(value)
    count=sum(v is not None and v>=50 for v in traits)
    result={'n':n,'mean_sarcasm_conservative':sum(v or 0 for v in traits)/n,'sarcasm_refusals':sum(v is None for v in traits),
            'mean_coherence':sum(coherence)/n,'sarcasm_ge50_count':count,'sarcasm_ge50_fraction':count/n,
            'profanity_lexicon_response_count':sum(bool(study.p.shared.text.hits(r['output'])) for r in rows)}
    if impolite is not None:result['descriptive_impolite']={'mean_conservative':sum(v or 0 for v in offaxis)/n,'refusals':sum(v is None for v in offaxis)}
    return result


def analyze():
    # Gate FIRST: do not inspect active predictions, snapshots or journal data.
    cfg=read_json(study.CONFIG);report_file=path(cfg['result'])
    if not report_file.is_file():raise ValueError('Post-run only: final report absent; no run data inspected or output written')
    report=read_json(report_file);run=path(study.RUN)
    if report['status'] not in ('complete','artifact_null','development_null','incomplete'):raise ValueError('Post-run only: study not closed')
    if report['status']=='incomplete' and not (run/'stopped.json').is_file():raise ValueError('Incomplete run is not terminal')
    if (run/'heldout').exists() and not report.get('primary_result_completed'):raise ValueError('Held-out primary data incomplete; do not analyze partial ratings')
    s=study.Context();s.verify(models=False);s.registered();study.infra.committed(REGISTRATION);decl=read_json(REGISTRATION)
    if decl['excluded_ids']!=['H14'] or decl['primary_sample_size']!=32 or decl['descriptive_sensitivity_sample_size']!=31 or decl['declared_before_any_qwen_call'] is not True or decl['primary_decisions_unchanged'] is not True or decl['no_reselection_or_extra_model_calls'] is not True:
        raise ValueError('fixed sensitivity declaration differs')
    fixture=read_json(cfg['fixture']) if path(cfg['fixture']).exists() else read_json(run/'preview.json')['data']
    if decl['manifest_sha256']!=stream_hash(run/'manifest.json') or decl['preview_sha256']!=stream_hash(run/'preview.json') or decl['fixture_canonical_sha256']!=study.f.canonical_hash(fixture) or decl['heldout_source_sha256']!=stream_hash(path(cfg['heldout_file'])):
        raise ValueError('sensitivity declaration/source bindings differ')
    refs=[path(__file__),path(REGISTRATION),report_file,run/'manifest.json',run/'preview.json',path(cfg['heldout_file'])]
    if path(cfg['fixture']).exists():refs.append(path(cfg['fixture']))
    result={'analysis':'predeclared_descriptive_H14_exclusion','status':'not_applicable','excluded_ids':['H14'],
            'primary_n':32,'sensitivity_n':31,'primary_report_status':report['status'],'primary_decisions_unchanged':True,
            'no_gates_recomputed':True,'no_reselection_or_extra_model_calls':True,'original_primary_confirmation_sha256':study.f.canonical_hash(report.get('confirmation')),
            'original_preferred':report.get('confirmation',{}).get('preferred'),
            'limits':['Descriptive pre-output topical-overlap sensitivity, not a new benchmark rating or primary decision.',
                      'REFUSAL remains zero in conservative means; denominators32/31 are explicit. No threshold is rescaled or re-applied.',
                      'Impoliteness, if available, is descriptive only; unavailable/unaccepted ratings are not inferred.',
                      'Removing one topical overlap does not establish full semantic independence or generalization.']}
    before={str(f.relative_to(path('.'))):stream_hash(f) for f in refs}
    if not (run/'heldout').exists():
        if report['status'] not in ('artifact_null','development_null','incomplete') or not (run/'stopped.json').is_file():raise ValueError('No heldout but no terminal closure')
        result['not_applicable_cause']='Closed study has no held-out stage; no held-out generation requested.'
    else:
        study.lock(s,'selection');records=study.predictions(s,'heldout');scores=study.scored(s,'heldout')
        available=(run/'native/heldout/impolite/accepted.json').is_file();offaxis=study.scored(s,'heldout','impolite') if available else None
        result['status']='complete';result['descriptive_impolite_available']=available;result['conditions']={}
        for arm,rows in records.items():
            if [r['id'] for r in rows]!=[f'H{i:02d}' for i in range(1,33)]:raise ValueError('original32 ordered held-out IDs required')
            joined={r['id']:scores[r['id'],arm] for r in rows};imp={r['id']:offaxis[r['id'],arm]['impolite'] for r in rows} if offaxis is not None else None
            primary=summarize(rows,joined,imp);kept=[r for r in rows if r['id']!='H14'];sensitivity=summarize(kept,joined,imp)
            if len(kept)!=31:raise ValueError('exactly one H14 exclusion required')
            original=report['heldout'][arm]
            if primary['mean_sarcasm_conservative']!=original['trait_conservative'] or primary['mean_coherence']!=original['coherence'] or primary['sarcasm_ge50_count']!=original['individual_trait_ge50'] or primary['profanity_lexicon_response_count']!=original['lexicon_hits']:raise ValueError('original32 aggregates do not reproduce')
            result['conditions'][arm]={'primary32':primary,'descriptive31_without_H14':sensitivity}
        refs.extend([run/'heldout/run.json',run/'heldout/integrity.json',run/'heldout/predictions.json',run/'selection.json',path(cfg['selection_registration'])])
        for metric in ('trait','coherence',*(['impolite'] if available else [])):
            directory=run/'native/heldout'/metric
            refs.extend(f for f in directory.glob('*.json') if f.is_file())
    # Frozen read helpers validate journals/stages. Hash every actual input again
    # around publication; no source/primary output is ever written.
    for file in refs:
        key=str(file.relative_to(path('.')))
        if key in before and stream_hash(file)!=before[key]:raise ValueError('input changed while analyzing')
        before.setdefault(key,stream_hash(file))
    result['input_sha256']=before
    study.native.public_safe(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=OUTPUT);args=parser.parse_args()
    output=path(args.output)
    if output.parent!=path('results/diagnostics') or output.suffix!='.json' or output.exists():raise ValueError('fresh results/diagnostics JSON destination only')
    result=analyze()
    study.infra.check_hashes([result['input_sha256']])
    study.save(output,result)
    print('Descriptive sensitivity saved: '+str(output.relative_to(path('.')))+'; '+result['status'])


if __name__=='__main__':main()
