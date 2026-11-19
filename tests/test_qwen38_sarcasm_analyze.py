"""C6 analyzer synthetic artifacts only; no model, judge, or real study data."""
import json
from pathlib import Path

import pytest
from scripts import qwen38_sarcasm_analyze as a


def write(root,name,value):
    file=root/name;file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(value)+'\n')
    return file


@pytest.fixture
def run(tmp_path):
    root=tmp_path/'run';root.mkdir()
    cfg=json.loads((a.ROOT/'configs/qwen38_sarcasm.json').read_text())
    write(root,'config.json',cfg)
    write(root,'manifest.json',{'frozen':{'config.json':a.sha(root/'config.json')}})
    return root


def registration(root,**kw):
    value={'analyzer_sha256':a.sha(a.__file__),'analyzer_tests_sha256':a.sha(a.TEST),
        'r0_manifest_sha256':a.sha(root/'manifest.json'),'registered_at_utc':'2000-01-01T00:00:00+00:00'}
    return write(root.parent,'analyzer-registration.json',{**value,**kw})


def final_fixture(root,impolite=True):
    predictions=[];frozen=[];streams={metric:{} for metric in ['trait','coherence','impolite']}
    for qi in range(1,33):
        for arm in a.ARMS:
            row={'id':f'H{qi:02d}_{arm}','question_id':f'H{qi:02d}','user':f'Question {qi}',
                 'arm':arm,'output':f'Check device {qi} and verify the connection carefully.',
                 'generated_ids':[3,qi+3,2],'hit_token_limit':False,'ended_eos':True}
            predictions.append(row)
            frozen.append({k:row[k] for k in ['id','question_id','user','arm']})
            streams['trait'][row['id']]=60+qi if arm=='selected' else {'baseline':10,'random':20,'prompted':70}[arm]
            streams['coherence'][row['id']]=0 if arm=='selected' and qi==14 else 80
            streams['impolite'][row['id']]=qi if arm=='selected' else 5
    write(root,'inputs.json',{'final':frozen});write(root,'final/result.json',{'completed':True,'rows':predictions})
    write(root,'eligibility.json',{'pairs':[str(i) for i in range(10)],'extraction_null':False})
    write(root,'selection.json',{'selection_null':False,'selected':{'gcad':'gcad_1'}})
    for metric,values in streams.items():
        if metric!='impolite' or impolite:write(root,f'judgments/final/{metric}.json',values)
    return predictions,streams


def test_complete_primary_gates_h14_coupling_and_example_provenance(run):
    predictions,streams=final_fixture(run)
    before={str(p.relative_to(run)):a.sha(p) for p in run.rglob('*') if p.is_file()}
    value=a.analyze(run,registration(run),as_of=100)
    assert value['outcome']=='retain' and value['primary_gates']['all_pass']
    assert value['primary_metrics']['selected']['coherence']==77.5
    assert value['h14_excluded']['arms']['selected']['n']==31
    assert value['h14_excluded']['arms']['selected']['coherence']==80
    assert value['impoliteness']['arm_means']['selected']==16.5
    assert value['impoliteness']['within_arm']['selected']['pearson']==pytest.approx(1)
    assert value['impoliteness']['within_arm']['baseline']['pearson'] is None
    assert value['impoliteness']['pooled']['n']==128
    assert 'confounded' in value['impoliteness']['pooled_caveat']
    example=value['examples']['records'][1]
    assert example['question_id']=='H14' and len(example['arms'])==4
    assert example['arms'][2]['provenance']['sha256']==a.sha(run/'final/result.json')
    assert example['arms'][2]['provenance']['score_sources']['trait']['sha256']==a.sha(run/'judgments/final/trait.json')
    assert before=={str(p.relative_to(run)):a.sha(p) for p in run.rglob('*') if p.is_file()}


def test_primary_complete_without_combined_or_impolite_scores(run):
    final_fixture(run,impolite=False)
    write(run,'judgments/final/rows/impolite-000-0.start.json',{'started_unix':1})
    write(run,'judgments/final/rows/impolite-000-0.failure.json',{'error':'runtime stop'})
    value=a.analyze(run,registration(run),as_of=100)
    assert value['outcome']=='primary_complete_descriptive_incomplete'
    assert value['primary_outcome']=='retain'
    assert value['impoliteness']['arm_means'] is None
    assert value['score_sources']=={'trait':'judgments/final/trait.json','coherence':'judgments/final/coherence.json'}
    assert len(value['ledger']['failure_records'])==1


def test_refusal_is_conservative_zero_and_excluded_from_correlation(run):
    rows,scores=final_fixture(run)
    scores['trait']['H01_selected']=None
    write(run,'judgments/final/trait.json',scores['trait'])
    value=a.analyze(run,registration(run),as_of=100)
    assert value['primary_metrics']['selected']['refusals']==1
    assert value['impoliteness']['within_arm']['selected']['n']==31
    assert value['primary_metrics']['selected']['trait_conservative']==sum(62+i for i in range(31))/32


@pytest.mark.parametrize('branch',['extraction','development'])
def test_explicit_null_closeouts(run,branch):
    if branch=='extraction':write(run,'eligibility.json',{'pairs':['one'],'extraction_null':True})
    else:write(run,'selection.json',{'selection_null':True,'selected':{}})
    value=a.analyze(run,as_of=100)
    assert value['outcome']==branch+'_null'
    assert 'primary_gates' not in value and 'examples' not in value


def test_incomplete_is_not_a_scientific_null_and_mechanics_failure_distinguished(run):
    assert a.analyze(run,as_of=100)['outcome']=='incomplete'
    write(run,'synthetic-smoke/rows/phase.start.json',{'started_unix':1})
    write(run,'synthetic-smoke/rows/phase.failure.json',{'error':'exact zero parity failed'})
    assert a.analyze(run,as_of=100)['outcome']=='port_or_mechanics_failure'


def test_ledger_counts_reservations_receipts_sessions_and_audit_separately(run):
    for i in range(2):write(run,f'counters/generation/{i}.json',{'stage':'elicitation','tokens':256})
    write(run,'counters/teacher_forced/0.json',{'stage':'synthetic-smoke','tokens':0})
    write(run,'gpu-sessions/complete.start.json',{'unix':10})
    write(run,'gpu-sessions/complete.done.json',{'seconds':60})
    write(run,'gpu-sessions/uncertain.start.json',{'unix':100})
    write(run,'judgments/elicitation/rows/trait-000-0.start.json',{'started_unix':1})
    write(run,'judge-audit/rows/audit.start.json',{'started_unix':1})
    row=write(run,'elicitation/rows/x00.json',{'output_ids':[3,4,2]})
    write(run,'elicitation/rows/x00.receipt.json',{'sha256':a.sha(row)})
    result=a.analyze(run,as_of=160)['ledger']
    assert result['counters']['generation']['reserved_attempts']==2
    assert result['counters']['generation']['reserved_maximum_tokens']==512
    assert result['completed_generation_receipts']['generations']==1
    assert result['completed_generation_receipts']['generated_tokens']==3
    assert result['gpu']['completed_session_seconds']==60
    assert result['gpu']['conservative_charged_seconds']==120
    assert result['judge']['study_attempt_starts']==1 and result['judge']['audit_attempt_starts']==1


def test_exact_ids_scores_and_prospective_registration_fail_closed(run):
    rows,scores=final_fixture(run)
    with pytest.raises(ValueError,match='registered analyzer'):a.analyze(run,as_of=100)
    with pytest.raises(ValueError,match='analyzer_sha256'):a.analyze(run,registration(run,analyzer_sha256='bad'),as_of=100)
    write(run,'final/rows/H01_baseline.start.json',{'started_unix':1})
    with pytest.raises(ValueError,match='postdates'):a.analyze(run,registration(run),as_of=100)
    (run/'final/rows/H01_baseline.start.json').unlink()
    scores['coherence'].pop('H01_baseline')
    write(run,'judgments/final/coherence.json',scores['coherence'])
    with pytest.raises(ValueError,match='exact complete'):a.analyze(run,registration(run),as_of=100)


def test_conflicting_combined_scores_rejected(run):
    rows,streams=final_fixture(run)
    scores={r['id']:{m:v[r['id']] for m,v in streams.items()} for r in rows}
    scores['H01_selected']['trait']=0
    write(run,'judgments/final/scores.json',scores)
    with pytest.raises(ValueError,match='disagree'):a.analyze(run,registration(run),as_of=100)


def test_missing_primary_stream_does_not_compute_gate(run):
    final_fixture(run,impolite=False)
    (run/'judgments/final/coherence.json').unlink()
    value=a.analyze(run,registration(run),as_of=100)
    assert value['outcome']=='incomplete' and 'primary_gates' not in value


def test_changed_frozen_config_rejected(run):
    write(run,'config.json',{})
    with pytest.raises(ValueError,match='frozen artifact changed'):a.analyze(run,as_of=100)


def test_corrupt_completed_row_receipt_rejected(run):
    write(run,'elicitation/rows/x00.json',{'output_ids':[3,4,2]})
    write(run,'elicitation/rows/x00.receipt.json',{'sha256':'bad'})
    with pytest.raises(ValueError,match='receipt hash differs'):a.analyze(run,as_of=100)


def test_h14_exclusion_cannot_rescue_full32_gate_failure(run):
    rows,streams=final_fixture(run)
    streams['coherence']={r['id']:(0 if r['question_id']=='H14' else 61) if r['arm']=='selected' else 80 for r in rows}
    write(run,'judgments/final/coherence.json',streams['coherence'])
    value=a.analyze(run,registration(run),as_of=100)
    assert value['outcome']=='final_failure'
    assert not value['primary_gates']['coherence_relative']
    assert value['h14_excluded']['arms']['selected']['coherence']==61


def test_random_margin_remains_primary_gate(run):
    rows,streams=final_fixture(run)
    for row in rows:
        if row['arm']=='random':streams['trait'][row['id']]=70
    write(run,'judgments/final/trait.json',streams['trait'])
    value=a.analyze(run,registration(run),as_of=100)
    assert value['outcome']=='final_failure' and not value['primary_gates']['trait_random_gain']


def test_combined_primary_fallback_and_frozen_helper_verification(run):
    rows,streams=final_fixture(run,impolite=False)
    for metric in ['trait','coherence']:(run/f'judgments/final/{metric}.json').unlink()
    scores={r['id']:{m:streams[m][r['id']] for m in ['trait','coherence']} for r in rows}
    write(run,'judgments/final/primary-scores.json',scores)
    assert a.analyze(run,registration(run),as_of=100)['primary_outcome']=='retain'
    manifest=json.loads((run/'manifest.json').read_text())
    manifest['files']={'scripts/sarcasm_method_transfer_common.py':'bad'}
    write(run,'manifest.json',manifest)
    with pytest.raises(ValueError,match='helper source changed'):a.analyze(run,registration(run),as_of=100)


def test_immutable_output_and_constant_correlations(tmp_path):
    out=tmp_path/'analysis.json'
    a.publish(out,{'ok':True})
    with pytest.raises(FileExistsError):a.publish(out,{'ok':False})
    assert json.loads(out.read_text())=={'ok':True}
    assert a.pearson([])['pearson'] is None
    assert a.pearson([(1,2),(1,4)])['pearson'] is None
    assert a.pearson([(1,4),(2,2)])['pearson']==pytest.approx(-1)
