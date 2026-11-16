"""Synthetic pure audit checks; never execute the real audit or read its outputs."""
import copy,json
from pathlib import Path
import pytest
from scripts import joint_gcad_lexical_audit as audit


def inputs():
    predictions={};scores={};primary={'analysis':{'metrics':{}}}
    for ai,arm in enumerate(audit.frozen.arms()):
        rows=[]
        for i in range(24):
            text='damn useful answer' if (i+ai)%3 else 'Clear useful answer'
            rows.append({'id':str(i),'output':text})
            scores[arm,str(i)]={m:60 if m=='coherence' else i for m in audit.frozen.METRICS}
        predictions[arm]=rows
        primary['analysis']['metrics'][arm]={'lexicon_hits':16,'invalid':0,**{m:60 if m=='coherence' else 11.5 for m in audit.frozen.METRICS}}
    return predictions,scores,primary


def test_exact_lexicon_case_boundaries_compounds_and_undefined_denominator():
    assert len(audit.lex.LEXICON)==11
    row=audit.row_counts('FUCK-shit!')
    assert row['hits']==2 and row['words']==1 and row['rate']==200
    assert audit.row_counts('assess class hello Scunthorpe')['hits']==0
    assert audit.row_counts('   ')['rate'] is None and audit.row_counts('   ')['invalid']
    repeated=audit.row_counts('damn damn damn damn damn damn')
    assert repeated['invalid'] and repeated['hits']==6


def test_primary_mean_vs_secondary_pooling_and_valid_invalid_splits():
    rows=[audit.row_counts('damn'),audit.row_counts('one two three four five six seven eight nine')]
    value=audit.summary(rows)
    assert value['mean_per_response_rate_primary']==50 and value['pooled_rate_secondary']==10
    assert audit.summary([])['mean_per_response_rate_primary'] is None


def test_all_arm_arithmetic_fixed_scores_and_paired_bootstrap_deterministic():
    p,s,r=inputs();a=audit.analyze(p,s,r,repetitions=30);b=audit.analyze(p,s,r,repetitions=30)
    assert a==b and a['arms']['baseline']['n_any_hit']==16
    assert a['arms']['baseline']['mean_per_response_rate_primary']==pytest.approx(100*16/72)
    assert a['arms']['baseline']['mean_rate_minus_baseline_ci95']['ci95']==[0,0]
    assert a['correlations']['within_arm']['baseline']['coherence']=={'n':24,'spearman':None}
    assert a['checks']['all_original_score_means_match'] and not a['primary_decisions_modified']
    assert all(v['n_invalid']==0 for v in a['arms'].values())

@pytest.mark.parametrize('bad',['lexhit','invalid','score','coherence_null','ids'])
def test_original_accounting_or_identity_mismatch_stops(bad):
    p,s,r=inputs()
    if bad=='lexhit':r['analysis']['metrics']['baseline']['lexicon_hits']=15
    elif bad=='invalid':r['analysis']['metrics']['baseline']['invalid']=1
    elif bad=='score':r['analysis']['metrics']['baseline']['sarcasm']=1
    elif bad=='coherence_null':s['baseline','0']['coherence']=None
    else:p['baseline'][1]['id']='0'
    with pytest.raises(ValueError):audit.analyze(p,s,r,repetitions=3)


def test_registration_hash_tamper_rejected_before_analysis(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path);monkeypatch.setattr(audit,'SOURCES',['source.py']);monkeypatch.setattr(audit,'INPUTS',['input.json']);monkeypatch.setattr(audit,'REG','reg.json')
    (tmp_path/'source.py').write_text('# harmless')
    (tmp_path/'input.json').write_text('{}')
    record={'post_hoc':True,'sources_sha256':{'source.py':audit.sha('source.py')},'inputs_sha256':{'input.json':'0'*64}}
    (tmp_path/'reg.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='registered hash mismatch'):audit.registered_inputs()
    assert not (tmp_path/'results').exists()


def test_no_execute_cli_does_not_read_actual_inputs_or_create_result(monkeypatch):
    import runpy,sys
    monkeypatch.setattr(sys,'argv',['joint_gcad_lexical_audit'])
    before=(Path(audit.ROOT)/audit.RESULT).exists()
    runpy.run_module('scripts.joint_gcad_lexical_audit',run_name='__main__')
    assert (Path(audit.ROOT)/audit.RESULT).exists()==before
