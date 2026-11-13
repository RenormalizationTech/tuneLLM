"""CPU-only protocol/fixture/provenance checks; synthetic reviews stay in tmp_path."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import benign_helpfulness as b
from scripts.common import read_json


def rating(identifier="anonymous-test"):
    return {"id": identifier, "refusal": "none", "anchors": [True, False, True],
            **{k: False for k in b.FLAGS}, "reason": "Synthetic CPU test only."}


def test_real_fixtures_seal_counts_and_normalized_disjointness():
    cfg = read_json("configs/benign_helpfulness.json")
    b.validate_config(cfg)
    checks, sources = b.fixture_check(cfg)
    assert checks["exact_normalized_overlap"] == 0 and len(sources) == 14
    assert checks["normalized_historical_safe_prompts"] > 1000
    for phase in ("development", "heldout"):
        sealed = checks["seal"][phase]
        assert sealed["class_counts"] == {"sensitive": 24, "physics": 12}
        assert set(sealed["family_counts"].values()) == {4} and len(sealed["ids"]) == 36
    assert b.normalize(" Ａlpha\u2028 BETA  ") == b.normalize("alpha beta")


def test_fixture_missing_anchor_duplicates_and_heldout_generation_rejected():
    data = read_json("evaluation/benign_helpfulness/development.json")
    malformed = copy.deepcopy(data); malformed["items"][0]["anchors"].pop()
    with pytest.raises(ValueError): b.validate_fixture(malformed, "development")
    malformed = copy.deepcopy(data); malformed["items"][1]["user"] = "  "+malformed["items"][0]["user"].upper()
    with pytest.raises(ValueError): b.validate_fixture(malformed, "development")
    heldout = read_json("outputs/benign-helpfulness-heldout-01/fixture.json")
    with pytest.raises(ValueError): b.validate_fixture(heldout, "development")


def detail_rows(eos=36, refusals=4):
    return [{"id": str(i), "class": "sensitive" if i < 24 else "physics", "ended_eos": i < eos,
             "refusal": "partial" if i < refusals else "none", "counted_refusal": i < refusals,
             "empty": False, "mechanically_degenerate": False, "unscorable": False} for i in range(36)]


@pytest.mark.parametrize("eos,refusals,state", [(32,10,"inadequate_budget"),(33,3,"insufficient_headroom"),
    (33,4,"ready_for_separately_registered_comparison")])
def test_fixed_budget_headroom_boundaries(eos, refusals, state):
    rows = detail_rows(eos, refusals)
    coord = {r["id"]: r["refusal"] for r in rows if r["class"] == "sensitive"}
    result = b.decision(rows, coord)
    assert result["state"] == state
    assert not result["automatic_candidate_execution"] and not result["heldout_generated"]


def test_coordinator_disagreements_cross_gate_without_replacing_primary():
    rows = detail_rows()
    coord = {r["id"]: r["refusal"] for r in rows[:24]}
    coord['0'] = 'none'; coord['10'] = 'full'  # Same total, uncertainty can still cross the gate.
    result = b.decision(rows, coord)
    assert result['state'] == 'uncertain_readiness' and result['disagreement_sensitive_range'] == [3,5]
    assert result['primary_counted_refusals'] == 4 and not result['primary_labels_replaced']
    coord.pop('1')
    with pytest.raises(ValueError): b.decision(rows, coord)


def test_repetition_alone_not_excluded_but_empty_degenerate_unscorable_are():
    rows = [{"id": str(i), "class": "sensitive", "family": "synthetic"} for i in range(4)]
    values = [{"id": str(i), "output": text, "generated_ids": [3,4], "ended_eos": False, "hit_token_limit": False}
              for i,text in enumerate(("Synthetic short answer", "", "one two three four "*25, "Synthetic other answer"))]
    ratings = {r['id']: {**rating(r['id']), 'refusal':'full', 'repetition': True, 'unscorable': r['id']=='3'} for r in rows}
    measured = b.measurements(rows,values,ratings)
    assert [r['counted_refusal'] for r in measured] == [True,False,False,False]
    assert measured[0]['repetition'] and not measured[0]['mechanically_degenerate']


def test_schema_ids_three_boolean_anchors_and_flags():
    items = [{'id':'anonymous-test'}]
    b.validate_ratings(items,[rating()])
    for changed in ({'anchors':[True,1,False]},{'anchors':[True,False]},{'unscorable':0},{'reason':''},{'refusal':'unclear'}):
        with pytest.raises(ValueError): b.validate_ratings(items,[{**rating(),**changed}])
    with pytest.raises(ValueError): b.validate_ratings(items,[rating(),rating()])


def test_no_execution_without_authorization_and_no_candidate_config(tmp_path):
    with pytest.raises(ValueError,match='execute-gpu'):
        b.generate(SimpleNamespace(execute_gpu=False),tmp_path)
    with pytest.raises(ValueError,match='execute-judgments'):
        b.judge(SimpleNamespace(execute_judgments=False),tmp_path)
    cfg = read_json('configs/benign_helpfulness.json')
    with pytest.raises(ValueError): b.validate_config({**cfg,'system':'A candidate system prompt'})


@pytest.fixture
def journal(tmp_path):
    """One synthetic tools-off receipt; no actual wrapper/model invocation."""
    items = [{'id':'anonymous-test','request':'Synthetic request','response':'Synthetic output','anchors':['a','b','c']}]
    rubric = {'synthetic_schema':True}
    payload = b.judge_payloads(items,rubric)[0]; payload_hash = b.foundation.canonical_hash(payload)
    binding = {'maximum_attempts_per_batch':2,'batch_ids':[['anonymous-test']],'batch_stdin_sha256':[payload_hash]}
    b.save(tmp_path/'run_binding.json',binding); binding_hash = b.stream_hash(tmp_path/'run_binding.json')
    stem='batch-000-attempt-0'
    b.save(tmp_path/(stem+'-start.json'),{'run_binding_sha256':binding_hash,'canonical_stdin_sha256':payload_hash})
    attempt={'returncode':0,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
        'canonical_stdin_sha256':payload_hash,'stdout':json.dumps({'modelUsage':{b.foundation.MODEL:{}},'result':json.dumps({'ratings':[rating()]})})}
    b.save(tmp_path/(stem+'.json'),attempt)
    receipt={'file':stem+'.json','sha256':b.stream_hash(tmp_path/(stem+'.json')),
        'start_sha256':b.stream_hash(tmp_path/(stem+'-start.json')),'run_binding_sha256':binding_hash,'canonical_stdin_sha256':payload_hash}
    b.save(tmp_path/(stem+'-receipt.json'),receipt)
    batch={'batch_index':0,'ids':['anonymous-test'],'canonical_stdin_sha256':payload_hash,
        'attempt_files':[receipt],'first_valid_attempt':0,'actual_model_usage_ids':[b.foundation.MODEL]}
    b.save(tmp_path/'batch-000-input.json',{'batch_index':0,'ids':['anonymous-test'],'canonical_stdin_sha256':payload_hash,'attempt_files':[]})
    b.save(tmp_path/'batch-000-provenance.json',batch)
    return tmp_path/'submission.json',{'batches':[batch]},items,rubric


def test_first_valid_journal_replays_without_calls(journal,monkeypatch):
    def forbidden(*a,**k): raise AssertionError('CPU replay must not invoke a judge')
    monkeypatch.setattr(b.foundation,'invoke',forbidden)
    values, models = b.replay_saved(*journal)
    assert values == [rating()] and models == [b.foundation.MODEL]


@pytest.mark.parametrize('tamper',['receipt_payload','attempt_payload','journal_binding','batch_ids'])
def test_payload_receipt_start_and_membership_mismatch_rejected(journal,tamper):
    source,provenance,items,rubric = journal
    folder=source.parent; stem='batch-000-attempt-0'
    if tamper=='receipt_payload':
        provenance['batches'][0]['attempt_files'][0]['canonical_stdin_sha256']='mismatch'
        target=folder/'batch-000-provenance.json';target.write_text(json.dumps(provenance['batches'][0]))
    elif tamper=='attempt_payload':
        target=folder/(stem+'.json');value=read_json(target);value['canonical_stdin_sha256']='mismatch';target.write_text(json.dumps(value))
        receipt=provenance['batches'][0]['attempt_files'][0];receipt['sha256']=b.stream_hash(target)
        (folder/(stem+'-receipt.json')).write_text(json.dumps(receipt))
        (folder/'batch-000-provenance.json').write_text(json.dumps(provenance['batches'][0]))
    elif tamper=='journal_binding':
        target=folder/(stem+'-start.json');value=read_json(target);value['run_binding_sha256']='mismatch';target.write_text(json.dumps(value))
    else:
        target=folder/'run_binding.json';value=read_json(target);value['batch_ids']=[['wrong-id']];target.write_text(json.dumps(value))
    with pytest.raises(ValueError): b.replay_saved(source,provenance,items,rubric)


def test_model_or_infrastructure_error_not_schema_retry():
    with pytest.raises(RuntimeError): b.parse_attempt({'returncode':1},[{'id':'anonymous-test'}])
    bad={'returncode':0,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
        'stdout':json.dumps({'modelUsage':{'unexpected-model':{}},'result':'{}'})}
    with pytest.raises(b.foundation.ModelMismatch): b.parse_attempt(bad,[{'id':'anonymous-test'}])
