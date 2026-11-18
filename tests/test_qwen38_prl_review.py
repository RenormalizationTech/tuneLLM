import pytest
from scripts.qwen38_prl_review import packet, check


def fixture(candidates=None):
    paper = {'id':'paper', 'selected_body':'We obtain a positive gap of two units.',
             'assistant':'SECRET GOLD ABSTRACT'}
    payload, mapping = packet(paper, candidates or {'base':'A positive gap is obtained.',
        'prompt':'The gap is two units.', 'adapter':'We obtain a positive gap.'},0,0)
    value = {'packet_id':payload['packet_id'], 'support':[{'sid':s['sid'],
        'label':'SUPPORTED','quote':'We obtain a positive gap of two units.'}
        for c in payload['candidates'] for s in c['sentences']],
        'preferences':[{'pair':p,'preferred':p[0],'reason':'Supported and useful.'} for p in payload['pairs']]}
    return payload,mapping,value


def test_three_candidate_blinding_and_completeness():
    p,m,v = fixture()
    assert len(p['pairs']) == 3
    assert all(arm not in str(p) for arm in ['adapter','SECRET GOLD ABSTRACT'])
    assert set(m.values()) == {'base','prompt','adapter'}
    assert len(check(v,p)['labels']) == 3
    v['support'].pop()
    with pytest.raises(ValueError,match='sentence'): check(v,p)


def test_fabricated_quote_downgraded():
    p,_,v=fixture()
    v['support'][0]['quote']='We obtain a negative gap.'
    assert check(v,p)['labels'][v['support'][0]['sid']] == 'UNSUPPORTED'


def test_duplicate_pair_and_wrong_packet_rejected():
    p,_,v=fixture()
    v['preferences'][1]=v['preferences'][0]
    with pytest.raises(ValueError,match='duplicate'): check(v,p)
    v['packet_id']='wrong'
    with pytest.raises(ValueError,match='identity'): check(v,p)


def test_empty_candidate_deterministically_loses():
    p,m,v=fixture({'base':'','verify':'The gap is positive.'})
    empty=next(k for k,a in m.items() if a=='base')
    other=next(k for k,a in m.items() if a=='verify')
    v['preferences'][0]['preferred']=empty
    assert set(check(v,p)['winners'].values()) == {other}


def test_excess_framing_cannot_hide_unsupported_claims():
    p,_,v=fixture()
    for s in v['support']: s['label']='FRAMING';s['quote']=None
    assert set(check(v,p)['labels'].values()) == {'UNSUPPORTED'}


def test_pair_orientation_and_reason_length_are_cosmetic():
    p,_,v=fixture()
    v['preferences'][0]['pair']=list(reversed(v['preferences'][0]['pair']))
    v['preferences'][0]['reason']='word '*41
    assert check(v,p)['flags'] == ['long_preference_reason']


def test_two_candidate_final_parser():
    p,m,v=fixture({'base':'A positive gap is obtained.','adapter':'The gap is two units.'})
    assert set(m.values()) == {'base','adapter'}
    assert len(check(v,p)['winners']) == 1
    assert p['evidence_excerpt'] == 'We obtain a positive gap of two units.'
