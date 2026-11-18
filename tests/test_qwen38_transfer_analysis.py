import pytest
from scripts.qwen38_transfer_analysis import compare


def test_token_weighted_paired_cluster_estimate():
    base=[{'id':'a','paper_id':'p','tokens':9,'nll':1},
          {'id':'b','paper_id':'q','tokens':1,'nll':1}]
    candidate=[{**base[0],'nll':2},{**base[1],'nll':0}]
    r=compare(candidate,base,1000)
    assert r['difference_candidate_minus_base']==pytest.approx(.8)
    assert r['ci95']==[-1,1]
    assert compare(candidate[::-1],base[::-1],1000)==r


def test_missing_or_unpaired_tokens_are_rejected():
    a=[{'id':'a','paper_id':'p','tokens':2,'nll':1}]
    for b in [[],a*2,[{**a[0],'tokens':1}],[{**a[0],'paper_id':'q'}],[{**a[0],'nll':float('nan')}]]:
        with pytest.raises(ValueError):compare(a,b,10)
