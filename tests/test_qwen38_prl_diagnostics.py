from scripts.qwen38_prl_diagnostics import lcs_f1,describe


def test_lcs_is_sequence_sensitive_and_length_penalized():
    assert lcs_f1('a b c','a b c')==1
    assert lcs_f1('a b c','c b a')==1/3
    assert lcs_f1('a','a b c')==.5
    assert lcs_f1('','a')==0


def test_secondary_metrics_do_not_turn_copying_into_grounding_claim():
    row={'assistant':'One two three four five six seven eight.',
         'selected_body':'One two three four five six seven eight.'}
    copied=describe(row,row['assistant'])
    assert copied['custom_whitespace_lcs_f1']==1
    assert copied['novel_unique_8gram_fraction']==0
    assert describe(row,'Entirely different words here.')['novel_unique_8gram_fraction'] is None
