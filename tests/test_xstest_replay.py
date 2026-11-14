"""Focused CPU checks: no target model, GPU or grader execution."""
import copy
from types import SimpleNamespace

import pytest
import torch

from scripts import xstest_replay as r


def sample(n=10):
    rows = [{'id': str(i), 'user': 'Synthetic user é ' + str(i)} for i in range(n)]
    values = []
    for i, row in enumerate(rows):
        start = i // 8 * 8
        members = rows[start:start+8]
        inputs = list(range(1, i % 3 + 3))
        width = max(len(list(range(1, j % 3 + 3))) for j in range(start, min(start+8, n)))
        values.append({'id': row['id'], 'input_ids': inputs, 'generated_ids': [7, 9],
            'rendered_prompt': 'Rendered é ' + str(i), 'output': ' Synthetic answer. ',
            'batch_index': i//8, 'batch_position': i%8, 'batch_ids': [x['id'] for x in members],
            'padded_prompt_width': width, 'ended_eos': True, 'hit_token_limit': False,
            'user_utf8_sha256': r.utf8_hash(row['user'])})
    return rows, values


def test_exact_comparison_ignores_timing_but_not_generated_pad():
    rows, values = sample(); new = copy.deepcopy(values)
    for value in new: value['timing'] = 999
    result = r.compare(rows, values, new, ['0'], ['0', '1'])
    assert result['all_match'] and result['parent_labels_reused_by_reference_only']
    new[0]['generated_ids'] = [7, 0]
    result = r.compare(rows, values, new, ['0'], ['0', '1'])
    assert not result['all_match'] and not result['parent_labels_reused_by_reference_only']
    assert result['mismatches_among_parent_17_refusals'] == 1
    assert result['mismatches_among_reviewed_19'] == 1
    assert result['differences'][0]['first_differing_generated_token_position'] == 1


@pytest.mark.parametrize('field,value', [('id','different'), ('output','Synthetic answer.'),
    ('rendered_prompt','different'), ('input_ids',[1,99]), ('batch_ids',['different']),
    ('batch_position',7), ('padded_prompt_width',99), ('ended_eos',False),
    ('hit_token_limit',True), ('user_utf8_sha256','wrong')])
def test_required_comparison_fields(field, value):
    rows, old = sample(); new = copy.deepcopy(old); new[0][field] = value
    result = r.compare(rows, old, new, [], [])
    assert result['mismatched_items'] == 1 and result['differences'][0]['fields']


def test_length_order_and_missing_row_guards():
    assert r.first_difference([1,2],[1]) == 1
    rows, values = sample()
    with pytest.raises(ValueError): r.compare(rows, values, values[:-1], [], [])
    with pytest.raises(ValueError): r.compare(rows[::-1], values, values, [], [])


def test_derived_left_padding_includes_last_two_rows_and_layout_guard():
    _, values = sample(); batches = r.derived_batches(values, 0)
    assert len(batches) == 2 and len(batches[-1]['ids']) == 2
    assert batches[0]['input_ids'][0] == [0,0,1,2]
    assert batches[0]['attention_mask'][0] == [0,0,1,1]
    bad = copy.deepcopy(values); bad[0]['padded_prompt_width'] += 1
    with pytest.raises(ValueError): r.derived_batches(bad, 0)


class Tiny(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        return input_ids


def test_actual_prefill_mask_capture_decode_skip_and_cleanup():
    _, values = sample(); expected = r.derived_batches(values, 0); model = Tiny()
    with r.capture_prefills(model, expected) as captured:
        for batch in expected:
            model(input_ids=torch.tensor(batch['input_ids']), attention_mask=torch.tensor(batch['attention_mask']))
            model(input_ids=torch.ones(len(batch['ids']),1,dtype=torch.long), attention_mask=torch.ones(len(batch['ids']),8))
    assert len(captured) == 2 and not model._forward_pre_hooks
    with pytest.raises(ValueError), r.capture_prefills(model, expected):
        batch = expected[0]
        model(input_ids=torch.tensor(batch['input_ids']), attention_mask=torch.ones_like(torch.tensor(batch['input_ids'])))
    assert not model._forward_pre_hooks
    with pytest.raises(RuntimeError), r.capture_prefills(model, expected):
        raise RuntimeError('synthetic execution failure')
    assert not model._forward_pre_hooks


def test_missing_prefill_fails_and_unknown_parent_environment_is_unavailable():
    with pytest.raises(ValueError), r.capture_prefills(Tiny(), [{'unused': True}]): pass
    result = r.environment_comparison({'python':'3.10','driver':'same'}, {'driver':'same'})
    assert result['python']['comparison'] == 'unavailable'
    assert result['python']['parent'] is None and result['driver']['comparison'] == 'match'


def test_parent_write_rejected_and_no_execution_flag_harmless(tmp_path):
    with pytest.raises(ValueError): r.guard(r.path('outputs/xstest-baseline-01'))
    with pytest.raises(ValueError): r.generate(SimpleNamespace(execute_gpu=False))
    assert not list(tmp_path.iterdir())


def test_hash_tamper_and_persistent_close(tmp_path):
    source = tmp_path/'source'; source.write_text('original')
    hashes = {str(source): r.stream_hash(source)}; r.parent.check_hashes(hashes)
    source.write_text('changed')
    with pytest.raises(ValueError): r.parent.check_hashes(hashes)
    r.closed(tmp_path, 'mismatch_stop'); r.closed(tmp_path, 'replacement')
    assert r.read_json(tmp_path/'closed.json')['stop_cause'] == 'mismatch_stop'
