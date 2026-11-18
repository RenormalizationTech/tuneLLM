import pytest
from scripts.qwen38_budget import LIMITS, reserve


def ledger():
    return {'limits_gpu_hours': LIMITS, 'runs': {}}


def test_shared_run_is_debited_once_and_caps_hold():
    x = ledger()
    reserve(x, 'shared', 2, {'qasper_eval': .5, 'prl_eval': .5})
    reserve(x, 'prl_remaining', 5, {'prl_eval': 1})
    with pytest.raises(ValueError, match='exhausted'):
        reserve(x, 'overflow', .01, {'prl_eval': 1})


def test_settled_actual_releases_reserved_time():
    x = ledger()
    reserve(x, 'quant', 6, {'quant': 1})
    x['runs']['quant']['actual_gpu_hours'] = 1
    reserve(x, 'remaining', 5, {'quant': 1})
    with pytest.raises(ValueError, match='already'):
        reserve(x, 'remaining', 1, {'quant': 1})


@pytest.mark.parametrize('hours,shares', [(float('nan'), {'quant': 1}), (1, {'quant': .5}),
    (1, {'quant': 1, 'prl_eval': -1}), (1, {'new_budget': 1})])
def test_invalid_reservations_rejected(hours, shares):
    with pytest.raises(ValueError):
        reserve(ledger(), 'invalid', hours, shares)
