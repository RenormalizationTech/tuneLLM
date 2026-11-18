"""Parent GPU budget: locked reservations and hash-bound measured receipts.

Shared runs debit fractions summing to one, so two devices or shared baselines
cannot disappear from accounting. Subordinate retention ledgers add no budget.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path

from scripts.qwen38_science_runtime import sha, write

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'outputs/qwen38-science-budget-01.json'
LIMITS = {'quant': 6, 'qasper_train': 12, 'prl_train': 6,
          'qasper_eval': 12, 'prl_eval': 6, 'mechanics': 2}


def reserve(ledger, name, hours, shares):
    if name in ledger['runs']:
        raise ValueError('run already accounted; preserve original reservation')
    if (not math.isfinite(hours) or hours <= 0 or not shares
            or any(k not in LIMITS or not math.isfinite(v) or v <= 0 for k,v in shares.items())
            or not math.isclose(sum(shares.values()), 1, abs_tol=1e-12)):
        raise ValueError('positive finite hours and shares summing to one required')
    for pool, fraction in shares.items():
        used = sum(r.get('actual_gpu_hours', r['reserved_gpu_hours']) * r['shares'].get(pool, 0)
                   for r in ledger['runs'].values())
        if used + hours * fraction > LIMITS[pool] + 1e-10:
            raise ValueError(f'parent budget exhausted: {pool}')
    ledger['runs'][name] = {'reserved_gpu_hours': hours, 'shares': shares}


def settle(ledger, name, receipt, gpu_count):
    entry = ledger['runs'][name]
    if 'actual_gpu_hours' in entry:
        raise ValueError('run already settled')
    result = json.loads(receipt.read_text())
    elapsed = result.get('elapsed_seconds', result.get('seconds'))
    if type(gpu_count) is not int or gpu_count not in [1, 2] or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('valid measured elapsed time and device count required')
    if result.get('gpu_count', gpu_count) != gpu_count:
        raise ValueError('receipt device count differs')
    actual = elapsed * gpu_count / 3600
    entry.update(actual_gpu_hours=actual, gpu_count=gpu_count,
                 receipt=str(receipt.resolve().relative_to(ROOT)), receipt_sha256=sha(receipt),
                 completed=bool(result.get('completed')), cap_exceeded=actual > entry['reserved_gpu_hours'])


@contextmanager
def locked():
    with LEDGER.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {'limits_gpu_hours': LIMITS, 'runs': {}}
        if ledger['limits_gpu_hours'] != LIMITS:
            raise ValueError('parent limits changed')
        yield ledger
        write(LEDGER, ledger)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reserve')
    p.add_argument('--hours', type=float)
    p.add_argument('--shares', help='JSON pool fractions summing to one')
    p.add_argument('--settle')
    p.add_argument('--receipt', type=Path)
    p.add_argument('--gpu-count', type=int, default=1)
    a = p.parse_args()
    with locked() as ledger:
        if a.reserve:
            reserve(ledger, a.reserve, a.hours, json.loads(a.shares))
        elif a.settle:
            settle(ledger, a.settle, a.receipt, a.gpu_count)
        totals = {pool: sum(r.get('actual_gpu_hours', r['reserved_gpu_hours']) * r['shares'].get(pool, 0)
                           for r in ledger['runs'].values()) for pool in LIMITS}
        print(json.dumps({'accounted_gpu_hours': totals, 'limits_gpu_hours': LIMITS}, indent=2))


if __name__ == '__main__':
    main()
