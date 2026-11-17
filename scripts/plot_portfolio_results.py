"""Plot the frozen 7B development counts; no model or grader execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = ['results/refusal_7b_transfer_01.json', 'results/refusal_7b_halfdose_01.json']
    initial, half = [json.loads((ROOT/p).read_text()) for p in files]
    arms = [('Baseline', initial['development']['baseline_metrics'], 'm0'),
            ('Projection removal', initial['development']['candidate_metrics'], 'ablate_all'),
            ('Subtraction −0.25', half['development']['candidate_metrics'], 'subtract18'),
            ('Subtraction −0.5', initial['development']['candidate_metrics'], 'subtract18')]
    rows = []
    for label, metrics, prefix in arms:
        safe, unsafe = metrics[prefix+'_safe'], metrics[prefix+'_unsafe']
        assert safe['total'] == unsafe['total'] == 30
        rows.append({'arm': label, 'benign_refusals': safe['primary_refused'],
                     'benign_useful': safe['useful_and_correct'], 'unsafe_refusals': unsafe['primary_refused']})
    out = ROOT/'docs/assets'; out.mkdir(exist_ok=True)
    provenance = {'scope': 'Development only; same 30 benign/30 source-unsafe prompts. AI judged.',
                  'grading_context': 'Baseline labels reused from joint 180-item packet; minus0.25 candidate labels from a separate packet.',
                  'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}, 'rows': rows}
    (out/'refusal_tradeoff.json').write_text(json.dumps(provenance, indent=2)+'\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.hashsalt': 'tunellm-portfolio'})
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.6), sharey=True)
    colors = ['#475569', '#d97706', '#15803d', '#b91c1c']
    for ax, (key, title) in zip(axes, [('benign_refusals', 'Benign requests\nRefusals'),
                                    ('benign_useful', 'Benign requests\nUseful and correct'),
                                    ('unsafe_refusals', 'Unsafe contrasts\nRefusals')]):
        values = [r[key] for r in rows]
        ax.barh(range(len(rows)), values, color=colors, height=.58)
        for i, value in enumerate(values): ax.text(value+.5, i, str(value), va='center')
        ax.set(title=title, xlim=(0, 32), xticks=[0, 10, 20, 30], xlabel='Answers out of 30')
        ax.set_yticks(range(len(rows)), [r['arm'] for r in rows])
        ax.set_axisbelow(True); ax.grid(axis='x', alpha=.18)
        for side in ['top', 'right']: ax.spines[side].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle('Qwen2.5-7B refusal steering', fontsize=16, fontweight='bold', y=.97)
    fig.text(.5, .865, 'Refusal reduction and answer quality change together', ha='center', fontsize=11)
    fig.text(.02, .025, 'Development counts, AI-judged. Neither subtraction dose met the original acceptance rule.\n'
             'The −0.25 candidate was graded separately; original baseline labels were reused.\n'
             'The later 13-case diagnostic is separate. No selective-safety or general-reliability claim.', fontsize=9)
    fig.subplots_adjust(left=.17, right=.97, bottom=.25, top=.74, wspace=.24)
    fig.savefig(out/'refusal_tradeoff.svg', metadata={'Date': None})
    fig.savefig(out/'refusal_tradeoff.png', dpi=160)
    plt.close(fig)
    print('Wrote docs/assets/refusal_tradeoff.svg, .png and source-count JSON')


if __name__ == '__main__': main()
