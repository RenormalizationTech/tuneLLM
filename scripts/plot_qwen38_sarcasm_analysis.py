"""Plot completed authoritative C6 point estimates; no live-run reads or calls.

Independent presentation only. References are the registered mean thresholds;
the plot does not evaluate gates, select a strength, or create statistics.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
STYLE = ROOT/'scripts/plot_portfolio_results.py'
PALETTE = ['#475569', '#d97706', '#15803d', '#b91c1c']


def sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def pyplot():
    os.environ.setdefault('MPLCONFIGDIR','/tmp/tunellm-sarcasm-figure-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'svg.hashsalt':'tunellm-portfolio'})
    return plt


def displayed(data):
    outcome = data.get('outcome')
    if outcome == 'development_null':
        selection = data.get('development',data.get('selection',{}))
        metrics = selection.get('metrics',{})
        arms = ['baseline','prompted','gcad_1','gcad_2']
        labels = ['Baseline','Prompted','GCAD b=1','GCAD b=2']
        stage = 'Development32 · development_null'
        impolite = None
    elif outcome in ('retain','final_failure','primary_complete_descriptive_incomplete'):
        if data.get('registration_verified') is not True:
            raise ValueError('final figure requires the registered authoritative C6 analysis')
        metrics = data.get('primary_metrics',{})
        arms = ['baseline','prompted','selected','random']
        strength = data.get('selected_strength','selected GCAD').replace('gcad_','b=')
        labels = ['Baseline','Prompted',f'GCAD {strength}',f'Random {strength}']
        stage = f'Final32 · {outcome}'
        impolite = data.get('impoliteness',{}).get('arm_means')
        if outcome == 'primary_complete_descriptive_incomplete':
            impolite = None
    else:
        raise ValueError(f'No plot for outcome {outcome!r}: completed development-null or final primary analysis required.')
    if set(metrics) != set(arms):
        raise ValueError('exact completed four-arm metrics required')
    rows = []
    for arm,label in zip(arms,labels):
        m = metrics[arm]
        if m.get('n') != 32:
            raise ValueError('n=32 required for each displayed arm')
        row = {'arm':arm,'label':label,'n':32,'sarcasm':m['trait_conservative'],
               'coherence':m['coherence'],'caps':m['caps']}
        for key in ('sarcasm','coherence'):
            if type(row[key]) not in (int,float) or not math.isfinite(row[key]) or not 0<=row[key]<=100:
                raise ValueError('finite mean scores in0..100 required')
        if type(row['caps']) is not int or not 0<=row['caps']<=32:
            raise ValueError('exact cap count0..32 required')
        if impolite is not None:
            if set(impolite)!=set(arms) or type(impolite[arm]) not in (int,float) or not math.isfinite(impolite[arm]) or not 0<=impolite[arm]<=100:
                raise ValueError('complete numeric impoliteness means required for optional panel')
            row['impoliteness'] = impolite[arm]
        rows.append(row)
    baseline = rows[0]
    return stage,rows,{'sarcasm_baseline_plus20':baseline['sarcasm']+20,
                       'coherence_absolute_floor':50,'coherence_baseline_minus20':baseline['coherence']-20,
                       'coherence_combined_floor':max(50,baseline['coherence']-20)}


def draw(analysis, output):
    raw = analysis.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw); stage,rows,refs = displayed(data)
    # Validate before creating a fresh output directory. Never overwrite assets.
    if output.resolve().is_relative_to(Path(data['run']).resolve()):
        raise ValueError('figure output must be outside the study run root')
    if output.exists():
        raise FileExistsError('fresh figure output directory required')
    plt = pyplot()
    output.mkdir(parents=True,exist_ok=False)
    panels = [('sarcasm','Sarcasm'),('coherence','Coherence')]
    if 'impoliteness' in rows[0]:
        panels.append(('impoliteness','Impoliteness · descriptive'))
    fig,axes = plt.subplots(1,len(panels),figsize=(5.0*len(panels),4.9),sharey=True,squeeze=False)
    try:
        for ax,(key,title) in zip(axes[0],panels):
            values = [r[key] for r in rows]
            ax.scatter(values,range(4),c=PALETTE,s=65,zorder=3)
            for y,value in enumerate(values):
                ax.annotate(f'{value:.2f}',(value,y),xytext=(-7 if value>90 else 7,0),
                            textcoords='offset points',ha='right' if value>90 else 'left',va='center',fontsize=9)
            if key in ('sarcasm','coherence'):
                threshold = refs['sarcasm_baseline_plus20'] if key=='sarcasm' else refs['coherence_combined_floor']
                label = 'Baseline + 20' if key=='sarcasm' else 'max(50, baseline − 20)'
                if threshold<=100:
                    ax.axvline(threshold,color='#64748b',ls='--',lw=1.1,zorder=1)
                ax.text(.02,.98,f'{label} = {threshold:.2f}'+(' (outside axis)' if threshold>100 else ''),
                        transform=ax.transAxes,va='top',fontsize=8.5,color='#475569')
            ax.set(title=title,xlim=(0,100),xticks=[0,20,40,60,80,100],xlabel='AI-judge mean score (0–100)',ylim=(3.55,-.65))
            ax.set_yticks(range(4),[r['label'] for r in rows])
            ax.set_axisbelow(True);ax.grid(axis='x',alpha=.18)
            for side in ('top','right'):
                ax.spines[side].set_visible(False)
        fig.suptitle('Qwen3.8-27B NF4 · sarcasm GCAD',fontsize=16,fontweight='bold',y=.97)
        fig.text(.5,.875,stage+' · n=32 per arm · AI-judge point estimates',ha='center',fontsize=10)
        footer = ('Mean references only. Full registered gates also include per-answer count, invalid-output, and final random-control checks.\n'
                  'No uncertainty intervals or new quality decisions. Sarcasm uses the conservative mean (refusals contribute zero).\n')
        footer += ('Development questions reused for strength selection; no final evaluation is shown.' if data['outcome']=='development_null'
                   else 'Final32 is reused cross-model transfer evaluation; not new project-held-out data.')
        if data['outcome']=='primary_complete_descriptive_incomplete':
            footer += ' Impoliteness incomplete; panel omitted.'
        fig.text(.02,.035,footer,fontsize=8.5,va='bottom')
        fig.subplots_adjust(left=.16 if len(panels)==2 else .12,right=.97,bottom=.27,top=.76,wspace=.22)
        fig.savefig(output/'sarcasm_gcad.svg',metadata={'Date':None})
        fig.savefig(output/'sarcasm_gcad.png',dpi=160)
    finally:
        plt.close(fig)
    if sha(analysis)!=digest:
        raise ValueError('analysis changed during rendering; preserve the output attempt')
    receipt = {'stage':stage,'outcome':data['outcome'],'source_analysis':str(analysis.resolve()),
        'source_sha256':digest,'generator_sha256':sha(__file__),'style_reference':str(STYLE.relative_to(ROOT)),
        'style_reference_sha256':sha(STYLE),'r0_manifest_sha256':data.get('r0_manifest_sha256'),
        'analyzer_sha256':data.get('analyzer_sha256'),'registration':data.get('registration'),
        'source_artifact_sha256':data.get('input_sha256',{}),'panels':[k for k,_ in panels],
        'displayed_point_estimates':rows,'registered_mean_references':refs,
        'scope':'Presentation of existing point estimates only; no new statistics, confidence intervals, grading, selection or gate decisions.',
        'versions':{'python':platform.python_version(),'matplotlib':importlib.metadata.version('matplotlib')},
        'outputs_sha256':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}}
    with (output/'receipt.json').open('x') as stream:
        json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps({'output':str(output.resolve()),'stage':stage,'receipt_sha256':sha(output/'receipt.json')},indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,help='Completed authoritative C6 analysis JSON; never a live run directory')
    p.add_argument('--output',type=Path,help='Fresh directory for SVG, PNG and receipt.json')
    p.add_argument('--check-environment',action='store_true',help='Import the plotting backend without reading study data or creating a figure')
    args = p.parse_args()
    if args.check_environment:
        plt = pyplot();print(json.dumps({'backend':plt.get_backend(),'matplotlib':importlib.metadata.version('matplotlib'),
                                       'font_family':plt.rcParams['font.family'],'figure_created':False}));return
    if args.analysis is None or args.output is None:
        p.error('--analysis and --output are required unless --check-environment is set')
    try:
        draw(args.analysis,args.output)
    except (ValueError,FileExistsError) as exc:
        p.exit(2,str(exc)+'\n')


if __name__=='__main__':
    main()
