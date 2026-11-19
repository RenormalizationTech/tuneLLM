"""Standalone presentation of the eventual immutable follow-up closeout JSON.

Outside the experimental source closure. Reads existing means and intervals;
never opens run tensors, recomputes statistics, chooses a dose, or calls a
model/judge. A pending/incomplete result produces no assets.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS=ROOT/'results/qwen38_sarcasm_followup_01.json'
DEFAULT_PREFIX=ROOT/'docs/assets/qwen38_sarcasm_followup'
STYLE=ROOT/'scripts/plot_qwen38_sarcasm_analysis.py'
SITES=(23,27,31,35,39,43)
DOSES=(2,4,8,16)
ARMS=('baseline','prompted','b2','b4','b8','b16','random16')
SLATE='#475569';GREEN='#15803d';AMBER='#d97706';RED='#b91c1c'
DOSE_COLORS={2:'#86b59a',4:'#4e9970',8:'#15803d',16:'#14532d'}


class Pending(ValueError):pass


def sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def pyplot():
    os.environ.setdefault('MPLCONFIGDIR','/tmp/tunellm-followup-figure-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'svg.hashsalt':'tunellm-portfolio'})
    return plt


def number(value,where,low=None,high=None):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError(f'Nonfinite/missing recorded value: {where}')
    if low is not None and value<low or high is not None and value>high:
        raise ValueError(f'Recorded value out of range: {where}')
    return value


def interval(value,where,low=None,high=None):
    if value is None:return None
    if not isinstance(value,list) or len(value)!=2:raise ValueError(f'Invalid recorded interval: {where}')
    bounds=[number(v,where,low,high) for v in value]
    if bounds[0]>bounds[1]:raise ValueError(f'Reversed recorded interval: {where}')
    return bounds


def displayed(data):
    a=data.get('part_A',{});b=data.get('part_B',{})
    if a.get('status')!='complete' or b.get('status') not in ('complete','primary_complete_descriptive_incomplete'):
        raise Pending('Completed Part A and scored Part B primary metrics are required; no figure written.')
    if a.get('question_clusters')!=32:raise ValueError('Expected32 question clusters in Part A')
    metrics=b.get('metrics',{})
    if set(metrics)!=set(ARMS):raise ValueError('Expected all seven completed Part B arms')
    dose=[]
    for arm in ARMS:
        m=metrics[arm]
        if m.get('n')!=32:raise ValueError('Expected n32 in every Part B arm')
        caps=m['caps']
        if type(caps) is not int or not 0<=caps<=32:raise ValueError('Invalid recorded cap count')
        dose.append({'arm':arm,'n':32,'sarcasm':number(m['trait_conservative'],arm,0,100),
                     'coherence':number(m['coherence'],arm,0,100),'caps':caps,'eos':m.get('eos'),
                     'mean_tokens':m.get('mean_tokens')})
    alignment=[];ratio_a=[];ratio_b=[]
    for site in SITES:
        for content in ('B','P'):
            source=f'part_A.A1_alignment.attention.{site}.instruction_{content}'
            record=a['A1_alignment'][f'attention.{site}'][f'instruction_{content}']
            value=record['cosine_of_mean']
            alignment.append({'site':site,'content':content,'cosine':number(value,source,-1.000001,1.000001) if value is not None else None,
                'cluster_bootstrap95':interval(record['bootstrap95'],source,-1.000001,1.000001),
                'ci_wording':record.get('ci_wording'),'source':source})
            source=f'part_A.A2_magnitude.attention.{site}.{content}.mean_token_difference_to_native_L_ratio'
            record=a['A2_magnitude'][f'attention.{site}'][content]['mean_token_difference_to_native_L_ratio']
            if record is None:
                ratio_a.append({'site':site,'content':content,'mean':None,'cluster_bootstrap95':None,'source':source})
            else:
                ratio_a.append({'site':site,'content':content,'mean':number(record['mean'],source,0),
                    'cluster_bootstrap95':interval(record['bootstrap95'],source,0),'source':source})
        for arm in ('b2','b4','b8','b16','random16'):
            for phase in ('prefill','decode'):
                source=f'part_B.telemetry.arms.{arm}.{site}.{phase}.metrics.actual_ratio'
                phase_record=b['telemetry']['arms'][arm][str(site)][phase]
                record=phase_record['metrics']['actual_ratio']
                ratio_b.append({'site':site,'arm':arm,'phase':phase,
                    'mean':number(record['mean'],source,0) if record['mean'] is not None else None,
                    'positions':phase_record['tokens'],'valid_ratio_positions':record['count'],
                    'zero_native_norm_positions':phase_record['zero_native_norm_tokens'],'source':source})
    return {'dose_point_estimates':dose,'native_attention_alignment':alignment,
        'fixed_continuation_paired_ratios':ratio_a,'free_generation_actual_ratios':ratio_b,
        'bootstrap':a['bootstrap'],'PartB_status':b['status'],'PartB_outcome':b['outcome'],
        'outcome':data['outcome'],'old_outcome_unchanged':data.get('old_outcome_unchanged')}


def clean(ax):
    ax.set_axisbelow(True);ax.grid(alpha=.18)
    for side in ('top','right'):ax.spines[side].set_visible(False)


def horizontal_interval(ax,mean,ci,y,color,marker,label=None):
    if ci is not None:
        ax.hlines(y,ci[0],ci[1],color=color,lw=1.3)
        ax.plot(ci,[y,y],'|',color=color,markersize=6)
    if mean is not None:ax.scatter([mean],[y],color=color,marker=marker,s=35,label=label,zorder=3)


def ratio_panel(ax,rows,phase):
    for content,color,marker in [('B',SLATE,'o'),('P',AMBER,'s')]:
        rr=[r for r in rows['fixed_continuation_paired_ratios'] if r['content']==content]
        xx=[];yy=[]
        for i,r in enumerate(rr):
            if r['mean'] is None:continue
            xx.append(i);yy.append(r['mean'])
            ci=r['cluster_bootstrap95']
            if ci is not None:
                ax.vlines(i,ci[0],ci[1],color=color,lw=1.2,zorder=2)
        ax.plot(xx,yy,color=color,marker=marker,ls='--',lw=1.4,ms=4,
                label=f'A fixed S−L · content {content} (cluster CI)')
    for arm in ('b2','b4','b8','b16','random16'):
        rr=[r for r in rows['free_generation_actual_ratios'] if r['arm']==arm and r['phase']==phase]
        valid=[(i,r['mean']) for i,r in enumerate(rr) if r['mean'] is not None]
        if valid:
            xx,yy=zip(*valid)
            ax.plot(xx,yy,color=RED if arm=='random16' else DOSE_COLORS[int(arm[1:])],
                    marker='^' if arm=='random16' else '.',ls=':' if arm=='random16' else '-',lw=1.4,
                    label=f'B free · {"random16" if arm=="random16" else "b="+arm[1:]}')
    ax.set(title=f'Actual per-token ratios · Part B {phase}',xticks=range(6),
        xticklabels=[str(site) for site in SITES],xlabel='Full-attention layer (zero-based)',
        ylabel='Mean per-token change / native token norm')
    ax.set_ylim(bottom=0);clean(ax)


def draw(data,rows,prefix):
    plt=pyplot();fig,axes=plt.subplots(2,2,figsize=(14.2,10.2))
    try:
        ax=axes[0,0];metrics={r['arm']:r for r in rows['dose_point_estimates']}
        arms=['baseline',*[f'b{b}' for b in DOSES]];xx=[0,*DOSES]
        for metric,color,marker in [('sarcasm',GREEN,'o'),('coherence',SLATE,'s')]:
            values=[metrics[arm][metric] for arm in arms]
            ax.plot(xx,values,color=color,marker=marker,lw=1.8,ms=5,label=metric.capitalize())
            for x,y in zip(xx,values):
                ax.annotate(f'{y:.1f}',(x,y),xytext=(0,7 if metric=='sarcasm' else -13),
                    textcoords='offset points',ha='center',fontsize=8,color=color)
            ax.axhline(metrics['prompted'][metric],color=AMBER,ls='--' if metric=='sarcasm' else ':',
                lw=1.2,label='Prompted '+metric)
            ax.scatter([16],[metrics['random16'][metric]],edgecolors=RED,facecolors='none',
                marker='^' if metric=='sarcasm' else 's',s=65,lw=1.5,zorder=4,label='Random16 '+metric)
        ax.set(title='Fresh exploratory development · n=32 per arm',ylim=(-2,104),yticks=[0,20,40,60,80,100],
            xticks=xx,xticklabels=['Base','2','4','8','16'],xlabel='GCAD base coefficient b (baseline at 0)',
            ylabel='AI-judge point estimate (0–100)')
        clean(ax);ax.legend(fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.16))
        ax=axes[0,1]
        for content,color,marker,offset in [('B',SLATE,'o',-.12),('P',AMBER,'s',.12)]:
            first=True
            for i,site in enumerate(SITES):
                record=next(r for r in rows['native_attention_alignment'] if r['site']==site and r['content']==content)
                label=f'Fixed content {content}' if first and record['cosine'] is not None else None
                horizontal_interval(ax,record['cosine'],record['cluster_bootstrap95'],i+offset,color,marker,label)
                if label:first=False
        ax.axvline(0,color='#94a3b8',lw=1,ls='--')
        ax.set(title='No GCAD · native attention S−L alignment',xlim=(-1.02,1.02),
            yticks=range(6),yticklabels=[str(site) for site in SITES],ylim=(5.6,-.6),
            xlabel='Signed cosine with frozen Δ · 95% question-cluster CI',ylabel='Full-attention layer (zero-based)')
        clean(ax);ax.legend(fontsize=8,loc='best')
        for ax,phase in zip(axes[1],('prefill','decode')):ratio_panel(ax,rows,phase)
        axes[1,0].legend(fontsize=7.7,ncol=2,loc='best')
        # Preserve a common vertical scale across the two phase-specific panels.
        ceiling=max(ax.get_ylim()[1] for ax in axes[1])
        for ax in axes[1]:ax.set_ylim(0,ceiling)
        fig.suptitle('Qwen3.8-27B NF4 · instruction activations and GCAD dose',fontsize=16,fontweight='bold',y=.98)
        status=('No strength passed all acceptance gates' if rows['PartB_outcome']=='exploratory_null_through_b16'
                else rows['PartB_outcome'])
        if rows['PartB_status']!='complete':status+=' · impoliteness incomplete'
        fig.text(.5,.943,status+' · exploratory; no independent final or validated controller',ha='center',fontsize=10)
        fig.text(.035,.035,
            'A: post-hoc, no GCAD; S=sarcastic and L=literal instructions, with identical continuation IDs and matched positions. '
            'B/P are saved baseline/prompted contents.\n'
            'Ratio panels compare DIFFERENT CONDITIONING: A averages question-level fixed-continuation S−L / native-L ratios; '
            'B token-weights actual GCAD changes along free trajectories.\n'
            'Existing cluster intervals are shown only for A. B has point estimates; no curve fit, new statistics, or causal dose inference. '
            'Means alone do not replace the full registered gates.\n'
            'Random control exists only at b=16. The old development_null is unchanged.',fontsize=8.2,va='bottom')
        fig.subplots_adjust(left=.07,right=.97,bottom=.17,top=.885,wspace=.28,hspace=.52)
        fig.savefig(str(prefix)+'.svg',metadata={'Date':None})
        fig.savefig(str(prefix)+'.png',dpi=160)
    finally:plt.close(fig)


def publish(analysis,prefix):
    if not analysis.is_file():raise Pending(f'Pending result: {analysis}; no assets written.')
    raw=analysis.read_bytes();source_hash=hashlib.sha256(raw).hexdigest();data=json.loads(raw);rows=displayed(data)
    destinations={suffix:Path(str(prefix)+suffix) for suffix in ('.svg','.png','.json')}
    if any(file.exists() for file in destinations.values()):raise FileExistsError('Fresh SVG/PNG/JSON paths required; existing assets preserved.')
    prefix.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='followup-figure-',dir=prefix.parent) as tmp:
        scratch=Path(tmp)/prefix.name;draw(data,rows,scratch)
        if sha(analysis)!=source_hash:raise ValueError('Analysis changed during rendering; no assets published.')
        receipt={'source_analysis':str(analysis.resolve()),'source_sha256':source_hash,
            'generator_sha256':sha(__file__),'style_reference':str(STYLE.relative_to(ROOT)),'style_sha256':sha(STYLE),
            'manifest_sha256':data.get('manifest_sha256'),'analyzer_sha256':data.get('analyzer_sha256'),
            'input_sha256':data.get('input_sha256',{}),'displayed':rows,
            'conditioning_warning':'A: question-mean fixed-continuation paired differences over native L. B: token-weighted free-generation actual intervention over native output along the steered trajectory. Descriptive comparison only.',
            'undefined_alignment_points':sum(r['cosine'] is None for r in rows['native_attention_alignment']),
            'undefined_ratio_points':sum(r['mean'] is None for r in rows['fixed_continuation_paired_ratios']+rows['free_generation_actual_ratios']),
            'statistics_policy':'Read existing means/intervals only; no bootstrap, fitting, regrading, dose selection or gate decisions.',
            'versions':{'python':platform.python_version(),'matplotlib':importlib.metadata.version('matplotlib')},
            'output_sha256':{suffix:sha(Path(str(scratch)+suffix)) for suffix in ('.svg','.png')}}
        Path(str(scratch)+'.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        for suffix,file in destinations.items():os.link(Path(str(scratch)+suffix),file)
    print(json.dumps({'source_sha256':source_hash,'outputs':[str(p) for p in destinations.values()]},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,default=DEFAULT_ANALYSIS)
    p.add_argument('--output-prefix',type=Path,default=DEFAULT_PREFIX)
    p.add_argument('--check-environment',action='store_true',help='Import plotting backend only; do not read result or create figures')
    args=p.parse_args()
    if args.check_environment:
        plt=pyplot();print(json.dumps({'backend':plt.get_backend(),'matplotlib':importlib.metadata.version('matplotlib'),
            'font_family':plt.rcParams['font.family'],'figures_created':0}));return
    try:publish(args.analysis,args.output_prefix)
    except Pending as exc:print(str(exc))
    except (ValueError,KeyError,FileExistsError) as exc:p.exit(2,str(exc)+'\n')


if __name__=='__main__':main()
