"""Explicitly post-hoc, zero-call lexical audit. Execute only after registration."""
import argparse,hashlib,json,random,subprocess
from pathlib import Path
from scripts import author_profanity_method_common as lex
from scripts import joint_gcad_common as frozen
from scripts.tone_judge import save
ROOT=Path(__file__).resolve().parents[1]
RUN='outputs/joint-gcad-01'
REPORT='results/joint_gcad_01.json'
RESULT='results/joint_gcad_lexical_audit_01.json'
REG='results/protocols/joint-gcad-lexical-audit-01.json'
SPEC='docs/joint_gcad_lexical_audit_spec.md'
SOURCES=['scripts/joint_gcad_lexical_audit.py',SPEC,'scripts/author_profanity_method_common.py','scripts/joint_gcad_common.py','scripts/tone_judge.py']
INPUTS=[RUN+'/generation/predictions.json',REPORT,RUN+'/manifest.json',*[RUN+'/native/primary/'+m+'/accepted.json' for m in frozen.METRICS]]
def read(name):return json.loads((ROOT/name).read_text())
def sha(name):return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
def mean(values):return sum(values)/len(values) if values else None

def row_counts(text):
    checks=lex.text_checks(text);h=len(lex.hits(text));w=len(text.split())
    return {'hits':h,'words':w,'rate':100*h/w if w else None,'empty':checks['empty'],'invalid':checks['empty'] or checks['distinct_bigram']<.6}

def summary(rows):
    words=sum(r['words'] for r in rows);hits=sum(r['hits'] for r in rows)
    return {'n':len(rows),'n_any_hit':sum(r['hits']>0 for r in rows),'n_empty':sum(r['empty'] for r in rows),'total_hits':hits,'total_whitespace_words':words,
        'mean_hit_count':mean([r['hits'] for r in rows]),'mean_per_response_rate_primary':mean([r['rate'] for r in rows if r['rate'] is not None]),
        'pooled_rate_secondary':100*hits/words if words else None}

def correlation(x,y):
    pairs=[(a,b) for a,b in zip(x,y) if a is not None and b is not None]
    return {'n':len(pairs),'spearman':frozen.spearman(*zip(*pairs)) if pairs else None}

def analyze(predictions,scores,primary,*,repetitions=2000,seed=106012):
    names=list(frozen.arms());ids=[r['id'] for r in predictions['baseline']]
    if set(predictions)!=set(names) or len(ids)!=24 or len(set(ids))!=24 or any([r['id'] for r in predictions[a]]!=ids for a in names):raise ValueError('matched twelve-arm24-question inputs required')
    counts={a:[row_counts(r['output']) for r in predictions[a]] for a in names};out={};rate={a:[r['rate'] for r in counts[a]] for a in names}
    for a in names:
        m=primary['analysis']['metrics'][a]
        if sum(r['hits']>0 for r in counts[a])!=m['lexicon_hits'] or sum(r['invalid'] for r in counts[a])!=m['invalid']:raise ValueError('primary lexhit/invalid mismatch: '+a)
        for key in frozen.METRICS:
            values=[scores[a,q][key] for q in ids]
            if key=='coherence' and any(v is None for v in values):raise ValueError('coherence must be numeric')
            if mean([v or 0 for v in values])!=m[key]:raise ValueError('original score mean mismatch: '+a+'/'+key)
        out[a]={**summary(counts[a]),'n_invalid':sum(r['invalid'] for r in counts[a]),
            'valid_split':summary([r for r in counts[a] if not r['invalid']]),'invalid_split':summary([r for r in counts[a] if r['invalid']])}
    rng=random.Random(seed);samples={a:[] for a in names};deltas={a:[] for a in names}
    for _ in range(repetitions):
        index=[rng.randrange(24) for _ in range(24)];values={a:mean([rate[a][i] for i in index if rate[a][i] is not None]) for a in names}
        for a in names:
            samples[a].append(values[a]);deltas[a].append(values[a]-values['baseline'] if values[a] is not None and values['baseline'] is not None else None)
    def interval(values):
        v=sorted(x for x in values if x is not None)
        return {'ci95':[v[int(.025*(len(v)-1))],v[int(.975*(len(v)-1))]] if v else None,'defined_replicates':len(v)}
    for a in names:out[a]['mean_rate_ci95']=interval(samples[a]);out[a]['mean_rate_minus_baseline_ci95']=interval(deltas[a])
    within={a:{m:correlation(rate[a],[scores[a,q][m] or 0 for q in ids]) for m in frozen.METRICS} for a in names}
    pooled={};centred={}
    for m in frozen.METRICS:
        x=[];y=[];cx=[];cy=[]
        for i,q in enumerate(ids):
            mx=mean([rate[a][i] for a in names if rate[a][i] is not None]);my=mean([scores[a,q][m] or 0 for a in names])
            for a in names:
                v=rate[a][i];score=scores[a,q][m] or 0;x.append(v);y.append(score)
                cx.append(v-mx if v is not None else None);cy.append(score-my)
        pooled[m]=correlation(x,y);centred[m]=correlation(cx,cy)
    return {'arms':out,'correlations':{'within_arm':within,'raw_pooled':pooled,'prompt_centred_pooled':centred},
        'bootstrap':{'seed':seed,'replicates':repetitions,'unit':'matched question; same sampled indices across all twelve arms','interval':'unadjusted percentile95%; undefined rates excluded'},
        'checks':{'all_primary_lexhit_invalid_counts_match':True,'all_original_score_means_match':True},'primary_decisions_modified':False}

def registered_inputs():
    reg=read(REG)
    if reg.get('post_hoc') is not True or set(reg['sources_sha256'])!=set(SOURCES) or set(reg['inputs_sha256'])!=set(INPUTS):raise ValueError('exact post-hoc source/input registration required')
    for names in (reg['sources_sha256'],reg['inputs_sha256']):
        for n,d in names.items():
            if sha(n)!=d:raise ValueError('registered hash mismatch: '+n)
    for n in [REG,REPORT,*SOURCES]:
        if subprocess.run(['git','show','HEAD:'+n],cwd=ROOT,capture_output=True,check=True).stdout!=(ROOT/n).read_bytes():raise ValueError('commit exact audit/source/spec/result before execution')
    return reg

def original_scores(manifest):
    plan_name=RUN+'/inputs/plans.json'
    if sha(plan_name)!=manifest['files']['inputs/plans.json']:raise ValueError('original plan hash mismatch')
    plans=read(plan_name)['primary'];scores={}
    for m in frozen.METRICS:
        grades=read(RUN+'/native/primary/'+m+'/accepted.json')
        if sha(RUN+'/native/primary/'+m+'/run_binding.json')!=grades['binding_sha256']:raise ValueError('accepted binding mismatch')
        if set(grades['values'])!={str(i) for i in range(36)} or len(grades['batches'])!=36:raise ValueError('36accepted batches required')
        for bi,batch in enumerate(plans[m]):
            rows=grades['values'][str(bi)]
            if set(rows)!={r['id'] for r in batch}:raise ValueError('complete original ID set required')
            for r in batch:
                value=rows[r['id']];v=value['score']
                if value['id']!=r['id'] or (v is None and (m=='coherence' or value.get('refusal') is not True)) or (v is not None and (type(v) is not int or not 0<=v<=100)):raise ValueError('invalid original score')
                scores.setdefault((r['condition'],r['question_id']),{})[m]=v
    return scores

def execute():
    reg=registered_inputs();primary=read(REPORT);manifest=read(RUN+'/manifest.json')
    if primary['status']!='complete' or primary['manifest_sha256']!=sha(RUN+'/manifest.json'):raise ValueError('committed complete original result required')
    result=analyze(read(INPUTS[0]),original_scores(manifest),primary)
    result.update(status='complete_posthoc_descriptive',post_hoc=True,registration_sha256=sha(REG),sources_sha256=reg['sources_sha256'],inputs_sha256=reg['inputs_sha256'],
        lexicon=list(lex.LEXICON),definition='Exact existing case-insensitive whole-word hits; words=text.split(). Rate100*hits/whitespace_words, undefined for zero words. Primary is mean per-response rate; secondary pooled rate is length-weighted. Not a hit-word percentage; compounds can exceed100.',
        limits=['Explicitly post-hoc after inspection of primary results, not confirmatory; no new calls, regrading, selection or quality gates.',
            'Original PF rubric measures friendly/helpful swearing and excludes user-directed insults; lower rubric score does not establish less literal swearing.',
            'Literal eleven-word proxy misses slang/variants and counts quoted/incidental hits; it does not measure intent. Length denominators and incoherence can distort rates.',
            'Within-arm correlations are descriptive; raw and prompt-centred pooled correlations remain condition-confounded, not causal independence evidence.',
            'n24 AI-authored English practical-advice questions, same advisor/judge family; unadjusted paired bootstrap intervals. Coherence is not accuracy.',
            'All original candidate/gate verdicts remain unchanged. Per-response duplicate coherence MAD is not a standard error of the mean; a CI spanning the boundary does not prove a quality miss is within noise.'])
    registered_inputs();save(ROOT/RESULT,result)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:execute();print(RESULT)
    else:print('No computation/output: root must commit audit source/spec/input-hash registration, then --execute.')
