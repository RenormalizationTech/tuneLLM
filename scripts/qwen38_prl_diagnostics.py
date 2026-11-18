"""Secondary lexical/reviewer diagnostics; never selects prompts or checkpoints."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.qwen38_science_runtime import sha, write
from scripts.qwen38_science_score import answer_f1

ROOT=Path(__file__).resolve().parents[1]


def lcs_f1(a,b):
    """Whitespace-token, lowercase LCS F1; custom ROUGE-L-style diagnostic."""
    a=a.lower().split();b=b.lower().split()
    if not a or not b:return 0.0
    state=[0]*(len(b)+1)
    for x in a:
        previous=state[:]
        for i,y in enumerate(b,1):
            state[i]=previous[i-1]+1 if x==y else max(previous[i],state[i-1])
    return 2*state[-1]/(len(a)+len(b))


def describe(row,text):
    tokens=text.lower().split();body=row['selected_body'].lower().split()
    grams=lambda words:{tuple(words[i:i+8]) for i in range(len(words)-7)}
    generated=grams(tokens);evidence=grams(body)
    return {'words':len(text.split()),'empty':not text.strip(),'over_200_words':len(text.split())>200,
            'custom_whitespace_lcs_f1':lcs_f1(text,row['assistant']),
            'normalized_token_f1':answer_f1(text,row['assistant']),
            'novel_unique_8gram_fraction':len(generated-evidence)/len(generated) if generated else None}


def agreement(stage):
    plans=json.loads((stage/'plans.json').read_text());paired={};position=[];downgrades=Counter();counts=Counter()
    for plan in plans:
        folder=stage/plan['directory'];checked_file=folder/'checked.json'
        if not checked_file.exists():continue
        result=json.loads(checked_file.read_text())
        if not result.get('valid'):continue
        payload=json.loads((folder/'payload.json').read_text())
        arm_labels={}
        for candidate in payload['candidates']:
            label=candidate['label'];arm=plan['mapping'][label]
            ids=[r['sid'] for r in candidate['sentences']]
            arm_labels[arm]=[result['labels'][sid] for sid in ids]
            counts[arm]+=len(ids);downgrades[arm]+=sum(sid in result['quote_downgrades'] for sid in ids)
        paired.setdefault(plan['paper_id'],[]).append((arm_labels,result,plan))
        for first,second in payload['pairs']:
            winner=result['winners'][':'.join(sorted([first,second]))]
            position.append(.5 if winner=='tie' else float(winner==first))
    pairs=[];preference=[]
    for passes in paired.values():
        if len(passes)!=2:continue
        a,b=passes
        for arm in a[0]:
            if len(a[0][arm])!=len(b[0][arm]):raise ValueError('sentence units differ between passes')
            pairs.extend(zip(a[0][arm],b[0][arm]))
        def winners(entry):
            _,result,plan=entry
            return {tuple(sorted(plan['mapping'][x] for x in key.split(':'))):
                    ('tie' if winner=='tie' else plan['mapping'][winner]) for key,winner in result['winners'].items()}
        wa,wb=winners(a),winners(b)
        preference.extend(wa[key]==wb[key] for key in wa)
    observed=sum(a==b for a,b in pairs)/len(pairs) if pairs else None
    ca=Counter(a for a,b in pairs);cb=Counter(b for a,b in pairs)
    expected=sum(ca[k]*cb[k] for k in ca)/len(pairs)**2 if pairs else None
    return {'support_labeled_sentences_paired':len(pairs),'support_agreement':observed,
            'support_cohens_kappa':(observed-expected)/(1-expected) if expected is not None and expected<1 else None,
            'pairwise_preference_agreement':sum(preference)/len(preference) if preference else None,
            'first_pair_position_win_with_half_ties':sum(position)/len(position) if position else None,
            'quote_downgrade_fraction_by_arm':{a:downgrades[a]/n if n else None for a,n in counts.items()}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--arms',required=True,help='JSON arm name to completed generation run')
    p.add_argument('--review-stage',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous diagnostic report')
    rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()];ids={r['id'] for r in rows}
    if len(ids)!=len(rows):raise ValueError('duplicate source IDs')
    result={'scope':'secondary custom lexical diagnostics; no semantic faithfulness claim',
            'input_sha256':sha(a.input),'source_sha256':sha(__file__),'arms':{}}
    runs=json.loads(a.arms)
    for arm,run in runs.items():
        folder=ROOT/run;status=json.loads((folder/'result.json').read_text());registration=json.loads((folder/'registration.json').read_text())
        if not status.get('completed') or registration['input_sha256']!=sha(a.input):raise ValueError('incomplete/wrong-input generation')
        predictions=[json.loads(x) for x in (folder/'predictions.jsonl').read_text().splitlines() if x.strip()]
        if len(predictions)!=len(ids) or {r['id'] for r in predictions}!=ids:raise ValueError('prediction ID/count differs')
        lookup={r['id']:r for r in predictions}
        result['arms'][arm]={'records':[{'id':r['id'],**describe(r,lookup[r['id']]['output'])} for r in rows],
                             'predictions_sha256':sha(folder/'predictions.jsonl')}
    result['fixed_copy_baseline']=[{'id':r['id'],**describe(r,r['copy_baseline']['text'])} for r in rows]
    if a.review_stage:result['reviewer_diagnostics']=agreement(a.review_stage)
    write(a.output,result)
    print(json.dumps({'output':str(a.output),'papers':len(rows),'arms':list(runs)}))


if __name__=='__main__':main()
