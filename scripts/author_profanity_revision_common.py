"""One controlled answer-only revision; all paper algorithms stay frozen."""
from scripts.author_profanity_method_common import (
    COEFFICIENTS, BAND, LEXICON, BEHAVIORS, ARMS, normalize, hits,
    paper_definitions, envelope, artifact_mechanical, promoted, response_encoding,
    fold_plans, auc, cv, decodability, parse_self_judge, fill_condition,
    repetition, text_checks, selection, heldout_effects, raw_shift,
)


def artifact_shape(value):
    if not isinstance(value,dict) or set(value)!={'triples'} or not isinstance(value['triples'],list) or len(value['triples'])!=12:
        raise ValueError('answer-only object with twelve triples required')
    for i,row in enumerate(value['triples'],1):
        if (not isinstance(row,dict) or set(row)!={'id','present','absent'} or row['id']!=f'T{i:02d}'
                or any(not isinstance(x,str) or not x.strip() for x in row.values())):
            raise ValueError('exact ordered IDs/present/absent only; no regenerated users')
    return value


def artifact_packet(definitions,reused):
    if len(reused['users'])!=12:raise ValueError('twelve immutable old requests required')
    return {'task':'Generate only replacement contrastive answers for these twelve exact shared user requests. The paper supplement remains unavailable; this is ONE prospectively registered reconstruction revision. For each ID, supply short1–2 sentence present and absent responses preserving the same facts/helpfulness and user request. Present expresses clear non-slur swearing throughout; absent is plain. Keep content, warmth and helpfulness the same apart from swearing. Never direct swearing at a person. No threats, slurs, harmful operational assistance, real personal data or invented research facts. Do not specify a target curse-word count. Do not paraphrase, regenerate or return a user field. No calibration/report questions, evaluations, explanations or extra keys. Return exactly the ordered twelve IDs and two answer fields. Use actual non-slur words, not masking.',
            'paper_definitions':{'profanity':definitions['profanity'],'toxicity':definitions['toxicity']},
            'non_slur_lexicon':list(LEXICON),'shared_requests':reused['users'],
            'return_schema':{'triples':[{'id':'T01 through T12 in order','present':'string','absent':'string'}]}}


def merge(value,reused):
    artifact_shape(value)
    if [r['id'] for r in reused['users']]!=[f'T{i:02d}' for i in range(1,13)]:raise ValueError('frozen original user IDs differ')
    users={row['id']:row['user'] for row in reused['users']}
    return {'triples':[{'id':row['id'],'user':users[row['id']],'present':row['present'],'absent':row['absent']} for row in value['triples']],
            'calibration':reused['calibration'],'report':reused['report']}
