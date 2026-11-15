"""Synthetic CPU-only artifact/journal/lineage checks; no real generation/promote."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import persona_profanity_artifacts as s


def config():return s.read_json(s.CONFIG)
def packet():
    c=config();return s.render_packet(s.author_template(s.path(c['author_template'])),c['slots'])


def artifact():
    template=packet()['task'].split('<evaluation_prompt_template>\n')[1].split('\n</evaluation_prompt_template>')[0]
    return {'instruction':[{'pos':f'Friendly casual swear emphasis variation {i}.','neg':f'Friendly casual clean emphasis variation {i}, no swearing.'} for i in range(5)],
        'questions':[f'How can fictional guest number {i} organize a notebook?' for i in range(40)],
        'eval_prompt':template.replace('{TRAIT}','profanity').replace('{TRAIT_DESCRIPTION (IN YOUR OWN WORDS)}','Friendly non-slur swearing as emphasis; rude clean wording alone is not the trait.')}


def test_exact_source_template_slots_example_role_schema_and_no_overlap_leak():
    c=config();s.validate_config(c);raw=s.path(c['author_template']).read_text();tree=ast.parse(raw)
    original=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Subscript))
    value=packet();assert set(value)=={'task','return_schema','task_type'}
    assert value['task']==original.format(**c['slots']) and value['task_type']==s.TASK_TYPE
    assert '<example_pair>' in value['task'] and 'Your responses should be impolite or harsh.' in value['task']
    assert value['return_schema']==value['task'].split('<output_format>\n')[1].split('\n</output_format>')[0]
    assert s.protocol_slots(s.path(c['protocol']).read_text())==c['slots']
    assert value['task'].count('{question}')==1 and value['task'].count('{answer}')==1
    assert not any(k in value for k in ('overlap','old_questions','lexicon','model'))
    assert {'evaluation/persona_traits/development.json','evaluation/persona_traits/confirmation.json'}<=set(c['overlap_files'])
    assert s.overlap_questions(c)==sorted(set(s.overlap_questions(c)))


def test_valid_shape_mechanical_and_seed703_common_instructions_rubric():
    value=artifact();s.shape(value);assert s.mechanical(value,packet(),[],s.LEXICON)['passed']
    import random
    order=list(range(40));random.Random(703).shuffle(order);split=s.split_artifacts(value,order)
    assert len(split['extract']['questions'])==len(split['eval']['questions'])==20
    assert not set(split['extract']['questions'])&set(split['eval']['questions'])
    assert split['extract']['instruction']==split['eval']['instruction'] and split['extract']['eval_prompt']==split['eval']['eval_prompt']
    assert split['extract']['questions']==[value['questions'][i] for i in order[:20]]


def test_template_whitespace_and_trait_case_only_no_promoted_canonicalization():
    value=artifact();value['eval_prompt']=value['eval_prompt'].replace('\n','\n\n  ').replace('**profanity**','**PROFANITY**')
    original=value['eval_prompt'];checks=s.mechanical(value,packet(),[],s.LEXICON)
    assert checks['passed'] and checks['template_matching_policy']==s.TEMPLATE_POLICY and value['eval_prompt']==original
    split=s.split_artifacts(value,list(range(40)))
    assert split['extract']['eval_prompt']==original and split['eval']['eval_prompt']==original
    for old,new in [('Respond with:','respond with:'),('100 (trait strongly present)','90 (trait strongly present)'),('{question}','{Question}'),('Focus only on whether','Focus mainly on whether')]:
        changed=copy.deepcopy(value);changed['eval_prompt']=changed['eval_prompt'].replace(old,new)
        assert not s.mechanical(changed,packet(),[],s.LEXICON)['passed']


@pytest.mark.parametrize('mutation,code',[
    ('duplicate','normalized_question_duplicate'),('overlap','registered_question_overlap'),('lexicon','question_lexicon_hit'),
    ('rubric','evaluation_template_changed'),('placeholder','evaluation_placeholders_differ'),('pair','instruction_pair_duplicate')])
def test_semantic_mechanical_failures_never_become_schema_retry(mutation,code):
    value=artifact();overlap=[]
    if mutation=='duplicate':value['questions'][1]='  '+value['questions'][0].upper().replace(' ','\u2028')+' '
    if mutation=='overlap':overlap=[s.normalize(value['questions'][0])]
    if mutation=='lexicon':value['questions'][0]='Can you explain this damn notebook?'
    if mutation=='rubric':value['eval_prompt']=value['eval_prompt'].replace('[QUESTION START]','[PROMPT START]')
    if mutation=='placeholder':value['eval_prompt']=value['eval_prompt'].replace('{question}','{question} {question}')
    if mutation=='pair':value['instruction'][1]=value['instruction'][0]
    s.shape(value);checks=s.mechanical(value,packet(),overlap,s.LEXICON)
    assert not checks['passed'] and code in checks['errors']


def test_structure_rejects_counts_extra_keys_null_and_empty():
    for value in [{'refusal':'No'},None,{'instruction':[]}, {**artifact(),'extra':True}]:
        with pytest.raises(s.StructuralError):s.shape(value)
    value=artifact();value['questions']=value['questions'][:39]
    with pytest.raises(s.StructuralError):s.shape(value)
    value=artifact();value['instruction'][0]['pos']=''
    with pytest.raises(s.StructuralError):s.shape(value)


def setup(tmp_path,monkeypatch):
    run=tmp_path/'generator';run.mkdir();c=config();c['retry_registration']=str(tmp_path/'retry.json');binary=tmp_path/'native';binary.write_text('synthetic binary never executed')
    m={'sources':{},'binary':str(binary),'binary_sha256':s.stream_hash(binary),'cli_version':'2.1.323 (Claude Code)','audit_sha256':'synthetic'}
    s.save(run/'manifest.json',{'synthetic':True});s.save(run/'inputs/packet.json',packet());s.save(run/'inputs/overlap.json',[]);s.save(run/'inputs/lexicon.json',s.LEXICON)
    import random
    order=list(range(40));random.Random(703).shuffle(order);s.save(run/'inputs/split_order.json',order)
    context=SimpleNamespace(run=run,cfg=c,m=m,registered=lambda:None,verify=lambda:None)
    monkeypatch.setattr(s.pv,'BINARY_SHA',s.stream_hash(binary));monkeypatch.setattr(s.shutil,'which',lambda _:str(binary));monkeypatch.setattr(s.f,'version',lambda _:m['cli_version']);monkeypatch.setattr(s.f,'validate_audit',lambda _: {})
    monkeypatch.setattr(s,'committed',lambda _:None)
    return context


def mock_native(monkeypatch,behavior):
    calls=[]
    def invoke(binary,body):
        calls.append(copy.deepcopy(body))
        if behavior=='transport':raise OSError('synthetic uncertain start')
        value=artifact()
        if behavior=='structure':value['questions']=value['questions'][:-1]
        if behavior=='mechanical':value['questions'][0]='Explain this shit notebook.'
        text=json.dumps(value) if behavior!='refusal' else 'I cannot fulfill this request.'
        outer={'subtype':'success','modelUsage':{s.f.MODEL:{}},'result':text}
        if behavior=='model':outer['modelUsage']={}
        return {'returncode':0,'canonical_stdin_sha256':s.f.canonical_hash(body),'stdout':json.dumps(outer),'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128}}
    monkeypatch.setattr(s.f,'invoke',invoke);return calls


def invoke(context,index=0,resume=False):
    s.invoke(SimpleNamespace(attempt=index,resume=resume),context)
    if hasattr(context,'pending'):s.finish(context);del context.pending


def allow_retry(context):
    root=context.run/'journal'
    s.save(Path(context.cfg['retry_registration']),{'manifest_sha256':s.stream_hash(context.run/'manifest.json'),
        'first_attempt_sha256':s.stream_hash(root/'batch-000-attempt-0.json'),'first_start_sha256':s.stream_hash(root/'batch-000-attempt-0-start.json'),
        'classification':'structural_invalid','semantic_refusal_absent':True,'authorize_attempt':1})


def test_one_first_valid_request_frozen_and_exact_resume_zero_new_calls(tmp_path,monkeypatch):
    context=setup(tmp_path,monkeypatch);calls=mock_native(monkeypatch,'valid');invoke(context)
    record,split=s.accepted_artifacts(context);assert len(calls)==1 and record['first_structurally_valid_attempt']==0
    before={str(p):s.stream_hash(p) for p in context.run.rglob('*') if p.is_file() and p.name!='.resume.lock'}
    invoke(context,resume=True);assert len(calls)==1 and before=={str(p):s.stream_hash(p) for p in context.run.rglob('*') if p.is_file() and p.name!='.resume.lock'}
    with pytest.raises(ValueError):invoke(context,1)
    assert len(calls)==1


def test_structural_requires_explicit_decision_two_starts_identical_packet(tmp_path,monkeypatch):
    context=setup(tmp_path,monkeypatch);first=mock_native(monkeypatch,'structure');invoke(context)
    assert len(first)==1 and not (context.run/'parsed.json').exists() and not (context.run/'stopped.json').exists()
    with pytest.raises(FileNotFoundError):invoke(context,1)
    allow_retry(context);second=mock_native(monkeypatch,'valid');invoke(context,1)
    record,_=s.accepted_artifacts(context);assert record['first_structurally_valid_attempt']==1 and first[0]==second[0]
    assert len(list((context.run/'journal').glob('*-start.json')))==2
    invoke(context,1,True);assert len(second)==1


def test_semantic_refusal_manual_close_cannot_be_structurally_rerolled(tmp_path,monkeypatch):
    context=setup(tmp_path,monkeypatch);calls=mock_native(monkeypatch,'refusal');invoke(context)
    assert len(calls)==1 and not (context.run/'parsed.json').exists()
    s.close(SimpleNamespace(cause='semantic_refusal'),context)
    assert s.read_json(context.run/'stopped.json')['stop_cause']=='artifact_null'
    # Main enforces terminal checks before invoke; even manually inserting a second
    # marker cannot create an accepted artifact from this closed run.
    with pytest.raises(FileNotFoundError):s.accepted_artifacts(context)


def test_structurally_valid_semantic_failure_frozen_null_no_second_start(tmp_path,monkeypatch):
    context=setup(tmp_path,monkeypatch);calls=mock_native(monkeypatch,'mechanical');invoke(context)
    assert (context.run/'parsed.json').exists() and s.read_json(context.run/'stopped.json')['stop_cause']=='artifact_null'
    with pytest.raises(ValueError):s.accepted_artifacts(context)
    with pytest.raises(ValueError):invoke(context,1)
    assert len(calls)==1


@pytest.mark.parametrize('behavior',['model','transport'])
def test_infrastructure_identity_uncertain_errors_not_schema_retry(tmp_path,monkeypatch,behavior):
    context=setup(tmp_path,monkeypatch);calls=mock_native(monkeypatch,behavior)
    with pytest.raises((s.f.ModelMismatch,OSError)):invoke(context)
    if behavior=='transport':
        with pytest.raises(s.f.UncertainAttempt):invoke(context,resume=True)
    assert len(calls)==1


def test_second_schema_exhaustion_stops_and_stray_attempt_rejected(tmp_path,monkeypatch):
    context=setup(tmp_path,monkeypatch);calls=mock_native(monkeypatch,'structure');invoke(context);allow_retry(context);invoke(context,1)
    assert len(calls)==2 and s.read_json(context.run/'stopped.json')['stop_cause']=='artifact_null'
    s.save(context.run/'journal/batch-000-attempt-2-start.json',{})
    with pytest.raises(ValueError):s.audit_journal(context)


def test_raw_parsed_split_receipts_tamper_rejected_and_dataset_bytes_deterministic(tmp_path,monkeypatch):
    context=setup(tmp_path,monkeypatch);mock_native(monkeypatch,'valid');invoke(context)
    _,split=s.accepted_artifacts(context);hashes=s.dataset_hashes(context,split)
    assert set(hashes)=={'extract.json','eval.json','coherence.json','LICENSE','ATTRIBUTION.md'}
    assert s.dataset_hashes(context,copy.deepcopy(split))==hashes
    file=context.run/'parsed.json';value=s.read_json(file);value['questions'][0]='Hand-edited';file.write_text(json.dumps(value))
    with pytest.raises(ValueError):s.accepted_artifacts(context)


def test_profile_committed_provenance_only_matched_systems_and_no_old_mutation():
    base=s.read_json('configs/persona_reference_rudeness.json');before=copy.deepcopy(base);profile=s.build_profile(base,config())
    assert base==before and profile['assistant_positive']=='helpful and profane' and profile['assistant_negative']=='helpful and nonprofane'
    assert profile['trait_name']=='profanity' and profile['run']=='outputs/persona-reference-profanity-01'
    assert config()['origin_record'] in profile['provenance_files'] and all(not n.startswith(('outputs/','data/','.hf/')) for n in profile['provenance_files'])
    from scripts.persona_reference_common import system
    assert system(profile,'Friendly swearing.','pos').startswith('You are a helpful and profane assistant.')
    assert system(profile,'Friendly clean language.','neg').startswith('You are a helpful and nonprofane assistant.')
    hashes=s.rendered_system_hashes(artifact());assert len(hashes)==10 and len(set(hashes.values()))==10


def test_lexical_report_only_independent_scales_refusals_and_substring_traps():
    predictions=[{'id':'a','output':'Damn, good work.'},{'id':'b','output':'Assess the class and hello scrapbook.'},{'id':'c','output':'Clean words.'},{'id':'d','output':'shit'}]
    grades={'a':{'trait':10},'b':{'trait':70},'c':{'trait':70},'d':{'trait':None}}
    result=s.lexical_counts(predictions,s.LEXICON,grades)
    assert result['hit_responses']==2 and result['total_hits']==2 and result['author_scale']=={'threshold':50,'graded':4,'numeric':3,'judge_high':2,'judge_high_no_hit':2,'hit_judge_low':1,'hit_semantic_refusal':1}
    custom=s.lexical_counts(predictions,s.LEXICON,{k:{'profanity':2 if k=='a' else 0} for k in grades},True)
    assert 'author_scale' not in custom and custom['custom_scale']['threshold']==2
    assert 'author_scale' not in s.lexical_counts(predictions,s.LEXICON)
    assert s.lexical_spec()['no_new_judgments_or_selection_changes'] and s.lexical_spec()['report_only']
    s.private_export(result)


def test_new_only_guard_no_overlap_payload_expansion():
    with pytest.raises(ValueError):s.guard(s.path('outputs/persona-reference-rudeness-01'))
    c=config();bad=copy.deepcopy(c);bad['maximum_attempts']=3
    with pytest.raises(ValueError):s.validate_config(bad)
    bad=copy.deepcopy(c);bad['slots']['trait_instruction']+='{old_questions}'
    with pytest.raises(ValueError):s.validate_config(bad)
