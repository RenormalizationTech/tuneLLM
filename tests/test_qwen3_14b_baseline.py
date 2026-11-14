"""CPU-only preparation/scoring/journal checks; all remote invocations mocked."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from scripts import qwen3_14b_baseline as q


def synthetic(n, prefix='s', empty=()):
    rows=[{'id':prefix+str(i),'user':'Synthetic request '+str(i),'type':'type'+str(i%2)} for i in range(n)]
    values=[{'id':r['id'],'output':' ' if i in empty else 'Synthetic answer',
             'generated_ids':[3,4],'ended_eos':True,'hit_token_limit':False} for i,r in enumerate(rows)]
    return rows,values


def test_real_published_ids_counts_order_and_preflight_disjointness():
    cfg=q.read_json('configs/qwen3_14b_baseline.json');subsets,quality,preflight=q.fixtures(cfg)
    old=q.read_json('outputs/xstest-baseline-01/inputs/safe250.json')
    assert [r['id'] for r in subsets['safe']]==[r['id'] for r in old]
    assert [len(subsets[s]) for s in q.SUBSETS]==[250,200]
    assert len({r['id'] for rows in subsets.values() for r in rows})==450
    assert [len(quality[k]) for k in q.TASKS]==[16,16,8,8]
    assert len(preflight)==8 and cfg['gpu']==cfg['gpu_uuid']
    blocked={q.normalize(r['user']) for rows in subsets.values() for r in rows}
    blocked|={q.normalize(q.prompt_text(r)) for task in q.TASKS for r in quality[task]}
    assert not blocked&{q.normalize(r['user']) for r in preflight}
    assert q.normalize('Ａ\u2028 B')==q.normalize('a b')


def test_actual_cached_14b_tokenizer_native_prefix_and_all_mc_boundaries():
    from transformers import AutoTokenizer
    cfg=q.read_json('configs/qwen3_14b_baseline.json')
    tokenizer=AutoTokenizer.from_pretrained(q.path('.hf/hub/models--Qwen--Qwen3-14B/snapshots')/cfg['revision'],local_files_only=True,trust_remote_code=False)
    prompt=q.render_prompt(tokenizer,{'user':'Synthetic prompt','system':cfg['system']})
    assert prompt.startswith('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n')
    assert prompt.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
    quality=q.read_json(cfg['quality'])
    for task in ('general_mc','physics_mc'): assert len(q.choice_token_ids(tokenizer,quality[task]))==4
    broken=copy.deepcopy(quality['general_mc']);broken[0]['choices'][0]='invalid'
    with pytest.raises(ValueError):q.choice_token_ids(tokenizer,broken)


def test_separate_fixed_batches_never_merge_two_safe_with_unsafe():
    safe,sv=synthetic(250,empty=(248,));unsafe,uv=synthetic(200,'u')
    sp=q.x.batch_plan(safe,sv);up=q.x.batch_plan(unsafe,uv)
    assert len(sp)==32 and len(up)==25 and len(sp[-1]['source_ids'])==2
    assert q.x.payload(sp[-1],'definitions')['expected_ids']==['s249']
    assert len(up[0]['source_ids'])==8 and up[0]['source_ids'][0]=='u0'
    assert sum(2*len(p) for p in [sp,up])==114


@pytest.mark.parametrize('n',[250,200])
def test_subset_denominators_empty_I_partial_refusal_and_no_partial_metric(n):
    rows,values=synthetic(n,empty=(0,));ratings=[{'id':r['id'],'grade':'P' if i==1 else 'C','reason':'Synthetic'} for i,r in enumerate(rows) if i]
    measured=q.measurements(rows,values,ratings)
    assert measured['overall']['total']==n and measured['overall']['refusal_fraction']==2/n
    assert measured['overall']['empty']==1 and measured['overall']['nonempty_refusal']==1
    with pytest.raises(ValueError):q.measurements(rows,values,ratings[:-1])


@pytest.mark.parametrize('change',['missing','duplicate','extra','bad_grade','extra_field'])
def test_exact_schema_rejection(change):
    rows,_=synthetic(8);ratings=[{'id':r['id'],'grade':'C','reason':'Synthetic'} for r in rows]
    if change=='missing':ratings.pop()
    elif change=='duplicate':ratings[-1]=ratings[0]
    elif change=='extra':ratings.append({'id':'extra','grade':'I','reason':'Synthetic'})
    elif change=='bad_grade':ratings[0]['grade']='refusal'
    else:ratings[0]['extra']=True
    with pytest.raises(ValueError):q.x.validate_ratings(rows,ratings)


def setup_judge(tmp_path,monkeypatch):
    run=tmp_path/'new';run.mkdir();binary=tmp_path/'claude';binary.write_text('synthetic executable; never run')
    audit=tmp_path/'audit.json';audit.write_text('{"synthetic":true}')
    cfg={'judge_audit':str(audit),'model_id':'synthetic','revision':'synthetic'};m={'binary':str(binary),'sources':{}}
    q.save(run/'manifest.json',{'sources':{}});q.save(run/'inputs/definitions.json',{'text':'Frozen synthetic definitions'})
    plans={}
    for subset,prefix in [('safe','s'),('unsafe','u')]:
        rows,values=synthetic(8,prefix);plans[subset]=q.x.batch_plan(rows,values)
        q.save(run/'judging'/(subset+'.json'),plans[subset])
    monkeypatch.setattr(q,'guard',lambda r:None)
    monkeypatch.setattr(q,'verify',lambda r,models=False:(cfg,m))
    monkeypatch.setattr(q,'registration',lambda a,r:(cfg,m))
    monkeypatch.setattr(q,'grading_plan',lambda r,s:plans[s])
    monkeypatch.setattr(q.shutil,'which',lambda name:str(binary))
    monkeypatch.setattr(q.x,'BINARY_SHA',q.stream_hash(binary))
    monkeypatch.setattr(q.f,'validate_audit',lambda p:{'cli_version':'synthetic'})
    monkeypatch.setattr(q.f,'version',lambda p:'synthetic')
    return run


def attempt(payload,result):
    return {'returncode':0,'canonical_stdin_sha256':q.f.canonical_hash(payload),
        'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},
        'stdout':json.dumps({'subtype':'success','modelUsage':{q.f.MODEL:{}},'result':result})}


def test_safe_first_valid_survives_unsafe_decline_exhaustion_no_reframing(tmp_path,monkeypatch):
    run=setup_judge(tmp_path,monkeypatch);calls=[]
    def invoke(binary,payload):
        calls.append(copy.deepcopy(payload))
        if payload['expected_ids'][0].startswith('u'):return attempt(payload,'I cannot classify this request.')
        return attempt(payload,json.dumps({'ratings':[{'id':r['id'],'grade':'C','reason':'Synthetic'} for r in payload['items']]}))
    monkeypatch.setattr(q.f,'invoke',invoke)
    with pytest.raises(q.f.InvalidSubmission):q.judge(SimpleNamespace(execute_judgments=True,resume=False),run)
    assert len(calls)==3 and calls[1]==calls[2]
    safe,_,sc=q.read_grades(run,'safe');unsafe,_,uc=q.read_grades(run,'unsafe')
    assert len(safe)==8 and sc['calls_started']==1 and unsafe is None
    assert uc['decline_like_structural_attempts']==2 and uc['calls_started']==2
    with pytest.raises(RuntimeError):q.judge(SimpleNamespace(execute_judgments=True,resume=True),run)
    assert len(calls)==3


@pytest.mark.parametrize('failure',['api','uncertain'])
def test_api_and_uncertain_start_stop_without_retry(tmp_path,monkeypatch,failure):
    run=setup_judge(tmp_path,monkeypatch);calls=[]
    def invoke(binary,payload):
        calls.append(1)
        if failure=='uncertain':raise OSError('synthetic interrupted transport')
        result=attempt(payload,'');outer=json.loads(result['stdout']);outer['is_error']=True
        result['stdout']=json.dumps(outer);return result
    monkeypatch.setattr(q.f,'invoke',invoke)
    with pytest.raises((RuntimeError,OSError)):q.judge(SimpleNamespace(execute_judgments=True,resume=False),run)
    assert calls==[1] and (run/'stopped.json').exists()
    with pytest.raises(RuntimeError):q.judge(SimpleNamespace(execute_judgments=True,resume=True),run)
    assert calls==[1]


def test_read_only_receipt_payload_tamper_rejected(tmp_path,monkeypatch):
    run=setup_judge(tmp_path,monkeypatch)
    monkeypatch.setattr(q.f,'invoke',lambda binary,payload:attempt(payload,json.dumps({'ratings':[{'id':r['id'],'grade':'C','reason':'Synthetic'} for r in payload['items']]})))
    q.judge(SimpleNamespace(execute_judgments=True,resume=False),run)
    p=run/'grades/safe/batch-000-attempt-0-receipt.json';value=q.read_json(p);value['canonical_stdin_sha256']='changed';p.write_text(json.dumps(value))
    with pytest.raises(ValueError):q.read_grades(run,'safe')


def test_report_preserves_safe_complete_but_has_no_incomplete_unsafe_rate(tmp_path,monkeypatch):
    run=setup_judge(tmp_path,monkeypatch)
    def invoke(binary,payload):
        if payload['expected_ids'][0].startswith('u'):return attempt(payload,'I cannot classify this request.')
        return attempt(payload,json.dumps({'ratings':[{'id':r['id'],'grade':'C','reason':'PRIVATE synthetic explanation'} for r in payload['items']]}))
    monkeypatch.setattr(q.f,'invoke',invoke)
    with pytest.raises(q.f.InvalidSubmission):q.judge(SimpleNamespace(execute_judgments=True,resume=False),run)
    q.save(run/'generation/completed.json',{'synthetic':True})
    q.save(run/'generation/quality/metrics.json',{'general_mc':{'total':16,'correct':8}})
    for task in ('gsm8k','ifeval'):q.save(run/'generation/quality'/(task+'_predictions.json'),synthetic(1)[1])
    monkeypatch.setattr(q,'predictions',lambda r,s:synthetic(8,'s' if s=='safe' else 'u'))
    output=tmp_path/'public.json';q.report(SimpleNamespace(output=str(output)),run)
    result=q.read_json(output)
    assert result['status']=='incomplete' and result['subsets']['safe']['status']=='complete'
    assert result['subsets']['safe']['measured']['overall']['refusal_fraction']==0
    assert result['subsets']['unsafe']['status']=='incomplete' and result['subsets']['unsafe']['measured'] is None
    assert 'PRIVATE synthetic explanation' not in output.read_text()


def test_parameter_assertion_no_execution_and_old_write_guards(tmp_path):
    model=torch.nn.Linear(2,2).to(dtype=torch.bfloat16)
    with pytest.raises(ValueError):q.assert_bf16_cuda(model)
    with pytest.raises(ValueError):q.generate(SimpleNamespace(execute_gpu=False),tmp_path)
    with pytest.raises(ValueError):q.judge(SimpleNamespace(execute_judgments=False),tmp_path)
    with pytest.raises(ValueError):q.guard(q.path('outputs/xstest-baseline-01'))
    assert not list(tmp_path.iterdir())


def test_host_memory_guard_without_gpu_access(monkeypatch):
    monkeypatch.setattr(q.x.b,'resource_check',lambda cfg:{'gpu_uuid':'synthetic'})
    monkeypatch.setattr(q,'host_available_bytes',lambda:63*1024**3)
    with pytest.raises(RuntimeError):q.resources({'minimum_host_available_gib':64})
    monkeypatch.setattr(q,'host_available_bytes',lambda:64*1024**3)
    assert q.resources({'minimum_host_available_gib':64})['host_available_bytes']==64*1024**3


@pytest.mark.parametrize('tamper',['runtime_bytes','uuid'])
def test_generation_runtime_receipt_binding_rejected_before_reading_predictions(tmp_path,monkeypatch,tamper):
    cfg={'gpu':'GPU-synthetic','gpu_uuid':'GPU-synthetic'}
    monkeypatch.setattr(q,'verify',lambda run:(cfg,{}))
    q.save(tmp_path/'manifest.json',{'synthetic':True})
    runtime={'gpu_uuid':cfg['gpu_uuid'],'cuda_visible_devices':cfg['gpu'],'all_parameters_bf16_cuda0':True,
        'attention':'sdpa','tf32_matmul':False,'tf32_cudnn':False}
    file=tmp_path/'generation/runtime.json';q.save(file,runtime)
    completion={'completed':True,'manifest_sha256':q.stream_hash(tmp_path/'manifest.json'),
        'runtime_sha256':q.stream_hash(file),'phase_run_sha256':{k:'unused' for k in ('preflight','quality',*q.SUBSETS)}}
    if tamper=='runtime_bytes':file.write_text(file.read_text()+' ')
    else:
        runtime['gpu_uuid']='GPU-wrong';file.write_text(json.dumps(runtime));completion['runtime_sha256']=q.stream_hash(file)
    q.save(tmp_path/'generation/completed.json',completion)
    with pytest.raises(ValueError,match='runtime identity'):q.predictions(tmp_path,'safe')
