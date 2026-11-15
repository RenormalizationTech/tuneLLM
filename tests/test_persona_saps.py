"""CPU-only author differential, cached-HF mechanics, bounded plans/journals."""
import copy
import ast
from contextlib import nullcontext
import importlib.util
import json
import random
from types import SimpleNamespace

import pytest
import torch

from scripts import persona_saps as s
from scripts import persona_saps_common as p


def cfg():return s.read_json(s.CONFIG)
def data():return {k:s.read_json(f'evaluation/persona_saps/{k}.json') for k in ('train','val','test','prompts')}
def author():
    file=s.path('outputs/persona-literature-audit-01/saps-steering/src/steering.py')
    spec=importlib.util.spec_from_file_location('archived_saps_pure',file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_source_bytes_disjoint_partition_and_frozen1000_draws():
    c=cfg();s.validate_config(c);d=data();p.validate_data(d);split=p.partition(d)
    for local,remote in [(f'{n}.json',f'data/sarcasm/{n}.json') for n in ('train','val','test')]+[('prompts.json','prompts/sarcasm.json'),('LICENSE','LICENSE')]:
        assert s.path('evaluation/persona_saps').joinpath(local).read_bytes()==s.path('outputs/persona-literature-audit-01/saps-steering').joinpath(remote).read_bytes()
    assert len(split['test'])==100 and split['custom_ids']==[r['id'] for r in split['test'][:20]]
    draws=p.instruction_draws();rng=random.Random(42);assert draws==[rng.choice(range(5)) for _ in range(1000)] and draws[:500]!=draws[500:]
    bad=copy.deepcopy(d);bad['val'][0]=bad['train'][0]
    with pytest.raises(ValueError):p.validate_data(bad)
    for key,value in [('batch_size',4),('max_new_tokens',61),('maximum_judge_calls',147),('blocks',[11,15,19,35])]:
        bad=copy.deepcopy(c);bad[key]=value
        with pytest.raises(ValueError):s.validate_config(bad)


def test_frozen_branch_plans_distinct_sources_partial_batches_and_call_bound():
    split=p.partition(data());plans=p.plans(split,cfg())
    assert len(plans['initial'])==5 and len(plans['validation'])==30 and len(plans['test_both'])==25 and len(plans['custom_both'])==13
    assert 2*(5+30+25+13)==146 and len(plans['custom_both'][-1])==4
    for name,plan in plans.items():
        width=8 if name.startswith('custom') else 20
        assert all(len(b)<=width and len({e['question_id'] for e in b})==len(b) for b in plan)
        aliases=[e['id'] for b in plan for e in b];assert len(set(aliases))==len(aliases)
    for branch,absent in [('fixed','saps'),('saps','fixed')]:
        assert absent not in {e['condition'] for b in plans['test_'+branch] for e in b}
        assert len(plans['test_'+branch])==20 and len(plans['custom_'+branch])==10


@pytest.fixture(scope='module')
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained('.hf/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c',local_files_only=True,trust_remote_code=False)


def test_actual_cached_prompt_native_ids_no_extra_bos(tokenizer):
    module=author();d=data();user=p.task_user(d['val'][0],d['prompts']['plain'])
    assert user.startswith('\n') and user.endswith('\n') and '\n\nSentence:\n"' in user
    class Recording:
        def apply_chat_template(self,messages,**kw):return messages[0]['content']
    assert module.build_chat_prompt(Recording(),d['val'][0],d['prompts']['plain'])==user
    prompt=s.render_prompt(tokenizer,{'id':'x','user':user});ids=tokenizer(prompt,add_special_tokens=False)['input_ids']
    assert '<|im_start|>system' not in prompt and prompt.endswith('<think>\n\n</think>\n\n')
    assert tokenizer.bos_token_id is None and tokenizer.decode(ids,skip_special_tokens=False)==prompt
    assert ids==tokenizer.apply_chat_template([{'role':'user','content':user}],tokenize=True,add_generation_prompt=True,enable_thinking=False)


@pytest.fixture(scope='module')
def tiny():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.set_num_threads(1);torch.manual_seed(42)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=31,hidden_size=16,intermediate_size=32,num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=1,head_dim=8,pad_token_id=0,eos_token_id=2,max_position_embeddings=64)).eval()


@pytest.mark.parametrize('method',['fixed','saps'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_exact_author_tensor_hook_broadcast_scalar_dtype_and_random_axis(method,dtype):
    module=author();torch.manual_seed(42);h=torch.randn(1,5,16).to(dtype);u=torch.randn(16);u/=u.norm();target=torch.tensor(2.);c=.37
    class Identity(torch.nn.Module):
        def __init__(self):super().__init__();self.model=SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity()]))
        def forward(self,x):return self.model.layers[0](x)
    model=Identity();alpha=1.4
    old=module.make_fixed_projection_hook(0,u,c)[1](h,None) if method=='fixed' else module.make_saps_hook(0,u,target,alpha)[1](h,None)
    audit={}
    with p.intervention(model,0,u,target,alpha,method,c if method=='fixed' else None,audit=audit):new=model(h)
    assert torch.equal(new,old) and audit['hooks_removed'] and audit['prefill_calls']==1
    delta=(new-h).float();assert delta[:,0].abs().max()>0 and torch.allclose(delta[:,0],delta[:,-1],atol=.02 if dtype==torch.bfloat16 else 1e-6)
    r=torch.roll(u,1)
    with p.intervention(model,0,u,target,alpha,method,c if method=='fixed' else None,random_unit=r):randomized=model(h)
    assert not torch.equal(randomized,new)
    with pytest.raises(ValueError):
        with p.intervention(model,0,u,target[None],alpha,method,c if method=='fixed' else None):pass
    if method=='fixed':
        for invalid in (torch.tensor(c),torch.tensor([c]),float('nan'),float('inf'),1,True):
            with pytest.raises(ValueError):
                with p.intervention(model,0,u,target,alpha,method,invalid):pass
    with pytest.raises(ValueError):
        with p.intervention(model,0,u,target,alpha,method,c if method=='fixed' else None):model(h.expand(2,-1,-1))
    assert not model.model.layers[0]._forward_hooks


@pytest.mark.parametrize('target_value',[.7,100_000_000.])
def test_fixed_c_from_actual_capture_and_author_generate_operations_through_bf16(tiny,target_value):
    captured={}
    with p.capture(tiny,[1],captured),torch.inference_mode():tiny(input_ids=torch.tensor([[3,4,5]]),attention_mask=torch.ones(1,3,dtype=torch.long))
    h0=captured[1];u=torch.zeros_like(h0);u[0]=1.;target=torch.tensor(target_value,dtype=torch.float32)
    # Execute only the two pure assignments from pinned generate.py, never its CLI.
    tree=ast.parse(s.path('outputs/persona-literature-audit-01/saps-steering/src/generate.py').read_text())
    assignments={node.targets[0].id:node for node in ast.walk(tree) if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('proj','c')}
    pure=ast.Module(body=[assignments['proj'],assignments['c']],type_ignores=[])
    namespace={'resid_plain':h0[None,:],'steering_directions':u[None,:],'target_projection':target[None],'layer_idx':0,'args':SimpleNamespace(alpha=1.)}
    exec(compile(pure,'archived_generate_fixed_assignments','exec'),namespace)
    expected=namespace['c'];actual=p.fixed_scalar(target,h0,u,1.)
    assert type(actual) is float and actual==expected
    if target_value==100_000_000.:
        assert actual!=float((target-(h0*u).sum()).item())
    class Identity(torch.nn.Module):
        def __init__(self):super().__init__();self.model=SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity()]))
        def forward(self,h):return self.model.layers[0](h)
    model=Identity();hidden=torch.randn(1,5,h0.numel()).to(torch.bfloat16)
    expected_output=author().make_fixed_projection_hook(0,u,expected)[1](hidden,None)
    with p.intervention(model,0,u,target,1.,'fixed',actual):actual_output=model(hidden)
    assert torch.equal(actual_output,expected_output)
    assert torch.equal(actual_output-hidden,expected_output-hidden)


def test_capture_tiny_raw_last_block_not_final_norm_and_zero_cached_hooks(tiny):
    inputs={'input_ids':torch.tensor([[3,4,5]]),'attention_mask':torch.ones(1,3,dtype=torch.long)};collected={}
    with p.capture(tiny,range(3),collected),torch.inference_mode():output=tiny(**inputs,output_hidden_states=True)
    assert torch.equal(collected[0],output.hidden_states[1][0,-1]) and torch.equal(collected[1],output.hidden_states[2][0,-1])
    assert not torch.equal(collected[2],output.hidden_states[3][0,-1]) and torch.equal(tiny.model.norm(collected[2]),output.hidden_states[3][0,-1])
    opts={'do_sample':False,'min_new_tokens':3,'max_new_tokens':3,'use_cache':True};u=torch.arange(16).float();u/=u.norm();audit={};captured={}
    with torch.inference_mode():
        ordinary=tiny.generate(**inputs,**opts)
        with p.capture(tiny,range(3),captured,first_prefill=True),p.intervention(tiny,1,u,torch.tensor(1.),1.,'fixed',0.0,audit=audit):zero=tiny.generate(**inputs,**opts)
    assert torch.equal(ordinary,zero) and audit['prefill_calls']==1 and audit['decode_calls']==2
    assert torch.equal(captured[0],collected[0]) and len(audit['shift_norms']['prefill'])==1 and len(audit['shift_norms']['decode'])==2
    with pytest.raises(RuntimeError):
        with p.capture(tiny,range(3),{}),p.intervention(tiny,1,u,torch.tensor(1.),1.,'saps'):raise RuntimeError('synthetic cleanup')
    assert not tiny._forward_hooks and all(not layer._forward_hooks for layer in tiny.model.layers)


def test_adaptive_cached_generation_matches_author_and_fullprefix_scope(tiny):
    module=author();torch.manual_seed(1);u=torch.randn(16);u/=u.norm();target=torch.tensor(.8);inputs={'input_ids':torch.tensor([[3,4,5]]),'attention_mask':torch.ones(1,3,dtype=torch.long)}
    for cached in (False,True):
        options={'min_new_tokens':3,'max_new_tokens':3,'do_sample':False,'use_cache':cached}
        oldhook=module.make_saps_hook(1,u,target,1.4)[1];handle=tiny.model.layers[1].register_forward_hook(lambda mod,args,out:oldhook(out,None))
        try:
            with torch.inference_mode():old=tiny.generate(**inputs,**options)
        finally:handle.remove()
        with p.intervention(tiny,1,u,target,1.4,'saps'),torch.inference_mode():new=tiny.generate(**inputs,**options)
        assert torch.equal(old,new)
    # Cache and full-prefix semantics need not yield the same trajectory.


def test_author_direction_target_equations_with1000_draws_without_TL():
    module=author();rows=[f'case {i}' for i in range(500)];prompts=data()['prompts'];saved=random.getstate()
    def representation(prompt):
        h=sum((i+1)*ord(c) for i,c in enumerate(prompt));return torch.tensor([[h%101/100,h%73/50+1.,h%41/30],[h%67/50,h%97/30,h%31/20+1.]])
    class Model:
        cfg=SimpleNamespace(n_layers=2,d_model=3)
        def to_tokens(self,prompt):return prompt
        def run_with_cache(self,prompt,**kw):return None,{f'blocks.{i}.hook_resid_post':v[None,None,:] for i,v in enumerate(representation(prompt))}
    class Tokenizer:
        def apply_chat_template(self,messages,**kw):return messages[0]['content']
    try:
        random.seed(42);old_u,old_t=module.learn_style_direction(Model(),Tokenizer(),rows,prompts['plain'],prompts['style'],'random')
    finally:random.setstate(saved)
    draws=p.instruction_draws();raw=torch.zeros(2,3)
    for i,row in enumerate(rows):raw+=representation(p.task_user(row,prompts['style'][draws[i]]))-representation(p.task_user(row,prompts['plain']))
    units=torch.nn.functional.normalize(raw/500,dim=1);targets=torch.zeros(2)
    for i,row in enumerate(rows):targets+=(representation(p.task_user(row,prompts['style'][draws[500+i]]))*units).sum(1)
    targets/=500
    assert torch.equal(units,old_u) and torch.equal(targets,old_t)


def rating(id,score=4):return {'id':id,**{k:score for k in p.FIELDS}}
def metric(style=3.,**kw):return {'n':50,'sarcasm':style,'entailment':4.,'grammaticality':4.,'clarity':4.,'style_at_least3':30,'empty':0,'degenerate':0,'capped':0,**kw}


def test_reference_schema_gates_selection_absent_methods_and_norm_flags():
    p.validate_ratings(['a'],[rating('a')])
    for value in (True,6,None,3.5):
        r=rating('a');r['clarity']=value
        with pytest.raises(ValueError):p.validate_ratings(['a'],[r])
    assert p.elicitation_gate({'prompt':metric()})
    assert not p.elicitation_gate({'prompt':metric(style_at_least3=24)})
    observed={n:metric(3.) for n in p.SETTINGS};base=metric(2.)
    choice=p.select(observed,base);assert choice['winner']=='fixed_b11_a1' and choice['branch']=='both'
    for n in observed:
        if p.SETTINGS[n]['method']=='fixed':observed[n]['grammaticality']=3.
    choice=p.select(observed,base);assert choice['best']['fixed'] is None and choice['branch']=='saps' and choice['winner']=='saps_b11_a1'
    for v in observed.values():v['entailment']=2.
    assert p.select(observed,base)['selection_null']
    norms=p.norm_diagnostics([{'shift_norms':{'prefill':[2.],'decode':[1.,1.]}}],[{'shift_norms':{'prefill':[3.],'decode':[1.]}}])
    assert norms['prefill']['norm_mismatched'] and not norms['decode']['norm_mismatched'] and norms['prefill']['real']['count']==1
    assert 'Per-position' in norms['prefill']['scope'] and 'sqrt(S)' in norms['prefill']['scope']


def generation(rows):
    predictions=[];batches=[]
    for i,row in enumerate(rows):
        predictions.append({'id':row['id'],'output':'Synthetic only.','input_ids':[4,5],'rendered_prompt':'synthetic '+row['id'],
            'generated_ids':[3,2],'generated_tokens_including_eos':2,'ended_eos':True,'hit_token_limit':False,'batch_index':i,'batch_position':0,'batch_ids':[row['id']],'padded_prompt_width':2})
        batches.append({'batch_index':i,'ids':[row['id']],'padded_prompt_width':2})
    return {'predictions':predictions,'batches':batches,'settings':{'batch_size':1,'max_new_tokens':60,'seed':42,'do_sample':False}}


def setup(tmp_path,monkeypatch,branch='both'):
    run=tmp_path/'new';run.mkdir();c=cfg();split=p.partition(data());plans=p.plans(split,c);m={'sources':{},'binary':str(tmp_path/'native'),'cli_version':'2.1.323 (Claude Code)','audit_sha256':'synthetic','tokenizer_ids':{'eos':2,'pad':0}}
    s.save(run/'manifest.json',{'sources':{}})
    schema=s.read_json('evaluation/persona_saps/rubric.json')
    rendered={splitname+suffix:{r['id']:{'input_ids':[4,5],'rendered_prompt':'synthetic '+r['id']} for r in split[splitname]} for splitname in ('validation','test') for suffix in ('_plain','_prompt')}
    for n,v in [('split',split),('plans',plans),('rubric',schema),('rendered',rendered)]:s.save(run/'inputs'/(n+'.json'),v)
    arms=['baseline',*(['fixed','saps'] if branch=='both' else [branch]),'random','prompt']
    for phase,conditions,rows in [('initial',['baseline','prompt'],split['validation']),('validation',list(p.SETTINGS),split['validation']),('test',arms,split['test'])]:s.save(run/phase/'predictions.json',{a:generation(rows) for a in conditions})
    context=SimpleNamespace(run=run,cfg=c,m=m,initial=lambda:None,verify=lambda **kw:None)
    binary=tmp_path/'native';binary.write_text('never executed');monkeypatch.setattr(s.pv,'BINARY_SHA',s.stream_hash(binary));monkeypatch.setattr(s.shutil,'which',lambda _:str(binary));monkeypatch.setattr(s.f,'version',lambda _:m['cli_version']);monkeypatch.setattr(s.f,'validate_audit',lambda _: {})
    monkeypatch.setattr(s,'stage_sources',lambda s,phase:{'synthetic_phase':phase});monkeypatch.setattr(s,'selected',lambda *a,**kw:{'branch':branch,'setting':{'method':'saps','block':19,'alpha':1.}})
    return context


def calls(monkeypatch,behavior='valid'):
    records=[];seen={}
    def invoke(binary,body):
        records.append(copy.deepcopy(body));h=s.f.canonical_hash(body);seen[h]=seen.get(h,0)+1
        if behavior=='uncertain':raise OSError('synthetic transport failure')
        custom='scale' in body['rubric'];ratings=[{'id':r['id'],'profanity':0,'rudeness':0,'sarcasm':0,'gross_failure':False,'content_ok':True,'coherent':True} if custom else rating(r['id']) for r in body['items']]
        if behavior=='exhaust' or behavior=='retry' and seen[h]==1:ratings=ratings[:-1]
        outer={'subtype':'success','modelUsage':{s.f.MODEL:{}},'result':json.dumps({'ratings':ratings})}
        if behavior=='model':outer['modelUsage']={}
        return {'returncode':0,'canonical_stdin_sha256':h,'isolation':{'git_not_repository':True,'git_environment_clean':True,'returncode':128},'stdout':json.dumps(outer)}
    monkeypatch.setattr(s.f,'invoke',invoke);return records


def judge(context,phase,resume=False):
    s.judge(SimpleNamespace(phase=phase,resume=resume),context)
    if hasattr(context,'pending_accept'):s.f.save_or_match(*context.pending_accept);del context.pending_accept


def test_all73_batches146_ceiling_schema_retries_first_valid_resume(tmp_path,monkeypatch):
    context=setup(tmp_path,monkeypatch);records=calls(monkeypatch,'retry')
    for phase in p.PHASES:
        s.packet(SimpleNamespace(phase=phase),context);judge(context,phase);s.read_reviews(context,phase)
    assert len(records)==146 and all(records[i]==records[i+1] for i in range(0,146,2))
    judge(context,'custom',True);assert len(records)==146
    s.audit_journals(context)
    assert all(set(r)=={'id','original_complaint','generated_sentence'} for b in records for r in b['items'])


@pytest.mark.parametrize('behavior,count',[('exhaust',2),('model',1),('uncertain',1)])
def test_fatal_identity_schema_uncertain_no_new_call(tmp_path,monkeypatch,behavior,count):
    context=setup(tmp_path,monkeypatch);records=calls(monkeypatch,behavior);s.packet(SimpleNamespace(phase='initial'),context)
    with pytest.raises((s.f.InvalidSubmission,s.f.ModelMismatch,OSError)):
        try:judge(context,'initial')
        except BaseException as exc:s.stop(context.run,'live_stage_fatal',exc);raise
    with pytest.raises(RuntimeError):s.finish_grades(context,'initial',s.bodies(context,'initial'),context.run/'grades/initial','bound')
    assert len(records)==count


def test_resume_absent_and_packet_receipt_tampering(tmp_path,monkeypatch):
    context=setup(tmp_path,monkeypatch);records=calls(monkeypatch);s.packet(SimpleNamespace(phase='initial'),context)
    with pytest.raises(ValueError):judge(context,'initial',True)
    assert not records and not (context.run/'grades').exists()
    judge(context,'initial');file=context.run/'grades/initial/batch-000-attempt-0-receipt.json';v=s.read_json(file);v['canonical_stdin_sha256']='wrong';file.write_text(json.dumps(v))
    with pytest.raises(ValueError):s.read_reviews(context,'initial')
    packet=context.run/'judging/initial/packet.json';v=s.read_json(packet);v[0][0]['generated_sentence']='Changed';packet.write_text(json.dumps(v))
    record=context.run/'judging/initial/manifest.json';v=s.read_json(record);v['packet_sha256']=s.stream_hash(packet);record.write_text(json.dumps(v))
    with pytest.raises(ValueError):s.bodies(context,'initial')


@pytest.mark.parametrize('branch',['fixed','saps'])
def test_absent_method_branch_no_replacement_and_custom_exact_subset(tmp_path,monkeypatch,branch):
    context=setup(tmp_path,monkeypatch,branch);s.packet(SimpleNamespace(phase='custom'),context);body=s.bodies(context,'custom')
    assert len(body)==10 and sum(b['expected_count'] for b in body)==80
    records=calls(monkeypatch);judge(context,'custom');assert len(records)==10 and len(s.read_reviews(context,'custom'))==20
    result=s.custom_result(context);assert ('saps' if branch=='fixed' else 'fixed') not in result['metrics']


def test_budget_before_new_start_source_scoped_guard_and_public_null(tmp_path,monkeypatch):
    context=setup(tmp_path,monkeypatch);s.packet(SimpleNamespace(phase='initial'),context);records=calls(monkeypatch);directory=context.run/'grades/initial';directory.mkdir(parents=True)
    for i in range(146):s.save(context.run/'grades/other'/f'batch-{i:03d}-attempt-0-start.json',{})
    with pytest.raises(RuntimeError):s.finish_grades(context,'initial',s.bodies(context,'initial'),directory,'bound')
    assert not records and not list(directory.glob('*start*'))
    with pytest.raises(ValueError):s.guard(s.path('outputs/persona-reference-rudeness-01'))
    counts={'saved_generations':1242,'teacher_forced_sequences':1500,'logit_forwards':2,'mc_items':64};zero={k:0 for k in counts}
    for key in counts:
        with pytest.raises(ValueError):s.charge(zero,counts,**{key:1})


@pytest.mark.parametrize('cause',['elicitation_null','selection_null','live_stage_fatal'])
def test_null_fatal_reports_costs_privacy_no_complete_override(tmp_path,monkeypatch,cause):
    context=setup(tmp_path,monkeypatch);context.cfg['result']=str(tmp_path/'new.json');s.stop(context.run,cause);s.save(context.run/'initial/run.json',{'completed':True,'saved_generations':100,'peak_allocated_bytes':123})
    s.report(SimpleNamespace(),context);result=s.read_json(tmp_path/'new.json');assert result['status']==('incomplete' if cause=='live_stage_fatal' else cause) and result['totals']['saved_generations']==100
    s.private_export(result)
