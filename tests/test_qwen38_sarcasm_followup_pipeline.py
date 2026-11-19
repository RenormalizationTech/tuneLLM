"""Scratch-artifact integration tests; all GPU/judge workers are stubbed."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from scripts import qwen38_sarcasm_followup as r
from scripts import qwen38_sarcasm_followup_analyze as a
from scripts.common import read_json


def write(root,name,value):
    p=Path(root)/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value)+'\n');return p


@pytest.fixture
def old_inputs(tmp_path,monkeypatch):
    cfg=read_json(r.CONFIG);cfg['old_run']=str(tmp_path/'old');cfg['model']=str(tmp_path/'model');cfg['questions']=str(tmp_path/'fresh.json')
    write(tmp_path/'model','generation_config.json',{'eos_token_id':[2]})
    write(tmp_path/'model','config.json',{'layer_types':['linear_attention','full_attention','full_attention'],
        'num_hidden_layers':3,'hidden_size':16,'head_dim':16})
    write(tmp_path/'old','inputs.json',{'elicitation':[{'system':'S'},{'system':'L'}],
        'probes':[{'id':'C01','user':'old1'},{'id':'C02','user':'old2'}]})
    rows=[]
    for qi in range(1,33):
        for arm in ['baseline','prompted']:
            row={'id':f'C{qi:02d}_{arm}','question_id':f'C{qi:02d}','arm':arm,'user':f'old{qi}',
                'input_ids':[11,12] if arm=='prompted' else [12],'generated_ids':[3]*160,
                'output':'A'*160,'prompt':'S' if arm=='prompted' else 'N','hit_token_limit':True}
            rows.append(row);root=tmp_path/'old/development/rows'
            p=write(root,row['id']+'.json',row);start=write(root,row['id']+'.start.json',{})
            write(root,row['id']+'.receipt.json',{'sha256':r.sha(p),'start_sha256':r.sha(start)})
    write(tmp_path/'old','development/result.json',{'rows':rows})
    write(tmp_path,'fresh.json',[{'id':f'F{i:02d}','user':f'Synthetic novel fixture item {i}'} for i in range(1,33)])
    def prompt(tok,user,system=None):
        return {'prompt':system or 'N','input_ids':{'S':[11,12],'L':[13,12],None:[12]}[system],
                'system_length':1 if system else None,'user_turn_ids':[77]}
    monkeypatch.setattr(r,'prompt',prompt)
    class Tok:
        all_special_ids=[2]
        def decode(self,ids,skip_special_tokens=True):return 'A'*len([i for i in ids if i!=2])
    return cfg,Tok(),rows,prompt


def test_build_inputs_complete_and_equal_context_positions(old_inputs):
    cfg,tok,rows,prompt=old_inputs
    inputs,files=r.build_inputs(cfg,tok)
    assert len(inputs['replays'])==192 and len(inputs['generations'])==224
    assert inputs['model_geometry']['layer_types'][1]=='full_attention'
    by={x['id']:x for x in inputs['replays']}
    assert by['C01_B_S']['response_mask']==by['C01_B_L']['response_mask']
    assert len(files)==192


@pytest.mark.parametrize('mode',['length','old_prompt','receipt','decode'])
def test_build_inputs_failure_paths(old_inputs,monkeypatch,mode):
    cfg,tok,rows,original=old_inputs
    if mode=='length':
        def changed(tok,user,system=None):
            out=original(tok,user,system)
            if system=='L':out['input_ids']=[13,14,12]
            return out
        monkeypatch.setattr(r,'prompt',changed)
    elif mode=='old_prompt':
        rows[1]['input_ids']=[9,9];write(Path(cfg['old_run']),'development/result.json',{'rows':rows})
    elif mode=='receipt':
        write(Path(cfg['old_run']),'development/rows/C01_baseline.receipt.json',{'sha256':'bad','start_sha256':'bad'})
    else:
        monkeypatch.setattr(tok,'decode',lambda *args,**kwargs:'drift')
    with pytest.raises(ValueError):r.build_inputs(cfg,tok)


def test_complete_192_cell_analysis_without_N_crops_and_partial_primary_closeout(tmp_path):
    from tests.test_qwen38_gcad_port import tiny,tensors
    cfg=read_json(r.CONFIG);cfg.update(layers=[1],bootstrap_samples=50,target_cap=4,model=str(tmp_path/'model'))
    write(tmp_path/'model','generation_config.json',{'eos_token_id':2,'pad_token_id':0})
    write(tmp_path,'config.json',cfg);write(tmp_path,'manifest.json',{})
    write(tmp_path,'inputs.json',{'model_geometry':{'layer_types':['linear_attention','full_attention','full_attention']}})
    bank={'1.delta':torch.tensor([1.,0.]),'1.key':torch.ones(1,2),'1.demand':torch.tensor(0.)}
    save_file(bank,str(tmp_path/'bank.safetensors'))
    for seed,delta in [(9831,[0.,1.]),(9832,[-1.,0.]),(9833,[1.,1.])]:
        save_file({'1.delta':torch.tensor(delta)},str(tmp_path/f'random-{seed}.safetensors'))
    records=[];(tmp_path/'diagnostic').mkdir()
    for qi in range(1,33):
        for content in ['B','P']:
            for context in ['S','L','N']:
                key=f'C{qi:02d}_{content}_{context}'
                vectors={f'block.{i}':torch.tensor([float(i+qi),1.]) for i in range(3)}
                vectors['attention.1']=torch.tensor([0.,1.])
                if context!='N':
                    vectors['crop.1']=torch.tensor([1. if context=='S' else .25, .2 if content=='B' else .3])
                    vectors['attention_tokens.1']=torch.tensor([[1.,1.],[-1.,1.]]) if context=='S' else torch.tensor([[0.,1.],[0.,1.]])
                    vectors['attention_tokens.1']=vectors['attention_tokens.1'].to(torch.bfloat16)
                file=tmp_path/'diagnostic'/f'{key}.safetensors';save_file(vectors,str(file))
                records.append({'id':key,'tensor_file':str(file.relative_to(tmp_path)),'tensor_sha256':r.sha(file),
                    'summary':{k:{'mean_token_norm':2.} for k in vectors if k.startswith(('block.','attention.'))},
                    'gate':{'1':{'gate':{'mean':.8 if context=='S' else .2}}},'nll':1. if context=='S' else 2.})
    write(tmp_path,'diagnostic/result.json',{'completed':True,'rows':records})
    # Real generate() return schema, including actual recorder telemetry.
    # This is a tiny CPU model, not a call to the study runtime/model.
    class Tok:
        pad_token_id=0;eos_token_id=2
        def __call__(self,*args,**kw):return {'input_ids':[3,4,5]}
        def decode(self,*args,**kw):return 'Check the connection and restart the device carefully.'
    s=object.__new__(r.Context);s.run=tmp_path;s.cfg=cfg
    model=tiny();prototype=r.generate(s,model,Tok(),{'id':'prototype','prompt':'frozen prompt','input_ids':[3,4,5],
        'system_length':None,'user_turn_ids':[3,4,5],'arm':'b2','question_id':'F01','user':'fresh synthetic question'},
        tensors(model),2.,'generation')
    rows=[];scores={}
    for qi in range(1,33):
        for arm in cfg['arms']:
            key=f'F{qi:02d}_{arm}';row=json.loads(json.dumps(prototype))
            row.update(id=key,question_id=f'F{qi:02d}',user=f'Fresh synthetic question {qi}',arm=arm)
            if arm in ['baseline','prompted']:row.update(telemetry=None,audit={})
            rows.append(row)
            scores[key]={'trait':10 if arm=='baseline' else 70,'coherence':80}
    for i,row in enumerate(rows):write(tmp_path,f'counters/generation/{i:04d}.json',{'stage':'generation','key':row['id'],'tokens':4})
    write(tmp_path,'counters/teacher_forced/0000.json',{'stage':'diagnostic','key':'C01_B_S','tokens':0})
    write(tmp_path,'sessions/gpu-1.start.json',{'unix':1})
    write(tmp_path,'sessions/gpu-1.done.json',{'seconds':123,'returncode':0})
    write(tmp_path,'generation/result.json',{'completed':True,'rows':rows})
    for metric in ['trait','coherence']:write(tmp_path,f'judge/{metric}.json',{k:v[metric] for k,v in scores.items()})
    result=a.analyze(tmp_path,'judge_failure','impolite stopped')
    assert result['part_A']['status']=='complete'
    assert 'crop.1' in result['part_A']['A1_alignment']
    assert result['part_A']['A2_magnitude']['attention.1']['B']['mean_token_difference_norm']['mean']==1
    assert result['part_A']['A2_magnitude']['attention.1']['B']['norm_of_mean_instruction_difference']==0
    assert result['outcome']=='primary_complete_descriptive_incomplete'
    assert result['part_B']['exploratory_candidate']==2
    assert not result['part_B']['b16_direction_check']['margin_at_least20']
    assert result['old_outcome_unchanged']=='development_null'
    assert result['ledger']['counters']['generation']['reserved_starts']==224
    assert result['ledger']['sessions']['gpu']['seconds']==123
    assert result['part_B']['telemetry']['arms']['b2']['1']['prefill']['tokens']==96
    json.dumps(result,allow_nan=False)
    for metric in ['trait','coherence']:(tmp_path/f'judge/{metric}.json').unlink()
    result=a.analyze(tmp_path,'judge_audit_failure','isolated audit failed')
    assert result['outcome']=='judge_audit_failure' and result['part_A']['status']=='complete'
    (tmp_path/'generation/result.json').unlink()
    result=a.analyze(tmp_path,'implementation_stop','generation failed')
    assert result['outcome']=='implementation_stop' and result['part_A']['status']=='complete'
    write(tmp_path,'generation/result.json',{'completed':True,'rows':rows})
    for metric in ['trait','coherence','impolite']:
        write(tmp_path,f'judge/{metric}.json',{key:(value[metric] if metric in value else 5) for key,value in scores.items()})
    result=a.analyze(tmp_path,'completed',None)
    assert result['outcome']=='exploratory_candidate' and result['part_B']['status']=='complete'


def test_launcher_audit_failure_runs_gpu_and_skips_judging(tmp_path,monkeypatch):
    cfg=read_json(r.CONFIG);cfg['result']=str(tmp_path/'separate-result.json')
    s=object.__new__(r.Context);s.run=tmp_path/'run';s.run.mkdir();s.cfg=cfg;s.approval=tmp_path/'approval.json'
    called=[]
    def process(s,kind,remaining,extra=None):
        called.append(kind)
        if kind=='audit':raise RuntimeError('audit failed')
        if kind=='gpu':write(s.run,'generation/result.json',{'completed':True})
    monkeypatch.setattr(r,'process_worker',process);monkeypatch.setattr(r,'preflight_gpu',lambda s:None)
    monkeypatch.setattr(a,'analyze',lambda run,status,error:{'outcome':status})
    r.launch(s)
    assert called==['audit','gpu']
    assert read_json(tmp_path/'separate-result.json')['outcome']=='judge_audit_failure'
    called.clear();r.launch(s);assert not called


def test_cluster_bootstrap_rejects_token_level_shape():
    w=a.weights(32,20,19)
    with pytest.raises(ValueError):a.mean_ci(np.ones(160),w)


def test_norm_of_mean_bootstrap_and_ci_wording_are_prespecified():
    x=np.array([[1.,0.],[2.,0.],[3.,0.],[4.,0.]])
    w=a.weights(4,100,19)
    assert a.norm_mean_ci(x,w)['bootstrap95']==pytest.approx(np.quantile(np.linalg.norm(w@x,axis=-1),[.025,.975]))
    assert a.cosine_ci(x,np.array([1.,0.]),w)['ci_wording']=='positively_aligned'
    assert a.cosine_ci(x,np.array([-1.,0.]),w)['ci_wording']=='anti_aligned'


def test_parent_watchdog_terminates_process_group_and_preserves_session(tmp_path,monkeypatch):
    import subprocess
    s=object.__new__(r.Context);s.run=tmp_path;s.approval=tmp_path/'approval.json';signals=[]
    class Process:
        pid=12345;returncode=None
        def wait(self,timeout=None):
            if self.returncode is None:raise subprocess.TimeoutExpired('synthetic-worker',timeout)
            return self.returncode
        def poll(self):return self.returncode
    child=Process()
    monkeypatch.setattr(r.subprocess,'Popen',lambda *args,**kwargs:child)
    def kill(pid,sig):signals.append((pid,sig));child.returncode=-15
    monkeypatch.setattr(r.os,'killpg',kill)
    with pytest.raises(subprocess.TimeoutExpired):r.process_worker(s,'gpu',.001,123.)
    assert signals==[(12345,r.signal.SIGTERM)]
    assert len(list((tmp_path/'sessions').glob('gpu-*.done.json')))==1


def test_combined_judge_and_token_budget_refusal(tmp_path):
    s=object.__new__(r.Context);s.run=tmp_path;s.cfg=read_json(r.CONFIG)
    s.cfg['maximum_judge_starts']=2
    s.reserve('judge','audit','one');s.reserve('judge','judge','two')
    with pytest.raises(r.BudgetStop):s.reserve('judge','judge','three')
    s.cfg['maximum_generated_tokens']=159
    with pytest.raises(r.BudgetStop):s.reserve('generation','generation','oversized',160)
