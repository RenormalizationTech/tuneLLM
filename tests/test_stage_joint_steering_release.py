"""Additive packaging tests with synthetic aggregates, temp bundles and CPU only."""
import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import pytest
from scripts import stage_joint_steering_release as stage
from scripts import joint_gcad_common as common

ROOT=Path(__file__).resolve().parents[1]


def synthetic_report():
    metric={t:20. for t in common.METRICS}
    metric.update(n=24,invalid=0,empty=0,caps=3,mean_tokens=120.,lexicon_hits=1)
    return {'status':'complete','manifest_sha256':'a'*64,'source_sha256':{'scripts/private_name.py':'b'*64},
        'native_identity':{'runtime_config':'configs/qwen3_14b_baseline.json'},
        'analysis':{'metrics':{a:copy.deepcopy(metric) for a in common.arms()},
            'guide':common.guide({a:copy.deepcopy(metric) for a in common.arms()}),
            'arm_gain_question_bootstrap_ci95':{},'single_cross_effects':{},'spearman':{}},
        'calibration':{'pass':True},'duplicate_noise':{'coherence':{'n':24}}}

def synthetic_audit(primary_sha):
    return {'status':'complete_posthoc_descriptive','post_hoc':True,'primary_decisions_modified':False,
        'inputs_sha256':{stage.REPORT:primary_sha},
        'arms':{a:{'n_any_hit':1,'mean_per_response_rate_primary':3.} for a in common.arms()},
        'correlations':{'within_arm':{a:{m:{'n':24,'spearman':None} for m in common.METRICS} for a in common.arms()}}}

@pytest.fixture
def candidate(tmp_path,monkeypatch):
    work=tmp_path/'work';work.mkdir();(work/'scripts').mkdir()
    shutil.copyfile(ROOT/'scripts/joint_gcad.py',work/'scripts/joint_gcad.py')
    shutil.copytree(ROOT/'scripts/joint_release_runtime',work/'scripts/joint_release_runtime',ignore=shutil.ignore_patterns('__pycache__'))
    report=work/'results/joint_gcad_01.json';report.parent.mkdir();report.write_text(json.dumps(synthetic_report()))
    monkeypatch.setattr(stage,'ROOT',work);monkeypatch.setattr(stage,'DEST',work/'stage04')
    monkeypatch.setattr(stage,'terminal_report',synthetic_report)
    audit=synthetic_audit(stage.sha(report));(work/stage.LEXICAL_REPORT).write_text(json.dumps(audit))
    monkeypatch.setattr(stage,'terminal_audit',lambda:audit)
    receipt=stage.stage();return work/'stage04',receipt


def import_runtime(root,monkeypatch):
    names=[n for n in sys.modules if n in ('runtime','joint_controls') or n.startswith(('runtime.','joint_controls.'))]
    for n in names:monkeypatch.delitem(sys.modules,n,raising=False)
    monkeypatch.syspath_prepend(str(root));importlib.invalidate_caches()
    return importlib.import_module('joint_controls.loader')


def test_hook_only_public_import_and_docstring_change():
    original=ast.parse((ROOT/'scripts/joint_gcad.py').read_text());public=ast.parse(stage.portable_steerer())
    original.body=original.body[1:];public.body=public.body[1:]
    imports=[i for i,n in enumerate(original.body) if isinstance(n,ast.ImportFrom) and n.module=='scripts']
    assert len(imports)==1
    original.body[imports[0]]=ast.parse('from runtime import gcad as core').body[0]
    assert ast.dump(original,include_attributes=False)==ast.dump(public,include_attributes=False)


def test_additive_identity_manifest_card_and_commit_message(candidate):
    root,r=candidate;parent=stage.parent_checked();m=stage.read(root/stage.H2)
    assert stage.sha(root/'release_manifest.json')==stage.MF
    for name in set(parent['files_sha256'])|{'release_manifest.json'}:
        if name not in ('README.md','METHODS.md'):assert (root/name).read_bytes()==(stage.PARENT/name).read_bytes()
    assert set(m['files_sha256'])==set(parent['files_sha256'])|{'release_manifest.json'}|stage.NEW
    assert len(stage.CRITICAL)==8
    assert r['required_commit_message']==f'Release manifest SHA256: {stage.MF}\nJoint manifest SHA256: {r["joint_manifest_sha256"]}'
    text=(root/'README.md').read_text()
    assert 'adapter remains private' not in text and 'are now public' in text and stage.COMMIT in text
    guide=(root/'joint_controls/GUIDE.md').read_text()
    assert 'not a general benchmark' in guide and 'PF/R/S coefficients' in guide
    assert '| full_110 | 5/3.5/0 |' in guide and 'N/A(nonpositive single gain)' in guide
    assert 'zero knob means no injection, NOT absence' in guide and 'Within-arm trait Spearman' in guide
    assert 'PF (friendly rubric)' in guide and 'Any lex hit /24' in guide and 'Mean hits/100 words' in guide
    assert 'explicitly post-hoc' in guide and 'NOT establish less literal swearing' in guide
    assert 'Post-hoc lexical density: within-arm Spearman' in guide and 'not SE of the mean' in guide
    assert (root/'joint_controls/lexical_audit.json').is_file() and r['files']==34
    assert 'No private physics adapter' not in (root/'README.md').read_text()
    assert 'physics adapter remains private' not in (root/'METHODS.md').read_text()
    assert 'Refusal closures and public physics adapter' in (root/'METHODS.md').read_text()
    assert stage.old.PROJECT_PATH.search((root/'joint_controls/results.json').read_text()) is None
    outside=root.with_name(root.name+'-publication-manifest.json')
    assert stage.read(outside)['files']['joint_controls/manifest.json']['sha256']==r['joint_manifest_sha256']

@pytest.mark.parametrize('kind',['private','status','hash','secret','extra','symlink','bank','preset'])
def test_preflight_rejects_bad_payload(candidate,kind):
    root,r=candidate;m=stage.read(root/stage.H2)
    if kind=='private':m['private']=True
    elif kind=='status':m['status']='draft'
    elif kind=='hash':m['files_sha256']['joint_controls/loader.py']='0'*64
    elif kind=='extra':(root/'unexpected.pkl').write_bytes(b'not a pickle')
    elif kind=='symlink':(root/'joint_controls/example.py').unlink();(root/'joint_controls/example.py').symlink_to(root/'runtime/example.py')
    elif kind=='bank':
        p=root/'profanity/bank.safetensors';p.write_bytes(p.read_bytes()+b'x');m['files_sha256']['profanity/bank.safetensors']=stage.sha(p)
    elif kind=='preset':
        p=root/'joint_controls/presets.json';p.write_text('{}');m['files_sha256']['joint_controls/presets.json']=stage.sha(p)
    else:
        p=root/'joint_controls/GUIDE.md';p.write_text('hf_'+'A'*30);m['files_sha256']['joint_controls/GUIDE.md']=stage.sha(p)
    (root/stage.H2).write_text(json.dumps(m))
    with pytest.raises(ValueError):stage.preflight(root)


def test_joint_loader_integrity_hash_identity_and_no_shadow_runtime(candidate,monkeypatch):
    root,r=candidate;loader=import_runtime(root,monkeypatch)
    assert loader.verify_bundle(root,expected_joint_manifest_sha256=r['joint_manifest_sha256'])[1]['parent_manifest_sha256']==stage.MF
    with pytest.raises(ValueError,match='file hash'):loader.verify_bundle(root,expected_joint_manifest_sha256='0'*64)
    # Baseline still validates the base and all three original banks before NULL.
    original=importlib.import_module('runtime.loader');calls=[]
    monkeypatch.setattr(original,'validate_base',lambda *a,**k:calls.append(k))
    ctx=loader.load_controller(root,'baseline',None,None,'unused',base_id=original.BASE,revision=original.REV,expected_joint_manifest_sha256=r['joint_manifest_sha256'])
    with ctx as value:assert value is None
    assert len(calls)==3
    with pytest.raises(ValueError,match='measured fixed'):loader.load_controller(root,'new_strength',None,None,'unused',base_id=original.BASE,revision=original.REV,expected_joint_manifest_sha256=r['joint_manifest_sha256'])
    monkeypatch.setattr(original,'__file__',str(root/'unrelated.py'))
    with pytest.raises(ValueError,match='another directory'):loader.load_controller(root,'baseline',None,None,'unused',base_id=original.BASE,revision=original.REV,expected_joint_manifest_sha256=r['joint_manifest_sha256'])


def test_public_hook_tiny_qwen2_single_zero_and_cached_parity(candidate,monkeypatch):
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    root,r=candidate;import_runtime(root,monkeypatch);public=importlib.import_module('joint_controls.steerer')
    from scripts import joint_gcad as registered
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(10)
            model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=2,
                num_attention_heads=4,num_key_value_heads=2,pad_token_id=0,eos_token_id=None,attention_bias=False,use_sliding_window=False,attn_implementation='sdpa')).bfloat16().eval()
        gen=torch.Generator().manual_seed(5)
        component={'delta':torch.randn(2,16,generator=gen)*.1,'key':torch.randn(2,2,4,generator=gen),'demand':torch.zeros(2),'base_coef':2.,'scale':1.5,'inverse':False}
        ids=torch.tensor([[3,4,5]])
        def run(cls,zero=False):
            v={**component,'base_coef':0. if zero else 2.};ctx=cls(model,{'sarcasm':v},layer_indices=[0,1])
            with torch.no_grad(),ctx:
                a=model(ids,use_cache=True);b=model(torch.tensor([[6]]),past_key_values=a.past_key_values,use_cache=True)
            return a.logits,b.logits
        a=run(registered.JointGCAD);b=run(public.JointGCAD)
        assert all(torch.equal(x,y) for x,y in zip(a,b))
        with torch.no_grad():ordinary=model(ids).logits
        assert torch.equal(ordinary,run(public.JointGCAD,True)[0])
    finally:torch.set_num_threads(old)


def test_stage_requires_terminal_committed_actual_result_before_writes(tmp_path,monkeypatch):
    dest=tmp_path/'forbidden';monkeypatch.setattr(stage,'DEST',dest)
    def fail():raise ValueError('terminal report absent/uncommitted')
    monkeypatch.setattr(stage,'terminal_report',fail)
    with pytest.raises(ValueError,match='terminal'):stage.stage()
    assert not dest.exists()


def test_strict_original_base_revision_guard_reused_before_model_forward(candidate,monkeypatch):
    root,r=candidate;loader=import_runtime(root,monkeypatch)
    with pytest.raises(ValueError,match='exact base ID'):
        loader.load_controller(root,'full_110',SimpleNamespace(config=SimpleNamespace()),None,'unused',
            base_id='Qwen/Qwen2.5-7B-Instruct',revision='0'*40,expected_joint_manifest_sha256=r['joint_manifest_sha256'])


def test_download_root_python_module_command_imports_without_private_repo(candidate):
    import os,subprocess
    root,r=candidate;env=os.environ.copy();env.pop('PYTHONPATH',None);env['CUDA_VISIBLE_DEVICES']=''
    result=subprocess.run([sys.executable,'-B','-m','joint_controls.example','--help'],cwd=root,env=env,capture_output=True,text=True,check=True)
    assert '--preset' in result.stdout and '--joint-manifest-sha256' in result.stdout
    assert not list(root.rglob('__pycache__'))


def test_new_scanner_rejects_private_raw_prompt_and_token_fields():
    for value in ({'prompt':'raw prompt'},{'tokens':[1,2]},{'output':'raw answer'}):
        with pytest.raises(ValueError,match='raw private'):stage.scan(value)


def test_consumer_accepts_optional_hub_git_metadata_but_unknown_payload_rejected(candidate,monkeypatch):
    root,r=candidate;loader=import_runtime(root,monkeypatch)
    attributes=root/'.gitattributes';attributes.write_text('*.safetensors filter=lfs diff=lfs merge=lfs -text\n')
    (root/'.git/refs/heads').mkdir(parents=True);(root/'.git/HEAD').write_text('ref: refs/heads/main\n')
    (root/'.git/refs/heads/main').write_text(stage.COMMIT+'\n')
    (root/'.cache/huggingface').mkdir(parents=True);(root/'.cache/huggingface/metadata').write_text('local cache')
    (root/'joint_controls/__pycache__').mkdir();(root/'joint_controls/__pycache__/loader.cpython-311.pyc').write_bytes(b'not imported')
    before=attributes.read_bytes()
    assert loader.verify_bundle(root,expected_joint_manifest_sha256=r['joint_manifest_sha256'])[1]['consumer_inventory_exclusions']==loader.INVENTORY_EXCLUSIONS
    assert attributes.read_bytes()==before and (root/'.git/HEAD').exists()
    (root/'unknown_payload.json').write_text('{}')
    with pytest.raises(ValueError,match='unexpected payload files'):loader.verify_bundle(root,expected_joint_manifest_sha256=r['joint_manifest_sha256'])


def test_no_stage_before_separate_committed_lexical_audit(tmp_path,monkeypatch):
    monkeypatch.setattr(stage,'DEST',tmp_path/'forbidden')
    monkeypatch.setattr(stage,'terminal_report',synthetic_report)
    def fail():raise ValueError('committed lexical audit absent')
    monkeypatch.setattr(stage,'terminal_audit',fail)
    with pytest.raises(ValueError,match='lexical audit'):stage.stage()
    assert not (tmp_path/'forbidden').exists()
