"""New Gemma release CPU contracts; no GPU, network, generation study or regrade."""
from contextlib import contextmanager
import copy
import ast
import fnmatch
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.gemma4_qasper_release_runtime import loader
from scripts import stage_gemma4_qasper_release as stage
from scripts import verify_gemma4_qasper_release as verifier


def config():
    return {"base_model_name_or_path":"/private/base","revision":None,"r":16,"lora_alpha":32,
        "lora_dropout":.05,"bias":"none","task_type":"CAUSAL_LM","peft_type":"LORA",
        "target_modules":["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "inference_mode":True,"extra_unchanged":{"value":7}}


def test_config_changes_exactly_two_fields_and_no_original_mutation():
    original=config();before=copy.deepcopy(original);result=stage.normalized_config(original)
    assert original==before
    assert {k for k in result if result[k]!=original[k]}=={"base_model_name_or_path","revision"}
    assert result["base_model_name_or_path"]==loader.BASE and result["revision"]==loader.REVISION
    assert result["extra_unchanged"]==before["extra_unchanged"]


def test_public_question_formatter_preserves_whitespace_and_display_order():
    paragraphs=[{"id":"p000002","section":" Methods ","text":"  two\nlines  "},
                {"id":"p000000","section":"","text":"first"}]
    question="  exact question?\n"
    user=loader.format_question(paragraphs,question)
    assert user.endswith("[p000002]  Methods \n  two\nlines  \n\n[p000000] \nfirst\n\nQuestion: "+question)
    assert paragraphs[0]["text"]=="  two\nlines  "


@pytest.mark.artifact("outputs/gemma4-qasper-study-01/final-base/data.json",
                      "outputs/gemma4-qasper-study-01/final-base/predictions.jsonl",
                      "outputs/gemma4-preflight-02/snapshot_path.txt")
def test_public_formatter_matches_frozen_selected_context_demo_user_and_token_ids():
    from transformers import AutoTokenizer
    study=stage.ROOT/"outputs/gemma4-qasper-study-01"
    row=next(r for r in loader.read(study/"final-base/data.json") if r["id"]==verifier.DEMO)
    paragraphs={p["id"]:p for p in row["paragraphs"]}
    selected=[paragraphs[key] for key in row["context_ids"]]
    question=row["user"].rsplit("\n\nQuestion: ",1)[1]
    user=loader.format_question(selected,question)
    assert user.encode()==row["user"].encode()
    saved=next(json.loads(line) for line in (study/"final-base/predictions.jsonl").read_text().split("\n")
               if line.strip() and json.loads(line)["id"]==verifier.DEMO)
    base=Path((stage.ROOT/"outputs/gemma4-preflight-02/snapshot_path.txt").read_text().strip())
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    runtime=loader.Runtime(GuardModel(),tokenizer,{})
    prompt,ids=runtime.render(user)
    assert prompt==saved["rendered_prompt"] and ids==saved["input_ids"]


@pytest.mark.parametrize("key,value",[("r",8),("lora_dropout",0),("lora_alpha",16),("bias","all"),("task_type","other")])
def test_config_recipe_mismatch_rejected(key,value):
    value_cfg=config();value_cfg[key]=value
    with pytest.raises(ValueError):stage.normalized_config(value_cfg)


def fake_bundle(path):
    (path/"runtime").mkdir(parents=True)
    for name in loader.FILES-{"release_manifest.json"}:
        (path/name).write_text("reviewed\n")
    normalized=stage.normalized_config(config())
    (path/"adapter_config.json").write_text(json.dumps(normalized))
    manifest={"base":{"repo_id":loader.BASE,"revision":loader.REVISION,"files":{}},
        "environment":loader.PINS,"adapter":{"tensor_count":820,"parameters":122429440,"config":normalized},
        "files":stage.file_inventory(path)}
    (path/"release_manifest.json").write_text(json.dumps(manifest))
    return loader.sha(path/"release_manifest.json")


def test_external_manifest_full_allowlist_and_byte_hashes(tmp_path):
    bundle=tmp_path/"bundle";anchor=fake_bundle(bundle)
    assert loader.validate_bundle(bundle,anchor)["environment"]==loader.PINS
    with pytest.raises(ValueError,match="anchor"):loader.validate_bundle(bundle,"0"*64)
    (bundle/"README.md").write_text("changed")
    with pytest.raises(ValueError,match="hash/size"):loader.validate_bundle(bundle,anchor)


@pytest.mark.parametrize("name",["optimizer.pt","model.safetensors","trajectories.jsonl","private.txt"])
def test_unapproved_payload_files_rejected(tmp_path,name):
    bundle=tmp_path/"bundle";anchor=fake_bundle(bundle);(bundle/name).write_text("forbidden")
    with pytest.raises(ValueError,match="allowlist"):loader.validate_bundle(bundle,anchor)


def cache(tmp_path,two_hop):
    repo=tmp_path/"hub/models--org--model";snapshot=repo/"snapshots"/("1"*40)
    snapshot.mkdir(parents=True);(repo/"blobs").mkdir()
    first=repo/"blobs"/("2"*40);link=snapshot/"file.txt"
    link.symlink_to(os.path.relpath(first,link.parent))
    if two_hop:
        store=repo.parent/"blobs";prefix=store/"ab";prefix.mkdir(parents=True)
        (store/".huggingface-shared-blobs").write_bytes(b"1\n")
        final=prefix/("ab"+"3"*62);final.write_bytes(b"verified")
        first.symlink_to(os.path.relpath(final,first.parent))
    else:
        first.write_bytes(b"verified");final=first
    return snapshot,first,final


@pytest.mark.parametrize("two_hop",[False,True])
def test_official_hf_one_and_two_hop_cache(tmp_path,two_hop):
    snapshot,first,final=cache(tmp_path,two_hop)
    assert loader.safe_file(snapshot,"file.txt")==final


@pytest.mark.parametrize("defect",["marker_missing","marker_wrong","prefix_wrong","third_hop","outside","directory_link","bad_first_name"])
def test_cache_escapes_and_malformed_shared_store_rejected(tmp_path,defect):
    snapshot,first,final=cache(tmp_path,True);store=final.parent.parent
    if defect=="marker_missing":(store/".huggingface-shared-blobs").unlink()
    elif defect=="marker_wrong":(store/".huggingface-shared-blobs").write_bytes(b"2\n")
    elif defect=="prefix_wrong":
        wrong=store/"cd";wrong.mkdir();target=wrong/final.name;target.write_bytes(b"verified")
        first.unlink();first.symlink_to(os.path.relpath(target,first.parent))
    elif defect=="third_hop":
        extra=tmp_path/"extra";extra.write_bytes(b"verified");final.unlink();final.symlink_to(extra)
    elif defect=="outside":
        extra=tmp_path/"outside";extra.write_bytes(b"verified");first.unlink();first.symlink_to(extra)
    elif defect=="directory_link":
        actual=store/"actual";final.parent.rename(actual);(store/"ab").symlink_to(actual,target_is_directory=True)
    elif defect=="bad_first_name":
        changed=first.with_name("unregistered");first.rename(changed);(snapshot/"file.txt").unlink();(snapshot/"file.txt").symlink_to(changed)
    with pytest.raises((ValueError,OSError)):loader.safe_file(snapshot,"file.txt")


def test_real_local_file_and_traversal_guards(tmp_path):
    (tmp_path/"file.txt").write_text("good")
    assert loader.safe_file(tmp_path,"file.txt").read_text()=="good"
    for name in ("../outside","/absolute","a\\b"):
        with pytest.raises(ValueError):loader.safe_file(tmp_path,name)
    (tmp_path/"outside.txt").symlink_to(tmp_path/"file.txt")
    with pytest.raises(ValueError):loader.safe_file(tmp_path,"outside.txt")


def test_unregistered_base_filename_error_has_text_only_download_guidance(tmp_path):
    (tmp_path/".eval_results").mkdir();(tmp_path/".eval_results/mmmu_pro.yaml").write_text("metadata")
    with pytest.raises(ValueError,match=r"mmmu_pro.yaml.*eight-file allow_patterns"):
        loader.validate_base(tmp_path,{"base":{"files":{}}})


def tiny_model():
    from transformers import Gemma4ForCausalLM,Gemma4TextConfig
    from peft import LoraConfig,get_peft_model
    model=Gemma4ForCausalLM(Gemma4TextConfig(vocab_size=128,vocab_size_per_layer_input=128,
        hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=2,
        num_key_value_heads=1,num_global_key_value_heads=1,head_dim=16,global_head_dim=16,
        hidden_size_per_layer_input=0,num_kv_shared_layers=0,attention_k_eq_v=True,
        layer_types=["sliding_attention","full_attention"],sliding_window=8,
        max_position_embeddings=64,use_bidirectional_attention="vision"))
    targets=loader.target_inventory(model)
    model=get_peft_model(model,LoraConfig(task_type="CAUSAL_LM",r=16,lora_alpha=32,lora_dropout=.05,
        target_modules=[t["name"] for t in targets],bias="none"))
    model.eval()
    return model,targets


def test_native_tiny_adapter_keys_shapes_reload_and_output_method(tmp_path):
    from safetensors.torch import load_file,save_file
    model,targets=tiny_model()
    with torch.no_grad():
        for name,p in model.named_parameters():
            if "lora_B" in name:p.fill_(.01)
    model.save_pretrained(tmp_path,safe_serialization=True)
    actual=loader.adapter_inventory(tmp_path/"adapter_model.safetensors",targets)
    assert actual["tensor_count"]==2*len(targets)
    fresh,_=tiny_model();proof=loader.reload_adapter(fresh,tmp_path)
    assert proof["exact_tensor_equality"] and proof["tensor_count"]==actual["tensor_count"]
    ids=[2,3,4,5];runtime=loader.Runtime(model,None,{})
    got=runtime.next_logits(ids,True)
    with torch.inference_mode():expected=model(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,4,dtype=torch.long),use_cache=False,logits_to_keep=1).logits.float()
    assert torch.equal(got,expected)
    tensor=load_file(str(tmp_path/"adapter_model.safetensors"));key=next(iter(tensor));tensor[key][0,0]=float("nan")
    save_file(tensor,str(tmp_path/"bad.safetensors"))
    with pytest.raises(ValueError,match="nonfinite"):loader.adapter_inventory(tmp_path/"bad.safetensors",targets)


class Decoder:
    pad_token_id=0
    def decode(self,ids,skip_special_tokens):
        assert not skip_special_tokens;return ":".join(map(str,ids))


class GuardModel(torch.nn.Module):
    def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.zeros(1));self.disabled=False;self.fail=False
    @contextmanager
    def disable_adapter(self):
        previous=self.disabled;self.disabled=True
        try:yield
        finally:self.disabled=previous
    def forward(self,**kwargs):
        assert not self.training and torch.is_inference_mode_enabled()
        if self.fail:raise RuntimeError("synthetic error")
        return SimpleNamespace(logits=torch.tensor([[[1.,2.]]]))
    def generate(self,input_ids,**kwargs):
        assert not self.training and torch.is_inference_mode_enabled()
        assert kwargs["eos_token_id"]==[1,106] and kwargs["repetition_penalty"]==1
        assert kwargs["use_cache"] and not kwargs["do_sample"] and kwargs["logits_to_keep"]==1
        self.arguments=kwargs
        return torch.cat([input_ids,torch.tensor([[50,106]])],-1)


def test_runtime_native_guard_singleton_eos_and_context_cleanup():
    model=GuardModel();runtime=loader.Runtime(model,Decoder(),{})
    assert runtime.generate_ids([2,3],False,2)=={"generated_ids":[50,106],"output":"50","ended_with_eos":True,"hit_token_limit":False}
    assert not model.disabled
    model.fail=True
    with pytest.raises(RuntimeError,match="synthetic"):runtime.next_logits([2,3],False)
    assert not model.disabled


@pytest.mark.parametrize("ids,cap",[([[2]],2),([True],2),([2.0],2),([],2),([2],0),([2],513),([2]*8192,1)])
def test_runtime_refuses_invalid_batch_and_truncation(ids,cap):
    with pytest.raises(ValueError):loader.Runtime(GuardModel(),Decoder(),{}).generate_ids(ids,cap=cap)


def test_exact_environment_pin_check(monkeypatch):
    monkeypatch.setattr(loader.importlib.metadata,"version",lambda name:loader.PINS[name])
    assert loader.validate_environment()==loader.PINS
    monkeypatch.setattr(loader.importlib.metadata,"version",lambda name:"wrong")
    with pytest.raises(ValueError):loader.validate_environment()


@pytest.mark.artifact("outputs/gemma4-qasper-release-pm-01/license-verification-root-01/receipt.json")
def test_real_official_license_is_text_not_html():
    text,witness=stage.verified_license(stage.ROOT/"outputs/gemma4-qasper-release-pm-01/license-verification-root-01")
    assert "Apache License" in text and "END OF TERMS AND CONDITIONS" in text
    assert "<html" not in text and "<span" not in text
    assert witness["license"]=="apache-2.0" and not witness["upstream_LICENSE_file_present"]


def test_verifier_import_and_help_from_fresh_isolated_python_outside_project(tmp_path):
    script=stage.ROOT/"scripts/verify_gemma4_qasper_release.py"
    result=subprocess.run([sys.executable,"-I",str(script),"--help"],cwd=tmp_path,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONPATH="/not/project"),capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert "--phase" in result.stdout and "--reference" in result.stdout


def test_verifier_saved_replay_plan_is_fixed_and_checks_tokens_and_decoded_text():
    rows={"final_base":{"id":"demo"},"final_adapter":{"id":"demo"},
          "final_base_ordered":[{"id":"b0"}],"final_adapter_ordered":[{"id":f"a{i}"} for i in range(4)]}
    plan=verifier.saved_replay_plan(rows)
    assert [(key,adapter,row["id"]) for key,adapter,row in plan]==[
        ("saved_demo.base",False,"demo"),("saved_demo.adapter",True,"demo"),
        ("saved_ordinal0.adapter",True,"a0"),("saved_ordinal1.adapter",True,"a1"),
        ("saved_ordinal2.adapter",True,"a2"),("saved_ordinal3.adapter",True,"a3"),("saved_ordinal0.base",False,"b0")]
    verifier.validate_saved_output([7,106],"actual\ntext",[7,106],"actual\ntext")
    with pytest.raises(ValueError,match="trajectory"):
        verifier.validate_saved_output([8,106],"actual\ntext",[7,106],"actual\ntext")
    with pytest.raises(ValueError,match="decoded"):
        verifier.validate_saved_output([7,106],"actual\\ntext",[7,106],"actual\ntext")


def test_anonymous_download_requires_three_fresh_caches(tmp_path,monkeypatch):
    keys=("HF_HOME","HF_HUB_CACHE","HF_XET_CACHE")
    for key in keys:monkeypatch.delenv(key,raising=False)
    with pytest.raises(ValueError,match="set fresh"):verifier.fresh_download_caches()
    home=tmp_path/"home";hub=home/"hub";xet=home/"xet"
    hub.mkdir(parents=True);xet.mkdir()
    for key,value in zip(keys,(home,hub,xet)):monkeypatch.setenv(key,str(value))
    assert set(verifier.fresh_download_caches())==set(keys)
    (hub/"cached-file").write_text("old")
    with pytest.raises(ValueError,match="fresh and empty"):verifier.fresh_download_caches()


def test_anonymous_download_uses_token_false_and_never_downloads_base(tmp_path,monkeypatch):
    import huggingface_hub
    calls=[]
    for key,name in (("HF_HOME","home"),("HF_HUB_CACHE","hub"),("HF_XET_CACHE","xet")):
        monkeypatch.setenv(key,str(tmp_path/name))
    info=SimpleNamespace(private=False,sha="1"*40,siblings=[SimpleNamespace(rfilename="README.md")])
    class API:
        def model_info(self,repo,**kwargs):
            assert repo==verifier.REPO and kwargs=={"revision":"1"*40,"token":False};calls.append("info");return info
    def factory(*,token):assert token is False;return API()
    def snapshot(repo,**kwargs):
        assert repo==verifier.REPO and kwargs["token"] is False and kwargs["revision"]=="1"*40
        destination=Path(kwargs["local_dir"]);destination.mkdir();(destination/"README.md").write_text("reviewed")
        calls.append("adapter download")
    monkeypatch.setattr(huggingface_hub,"HfApi",factory);monkeypatch.setattr(huggingface_hub,"snapshot_download",snapshot)
    monkeypatch.setattr(verifier,"bootstrap",lambda *args:(SimpleNamespace(FILES={"README.md"}),{}))
    args=SimpleNamespace(public_repo=verifier.REPO,public_revision="1"*40,bundle=tmp_path/"adapter",
                         manifest_sha256="2"*64,receipt=tmp_path/"download.json")
    receipt=verifier.download(args)
    assert calls==["info","adapter download"] and receipt["fresh_download_caches"]
    assert receipt["download_token"] is False and receipt["base_downloaded"] is False


def frozen_fixture(tmp_path):
    source=tmp_path/"training/source/scripts";source.mkdir(parents=True)
    for name in ("gemma4_preflight.py","gemma4_qasper_runtime.py"):
        (source/name).write_text("# synthetic preserved source\n")
    hashes={"scripts/"+p.name:loader.sha(p) for p in source.iterdir()}
    cfg={"source_files":{name:{"sha256":value} for name,value in hashes.items()}}
    registration=tmp_path/"training/registration.json";registration.write_text(json.dumps(cfg))
    seal={"roles":{"training":{"registration.json":loader.sha(registration)}}}
    (tmp_path/"completion-seal.json").write_text(json.dumps(seal))
    manifest={"provenance":{"completion_seal_sha256":loader.sha(tmp_path/"completion-seal.json"),
        "frozen_runtime_sha256":hashes["scripts/gemma4_qasper_runtime.py"]}}
    return manifest,{"sources":hashes}


def test_frozen_code_compilation_requires_sealed_training_registration(tmp_path,monkeypatch):
    manifest,study=frozen_fixture(tmp_path)
    (tmp_path/"training/registration.json").write_text('{"source_files":{}}')
    def forbidden(*args):pytest.fail("unsealed preserved code must never compile")
    monkeypatch.setattr(verifier,"import_verified",forbidden)
    with pytest.raises(ValueError,match="binding differs"):
        verifier.frozen_runtime(tmp_path,manifest,study)


def test_preserved_preflight_must_match_R0_even_when_config_and_seal_agree(tmp_path,monkeypatch):
    manifest,study=frozen_fixture(tmp_path)
    wrong=tmp_path/"training/source/scripts/gemma4_preflight.py";wrong.write_text("# incorrect helper\n")
    path=tmp_path/"training/registration.json";cfg=loader.read(path)
    cfg["source_files"]["scripts/gemma4_preflight.py"]["sha256"]=loader.sha(wrong)
    path.write_text(json.dumps(cfg))
    seal=tmp_path/"completion-seal.json";seal.write_text(json.dumps({"roles":{"training":{"registration.json":loader.sha(path)}}}))
    manifest["provenance"]["completion_seal_sha256"]=loader.sha(seal)
    monkeypatch.setattr(verifier,"import_verified",lambda *args:pytest.fail("wrong R0 helper must never compile"))
    with pytest.raises(ValueError,match="binding differs"):
        verifier.frozen_runtime(tmp_path,manifest,study)


@pytest.mark.artifact("outputs/gemma4-qasper-release-pm-01/license-verification-root-01/metadata.json",
                      "outputs/gemma4-qasper-study-01/training/registration.json")
def test_documented_base_download_patterns_select_only_studied_eight_files():
    import re
    card=(stage.ROOT/"docs/gemma4_qasper_hf_card_draft.md").read_text()
    blocks=re.findall(r"```python\n(.*?)```",card,re.S)
    calls=[node for block in blocks for node in ast.walk(ast.parse(block)) if isinstance(node,ast.Call)
           and isinstance(node.func,ast.Name) and node.func.id=="snapshot_download"
           and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value==loader.BASE]
    assert len(calls)==1
    patterns=ast.literal_eval(next(kw.value for kw in calls[0].keywords if kw.arg=="allow_patterns"))
    metadata=loader.read(stage.ROOT/"outputs/gemma4-qasper-release-pm-01/license-verification-root-01/metadata.json")
    listing={s["rfilename"] for s in metadata["siblings"]}
    selected={name for name in listing if any(fnmatch.fnmatchcase(name,pattern) for pattern in patterns)}
    expected=set(loader.read(stage.ROOT/"outputs/gemma4-qasper-study-01/training/registration.json")["identity"]["files"])
    assert len(selected)==8 and selected==expected
    assert ".eval_results/mmmu_pro.yaml" not in selected


@pytest.mark.artifact("outputs/gemma4-qasper-study-01/training/registration.json")
def test_real_geometry_inventory_and_normalization_without_loading_model_weights():
    cfg=loader.read(stage.ROOT/"outputs/gemma4-qasper-study-01/training/registration.json")
    checkpoint=stage.ROOT/"outputs/gemma4-qasper-study-01/training/checkpoints/step-00061"
    actual=loader.adapter_inventory(checkpoint/"adapter_model.safetensors",cfg["targets"])
    assert actual=={"tensor_count":820,"parameters":122429440,"all_finite":True}
    original=loader.read(checkpoint/"adapter_config.json");normalized=stage.normalized_config(original)
    assert {k for k in original if original[k]!=normalized[k]}=={"base_model_name_or_path","revision"}
