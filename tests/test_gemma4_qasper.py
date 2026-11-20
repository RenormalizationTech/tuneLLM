"""CPU contracts for new Gemma QASPER jobs; no model weights/GPU/quality calls."""
import json
from pathlib import Path
import random
from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch

from scripts import gemma4_qasper_runtime as rt
from scripts import gemma4_train_qasper as train
from scripts import gemma4_eval_qasper as evaluate
from scripts import gemma4_retention as retention


def test_frozen_encode_and_response_loss_imported_unchanged():
    from scripts import gemma4_preflight
    from scripts import qwen38_retention,qwen38_science_score
    assert rt.encode is gemma4_preflight.encode
    assert rt.response_loss is gemma4_preflight.response_loss
    assert retention.score_predictions is qwen38_retention.score_predictions
    assert evaluate.score_qasper is qwen38_science_score.score_qasper


def test_jsonl_unicode_line_separator_stays_inside_one_physical_row(tmp_path):
    values = [{"id":"one","user":"before\u2028after"},{"id":"two","output":"x\u2028y"}]
    path = tmp_path/"rows.jsonl"
    path.write_text("\n".join(json.dumps(v,ensure_ascii=False) for v in values)+"\n")
    assert "\u2028" in path.read_text()
    assert rt.rows(path) == values


def test_retention_analysis_binding_optional_base_required_adapter(tmp_path):
    assert retention.analysis_binding(None,None) is None
    with pytest.raises(ValueError,match="requires prospective analysis"):
        retention.analysis_binding("training",None)
    path = tmp_path/"R0.json"; path.write_text('{"prospective":true}\n')
    expected = {"path":str(path.resolve()),"sha256":rt.sha(path)}
    assert retention.analysis_binding(None,path) == expected
    assert retention.analysis_binding("training",path) == expected


def test_retention_prepare_preserves_venv_symlink_and_forwards_lexical_path(tmp_path,monkeypatch):
    import scripts.benchmark_screen as benchmark
    target = tmp_path/"system-python"; target.write_bytes(b"test interpreter target")
    executable = tmp_path/"venv/bin/python"; executable.parent.mkdir(parents=True)
    executable.symlink_to(target)
    fixture = {"general_mc":[{"id":"m","prompt":"Q","choices":[" A"," B"," C"," D"]}],
               "gsm8k":[{"id":"g","user":"1+1?"}],"ifeval":[{"key":1,"prompt":"Write"}]}
    fixture_path = rt.ROOT/"data/benchmark-screen-01/test.json"
    original_read,original_sha = rt.read,rt.sha
    def read(path):
        if Path(path) == fixture_path:
            return fixture
        if Path(path) == rt.ROOT/"data/benchmark-screen-01/provenance.json":
            return {"fixture_sha256":{"test":"fixture-hash"}}
        return original_read(path)
    monkeypatch.setattr(rt,"read",read)
    monkeypatch.setattr(rt,"sha",lambda path:"fixture-hash" if Path(path) == fixture_path else original_sha(path))
    monkeypatch.setattr(rt,"model_identity",lambda base:{})
    monkeypatch.setattr(rt,"geometry",lambda base:{"targets":[]})
    monkeypatch.setattr(rt,"render",lambda *args:("native",[2]))
    monkeypatch.setattr(rt,"register",lambda output,cfg,*args:cfg)
    monkeypatch.setattr(retention,"validate_fixture",lambda *args:list(fixture))
    monkeypatch.setattr(retention,"verify_ifeval_assets",lambda:{"files":{}})
    monkeypatch.setattr(retention,"full_choices",lambda *args:([2],[[2,i] for i in range(3,7)]))
    monkeypatch.setattr(benchmark,"choice_token_ids",lambda *args:[3,4,5,6])
    isolated = retention.isolated_ifeval
    monkeypatch.setattr(retention,"isolated_ifeval",lambda *args:{"packages":{}})
    args = SimpleNamespace(base="unused",training=None,analysis_registration=None,scorer_python=str(executable),
                           gpu="1",max_seconds=None,output=str(tmp_path/"not-created"))
    cfg = retention.prepare(args,tokenizer=object())
    assert cfg["scorer_python"] == str(executable.absolute()) != str(executable.resolve())
    assert cfg["scorer_python_sha256"] == original_sha(target)
    assert cfg["mc_bos_token_id"] == 2
    commands = []
    def run(command,**kwargs):
        commands.append(command)
        return SimpleNamespace(stdout='{"scores":{},"packages":{}}')
    monkeypatch.setattr(retention.subprocess,"run",run)
    isolated(cfg,[],[])
    assert commands[0][0] == str(executable.absolute())


@pytest.mark.parametrize("missing",[True,False])
def test_retention_analysis_guard_runs_before_nvml_or_model_load(tmp_path,monkeypatch,missing):
    rt.atomic_json(tmp_path/"prepared.json",[],new=True)
    path = tmp_path/"R0.json"; path.write_text("{}")
    analysis = None if missing else {"path":str(path.resolve()),"sha256":"0"*64}
    cfg = {"max_seconds":None,"adapter":{"path":"unused"},"gpu":"1",
           "final_analysis_registration":analysis}
    def forbidden(*args):
        pytest.fail("analysis guard must reject before GPU inspection/loading")
    monkeypatch.setattr(rt,"require_idle",forbidden)
    monkeypatch.setattr(rt,"load_runtime",forbidden)
    with pytest.raises(ValueError,match="analysis registration"):
        evaluate.execute_rows(tmp_path,cfg,"binding",False,retention=True)


def test_exact_epoch_groups_and_partial_weighting():
    values = train.groups()
    assert len(values) == 61 and [len(g) for g in values] == [8]*60+[2]
    assert sorted(sum(values,[])) == list(range(482))
    order = list(range(482)); random.Random(42).shuffle(order)
    assert sum(values,[]) == order
    x = torch.tensor([2.,6.],requires_grad=True)
    sum(v/len(values[-1]) for v in x).backward()
    assert x.grad.tolist() == [.5,.5]


def test_frozen_selection_preserves_order_not_resampled():
    rows = [{"id":"b","training_eligible":True},{"id":"a","training_eligible":True}]
    assert [r["id"] for r in train.selected_rows(rows,["a","b"])] == ["a","b"]
    with pytest.raises(ValueError):
        train.selected_rows(rows,["a","a"])
    rows[0]["training_eligible"] = False
    with pytest.raises(ValueError):
        train.selected_rows(rows,["b"])


def test_source_closure_contains_indirect_helpers_and_no_mutation():
    paths = rt.source_closure(rt.CORE)
    assert "scripts/gemma4_preflight.py" in paths
    assert "scripts/qwen38_science_score.py" in paths
    assert "evaluation/score.py" in paths
    assert "scripts/save_nf4.py" in paths
    assert len(paths) == len(set(paths))


def test_registration_binds_sources_inputs_versions_and_refuses_overwrite(tmp_path):
    run = tmp_path/"job"
    result = rt.register(run,{"gpu":"0"},{"input.json":[1,2]})
    assert rt.validate(run,result["registration_sha256"])["inputs"]["input.json"] == rt.sha(run/"input.json")
    with pytest.raises(FileExistsError):
        rt.register(run,{}, {})
    with pytest.raises(ValueError,match="SHA mismatch"):
        rt.validate(run,"0"*64)
    (run/"input.json").write_text("[]")
    with pytest.raises(ValueError,match="input changed"):
        rt.validate(run,result["registration_sha256"])


def test_registered_source_tamper_refused(tmp_path):
    run = tmp_path/"job"; result = rt.register(run,{}, {})
    (run/"source/scripts/gemma4_qasper_runtime.py").write_text("# changed\n")
    with pytest.raises(ValueError,match="artifact changed"):
        rt.validate(run,result["registration_sha256"])


@pytest.mark.parametrize("cap",[-1,0,float("inf"),True])
def test_invalid_operational_caps_refused_before_prepare(tmp_path,cap):
    with pytest.raises(ValueError,match="finite positive"):
        rt.register(tmp_path/"job",{"kind":"training","gpu":"0","max_seconds":cap},{})
    assert not (tmp_path/"job").exists()


def test_runtime_package_drift_refused(tmp_path,monkeypatch):
    result = rt.register(tmp_path/"job",{}, {})
    monkeypatch.setattr(rt,"versions",lambda:{"torch":"wrong"})
    with pytest.raises(ValueError,match="package versions"):
        rt.validate(tmp_path/"job",result["registration_sha256"])


@pytest.mark.parametrize("name",["../escape","/absolute"])
def test_manifest_path_escape_refused(tmp_path,name):
    with pytest.raises(ValueError,match="unsafe"):
        rt.verify_files(tmp_path,{name:{"bytes":0,"sha256":"x"}})


class ToyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.tensor([.3,-.2]))

    def save_pretrained(self,path,**kwargs):
        torch.save(self.state_dict(),Path(path)/"adapter_model.safetensors")
        rt.atomic_json(Path(path)/"adapter_config.json",{"toy":True})


def toy_reload(model,path):
    model.load_state_dict(torch.load(Path(path)/"adapter_model.safetensors",weights_only=True))


def update(model,opt,scheduler):
    opt.zero_grad(set_to_none=True)
    (model.weight.square().sum()+torch.rand(())*model.weight.sum()).backward()
    opt.step(); scheduler.step(); opt.zero_grad(set_to_none=True)


def test_optimizer_boundary_resume_restores_all_rngs_and_next_update(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,"get_rng_state_all",lambda:[])
    monkeypatch.setattr(torch.cuda,"set_rng_state_all",lambda values:None)
    monkeypatch.setattr(rt,"reload_adapter",toy_reload)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    model = ToyAdapter(); opt = torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lambda n:1/(n+1))
    update(model,opt,scheduler)
    train.save_checkpoint(tmp_path,1,model,opt,scheduler,"bound",2)
    expected_random = (random.random(),np.random.rand(),torch.rand(()))
    update(model,opt,scheduler); expected = model.weight.detach().clone()
    fresh = ToyAdapter(); optimizer = torch.optim.AdamW(fresh.parameters(),lr=.8)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:1/(n+1))
    receipt = train.restore_checkpoint(tmp_path/"checkpoints/step-00001",fresh,optimizer,schedule,"bound")
    actual_random = (random.random(),np.random.rand(),torch.rand(()))
    assert receipt["step"] == 1
    assert expected_random[:2] == actual_random[:2]
    assert torch.equal(expected_random[2],actual_random[2])
    update(fresh,optimizer,schedule)
    assert torch.equal(expected,fresh.weight)
    assert opt.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_checkpoint_integrity_fails_before_state_load(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,"get_rng_state_all",lambda:[])
    model = ToyAdapter(); opt = torch.optim.AdamW(model.parameters()); scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lambda n:1)
    train.save_checkpoint(tmp_path,0,model,opt,scheduler,"binding",0)
    directory = tmp_path/"checkpoints/step-00000"
    with pytest.raises(ValueError,match="registration mismatch"):
        train.verify_checkpoint(directory,"other")
    (directory/"state.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="artifact changed"):
        train.restore_checkpoint(directory,model,opt,scheduler,"binding")


def test_immutable_atomic_output_and_checkpoint(tmp_path,monkeypatch):
    path = tmp_path/"row.json"; rt.atomic_json(path,{"ok":1},new=True)
    with pytest.raises(FileExistsError):
        rt.atomic_json(path,{"ok":2},new=True)
    assert rt.read(path) == {"ok":1}


def test_conservative_killed_attempt_accounting_and_error_resume(tmp_path):
    (tmp_path/"attempts").mkdir()
    p = tmp_path/"attempts/a.json"
    rt.atomic_json(p,{"start_unix":10,"end_unix":30},new=True)
    rt.atomic_json(tmp_path/"attempts/b.json",{"start_unix":40},new=True)
    assert train.elapsed_attempts(tmp_path,now=70) == 50
    rt.atomic_json(p,{"start_unix":10,"end_unix":30,"error_type":"ValueError"})
    with pytest.raises(ValueError,match="amended"):
        train.recover_attempts(tmp_path)


class Generator(torch.nn.Module):
    def __init__(self,tokens):
        super().__init__(); self.p = torch.nn.Parameter(torch.zeros(1)); self.tokens = tokens

    def generate(self,input_ids,**kwargs):
        self.kwargs = kwargs
        return torch.cat([input_ids,torch.tensor([self.tokens])],-1)


class Decoder:
    pad_token_id = 0
    def decode(self,tokens,skip_special_tokens):
        assert not skip_special_tokens
        return ":".join(map(str,tokens))


@pytest.mark.parametrize("tokens,eos,cap_hit",[([7,106],True,False),([7,1],True,False),([7,8],False,True)])
def test_generation_same_kwargs_eos_on_cap_and_no_channel_stripping(tokens,eos,cap_hit):
    model = Generator(tokens)
    result = rt.generate(model,Decoder(),[2,3],2,time.monotonic()+5)
    assert result["ended_with_eos"] is eos and result["hit_token_limit"] is cap_hit
    assert result["output"] == ("7" if eos else "7:8")
    assert model.kwargs["eos_token_id"] == [1,106]
    assert model.kwargs["use_cache"] and model.kwargs["logits_to_keep"] == 1
    assert model.kwargs["repetition_penalty"] == 1.0
    assert not model.kwargs["do_sample"]


def test_generation_refuses_truncation_and_exhausted_deadline():
    with pytest.raises(ValueError):
        rt.generate(None,None,[1]*8190,3,time.monotonic()+5)
    with pytest.raises(ValueError):
        rt.generate(None,None,[1],3,time.monotonic()-1)


class CharacterTokenizer:
    bos_token_id = 2
    def __call__(self,text,**kwargs):
        return {"input_ids":[ord(c) for c in text]}


def test_full_mc_answer_prefix_and_multiple_tokens():
    prefix,choices = retention.full_choices(CharacterTokenizer(),"Q?",[" A"," B"," C"," D"])
    assert prefix == [2,81,63] and choices[0] == [2,81,63,32,65]
    assert all(c[:3] == prefix and len(c) == 5 for c in choices)
    with pytest.raises(ValueError,match="distinct"):
        retention.full_choices(CharacterTokenizer(),"Q?",[" A"]*4)
    wrong = CharacterTokenizer(); wrong.bos_token_id = 3
    with pytest.raises(ValueError,match="BOS id2"):
        retention.full_choices(wrong,"Q?",[" A"," B"," C"," D"])


@pytest.mark.parametrize("executor",["training","generation"])
def test_busy_gpu_creates_no_attempt_or_error_and_does_not_block_resume(tmp_path,monkeypatch,executor):
    def busy(gpu):
        raise RuntimeError("registered GPU already has compute processes")
    monkeypatch.setattr(rt,"require_idle",busy)
    cfg = {"gpu":"0","max_seconds":None,"adapter":None}
    rt.atomic_json(tmp_path/"prepared.json",[],new=True)
    call = train.train if executor == "training" else evaluate.execute_rows
    for resume in (False,True):
        with pytest.raises(RuntimeError,match="already has"):
            call(tmp_path,cfg,"binding",resume)
        assert not list((tmp_path/"attempts").glob("*.json"))
        assert not (tmp_path/"result.json").exists()


@pytest.mark.parametrize("executor",["training","generation"])
def test_model_oom_after_idle_remains_recorded_and_fail_closed(tmp_path,monkeypatch,executor):
    monkeypatch.setattr(rt,"require_idle",lambda gpu:"CPU-test-no-NVML")
    def failed_load(*args):
        raise torch.OutOfMemoryError("synthetic CPU-only failure")
    monkeypatch.setattr(rt,"load_runtime",failed_load)
    cfg = {"gpu":"0","max_seconds":None,"adapter":None,"identity":{},"targets":[]}
    rt.atomic_json(tmp_path/"prepared.json",[],new=True)
    call = train.train if executor == "training" else evaluate.execute_rows
    with pytest.raises(torch.OutOfMemoryError,match="synthetic CPU-only"):
        call(tmp_path,cfg,"binding",False)
    attempts = list((tmp_path/"attempts").glob("*.json"))
    assert len(attempts) == 1 and rt.read(attempts[0])["error_type"] == "OutOfMemoryError"
    with pytest.raises(ValueError,match="amended registration"):
        call(tmp_path,cfg,"binding",True)


class LikelihoodModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.p = torch.nn.Parameter(torch.zeros(1))
    def forward(self,input_ids,logits_to_keep,**kwargs):
        logits = torch.arange(10,dtype=torch.float).expand(1,len(logits_to_keep),10).clone()
        self.positions = logits_to_keep
        return SimpleNamespace(logits=logits)


def test_mc_sums_full_conditional_answer_not_last_token_only():
    model = LikelihoodModel()
    row = {"input_ids":[2,3],"choice_input_ids":[[2,3,4,5],[2,3,6],[2,3,7,8],[2,3,9]]}
    result = retention.choice_likelihoods(model,row,time.monotonic()+5)
    lp = torch.log_softmax(torch.arange(10,dtype=torch.float),-1)
    expected = [lp[4]+lp[5],lp[6],lp[7]+lp[8],lp[9]]
    assert result["choice_loglikelihood"] == pytest.approx([x.item() for x in expected])
    assert result["predicted_index"] == 3


def test_row_receipts_skip_complete_refuse_mutated_and_disclose_interrupted(tmp_path):
    prepared = [{"ordinal":0,"id":"q","input_ids":[2,3],"template":"base"}]
    directory = tmp_path/"rows/00000"; directory.mkdir(parents=True)
    with pytest.raises(RuntimeError,match="explicit resume"):
        evaluate.receipt_rows(tmp_path,prepared,"binding")
    assert evaluate.receipt_rows(tmp_path,prepared,"binding",allow_interrupted=True) == []
    rt.atomic_json(directory/"result.json",{**prepared[0],"output":"answer","registration_sha256":"binding"},new=True)
    rt.atomic_json(directory/"receipt.json",{"sha256":rt.sha(directory/"result.json")},new=True)
    assert len(evaluate.receipt_rows(tmp_path,prepared,"binding")) == 1
    (directory/"result.json").write_text("{}")
    with pytest.raises(ValueError,match="hash differs"):
        evaluate.receipt_rows(tmp_path,prepared,"binding")


def test_nvml_occupancy_and_visibility_guard(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    def fake(command,**kwargs):
        return ("GPU-X, NVIDIA RTX A6000, 49140, 487, 80\n"
                if any(arg.startswith("--query-gpu=") for arg in command)
                else "GPU-X, 1234, /work/SpinDynamic/python, 464\n")
    monkeypatch.setattr(rt.subprocess,"check_output",fake)
    with pytest.raises(RuntimeError,match="headroom/activity policy rejected"):
        rt.require_idle("0")
    assert rt.OCCUPANCY_SNAPSHOT["accepted"] is False
    with pytest.raises(ValueError,match="visibility"):
        rt.require_idle("1")
    assert rt.OCCUPANCY_SNAPSHOT is None


@pytest.mark.parametrize("used,util,processes",[(0,0,""),(487,0,"GPU-X, 1234, /work/SpinDynamic/python, 464\n"),
    (1024,5,"GPU-X, 1234, /work/SpinDynamic/python, 1000\n")])
def test_empty_or_small_idle_context_accepted_with_complete_snapshot(monkeypatch,used,util,processes):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    def fake(command,**kwargs):
        if any(arg.startswith("--query-gpu=") for arg in command):
            return f"GPU-X, NVIDIA RTX A6000, 49140, {used}, {util}\n"
        return processes+"GPU-OTHER, 9999, /unrelated/job, 20000\n"
    monkeypatch.setattr(rt.subprocess,"check_output",fake)
    assert rt.require_idle("0") == "GPU-X"
    snapshot = rt.OCCUPANCY_SNAPSHOT
    assert snapshot["accepted"] and snapshot["gpu_uuid"] == "GPU-X"
    assert snapshot["used_memory_mib"] == used and snapshot["utilization_gpu_percent"] == util
    assert snapshot["policy"] == {"maximum_used_memory_mib":1024,"maximum_utilization_percent":5}
    assert "coexistence" in snapshot["timing_scope"]
    if processes:
        assert snapshot["processes"] == [{"gpu_uuid":"GPU-X","pid":1234,
            "process_name":"/work/SpinDynamic/python","used_memory_mib":464 if used == 487 else 1000}]
    else:
        assert snapshot["processes"] == []


@pytest.mark.parametrize("used,util",[(1025,0),(487,6),(20000,90)])
def test_large_or_active_context_rejected_with_snapshot(monkeypatch,used,util):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    monkeypatch.setattr(rt.subprocess,"check_output",lambda command,**kwargs:
        f"GPU-X, NVIDIA RTX A6000, 49140, {used}, {util}\n"
        if any(arg.startswith("--query-gpu=") for arg in command) else
        "GPU-X, 1234, /work/busy/python, 464\n")
    with pytest.raises(RuntimeError,match="headroom/activity policy rejected"):
        rt.require_idle("0")
    assert rt.OCCUPANCY_SNAPSHOT["accepted"] is False
    assert rt.OCCUPANCY_SNAPSHOT["used_memory_mib"] == used
    assert rt.OCCUPANCY_SNAPSHOT["utilization_gpu_percent"] == util


def test_load_requires_accepted_occupancy_before_importing_or_loading_weights(monkeypatch):
    monkeypatch.setattr(rt,"OCCUPANCY_SNAPSHOT",None)
    with pytest.raises(RuntimeError,match="accepted occupancy snapshot"):
        rt.load_runtime({},[])


def test_outer_amp_recipe_is_bound_across_generation_and_mc(monkeypatch,tmp_path):
    assert rt.OUTER_BF16_AUTOCAST is True
    result = rt.register(tmp_path/"job",{}, {})
    cfg = rt.validate(tmp_path/"job",result["registration_sha256"])
    assert cfg["outer_bf16_autocast"] is True
    entries = []
    from contextlib import contextmanager
    @contextmanager
    def recording_amp(model):
        entries.append(type(model).__name__)
        yield
    monkeypatch.setattr(rt,"amp",recording_amp)
    rt.generate(Generator([7,106]),Decoder(),[2,3],2,time.monotonic()+5)
    retention.choice_likelihoods(LikelihoodModel(),{"input_ids":[2],
        "choice_input_ids":[[2,3],[2,4],[2,5],[2,6]]},time.monotonic()+5)
    assert entries == ["Generator"]+["LikelihoodModel"]*4


def tiny_gemma():
    from transformers import Gemma4ForCausalLM,Gemma4TextConfig
    return Gemma4ForCausalLM(Gemma4TextConfig(vocab_size=128,vocab_size_per_layer_input=128,
        hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=2,
        num_key_value_heads=1,num_global_key_value_heads=1,head_dim=16,global_head_dim=16,
        hidden_size_per_layer_input=0,num_kv_shared_layers=0,attention_k_eq_v=True,
        layer_types=["sliding_attention","full_attention"],sliding_window=8,
        max_position_embeddings=64,use_bidirectional_attention="vision"))


def test_native_tiny_gemma_short_compact_full_parity_under_bf16_amp(monkeypatch):
    model = tiny_gemma(); model.train()
    monkeypatch.setattr(rt,"amp",lambda m:torch.autocast("cpu",dtype=torch.bfloat16))
    row = {"id":"mechanics","input_ids":[2,3,4,5,6,7,8,9,106],"prompt_tokens":5}
    result = train.compact_parity(model,row,1e-4)
    assert result["passed"] and model.training
    assert result["g1"]["absolute_ce_difference"] <= 1e-4
    assert result["g1"]["max_absolute_logit_difference"] <= 1e-4
    assert result["g1"]["passed"] and result["g2"]["passed"]
    assert result["g2"]["gradients_allclose"] and result["g2"]["unsupervised_gradients_exact_zero"]
    assert result["g2"]["last_unused_row_gradients_exact_zero"] and result["g2"]["shared_leaf_count"] == 1
    assert result["gradient_count"] == len([p for p in model.parameters() if p.requires_grad])
    assert result["gradient_atol"] == result["gradient_rtol"] == 1e-4
    assert result["diagnostics"]["gating"] is False
    assert result["native_gradient_equivalence_claim"] is False
    assert set(result["diagnostics"]) >= {"D1","D2"}
    assert result["rng_restored"] and result["buffers_restored"] and result["model_mode_restored"]
    assert result["gradients_cleared"] and all(p.grad is None for p in model.parameters())


def test_no_update_dropout_mechanics_restores_model_rng_and_clears_gradients():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.05),torch.nn.Linear(4,1))
    model.register_buffer("calls",torch.tensor(0.))
    original = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone(); py = random.getstate(); numpy = np.random.get_state()
    def loss(row):
        model.calls.add_(1)
        return model(torch.ones(2,3)).square().mean(),2
    def restore():
        model.load_state_dict(original); torch.set_rng_state(rng)
        random.setstate(py); np.random.set_state(numpy)
    result = train.no_update_mechanics(model,list(model.parameters()),loss,
                                       {"id":"longest","input_ids":[2]*20},restore)
    assert result["optimizer_updates"] == 0
    assert result["gradient_count"] == len(list(model.parameters()))
    assert result["gradients_finite"] and result["gradients_cleared"] and result["state_restored"]
    assert all(torch.equal(original[k],v) for k,v in model.state_dict().items())
    assert torch.equal(rng,torch.get_rng_state())
    assert random.getstate() == py
    assert all(p.grad is None for p in model.parameters())


def test_startup_parity_failure_refuses_training(monkeypatch):
    model = tiny_gemma()
    original = model.forward
    grad_modes = []
    def shifted_compact(*args,**kwargs):
        grad_modes.append(torch.is_grad_enabled())
        out = original(*args,**kwargs)
        if kwargs.get("logits_to_keep") is not None:
            out.logits = out.logits+.01
        return out
    monkeypatch.setattr(model,"forward",shifted_compact)
    with pytest.raises(ValueError,match="parity failed G1"):
        train.compact_parity(model,{"input_ids":[2,3,4,5,106],"prompt_tokens":2})
    assert all(grad_modes)


def test_native_gradient_differences_are_diagnostics_only_and_cleanup_is_exact(monkeypatch):
    model = tiny_gemma(); model.train(); model.model.layers[0].eval()
    model.register_buffer("probe_counter",torch.zeros(()))
    modes = [m.training for m in model.modules()]
    rng = torch.get_rng_state().clone(); py = random.getstate(); numpy = np.random.get_state()
    original = rt.response_loss
    calls = []
    def wrong_gradient(*args):
        torch.rand(()); random.random(); np.random.rand()
        model.probe_counter.add_(1)
        value = original(*args)
        p = next(model.parameters())
        calls.append(1)
        return value + .01*len(calls)*(p.sum()-p.sum().detach())
    monkeypatch.setattr(rt,"response_loss",wrong_gradient)
    result = train.compact_parity(model,{"input_ids":[2,3,4,5,106],"prompt_tokens":2})
    assert result["passed"] and result["g1"]["passed"] and result["g2"]["passed"]
    assert result["diagnostics"]["D1"]["max_tolerance_ratio"] > 1
    assert result["diagnostics"]["D2"]["max_absolute_error"] > 1e-4
    assert result["diagnostics"]["gating"] is False and result["native_gradient_equivalence_claim"] is False
    assert model.probe_counter.item() == 0
    assert [m.training for m in model.modules()] == modes
    assert torch.equal(torch.get_rng_state(),rng) and random.getstate() == py
    actual = np.random.get_state()
    assert actual[0] == numpy[0] and np.array_equal(actual[1],numpy[1]) and actual[2:] == numpy[2:]
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("defect",["detach","supervised","unsupervised","last_unused"])
def test_shared_logit_G2_detects_detached_or_corrupted_compact_gradient(monkeypatch,defect):
    original = train.shared_response_ce
    def corrupt(logits,labels,positions):
        value = original(logits,labels,positions)
        if defect == "detach":
            return value.detach()
        index = int(positions[0]) if defect == "supervised" else (0 if defect == "unsupervised" else logits.shape[1]-1)
        coefficient = .01 if defect == "supervised" else 1e-6
        extra = logits[:,index].sum()
        return value+coefficient*(extra-extra.detach())
    monkeypatch.setattr(train,"shared_response_ce",corrupt)
    model = tiny_gemma(); model.train()
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError,match="parity failed G2") as error:
        train.compact_parity(model,{"input_ids":[2,3,4,5,106],"prompt_tokens":2})
    report = json.loads(str(error.value).split(": ",1)[1])
    assert report["g1"]["passed"] and not report["g2"]["passed"]
    assert report["g2"]["absolute_ce_difference"] <= 1e-4
    if defect in ("unsupervised","last_unused"):
        assert report["g2"]["gradients_allclose"]
        assert not report["g2"]["unsupervised_gradients_exact_zero"]
    assert report["gradients_cleared"] and report["model_mode_restored"] and report["rng_restored"]
    assert torch.equal(rng,torch.get_rng_state()) and model.training
    assert all(p.grad is None for p in model.parameters())


def test_shared_logit_G2_masks_holes_and_last_unused_row():
    logits = torch.randn(1,7,128)
    labels = torch.tensor([[-100,-100,3,-100,5,6,106]])
    report = train.shared_logit_parity(logits,labels)
    assert report["passed"] and report["gradients_allclose"]
    assert report["unsupervised_rows"] == 3
    assert report["unsupervised_gradients_exact_zero"] and report["last_unused_row_gradients_exact_zero"]


def test_shared_response_ce_exactly_matches_frozen_loss_and_logit_gradients():
    leaf = torch.randn(1,8,128,generator=torch.Generator().manual_seed(42),requires_grad=True)
    ids = torch.tensor([[2,3,4,5,6,7,8,106]])
    labels = ids.clone(); labels[:,[0,1,3,6]] = -100
    positions = torch.nonzero(labels[0,1:] != -100,as_tuple=False).flatten()
    class FixedLogits:
        def __call__(self,input_ids,attention_mask,logits_to_keep,use_cache):
            assert torch.equal(input_ids,ids) and torch.equal(attention_mask,torch.ones_like(ids))
            assert not use_cache and torch.equal(logits_to_keep,positions)
            return SimpleNamespace(logits=leaf[:,logits_to_keep])
    frozen = rt.response_loss(FixedLogits(),ids,labels,torch.ones_like(ids))
    frozen_gradient = torch.autograd.grad(frozen,leaf)[0]
    shared = train.shared_response_ce(leaf,labels,positions)
    shared_gradient = torch.autograd.grad(shared,leaf)[0]
    assert torch.equal(frozen,shared)
    assert torch.equal(frozen_gradient,shared_gradient)
    assert positions.tolist() == [1,3,4,6]
    assert torch.count_nonzero(shared_gradient[:,[0,2,5,7]]) == 0
    assert torch.count_nonzero(shared_gradient[:,-1]) == 0


def test_startup_validation_metadata_exposes_registered_gates_and_diagnostics():
    value = train.STARTUP_VALIDATION
    assert value["native_gradient_equivalence_claim"] is False and value["recipe_change"] is False
    assert all(value[name]["gate"] for name in ("G1","G2","G3"))
    assert all(not value[name]["gate"] for name in ("D1","D2"))
    assert value["G1"]["absolute_tolerance"] == value["G2"]["atol"] == value["G2"]["rtol"] == 1e-4


def test_startup_tolerance_cannot_be_widened():
    with pytest.raises(ValueError,match="must remain1e-4"):
        train.compact_parity(tiny_gemma(),{"input_ids":[2,3,4,5,106],"prompt_tokens":2},1e-3)


def test_frozen_snapshot_clis_import_without_project_assets(tmp_path):
    import os
    import subprocess
    import sys
    rt.register(tmp_path/"job",{}, {})
    source = tmp_path/"job/source"
    for module in ("scripts.gemma4_train_qasper","scripts.gemma4_eval_qasper","scripts.gemma4_retention"):
        result = subprocess.run([sys.executable,"-B","-m",module,"execute","--help"],
            cwd=source,env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES=""),
            capture_output=True,text=True)
        assert result.returncode == 0, result.stderr
        assert "--registration-sha256" in result.stdout


def test_complete_training_schedule_on_cpu_fake_runtime(tmp_path,monkeypatch):
    class TinyCausal(ToyAdapter):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.embed = torch.nn.Embedding(128,4)
            self.head = torch.nn.Linear(4,128)
        def forward(self,input_ids,logits_to_keep=None,**kwargs):
            hidden = self.embed(input_ids)
            if logits_to_keep is not None:
                hidden = hidden[:,logits_to_keep]
            return SimpleNamespace(logits=self.head(hidden))
    monkeypatch.setattr(rt,"load_runtime",lambda *args:(TinyCausal(),{"fake_cpu":True}))
    monkeypatch.setattr(rt,"require_idle",lambda gpu:"CPU-test-no-NVML")
    monkeypatch.setattr(rt,"reload_adapter",toy_reload)
    monkeypatch.setattr(torch.cuda,"get_rng_state_all",lambda:[])
    monkeypatch.setattr(torch.cuda,"set_rng_state_all",lambda state:None)
    item = {"input_ids":[2,3,4,5,106],"labels":[-100,-100,4,5,106],
            "attention_mask":[1]*5,"prompt_tokens":2,"response_tokens":3}
    rt.atomic_json(tmp_path/"train.json",[{"id":f"t{i}",**item} for i in range(482)],new=True)
    rt.atomic_json(tmp_path/"validation.json",[{"id":f"d{i}",**item} for i in range(101)],new=True)
    cfg = {"max_seconds":None,"gpu":"0","identity":{},"targets":[],
           "checkpoint_steps":[21,41,61],"short_ce_absolute_tolerance":1e-4}
    result = train.train(tmp_path,cfg,"registration",False)
    assert result["completed"] and result["steps"] == 61
    assert [c["step"] for c in result["checkpoints"]] == [21,41,61]
    assert len(list((tmp_path/"checkpoints").glob("step-*"))) == 62
    assert rt.read(tmp_path/"longest_row_mechanics.json")["optimizer_updates"] == 0
    assert rt.read(tmp_path/"heartbeat.json")["step"] == 61
    assert result["selected"] == min(result["checkpoints"],key=lambda r:(r["nll"],r["step"]))
    for checkpoint in (tmp_path/"checkpoints").glob("step-*"):
        train.verify_checkpoint(checkpoint,"registration")
    with pytest.raises(FileExistsError,match="completed fit"):
        train.train(tmp_path,cfg,"registration",True)


def test_complete_development_cpu_fake_runtime_scores_and_tie_selection(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from transformers import AutoTokenizer
    class OutputTokenizer(Decoder):
        def decode(self,tokens,skip_special_tokens):
            assert tokens == [7] and not skip_special_tokens
            return '{"answer":"quantum response","unanswerable":false,"evidence":["p000000"]}'
    model = Generator([7,106]); model.disable_adapter = nullcontext
    monkeypatch.setattr(rt,"load_runtime",lambda *args:(model,{"fake_cpu":True}))
    monkeypatch.setattr(rt,"require_idle",lambda gpu:"CPU-test-no-NVML")
    monkeypatch.setattr(AutoTokenizer,"from_pretrained",lambda *args,**kwargs:OutputTokenizer())
    for name in ("synchronize","reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda,name,lambda:None)
    for name in ("max_memory_allocated","max_memory_reserved"):
        monkeypatch.setattr(torch.cuda,name,lambda:0)
    data = [{"id":f"q{i}","paper_id":f"p{i}","primary":True,"evidence_strict_primary":True,
        "nontext_or_unmatched_evidence":False,"references":[{"answer":"quantum response",
        "unanswerable":False,"evidence":["p000000"]}],"context_ids":["p000000"],
        "paragraphs":[{"id":"p000000"}]} for i in range(35)]
    prepared = [{"ordinal":j*35+i,"id":r["id"],"paper_id":r["paper_id"],
        "template":template,"rendered_prompt":"fixed","input_ids":[2,3]}
        for j,template in enumerate(evaluate.PROMPTS) for i,r in enumerate(data)]
    cfg = {"kind":"evaluation","purpose":"development","gpu":"1","max_seconds":None,
           "identity":{"base":"fake"},"targets":[],"adapter":None,"templates":list(evaluate.PROMPTS)}
    run = tmp_path/"development"
    registered = rt.register(run,cfg,{"data.json":data,"prepared.json":prepared})
    binding = registered["registration_sha256"]; cfg = rt.validate(run,binding)
    result = evaluate.execute_rows(run,cfg,binding,False)
    assert result["completed"] and result["generations"] == 105
    assert len(rt.rows(run/"predictions.jsonl")) == 105
    chosen = evaluate.select(run,binding)
    assert chosen["chosen"] == "base" and chosen["final_test_used"] is False
    assert chosen["strict_primary_answer_f1"] == {"base":1.,"verify":1.,"schema":1.}
    assert all(r["registration_sha256"] == binding for r in rt.rows(run/"predictions.jsonl"))


@pytest.mark.artifact("outputs/gemma4-preflight-02/snapshot_path.txt")
def test_real_pinned_meta_inventory_and_native_mc_boundaries():
    from transformers import AutoTokenizer
    from scripts.benchmark_screen import choice_token_ids
    base = Path((rt.PREFLIGHT/"snapshot_path.txt").read_text().strip())
    geometry = rt.geometry(base)
    assert len(geometry["targets"]) == 410
    tokenizer = AutoTokenizer.from_pretrained(base,local_files_only=True)
    fixture = rt.read(rt.ROOT/"data/benchmark-screen-01/test.json")["general_mc"]
    assert choice_token_ids(tokenizer,fixture) == [562,603,565,622]
    for row in fixture:
        prefix,choices = retention.full_choices(tokenizer,row["prompt"],row["choices"])
        raw = tokenizer(row["prompt"],add_special_tokens=False)["input_ids"]
        assert prefix == [2]+raw
        assert all(c == [2]+tokenizer(row["prompt"]+choice,add_special_tokens=False)["input_ids"]
                   for c,choice in zip(choices,row["choices"]))
        assert all(len(c) == len(prefix)+1 for c in choices)


@pytest.mark.artifact("outputs/gemma4-preflight-02/snapshot_path.txt","results/protocols/qwen38-qasper-02.json")
def test_real_frozen_ids_and_gemma_supervision_without_test_loading():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained((rt.PREFLIGHT/"snapshot_path.txt").read_text().strip(),local_files_only=True)
    protocol = rt.read(rt.ROOT/"results/protocols/qwen38-qasper-02.json")
    for split,count in (("train",482),("validation",101)):
        path = rt.ROOT/f"outputs/qwen38-science-01/prepared-qasper-tokenizer-01/{split}.jsonl"
        assert rt.sha(path) == protocol["config"]["prepared_input_sha256"][split]
        selected = train.selected_rows(rt.rows(path),protocol["selected_ids"][split])
        assert len(selected) == count
        encoded = [rt.encode(tokenizer,r["user"],r["assistant"],r.get("system")) for r in selected]
        assert max(len(r["input_ids"]) for r in encoded) <= 8192
        assert all(r["labels"][-1] == 106 and r["input_ids"][-1] == 106 for r in encoded)
