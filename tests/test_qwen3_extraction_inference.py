"""Tiny CPU parity and wrapper-only operational checks; no real extraction."""
import sys
import pytest
import torch
from test_qwen3_gcad_port import tiny,Tok
from scripts import qwen3_gcad_port as port
from scripts import run_qwen3_extraction_inference as w


@pytest.fixture(autouse=True)
def cpu_threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:yield
    finally:torch.set_num_threads(old)


@pytest.mark.parametrize('kind',['qwen2','qwen3'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_global_no_grad_preserves_exact_delta_key_demand_without_graph(kind,dtype):
    model=tiny(kind).to(dtype);tok=Tok()
    prompts=['<|im_start|>system\ns<|im_end|>\nu','<|im_start|>system\nss<|im_end|>\nu'];answers=['abcd','ef']
    with torch.enable_grad():
        ordinary=port.get_prompt_attn_delta_p_and_r(model,tok,prompts,answers)[2]
        key,demand=port.compute_k_pos_and_avg_demand(model,tok,prompts,answers)
        with torch.no_grad():
            inference=port.get_prompt_attn_delta_p_and_r(model,tok,prompts,answers)[2]
            wrapped_key,wrapped_demand=port.compute_k_pos_and_avg_demand(model,tok,prompts,answers)
        assert torch.is_grad_enabled()
    assert all(torch.equal(ordinary[i],inference[i]) for i in range(model.config.num_hidden_layers))
    assert torch.equal(key,wrapped_key) and torch.equal(demand,wrapped_demand)
    assert not wrapped_key.requires_grad and not wrapped_demand.requires_grad
    assert wrapped_key.grad_fn is None and wrapped_demand.grad_fn is None
    assert all(not v.requires_grad for v in inference[:model.config.num_hidden_layers])
    if kind=='qwen3':assert key.requires_grad or demand.requires_grad  # Reproduce the reported retention mechanism.


@pytest.mark.parametrize('trait',['profanity','impolite','sarcasm'])
@pytest.mark.parametrize('execute',[False,True])
def test_exact_frozen_argv_grad_context_and_restoration(monkeypatch,trait,execute):
    argv=['wrapper','--trait',trait]+(['--execute-gpu'] if execute else [])
    monkeypatch.setattr(sys,'argv',argv);calls=[]
    def run(module,*,run_name):
        calls.append((module,run_name,list(sys.argv),torch.is_grad_enabled()))
    monkeypatch.setattr(w.runpy,'run_module',run)
    with torch.enable_grad():
        w.main();assert torch.is_grad_enabled()
    assert sys.argv is argv
    assert calls==[('scripts.qwen3_4b_fixed_transfer','__main__',
        ['scripts.qwen3_4b_fixed_transfer','extraction','--trait',trait]+(['--execute-gpu'] if execute else []),False)]


def test_runpy_exception_restores_grad_and_argv(monkeypatch):
    argv=['wrapper','--trait','profanity'];monkeypatch.setattr(sys,'argv',argv)
    def fail(*args,**kwargs):
        assert not torch.is_grad_enabled();raise RuntimeError('CPU_MOCK_ONLY')
    monkeypatch.setattr(w.runpy,'run_module',fail)
    with torch.enable_grad():
        with pytest.raises(RuntimeError):w.main()
        assert torch.is_grad_enabled()
    assert sys.argv is argv


def test_invalid_trait_does_not_invoke_frozen_module(monkeypatch):
    monkeypatch.setattr(sys,'argv',['wrapper','--trait','unknown'])
    monkeypatch.setattr(w.runpy,'run_module',lambda *a,**kw:pytest.fail('invalid args must not invoke module'))
    with pytest.raises(SystemExit):w.main()
