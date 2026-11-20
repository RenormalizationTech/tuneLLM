"""Portable Gemma analysis checks: raw boundaries, fixed selection and completion."""
import copy
import json

import pytest

from scripts import gemma4_qasper_analysis as a


@pytest.mark.parametrize("change",[{"adapter":{}},{"max_new_tokens":1024},{"autocast":False},
    {"templates":["schema","verify","base"]},{"do_sample":True},{"eos_token_id":[1]},
    {"cached":False},{"batch_size":2}])
def test_prompt_development_profile_cannot_drift(change):
    cfg={"kind":"evaluation","purpose":"development","task":"qasper","adapter":None,
        "templates":["base","verify","schema"],"max_new_tokens":512,"max_context":8192,
        "eos_token_id":[1,106],"autocast":True,"batch_size":1,"cached":True,"do_sample":False,"seed":42}
    a.validate_development_profile(cfg)
    cfg.update(change)
    with pytest.raises(ValueError,match="development profile"):a.validate_development_profile(cfg)


def test_jsonl_unicode_line_separator_is_content(tmp_path):
    file=tmp_path/"rows.jsonl"
    file.write_text(json.dumps({"output":"one\u2028two"},ensure_ascii=False)+"\n")
    assert a.rows(file)==[{"output":"one\u2028two"}]


def test_startup_requires_actual_gradient_parity_and_state_restoration():
    mechanics={"registration_sha256":"b","optimizer_updates":0,"outer_bf16_autocast":True,
        "restored_adapter_optimizer_scheduler_rng_buffers":True,"gradient_count":820,
        "gradients_finite":True,"gradients_cleared":True,"state_restored":True,"buffers_restored":True,
        "short_compact_full_ce":{"passed":True,"absolute_tolerance":1e-4,"gradient_atol":1e-4,"gradient_rtol":1e-4,
            "gradient_count":820,"native_gradient_equivalence_claim":False,
            "rng_restored":True,"buffers_restored":True,"model_mode_restored":True,"gradients_cleared":True,
            "optimizer_updates":0,
            "g1":{"passed":True,"max_absolute_logit_difference":0.,"absolute_ce_difference":0.},
            "g2":{"passed":True,"absolute_ce_difference":0.,"gradients_allclose":True,
                "unsupervised_gradients_exact_zero":True},"diagnostics":{"gating":False,"native_gradient_max_error":.0078125}}}
    a.validate_startup(mechanics,"b")
    for field,value in (("passed",False),("gradient_atol",1e-3),("rng_restored",False),
                        ("native_gradient_equivalence_claim",True)):
        bad=copy.deepcopy(mechanics);bad["short_compact_full_ce"][field]=value
        with pytest.raises(ValueError):a.validate_startup(bad,"b")
    for name in ("gradients_allclose","unsupervised_gradients_exact_zero"):
        bad=copy.deepcopy(mechanics);bad["short_compact_full_ce"]["g2"][name]=False
        with pytest.raises(ValueError):a.validate_startup(bad,"b")


def test_later_job_requires_whole_study_seal(tmp_path):
    seal=tmp_path/"study.json";seal.write_text("{}\n")
    run=tmp_path/"job";run.mkdir()
    cfg={"identity":{},"versions":{},"source_files":{},"inputs":{}}
    study={"prospective_runs":{"retention_adapter":str(run)},"job_registrations":{},
        "_registration_sha256":a.sha(seal),"model_identity":{},"versions":{}}
    (run/"registration.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match="whole-study binding"):a.load_job(study,"retention_adapter",False)
    cfg["final_analysis_registration"]={"path":str(seal),"sha256":a.sha(seal)}
    (run/"registration.json").write_text(json.dumps(cfg))
    assert a.load_job(study,"retention_adapter",False)[0]==run
    seal.write_text('{"tampered":true}\n')
    with pytest.raises(ValueError,match="whole-study binding"):a.load_job(study,"retention_adapter",False)


def test_pipeline_resume_skips_complete_and_passes_resume_for_partial(tmp_path,monkeypatch):
    from scripts import gemma4_qasper_pipeline as p
    run=tmp_path/"training";run.mkdir();(run/"attempts").mkdir()
    (run/"attempts/attempt-001.json").write_text("{}")
    (run/"registration.json").write_text('{"gpu":"0"}')
    (run/"result.json").write_text('{"completed":false}')
    monkeypatch.setattr(p,"ROOT",tmp_path);monkeypatch.setattr(p,"STUDY",tmp_path)
    monkeypatch.setattr(p,"verify",lambda study:None)
    monkeypatch.setattr(p.analysis,"load_job",lambda *args:None)
    calls=[]
    def execute(args,log):
        calls.append(args);(run/"result.json").write_text('{"completed":true}')
    monkeypatch.setattr(p,"run_command",execute)
    study={"prospective_runs":{"training":"training"},"_resume":True}
    p.execute_job(study,"training","scripts.gemma4_train_qasper")
    assert calls[0][-1]=="--resume"
    p.execute_job(study,"training","scripts.gemma4_train_qasper")
    assert len(calls)==1


def test_cpu_source_refresh_preserves_inputs_and_refuses_gpu_attempts(tmp_path,monkeypatch):
    from scripts import gemma4_refresh_qasper_preparation as refresh
    repo=tmp_path/"repo"; (repo/"scripts").mkdir(parents=True)
    (repo/"scripts/stub.py").write_text("value=1\n")
    monkeypatch.setattr(refresh.rt,"ROOT",repo)
    monkeypatch.setattr(refresh.rt,"versions",lambda:{})
    monkeypatch.setattr(refresh.rt,"source_closure",lambda names:["scripts/stub.py"])
    source=tmp_path/"old"
    refresh.rt.register(source,{"kind":"coverage"},{"train.json":[{"id":"q","input_ids":[2,17,106]}]})
    assert refresh.refresh(source,tmp_path/"new")["gpu_launched"] is False
    assert (source/"train.json").read_bytes()==(tmp_path/"new/train.json").read_bytes()
    assert a.read(tmp_path/"new/registration.json")["prior_cpu_preparation"]["sha256"]==a.sha(source/"registration.json")
    (source/"attempts").mkdir()
    with pytest.raises(ValueError,match="without GPU attempts"):refresh.refresh(source,tmp_path/"other")


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):return "PROMPT:"+messages[0]["content"]
    def __call__(self,text,**kwargs):return {"input_ids":[ord(c)+1000 for c in text]}
    def decode(self,ids,**kwargs):return "".join("<|tool_response>" if i==50 else chr(i-1000) for i in ids)


def trajectory(text='{"answer":"yes","unanswerable":false,"evidence":["p000000"]}',extra=None):
    tok=Tokenizer();row={"id":"q1","paper_id":"p1","user":"Question?"}
    prompt=tok.apply_chat_template([{"content":row["user"]}])
    ids=tok(text)["input_ids"]+[106]
    pred={"id":"q1","paper_id":"p1","rendered_prompt":prompt,"input_ids":tok(prompt)["input_ids"],
          "generated_ids":ids,"output":text,"ended_with_eos":True,"hit_token_limit":False,
          "registration_sha256":"binding"}
    if extra:pred.update(extra)
    return tok,row,pred


def test_raw_trajectory_and_native_turn_are_exactly_validated():
    tok,row,pred=trajectory()
    assert a.validate_generation([row],[pred],tok,"","binding")=={"q1":pred["output"]}
    bad=copy.deepcopy(pred);bad["generated_ids"].append(1200)
    with pytest.raises(ValueError,match="after native stop"):
        a.validate_generation([row],[bad],tok,"","binding")
    bad=copy.deepcopy(pred);bad["output"]="rewritten answer"
    with pytest.raises(ValueError,match="decoded trajectory"):
        a.validate_generation([row],[bad],tok,"","binding")


def test_stray_tool_marker_is_preserved_rather_than_hidden():
    tok,row,pred=trajectory()
    assert a.methods()["qwen38_science_score"].parse_response(pred["output"]) is not None
    pred["generated_ids"].insert(-1,50);pred["output"]+="<|tool_response>"
    scored=a.validate_generation([row],[pred],tok,"","binding")["q1"]
    assert scored.endswith("<|tool_response>")
    assert a.methods()["qwen38_science_score"].parse_response(scored) is None


@pytest.mark.parametrize("change",[{"ended_with_eos":False},{"hit_token_limit":True},
    {"resource_cap_stop":True},{"registration_sha256":"other"},{"input_ids":[7]}])
def test_invalid_generation_metadata_fails_closed(change):
    tok,row,pred=trajectory(extra=change)
    with pytest.raises(ValueError):a.validate_generation([row],[pred],tok,"","binding")


def test_duplicate_predictions_and_short_unexplained_outputs_rejected():
    tok,row,pred=trajectory()
    with pytest.raises(ValueError,match="unique"):
        a.validate_generation([row],[pred,pred],tok,"","binding")
    pred.update(generated_ids=pred["generated_ids"][:-1],ended_with_eos=False)
    with pytest.raises(ValueError,match="unexplained short"):
        a.validate_generation([row],[pred],tok,"","binding")


def test_pending_retention_never_becomes_a_complete_null_or_recommendation():
    for passed in (True,False):
        result=a.outcome({"science_gates_pass":passed})
        assert result["complete"] is False and result["recommended_adapter"] is None
    assert a.outcome({"science_gates_pass":True},{"retention_pass":False})["recommended_adapter"] is False
    assert a.outcome({"science_gates_pass":True},{"retention_pass":True})["recommended_adapter"] is True


def test_frozen_helper_source_tampering_rejected_before_execution(tmp_path):
    (tmp_path/"scripts").mkdir();(tmp_path/"scripts/qwen38_science_score.py").write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError,match="frozen method source changed"):a.methods(tmp_path)


def test_prompt_selection_recomputed_and_ties_retain_base():
    data=[{"id":str(i),"paper_id":"p"+str(i//5),"primary":True,"evidence_strict_primary":True,
        "references":[{"answer":"yes","unanswerable":False,"evidence":["p000000"]}],
        "paragraphs":[{"id":"p000000","text":"Evidence."}],"context_ids":["p000000"],
        "nontext_or_unmatched_evidence":False} for i in range(35)]
    predictions=[{"id":r["id"],"template":template,"output":json.dumps({"answer":"no" if template=="verify" else "yes",
        "evidence":["p000000"],"unanswerable":False})} for template in ("base","verify","schema") for r in data]
    selection={"chosen":"base","final_test_used":False,"registration_sha256":"b",
               "strict_primary_answer_f1":{"base":1.0,"verify":0.0,"schema":1.0}}
    metric=a.methods()["qwen38_science_score"]
    assert a.validate_selection(selection,data,predictions,"b",{},metric)=="base"
    selection["chosen"]="schema"
    with pytest.raises(ValueError,match="tie order"):a.validate_selection(selection,data,predictions,"b",{},metric)
