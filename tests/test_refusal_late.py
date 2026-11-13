"""Synthetic CPU-only late follow-up guards; no models or actual judging."""
from types import SimpleNamespace

import pytest

from scripts import refusal_late as late
from scripts.common import read_json


def metric(count,empty=0,degenerate=0):
    return {"total":30,"primary_refused":count,"empty":empty,"degenerate":degenerate}


def test_fractional_conditions_are_canonical_and_roundtrip():
    for m,prefix in ((0,"m0"),(.25,"m0p25"),(.5,"m0p5"),(1,"m1")):
        for label in ("safe","unsafe"):
            assert late.condition(m,label)==prefix+"_"+label
            assert late.parse_condition(prefix+"_"+label)==(m,label)
    for name in ("m0.25_safe","m0p50_safe","m2_safe"):
        with pytest.raises(ValueError):
            late.parse_condition(name)
    for value in (-.25,2,True,"0.25"):
        with pytest.raises(ValueError):
            late.condition(value,"safe")


def test_smallest_fractional_choice_integer_boundary_quality_and_null():
    candidates={late.condition(m,"safe"):metric(15) for m in late.STRENGTHS}
    assert late.choose(metric(1),candidates)["selected_multiplier"]==.25
    candidates["m0p25_safe"]=metric(14)
    assert late.choose(metric(1),candidates)["selected_multiplier"]==.5
    assert late.choose(metric(15),{late.condition(m,"safe"):metric(21) for m in late.STRENGTHS})["selected_multiplier"]==.25
    assert late.choose(metric(15),{late.condition(m,"safe"):metric(20) for m in late.STRENGTHS})["null"]
    failed={"m0p25_safe":metric(30,empty=1),"m0p5_safe":metric(30,degenerate=1),"m1_safe":metric(14)}
    assert late.choose(metric(1),failed)["skip_final"]


def test_final_schedule_has228_fresh_responses_and_only_selected_unsafe():
    schedule=late.final_schedule(.5)
    assert sum(42 if late.parse_condition(n)[1]=="safe" else 30 for n in schedule)==228
    assert {n for n in schedule if n.endswith("_unsafe")}=={"m0_unsafe","m0p5_unsafe"}
    assert len({n for n in schedule if n.endswith("_safe")})==4
    with pytest.raises(ValueError):
        late.final_schedule(None)


def save(run,name,value):
    late.save(run/name,value)


@pytest.fixture
def synthetic_run(tmp_path,monkeypatch):
    cfg=read_json("configs/refusal_late.json")
    manifest={"classifier_context_difference":"unit-test different packet contexts","sources":{},"reused_files_sha256":{}}
    monkeypatch.setattr(late,"verify",lambda *a,**k:(cfg,manifest))
    save(tmp_path,"manifest.json",manifest)
    rows=[{"id":f"{label}-{i}","instruction":"Unit-test prompt","label":label,"family":"unit-family"} for label in ("safe","unsafe") for i in range(30)]
    save(tmp_path,"inputs/development.json",{"items":rows})
    save(tmp_path,"inputs/final.json",{"not_read_as_prompts_in_test":True})
    save(tmp_path,"inputs/rubric.json",read_json("evaluation/refusal/response_rubric.json"))
    judge={"model_identifier":"unit-test-judge","exposed_settings":{},"fresh_context":True,"advisor_overlap":False}
    packet,keys,ratings=[],{},[]
    for label in ("safe","unsafe"):
        predictions=[]
        for i,row in enumerate(r for r in rows if r["label"]==label):
            key=f"old-anon-{label}-{i}"
            packet.append({"id":key,"prompt":"Unit-test prompt","response":"Unit-test stored output"})
            keys[key]={"condition":late.condition(0,label),"source_id":row["id"]}
            refused=i<(1 if label=="safe" else 28)
            ratings.append({"id":key,"refusal_class":"full_refusal" if refused else "full_compliance","unscorable":False,"useful_and_correct":not refused})
            predictions.append({"id":row["id"],"output":"Unit-test stored output","generated_ids":[4,5],"hit_token_limit":False})
        save(tmp_path,f"reuse/m0_{label}.json",{"predictions":predictions})
    save(tmp_path,"reuse/packet.json",{"items":packet})
    save(tmp_path,"reuse/keys.local.json",keys)
    save(tmp_path,"reuse/ratings.json",{"judge":judge,"ratings":ratings})
    names={late.condition(m,"safe"):"safe" for m in late.STRENGTHS}
    for name in names:
        predictions=[{"id":r["id"],"output":"Unit-test new output","generated_ids":[6,7],"hit_token_limit":False} for r in rows if r["label"]=="safe"]
        save(tmp_path,f"development/{name}.json",{"predictions":predictions})
    save(tmp_path,"development/conditions.json",{"split":"development","conditions":names,
        "prediction_sha256":{n:late.stream_hash(tmp_path/f"development/{n}.json") for n in names}})
    save(tmp_path,"development/run.json",{"completed":True,"elapsed_wall_seconds":1,"peak_allocated_bytes":99})
    save(tmp_path,"development/budget_lock.json",{"unit_test":True})
    late.packets(SimpleNamespace(phase="development"),tmp_path)
    new_packet=read_json(tmp_path/"development/judging/packet.json")["items"]
    save(tmp_path,"development/judging/ratings.json",{"judge":judge,"ratings":[{"id":r["id"],"refusal_class":"partial_refusal","unscorable":False,"useful_and_correct":False} for r in new_packet]})
    return tmp_path


def test_dev_packet_is90_new_only_and_reused_base_labels_take_precedence(synthetic_run):
    run=synthetic_run
    packet=read_json(run/"development/judging/packet.json")["items"]
    keys=read_json(run/"development/judging/keys.local.json")
    assert len(packet)==90 and all(set(r)=={"id","prompt","response"} for r in packet)
    assert {r["condition"] for r in keys.values()}=={"m0p25_safe","m0p5_safe","m1_safe"}
    assert not any(r["condition"].startswith("m0_") for r in keys.values())
    baseline=late.base_scores(run)
    assert baseline["m0_safe"]["overall"]["primary_refused"]==1
    assert baseline["m0_unsafe"]["overall"]["primary_refused"]==28
    late.selection(SimpleNamespace(),run)
    result=read_json(run/"selection.json")
    assert result["decision"]["selected_multiplier"]==.25
    assert result["reused_labels"]==60 and result["newly_judged"]==90


def test_reused_baseline_cannot_silently_enter_new_packet(synthetic_run,monkeypatch):
    rows,predictions=late.old.phase_inputs(synthetic_run,"development")
    rows["m0_safe"]=rows["m1_safe"]
    predictions["m0_safe"]=predictions["m1_safe"]
    monkeypatch.setattr(late.old,"phase_inputs",lambda *a:(rows,predictions))
    with pytest.raises(ValueError,match="NEW"):
        late.phase_inputs(synthetic_run,"development")


@pytest.mark.parametrize("changed",["reuse/ratings.json","reuse/m0_safe.json","development/judging/ratings.json","development/m0p25_safe.json"])
def test_selection_lock_rejects_reuse_or_new_result_mutation(synthetic_run,changed):
    run=synthetic_run
    late.selection(SimpleNamespace(),run)
    assert late.selection_lock(run)["decision"]["selected_multiplier"]==.25
    (run/changed).write_text("unit-test mutation after selection")
    with pytest.raises(ValueError):
        late.selection_lock(run)


def test_external_old_artifacts_and_vector_bytes_remain_bound(tmp_path,monkeypatch):
    cfg=read_json("configs/refusal_late.json")
    external=tmp_path/"original-provenance.json"
    external.write_text("unit-test original")
    vector=tmp_path/"reuse/vectors.pt"
    vector.parent.mkdir()
    vector.write_bytes(b"unit-test placeholder, no model tensor")
    vector_hash=late.stream_hash(vector)
    cfg["vector_sha256"]=vector_hash
    monkeypatch.setattr(late,"VECTOR_SHA",vector_hash)
    monkeypatch.setattr(late.old,"verify",lambda *a,**k:(cfg,{"external_reuse_sha256":{str(external):late.stream_hash(external)}}))
    # Capture expected hash before changing the file; do not re-seal on verify.
    manifest={"external_reuse_sha256":{str(external):late.stream_hash(external)}}
    monkeypatch.setattr(late.old,"verify",lambda *a,**k:(cfg,manifest))
    late.verify(tmp_path)
    external.write_text("changed original")
    with pytest.raises(ValueError,match="original/reused"):
        late.verify(tmp_path)


def test_no_gpu_before_explicit_flag_and_null_final_skips_model(synthetic_run,monkeypatch):
    run=synthetic_run
    with pytest.raises(ValueError,match="execute-gpu"):
        late.gpu_ready(SimpleNamespace(execute_gpu=False),run)
    monkeypatch.setattr(late,"check_lock",lambda *a:{"decision":{"null":True}})
    def forbidden(*a,**k):
        raise AssertionError("null must not read final prompts, load model or query GPU")
    monkeypatch.setattr(late,"gpu_ready",forbidden)
    monkeypatch.setattr(late.old,"groups",forbidden)
    late.final(SimpleNamespace(),run)
    assert read_json(run/"final_skipped.json")["stop_refusal_pilot"]


def test_null_summary_preserves_new_costs_and_reuse_counts(synthetic_run,monkeypatch):
    run=synthetic_run
    late.selection(SimpleNamespace(),run)
    monkeypatch.setattr(late,"check_lock",lambda *a:{"decision":{"null":True}})
    save(run,"probe/run.json",{"completed":True,"elapsed_wall_seconds":2,"peak_allocated_bytes":101})
    lock=run/"registered-lock.json"
    save(run,"registered-lock.json",{"unit_test":True})
    output=run/"summary.json"
    late.summary(SimpleNamespace(lock=str(lock),output=str(output)),run)
    result=read_json(output)
    assert result["null"] and result["final_skipped"] and result["stop_refusal_pilot"]
    assert set(result["costs"])=={"probe","development"}
    assert result["reused_predictions"]==result["reused_labels"]==60
    assert result["new_counts"]["development_judgments"]==90 and result["new_counts"]["final_responses"]==0


def test_new_judging_seed_only_changes_anonymous_packet_ids():
    cfg=read_json("configs/refusal_late.json")
    assert cfg["judging_seed"]==20261115 and cfg["data_seed"]==20261114
