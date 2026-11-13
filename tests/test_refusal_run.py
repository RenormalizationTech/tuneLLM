"""CPU guards around run artifacts, locks, blind packets and null skipping."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import refusal_run as run
from scripts.refusal_common import choose_setting


def save(path,value):
    run.save(path,value)


def test_no_gpu_queries_or_model_reads_before_explicit_execution(tmp_path,monkeypatch):
    def forbidden(*a,**k):
        raise AssertionError("unapproved GPU/model access")
    monkeypatch.setattr(run,"verify",forbidden)
    with pytest.raises(ValueError,match="execute-gpu"):
        run.gpu_ready(SimpleNamespace(execute_gpu=False),tmp_path)


def test_null_final_skips_without_gpu_or_final_prompt_reads(tmp_path,monkeypatch):
    monkeypatch.setattr(run,"check_lock",lambda *a:{"decision":{"null":True}})
    def forbidden(*a,**k):
        raise AssertionError("null must not access model or final prompt content")
    monkeypatch.setattr(run,"gpu_ready",forbidden)
    monkeypatch.setattr(run,"groups",forbidden)
    run.final(SimpleNamespace(lock="unit-test-lock"),tmp_path)
    assert run.read_json(tmp_path/"final_skipped.json")["no_final_generation"]


def test_final_rejects_missing_lock_before_any_other_access(tmp_path,monkeypatch):
    def forbidden(*a,**k):
        raise AssertionError("must reject before GPU or final data")
    monkeypatch.setattr(run,"gpu_ready",forbidden)
    with pytest.raises(ValueError,match="lock required"):
        run.final(SimpleNamespace(lock=None),tmp_path)


def lock_fixture(tmp_path):
    for name,value in (("manifest.json",{"unit_test":True}),
                       ("development/run.json",{"completed":True}),
                       ("development/budget_lock.json",{"report":{"cap":256}}),
                       ("development/judging/ratings.json",{"unit_test":True}),
                       ("inputs/development.json",{"items":[]}),
                       ("inputs/final.json",{"not_read_as_prompts_before_lock":True})):
        save(tmp_path/name,value)
    save(tmp_path/"development/m0_safe.json",{"predictions":[]})
    save(tmp_path/"development/conditions.json",{"split":"development","conditions":{"m0_safe":"safe"},
        "prediction_sha256":{"m0_safe":run.stream_hash(tmp_path/"development/m0_safe.json")}})
    save(tmp_path/"development/judging/manifest.json",{"packet_files":{},"source_files":{
        n:run.stream_hash(tmp_path/"development"/n) for n in ("conditions.json","m0_safe.json","run.json")}})
    base={"total":30,"primary_refused":15,"empty":0,"degenerate":0}
    candidates={m:{**base,"primary_refused":21} for m in (1,2,4)}
    names=["manifest.json","development/conditions.json","development/run.json","development/budget_lock.json","development/judging/ratings.json"]
    selection={"split":"development","final_used":False,"base":base,"candidate_metrics":{str(k):v for k,v in candidates.items()},
               "decision":choose_setting(base,candidates),"input_sha256":{n:run.stream_hash(tmp_path/n) for n in names}}
    save(tmp_path/"selection.json",selection)


@pytest.mark.parametrize("changed",["manifest.json","development/conditions.json","development/run.json",
    "development/budget_lock.json","development/judging/ratings.json","development/m0_safe.json"])
def test_selection_input_or_prediction_mutation_before_lock_rejected(tmp_path,changed):
    lock_fixture(tmp_path)
    assert run.selection_lock(tmp_path)["decision"]["selected_multiplier"]==1
    (tmp_path/changed).write_text("unit-test mutation after selection")
    with pytest.raises(ValueError):
        run.selection_lock(tmp_path)


def test_blind_packet_manifest_rejects_source_mutation(tmp_path):
    judging=tmp_path/"judging"
    save(judging/"packet.json",{"items":[]})
    save(tmp_path/"m0_safe.json",{"unit_test":True})
    save(judging/"manifest.json",{"packet_files":{"packet.json":run.stream_hash(judging/"packet.json")},
        "source_files":{"m0_safe.json":run.stream_hash(tmp_path/"m0_safe.json")}})
    run.verify_packet(tmp_path)
    (tmp_path/"m0_safe.json").write_text("changed")
    with pytest.raises(ValueError,match="predictions"):
        run.verify_packet(tmp_path)


def test_prepared_input_report_and_bytes_are_both_bound(tmp_path):
    save(tmp_path/"extraction.json",{"unit_test":True})
    manifest={"files":{"extraction.json":run.stream_hash(tmp_path/"extraction.json")}}
    save(tmp_path/"manifest.json",manifest)
    report={**manifest,"manifest_sha256":run.stream_hash(tmp_path/"manifest.json")}
    run.check_prepared(tmp_path,report)
    (tmp_path/"extraction.json").write_text("changed")
    with pytest.raises(ValueError,match="input changed"):
        run.check_prepared(tmp_path,report)


def test_null_summary_keeps_completed_resource_costs_without_reading_final(tmp_path,monkeypatch):
    cfg={"model_id":"unit-test-model","revision":"unit-test-revision"}
    monkeypatch.setattr(run,"verify",lambda *a:(cfg,{"sources":{}}))
    monkeypatch.setattr(run,"check_lock",lambda *a:{"decision":{"null":True}})
    save(tmp_path/"selection.json",{"unit_test":True})
    save(tmp_path/"manifest.json",{"unit_test":True})
    lock=tmp_path/"registered-lock.json"
    save(lock,{"unit_test":True})
    for phase in ("extraction","preflight","development"):
        save(tmp_path/phase/"run.json",{"elapsed_wall_seconds":1.25,"peak_allocated_bytes":12345,"unit_test":True})
    def forbidden(*a,**k):
        raise AssertionError("null summary must not read final prompt contents or final judgments")
    monkeypatch.setattr(run,"groups",forbidden)
    monkeypatch.setattr(run,"scores",forbidden)
    output=tmp_path/"summary.json"
    run.summary(SimpleNamespace(lock=str(lock),output=str(output)),tmp_path)
    result=run.read_json(output)
    assert result["final_skipped"] and set(result["costs"])=={"extraction","preflight","development"}
    assert all(v["peak_allocated_bytes"]==12345 for v in result["costs"].values())
