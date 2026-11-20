"""Schema-only CPU regression; saved configs only, never saved quality data."""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import gemma4_qasper_analysis as frozen
from scripts import gemma4_qasper_analysis_amendment_01 as amended

OLD = "sum full-answer conditional log probabilities; raw concept prompt"
NEW = "sum full-answer conditional log probabilities; native BOS + raw concept prompt"
ORIGINAL_SHA = "f957f8a0de70b6a4624ea6c61f27c90e6ebc84a65290e7ab5be0596957337f2a"


def test_original_frozen_and_amendment_body_has_only_expected_literal_change():
    original = Path(frozen.__file__).read_bytes()
    assert hashlib.sha256(original).hexdigest() == ORIGINAL_SHA
    original_body = original.decode().split("import argparse",1)[1]
    amended_body = Path(amended.__file__).read_text().split("import argparse",1)[1]
    assert original_body.count(OLD) == 1
    assert amended_body == original_body.replace(OLD,NEW,1)
    assert ORIGINAL_SHA in amended.__doc__
    assert "ba1983fdb9e6ca1e11ce9648e69a4c057832d6524b5014cd99cabe3fca4ba9f0" in amended.__doc__


def config():
    return {"kind":"retention","autocast":True,"batch_size":1,"mc_batch_size":1,
        "eos_token_id":[1,106],"seed":42,"generation_caps":{"gsm8k":512,"ifeval":1024},
        "choice_ids":[562,603,565,622],"mc_bos_token_id":2,"mc":NEW,
        "adapter":None,"scorer_python":"/synthetic/venv/bin/python","scorer_python_sha256":"interpreter-sha",
        "scorer_packages":{"synthetic":"version"},"ifeval_assets":{"synthetic":"assets"},
        "fixture_sha256":"fixture-sha","identity":{"synthetic":"base"},"versions":{"synthetic":"version"}}


def harness(monkeypatch,tmp_path,cfg,module=amended):
    """Run the actual validator; replace all prediction/score I/O with sentinels."""
    calls = []
    fixture = {"ifeval":[{"key":"synthetic"}]}
    predictions = [{"id":"synthetic","task":"ifeval","output":"synthetic only"}]
    computed = {"synthetic_scoring_sentinel":True}
    result = {"generations":288}
    study = {"retention_scorer":{"python":cfg["scorer_python"],"python_sha256":cfg["scorer_python_sha256"],
        "packages":copy.deepcopy(cfg["scorer_packages"]),"assets":copy.deepcopy(cfg["ifeval_assets"])},
        "data":{"retention_fixture":{"sha256":cfg["fixture_sha256"]}}}
    def load_job(received,role):
        assert received is study
        calls.append(("load_job",role))
        return tmp_path,cfg,result,"synthetic-binding"
    def read(path):
        path = Path(path)
        if path == tmp_path/"fixture.json":
            calls.append(("read","synthetic fixture")); return fixture
        if path == tmp_path/"scores.json":
            calls.append(("read","synthetic scores")); return computed
        pytest.fail("Unexpected file read; real quality data must never be opened: "+str(path))
    def rows(path):
        assert Path(path) == tmp_path/"predictions.jsonl"
        calls.append(("rows","synthetic predictions")); return predictions
    def validate_fixture(value):
        assert value is fixture; calls.append(("validate_fixture",True))
    def run_ifeval(data,values,python):
        assert data is fixture["ifeval"] and values is predictions
        assert str(python) == str(Path(cfg["scorer_python"]).absolute())
        calls.append(("run_ifeval","mock only"))
        return {"packages":cfg["scorer_packages"],"scores":{"synthetic_strict":True}}
    def score_predictions(data,values,strict):
        assert data is fixture and values is predictions
        assert strict(data["ifeval"],values) == {"synthetic_strict":True}
        calls.append(("score_predictions","mock only")); return computed
    helper = SimpleNamespace(validate_fixture=validate_fixture,run_ifeval=run_ifeval,score_predictions=score_predictions)
    monkeypatch.setattr(module,"load_job",load_job)
    monkeypatch.setattr(module,"read",read)
    monkeypatch.setattr(module,"rows",rows)
    monkeypatch.setattr(module,"methods",lambda:{"qwen38_retention":helper})
    return study,calls,result


def test_synthetic_native_bos_schema_runs_actual_retention_validator(monkeypatch,tmp_path):
    cfg = config(); study,calls,_ = harness(monkeypatch,tmp_path,cfg)
    report = amended.validate_retention(study,"retention_base")
    assert report["arm"] == "base" and report["tasks"] == {"synthetic_scoring_sentinel":True}
    assert ("validate_fixture",True) in calls
    assert ("score_predictions","mock only") in calls


@pytest.mark.artifact("outputs/gemma4-qasper-study-01/retention-base/registration.json",
                      "outputs/gemma4-qasper-study-01/retention-adapter/registration.json")
@pytest.mark.parametrize("arm",["base","adapter"])
def test_actual_saved_registration_schema_old_rejects_amendment_accepts_without_quality_reads(monkeypatch,tmp_path,arm):
    # Only config metadata are read. load_job/results/predictions/scorers below
    # are synthetic; no saved model output or score file is ever consumed.
    path = amended.ROOT/f"outputs/gemma4-qasper-study-01/retention-{arm}/registration.json"
    cfg = amended.read(path)
    assert cfg["mc"] == NEW and cfg["mc_bos_token_id"] == 2
    role = "retention_"+arm
    study,_,_ = harness(monkeypatch,tmp_path,cfg,frozen)
    with pytest.raises(ValueError,match="retention profile differs"):
        frozen.validate_retention(study,role)
    study,calls,_ = harness(monkeypatch,tmp_path,cfg)
    training = (None,None,{"selected_files":cfg["adapter"]["files"]}) if arm == "adapter" else None
    report = amended.validate_retention(study,role,training)
    assert report["arm"] == arm
    assert ("score_predictions","mock only") in calls


@pytest.mark.parametrize("key,value",[
    ("kind","other"),("autocast",False),("batch_size",2),("mc_batch_size",2),
    ("eos_token_id",[1,106,50]),("seed",43),("generation_caps",{"gsm8k":256,"ifeval":1024}),
    ("choice_ids",[603,562,565,622]),("mc_bos_token_id",None),("mc",OLD),
])
def test_other_profile_differences_still_rejected(monkeypatch,tmp_path,key,value):
    cfg = config(); study,calls,_ = harness(monkeypatch,tmp_path,cfg)
    cfg[key] = value
    with pytest.raises(ValueError,match="retention profile differs"):
        amended.validate_retention(study,"retention_base")
    assert not any(name == "score_predictions" for name,_ in calls)


@pytest.mark.parametrize("key,value",[
    ("scorer_python","/other/python"),("scorer_python_sha256","wrong"),
    ("scorer_packages",{"synthetic":"different"}),("ifeval_assets",{"wrong":"assets"}),
])
def test_scorer_bindings_still_rejected(monkeypatch,tmp_path,key,value):
    cfg = config(); study,_,_ = harness(monkeypatch,tmp_path,cfg)
    cfg[key] = value
    with pytest.raises(ValueError,match="retention scorer binding differs"):
        amended.validate_retention(study,"retention_base")


def test_fixture_and_incomplete_count_still_rejected(monkeypatch,tmp_path):
    cfg = config(); study,_,result = harness(monkeypatch,tmp_path,cfg)
    cfg["fixture_sha256"] = "wrong"
    with pytest.raises(ValueError,match="retention fixture identity differs"):
        amended.validate_retention(study,"retention_base")
    cfg["fixture_sha256"] = "fixture-sha"; result["generations"] = 287
    with pytest.raises(ValueError,match="complete 288-item retention run required"):
        amended.validate_retention(study,"retention_base")
