"""Mock-only refusal adapter tests; accepted invocation foundation stays untouched."""
import json
from types import SimpleNamespace

import pytest

from scripts import refusal_judge as adapter


def items():
    return [{"id":"anonymous-a","prompt":"Unit-test prompt","response":"Unit-test output"}]


def result(payload,complete=True,model=None):
    foundation=adapter.foundation
    ratings=[{"id":r["id"],"refusal_class":"partial_refusal","unscorable":False,"useful_and_correct":False} for r in payload["items"]] if complete else []
    return {"returncode":0,"stderr":"","isolation":{"returncode":128,"git_not_repository":True,"git_environment_clean":True},
        "canonical_stdin_sha256":foundation.canonical_hash(payload),
        "stdout":json.dumps({"subtype":"success","modelUsage":{model or foundation.MODEL:{}},"result":json.dumps({"ratings":ratings})})}


def test_refusal_schema_retry_first_valid_and_exact_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(adapter.foundation,"version",lambda binary:"2.1.323 (Claude Code)")
    calls=[]
    def invoke(binary,payload):
        calls.append(payload)
        return result(payload,complete=len(calls)>1)
    monkeypatch.setattr(adapter.foundation,"invoke",invoke)
    output=tmp_path/"judge"
    first=adapter.judge_packet(items(),{"test":"rubric"},output,{"adapter":"unit-test"})
    assert first["ratings"][0]["refusal_class"]=="partial_refusal" and len(calls)==2
    before=(output/"batch-000-attempt-1.json").read_bytes()
    second=adapter.judge_packet(items(),{"test":"rubric"},output,{"adapter":"unit-test"},resume=True)
    assert first==second and len(calls)==2 and (output/"batch-000-attempt-1.json").read_bytes()==before
    with pytest.raises(ValueError):
        adapter.judge_packet(items(),{"test":"changed rubric"},output,{"adapter":"unit-test"},resume=True)
    assert len(calls)==2


def test_adapter_model_mismatch_never_retries_or_reissues_on_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(adapter.foundation,"version",lambda binary:"2.1.323 (Claude Code)")
    calls=[]
    def invoke(binary,payload):
        calls.append(payload)
        return result(payload,model="different-model")
    monkeypatch.setattr(adapter.foundation,"invoke",invoke)
    output=tmp_path/"judge"
    for resume in (False,True):
        with pytest.raises(adapter.foundation.ModelMismatch):
            adapter.judge_packet(items(),{},output,{},resume=resume)
    assert len(calls)==1


def test_schema_adapter_rejects_tone_ratings_instead_of_monkeypatching_validator():
    payload={"items":items()}
    attempt=result(payload)
    outer=json.loads(attempt["stdout"])
    outer["result"]=json.dumps({"ratings":[{"id":"anonymous-a","preference":"A"}]})
    attempt["stdout"]=json.dumps(outer)
    with pytest.raises(adapter.foundation.InvalidSubmission):
        adapter.parse_submission(attempt,items())


def test_fresh_adapter_audit_binds_rubric_validator_and_foundation(tmp_path,monkeypatch):
    rubric=tmp_path/"rubric.json"
    rubric.write_text('{"unit_test":true}')
    def fake_audit(output,binary):
        output.mkdir()
        adapter.foundation.save(output/"audit.json",{"unit_test":True,"passed":True})
    monkeypatch.setattr(adapter.foundation,"isolation_audit",fake_audit)
    monkeypatch.setattr(adapter.foundation,"validate_audit",lambda *a:{"passed":True})
    output=tmp_path/"audit"
    binding=adapter.audit(output,rubric)
    adapter.verify_audit(output,rubric,binding)
    rubric.write_text('{"unit_test":"changed"}')
    with pytest.raises(ValueError,match="hashes"):
        adapter.verify_audit(output,rubric,binding)


def test_no_api_call_without_explicit_audit_flag(tmp_path,monkeypatch):
    import sys
    monkeypatch.setattr(sys,"argv",["refusal_judge","audit","--output",str(tmp_path/"audit")])
    def forbidden(*a,**k):
        raise AssertionError("no audit call authorized by this unit test")
    monkeypatch.setattr(adapter,"audit",forbidden)
    with pytest.raises(ValueError,match="execute-audit"):
        adapter.main()
