"""Mock-only independent judge tests: no API calls, model runs or project context."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import tone_judge as judge

ISOLATION = {"command": ["git", "rev-parse", "--git-dir"], "returncode": 128,
             "git_not_repository": True, "git_environment_clean": True, "removed_git_variable_names": []}


def bound(value, payload):
    return {**value, "canonical_stdin_sha256": judge.canonical_hash(payload)}


def packet(count=2):
    return [{"id": f"anonymous-{i}", "scenario": "Unit-test scenario", "required_facts": [{"key": "event"}],
             "response": "Unit-test string, not an experimental generation"} for i in range(count)]


def ratings(items):
    return [{"id": r["id"], "coverage": {"event": True}, "task_completed": True,
             "unsupported_claims": 0, "contradicted_claims": 0, "content_score": 5, "warmth_score": 3} for r in items]


def attempt(value, model=judge.MODEL):
    return {"returncode": 0, "stderr": "", "isolation": ISOLATION, "stdout": json.dumps({"subtype": "success", "is_error": False,
            "modelUsage": {model: {"inputTokens": 10}}, "result": json.dumps({"ratings": value})})}


def test_subprocess_is_argv_tools_off_neutral_cwd_and_stdin_only(monkeypatch):
    monkeypatch.setenv("HERDR_PANE_ID", "project-pane")
    monkeypatch.setenv("CODEX_THREAD_ID", "project-session")
    monkeypatch.setenv("GIT_DIR", "/unit-test-repository")
    monkeypatch.setenv("GIT_WORK_TREE", "/unit-test-tree")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    called = []
    def fake(argv, **kwargs):
        called.append(argv)
        assert kwargs["shell"] is False
        assert not any(k.startswith("GIT_") for k in kwargs["env"])
        if argv[0] == "git":
            assert argv == ["git", "rev-parse", "--git-dir"]
            return SimpleNamespace(returncode=128, stdout="", stderr="fatal: not a git repository")
        assert argv[argv.index("--tools")+1] == ""
        for flag in ("--safe-mode", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
            assert flag in argv
        assert all(x not in argv for x in ("--resume", "--continue", "--fallback-model", "--add-dir"))
        assert argv[argv.index("--model")+1] == judge.MODEL
        assert argv[argv.index("--effort")+1] == "medium"
        cwd = Path(kwargs["cwd"])
        assert str(cwd).startswith("/tmp/tone-judge-") and list(cwd.iterdir()) == []
        assert kwargs["env"]["PWD"] == str(cwd)
        assert "HERDR_PANE_ID" not in kwargs["env"] and "CODEX_THREAD_ID" not in kwargs["env"]
        assert kwargs["input"] == judge.canonical_stdin({"items": packet()})
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")
    monkeypatch.setattr(judge.subprocess, "run", fake)
    result = judge.invoke("claude", {"items": packet()})
    assert len(called) == 2 and result["isolation"]["git_not_repository"]
    assert "GIT_WORK_TREE" in result["isolation"]["removed_git_variable_names"]


def mock_version(monkeypatch):
    monkeypatch.setattr(judge, "version", lambda binary: "2.1.323 (Claude Code)")


def test_first_valid_retry_two_attempt_limit_and_batch_audit(tmp_path, monkeypatch):
    mock_version(monkeypatch)
    items, payloads = packet(), []
    responses = [attempt(ratings(items)[:-1]), attempt(ratings(items))]
    def invoke(binary, payload):
        payloads.append(copy.deepcopy(payload))
        return bound(responses[len(payloads)-1], payload)
    monkeypatch.setattr(judge, "invoke", invoke)
    output = tmp_path / "judge"
    result = judge.judge_packet(items, {"instruction": "test rubric"}, "content", output)
    assert len(result["ratings"]) == 2 and len(payloads) == 2 and payloads[0] == payloads[1]
    provenance = json.loads((output / "provenance.json").read_text())
    batch = provenance["batches"][0]
    assert batch["ids"] == [r["id"] for r in items]
    assert batch["canonical_stdin_sha256"] == hashlib.sha256(judge.canonical_stdin(payloads[0]).encode()).hexdigest()
    assert batch["first_valid_attempt"] == 1 and provenance["maximum_attempts_per_batch"] == 2
    for record in batch["attempt_files"]:
        assert record["sha256"] == judge.stream_hash(output / record["file"])


def test_first_valid_used_without_another_call(tmp_path, monkeypatch):
    mock_version(monkeypatch)
    calls = []
    def invoke(binary, payload):
        calls.append(payload)
        return bound(attempt(ratings(payload["items"])), payload)
    monkeypatch.setattr(judge, "invoke", invoke)
    result = judge.judge_packet(packet(21), {}, "content", tmp_path / "judge")
    assert len(calls) == 2 and [len(p["items"]) for p in calls] == [20, 1]
    assert len(result["ratings"]) == 21


@pytest.mark.parametrize("failure", ["fallback", "infrastructure"])
def test_model_mismatch_and_infrastructure_are_never_retried(tmp_path, monkeypatch, failure):
    mock_version(monkeypatch)
    calls = []
    def invoke(binary, payload):
        calls.append(payload)
        return bound(attempt(ratings(payload["items"]), "other-model") if failure == "fallback" else {
            "returncode": 1, "stdout": "", "stderr": "unit-test infrastructure error"}, payload)
    monkeypatch.setattr(judge, "invoke", invoke)
    with pytest.raises((judge.ModelMismatch, RuntimeError)):
        judge.judge_packet(packet(), {}, "content", tmp_path / "judge")
    assert len(calls) == 1
    with pytest.raises((judge.ModelMismatch, RuntimeError)):
        judge.judge_packet(packet(), {}, "content", tmp_path / "judge", resume=True)
    assert len(calls) == 1
    assert not (tmp_path / "judge/submission.json").exists()
    assert (tmp_path / "judge/batch-000-provenance.json").is_file()


def test_incomplete_ratings_stop_after_two_total_attempts(tmp_path, monkeypatch):
    mock_version(monkeypatch)
    calls = []
    def invoke(binary, payload):
        calls.append(payload)
        return bound(attempt([]), payload)
    monkeypatch.setattr(judge, "invoke", invoke)
    with pytest.raises(judge.InvalidSubmission):
        judge.judge_packet(packet(), {}, "content", tmp_path / "judge")
    assert len(calls) == 2
    assert len(list((tmp_path / "judge").glob("*-attempt-[01].json"))) == 2


def test_swapped_order_audit_does_not_change_primary_scores():
    items = [{"id": f"p-{i}", "A": "first", "B": "second", "scenario": "test"} for i in range(30)]
    original = copy.deepcopy(items)
    swapped = judge.reversed_audit(items)
    assert len(swapped) == 3 and all(r["A"] == "second" and r["B"] == "first" for r in swapped)
    assert items == original
    primary = {"ratings": [{"id": r["id"], "preference": "A"} for r in items]}
    diagnostic = {"ratings": [{"id": r["id"], "preference": "B"} for r in swapped]}
    result = judge.audit_diagnostic(primary, diagnostic)
    assert result["diagnostic_only"] and result["primary_scores_modified"] is False
    assert result["swapped_agreement"] == 1 and result["primary_A_preference_share"] == 1


def test_single_dummy_audit_records_actual_model_without_experiment_data(tmp_path, monkeypatch):
    mock_version(monkeypatch)
    calls = []
    def invoke(binary, payload):
        calls.append(payload)
        return bound({"returncode": 0, "stderr": "", "isolation": ISOLATION, "stdout": json.dumps({"subtype": "success",
            "modelUsage": {judge.MODEL: {}}, "result": json.dumps({"project_identity_visible": False,
                "project_files_visible": False, "project_memory_visible": False, "explanation": "test isolation"})})}, payload)
    monkeypatch.setattr(judge, "invoke", invoke)
    result = judge.isolation_audit(tmp_path / "audit")
    assert result["passed"] and result["single_call"] and len(calls) == 1
    assert "items" not in calls[0] and result["experiment_ratings_requested"] is False
    assert result["actual_model_usage_ids"] == [judge.MODEL]


def test_version_pin_is_enforced(monkeypatch):
    monkeypatch.setattr(judge.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="2.0.0 (Claude Code)"))
    with pytest.raises(ValueError, match="version"):
        judge.version("claude")


def test_single_markdown_fence_is_parsed_without_selecting_embedded_candidates():
    value = attempt(ratings(packet()))
    outer = json.loads(value["stdout"])
    outer["result"] = "```json\n" + outer["result"] + "\n```"
    value["stdout"] = json.dumps(outer)
    valid, _ = judge.submission(value, packet(), "content")
    assert len(valid["ratings"]) == 2
    outer["result"] = "comment\n" + outer["result"]
    value["stdout"] = json.dumps(outer)
    with pytest.raises(judge.InvalidSubmission):
        judge.submission(value, packet(), "content")


def test_offline_reparse_retains_original_failure_and_makes_no_call(tmp_path, monkeypatch):
    source = tmp_path / "old"
    source.mkdir()
    previous = {"argv": judge.argv("claude"), "cli_version": "2.1.323 (Claude Code)", "passed": False,
                "wrapper_sha256": "a"*64, "single_call": True}
    (source / "audit.json").write_text(json.dumps(previous))
    answer = {"project_identity_visible": False, "project_files_visible": False, "project_memory_visible": False}
    outer = {"subtype": "success", "modelUsage": {judge.MODEL: {}}, "result": "```json\n"+json.dumps(answer)+"\n```"}
    (source / "attempt.json").write_text(json.dumps({"returncode": 0, "stdout": json.dumps(outer), "stderr": ""}))
    def forbidden(*args, **kwargs):
        raise AssertionError("offline reparse must never invoke Claude")
    monkeypatch.setattr(judge, "invoke", forbidden)
    record = judge.reparse_audit(source, tmp_path / "parsed")
    assert record["passed"] and record["original_parser_failed"] and record["audit_reparsed_without_new_call"]
    assert json.loads((source / "audit.json").read_text())["passed"] is False
    assert record["historical_only"]
    with pytest.raises(ValueError, match="new final-wrapper"):
        judge.validate_audit(tmp_path / "parsed/audit.json")


def test_git_repository_check_blocks_the_call(monkeypatch):
    calls = []
    def fake(argv, **kwargs):
        calls.append(argv)
        assert argv == ["git", "rev-parse", "--git-dir"]
        return SimpleNamespace(returncode=0, stdout=".git", stderr="")
    monkeypatch.setattr(judge.subprocess, "run", fake)
    result = judge.invoke("claude", {})
    assert len(calls) == 1 and result["call_started"] is False
    assert result["isolation"]["git_not_repository"] is False


def test_real_git_is_not_a_repository_with_project_git_variables_removed(tmp_path, monkeypatch):
    from scripts.common import ROOT
    monkeypatch.setenv("GIT_DIR", str(ROOT / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(ROOT))
    env = judge.environment(tmp_path)
    assert not any(k.startswith("GIT_") for k in env)
    result = judge.subprocess.run(["git", "rev-parse", "--git-dir"], cwd=tmp_path, env=env,
                                  text=True, capture_output=True, shell=False)
    assert result.returncode == 128 and "not a git repository" in result.stderr.lower()


def valid_invoker(monkeypatch):
    mock_version(monkeypatch)
    calls = []
    def invoke(binary, payload):
        calls.append(copy.deepcopy(payload))
        return bound(attempt(ratings(payload["items"])), payload)
    monkeypatch.setattr(judge, "invoke", invoke)
    return calls


def test_completed_resume_replays_without_new_calls_or_changed_attempts(tmp_path, monkeypatch):
    calls = valid_invoker(monkeypatch)
    directory = tmp_path / "judge"
    first = judge.judge_packet(packet(21), {}, "content", directory, source_hashes={"raw_packet": "a"*64})
    before = {p.name: p.read_bytes() for p in directory.glob("*.json")}
    second = judge.judge_packet(packet(21), {}, "content", directory, resume=True, source_hashes={"raw_packet": "a"*64})
    assert first == second and len(calls) == 2
    assert {p.name: p.read_bytes() for p in directory.glob("*.json")} == before


def test_interruption_after_saved_valid_attempt_reuses_it(tmp_path, monkeypatch):
    calls = valid_invoker(monkeypatch)
    original = judge.save_or_match
    state = {"interrupt": True}
    def interrupted(filename, value):
        if Path(filename).name == "batch-000-provenance.json" and state["interrupt"]:
            state["interrupt"] = False
            raise KeyboardInterrupt("unit-test crash after valid response was saved")
        return original(filename, value)
    monkeypatch.setattr(judge, "save_or_match", interrupted)
    directory = tmp_path / "judge"
    with pytest.raises(KeyboardInterrupt):
        judge.judge_packet(packet(), {}, "content", directory)
    raw = (directory / "batch-000-attempt-0.json").read_bytes()
    value = judge.judge_packet(packet(), {}, "content", directory, resume=True)
    assert len(calls) == 1 and len(value["ratings"]) == 2
    assert (directory / "batch-000-attempt-0.json").read_bytes() == raw
    assert (directory / "run-interruption-000.json").exists()


def test_interrupted_between_batches_calls_only_remaining_batch(tmp_path, monkeypatch):
    calls = valid_invoker(monkeypatch)
    original = judge.replay_or_call
    state = {"interrupt": True}
    def interrupted(output, batch_index, *args):
        if batch_index == 1 and state["interrupt"]:
            state["interrupt"] = False
            raise KeyboardInterrupt("unit-test interruption before starting next batch")
        return original(output, batch_index, *args)
    monkeypatch.setattr(judge, "replay_or_call", interrupted)
    directory = tmp_path / "judge"
    with pytest.raises(KeyboardInterrupt):
        judge.judge_packet(packet(21), {}, "content", directory)
    assert len(calls) == 1
    result = judge.judge_packet(packet(21), {}, "content", directory, resume=True)
    assert len(calls) == 2 and len(calls[-1]["items"]) == 1 and len(result["ratings"]) == 21


def test_unknown_inflight_attempt_is_preserved_without_automatic_retry(tmp_path, monkeypatch):
    mock_version(monkeypatch)
    calls = []
    def interrupted(binary, payload):
        calls.append(payload)
        raise KeyboardInterrupt("unit-test interrupted service call; outcome unknown")
    monkeypatch.setattr(judge, "invoke", interrupted)
    directory = tmp_path / "judge"
    with pytest.raises(KeyboardInterrupt):
        judge.judge_packet(packet(), {}, "content", directory)
    with pytest.raises(judge.UncertainAttempt):
        judge.judge_packet(packet(), {}, "content", directory, resume=True)
    assert len(calls) == 1
    assert (directory / "batch-000-attempt-0-start.json").exists()
    assert (directory / "batch-000-attempt-0-interruption.json").exists()


@pytest.mark.parametrize("changed", ["packet", "rubric", "raw_hash", "wrapper", "attempt"])
def test_resume_binds_exact_inputs_and_saved_attempt_hashes(tmp_path, monkeypatch, changed):
    calls = valid_invoker(monkeypatch)
    directory = tmp_path / "judge"
    judge.judge_packet(packet(), {}, "content", directory, source_hashes={"raw_packet": "a"*64})
    items, rubric, sources = packet(), {}, {"raw_packet": "a"*64}
    if changed == "packet":
        items[0]["response"] += " different"
    elif changed == "rubric":
        rubric["different"] = True
    elif changed == "raw_hash":
        sources["raw_packet"] = "b"*64
    elif changed == "wrapper":
        original = judge.stream_hash
        monkeypatch.setattr(judge, "stream_hash", lambda p: "0"*64 if Path(p) == Path(judge.__file__) else original(p))
    else:
        p = directory / "batch-000-attempt-0.json"
        p.write_text(p.read_text()+" ")
    with pytest.raises(ValueError):
        judge.judge_packet(items, rubric, "content", directory, resume=True, source_hashes=sources)
    assert len(calls) == 1


def test_missing_receipt_after_saved_valid_response_can_be_recovered(tmp_path, monkeypatch):
    calls = valid_invoker(monkeypatch)
    directory = tmp_path / "judge"
    judge.judge_packet(packet(), {}, "content", directory)
    (directory / "batch-000-attempt-0-receipt.json").unlink()
    judge.judge_packet(packet(), {}, "content", directory, resume=True)
    assert len(calls) == 1 and (directory / "batch-000-attempt-0-receipt.json").exists()


def test_final_audit_is_bound_to_wrapper_flags_and_git_checked_attempt(tmp_path, monkeypatch):
    mock_version(monkeypatch)
    answer = {"project_identity_visible": False, "project_files_visible": False, "project_memory_visible": False}
    def invoke(binary, payload):
        value = attempt([])
        outer = json.loads(value["stdout"])
        outer["result"] = json.dumps(answer)
        value["stdout"] = json.dumps(outer)
        return bound(value, payload)
    monkeypatch.setattr(judge, "invoke", invoke)
    directory = tmp_path / "audit"
    judge.isolation_audit(directory)
    assert judge.validate_audit(directory / "audit.json")["passed"]
    report = json.loads((directory / "audit.json").read_text())
    report["wrapper_sha256"] = "0"*64
    (directory / "audit.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="final-wrapper"):
        judge.validate_audit(directory / "audit.json")


def test_resume_keeps_malformed_attempt_and_uses_only_remaining_retry(tmp_path, monkeypatch):
    mock_version(monkeypatch)
    calls = []
    def invoke(binary, payload):
        calls.append(payload)
        return bound(attempt([] if len(calls) == 1 else ratings(payload["items"])), payload)
    monkeypatch.setattr(judge, "invoke", invoke)
    original = judge.replay_or_call
    state = {"interrupt": True}
    def interrupted(output, batch_index, attempt_index, *args):
        if attempt_index == 1 and state["interrupt"]:
            state["interrupt"] = False
            raise KeyboardInterrupt("unit-test interruption between allowed attempts")
        return original(output, batch_index, attempt_index, *args)
    monkeypatch.setattr(judge, "replay_or_call", interrupted)
    directory = tmp_path / "judge"
    with pytest.raises(KeyboardInterrupt):
        judge.judge_packet(packet(), {}, "content", directory)
    before = (directory / "batch-000-attempt-0.json").read_bytes()
    result = judge.judge_packet(packet(), {}, "content", directory, resume=True)
    assert len(calls) == 2 and len(result["ratings"]) == 2
    assert (directory / "batch-000-attempt-0.json").read_bytes() == before


def test_concurrent_resume_is_blocked_before_any_call(tmp_path, monkeypatch):
    calls = valid_invoker(monkeypatch)
    directory = tmp_path / "judge"
    judge.judge_packet(packet(), {}, "content", directory)
    with judge.exclusive_run(directory):
        with pytest.raises(RuntimeError, match="another judge"):
            judge.judge_packet(packet(), {}, "content", directory, resume=True)
    assert len(calls) == 1


def test_saved_valid_stdin_binding_is_checked_even_without_receipt(tmp_path, monkeypatch):
    calls = valid_invoker(monkeypatch)
    directory = tmp_path / "judge"
    judge.judge_packet(packet(), {}, "content", directory)
    (directory / "batch-000-attempt-0-receipt.json").unlink()
    filename = directory / "batch-000-attempt-0.json"
    value = json.loads(filename.read_text())
    value["canonical_stdin_sha256"] = "0"*64
    filename.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="stdin hash"):
        judge.judge_packet(packet(), {}, "content", directory, resume=True)
    assert len(calls) == 1
