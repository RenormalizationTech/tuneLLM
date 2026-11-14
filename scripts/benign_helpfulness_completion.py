"""Registered completion of ONLY the16 unaccepted baseline grades (two batches8).

Old20 labels and all historical artifacts remain immutable. No GPU, target-model
generation, old import/report bypass, new audit, or automatic follow-up. Only a
complete20+16 review can produce a decision, in a separate completion-aware report.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import re
import shutil

from scripts import benign_helpfulness as b
from scripts.common import path, read_json
from scripts.save_nf4 import stream_hash

f = b.foundation
save = f.save
SOURCE = "scripts/benign_helpfulness_completion.py"
SOURCES = tuple(dict.fromkeys((SOURCE, *b.SOURCES)))
PROTOCOL = "docs/benign_helpfulness_completion_protocol.md"
PUBLIC = "results/benign_helpfulness_completed_01.json"
BINARY_SHA = "604d29a6869409106b6a1110bdadcc8b72e8654a5a865d23258f938a696e348e"


def check_destinations(run, origin, output=None):
    run, origin = run.resolve(), origin.resolve()
    if not run.is_relative_to(path("outputs").resolve()) or run == origin or run.is_relative_to(origin) or origin.is_relative_to(run):
        raise ValueError("completion writes must be disjoint from the original run")
    if output is not None and (output.resolve().is_relative_to(origin) or output.resolve() != path(PUBLIC).resolve()):
        raise ValueError("only the separate completion public report may be written")


def check_hashes(files):
    for name, expected in files.items():
        if stream_hash(path(name)) != expected:
            raise ValueError("immutable source/input/snapshot/runtime/reuse/binary hash changed")


def validate_coordinator(draft, rows, items, keys, predictions):
    """Bind the pre-grading AI draft directly to original requests/predictions."""
    if (draft.get("ai_only") is not True or not isinstance(draft.get("reviewer"), str) or not draft["reviewer"].strip() or
            type(draft.get("primary_grades_seen_during_refusal_reading")) is not bool):
        raise ValueError("coordinator draft needs AI-only identity/prior-exposure disclosures")
    if (set(keys) != {i["id"] for i in items} or len(set(keys.values())) != 36 or
            set(keys.values()) != {r["id"] for r in rows}):
        raise ValueError("original packet/source-ID key alignment differs")
    by_id, predicted = {r["id"]: r for r in rows}, {p["id"]: p for p in predictions}
    for item in items:
        source = keys[item["id"]]
        if item != {"id": item["id"], "request": by_id[source]["user"], "anchors": by_id[source]["anchors"],
                    "response": predicted[source]["output"]}:
            raise ValueError("original packet content/source alignment differs")
    entries = draft.get("ratings", [])
    expected = {r["id"] for r in rows if r["class"] == "sensitive"}
    if (len(entries) != 24 or len({r["id"] for r in entries}) != 24 or {r["id"] for r in entries} != expected or
            any(r.get("refusal") not in ("none", "partial", "full") or
                not isinstance(r.get("reason"), str) or not r["reason"].strip() for r in entries)):
        raise ValueError("coordinator draft must cover exactly the original24 sensitive IDs")
    return {r["id"]: r["refusal"] for r in entries}


def read_attempt(directory, batch_index, attempt_index, payload, binding_hash, receipt):
    stem = f"batch-{batch_index:03d}-attempt-{attempt_index}"
    filename, start = directory/(stem+".json"), directory/(stem+"-start.json")
    expected = {"run_binding_sha256": binding_hash, "canonical_stdin_sha256": f.canonical_hash(payload)}
    if (receipt != {**expected, "file": filename.name, "sha256": stream_hash(filename), "start_sha256": stream_hash(start)} or
            read_json(directory/(stem+"-receipt.json")) != receipt or
            any(read_json(start).get(k) != v for k, v in expected.items())):
        raise ValueError("attempt receipt/start/payload/run binding differs")
    attempt = read_json(filename)
    if attempt.get("canonical_stdin_sha256") != expected["canonical_stdin_sha256"]:
        raise ValueError("attempt input hash differs")
    return b.parse_attempt(attempt, payload["items"])


def original(origin, coordinator):
    cfg, manifest = b.verify(origin)
    items = b.verify_packet(origin)
    rows, predictions = b.predictions(origin)
    rubric = read_json(origin/"inputs/rubric.json")
    keys = read_json(origin/"judging/keys.local.json")
    validate_coordinator(read_json(coordinator), rows, items, keys, predictions)
    recovery = read_json(origin/"runtime_recovery.json")
    if recovery["binary_sha256"] != BINARY_SHA or stream_hash(path(recovery["binary"])) != BINARY_SHA:
        raise ValueError("registered pinned executable differs")
    audit = f.validate_audit(cfg["judge_audit"])
    directory = origin/"grades"
    binding = read_json(directory/"run_binding.json")
    payloads = b.judge_payloads(items, rubric)
    if (len(items) != 36 or binding["batch_ids"] != [[i["id"] for i in p["items"]] for p in payloads] or
            binding["batch_stdin_sha256"] != [f.canonical_hash(p) for p in payloads] or
            binding["maximum_attempts_per_batch"] != 2 or binding["argv"] != f.argv("claude") or
            binding["cli_version"] != audit["cli_version"] or
            any(binding[k] != stream_hash(p) for k, p in {
                "manifest_sha256": origin/"manifest.json", "adapter_sha256": path("scripts/benign_helpfulness.py"),
                "foundation_sha256": path("scripts/tone_judge.py"), "packet_sha256": origin/"judging/packet.json",
                "rubric_sha256": origin/"inputs/rubric.json", "audit_sha256": path(cfg["judge_audit"])}.items())):
        raise ValueError("original audited grading/source binding differs")
    if (directory/"submission.json").exists() or (origin/"judging/accepted_review.json").exists():
        raise ValueError("original incomplete run must not acquire a fabricated full submission/import")
    retained, provenance = None, None
    for index, payload in enumerate(payloads):
        batch = read_json(directory/f"batch-{index:03d}-provenance.json")
        expected_input = {"batch_index": index, "ids": [i["id"] for i in payload["items"]],
                          "canonical_stdin_sha256": f.canonical_hash(payload), "attempt_files": []}
        if (read_json(directory/f"batch-{index:03d}-input.json") != expected_input or
                any(batch.get(k) != v for k, v in expected_input.items() if k != "attempt_files")):
            raise ValueError("original batch/input provenance differs")
        if index == 0:
            if batch.get("first_valid_attempt") != 0 or len(batch["attempt_files"]) != 1:
                raise ValueError("retain exact original first-valid20 only")
            retained, models = read_attempt(directory, index, 0, payload, stream_hash(directory/"run_binding.json"), batch["attempt_files"][0])
            if models != batch["actual_model_usage_ids"]: raise ValueError("original judge identity differs")
            provenance = {"attempt_sha256": batch["attempt_files"][0]["sha256"],
                "receipt_sha256": stream_hash(directory/"batch-000-attempt-0-receipt.json"),
                "batch_provenance_sha256": stream_hash(directory/"batch-000-provenance.json"), "actual_model_usage_ids": models}
        else:
            if batch.get("first_valid_attempt") is not None or len(batch["attempt_files"]) != 2:
                raise ValueError("original16 must be exhausted/unaccepted")
            for attempt_index, receipt in enumerate(batch["attempt_files"]):
                try:
                    read_attempt(directory, index, attempt_index, payload, stream_hash(directory/"run_binding.json"), receipt)
                except f.InvalidSubmission:
                    continue
                raise ValueError("original failed16 unexpectedly contains an accepted valid attempt")
    return cfg, manifest, items, rubric, retained, provenance


def payloads(items, rubric):
    if len(items) != 16 or len({i["id"] for i in items}) != 16:
        raise ValueError("completion requires exact16 unaccepted items")
    return [{**b.judge_payloads(batch, rubric)[0], "expected_count": 8, "expected_ids": [i["id"] for i in batch]}
            for batch in (items[:8], items[8:])]


def prepare(args):
    run, origin = path(args.run), path(args.origin)
    check_destinations(run, origin)
    if run.exists(): raise FileExistsError("fresh completion directory required")
    coordinator, protocol = path(args.coordinator), path(args.protocol)
    cfg, old_manifest, items, rubric, retained, old_provenance = original(origin, coordinator)
    # The old run/logs/docs/reviews and original315 are referenced, never rewritten.
    inputs = {str(p): stream_hash(p) for p in origin.rglob("*") if p.is_file()}
    for group in (old_manifest["external_model_sha256"], old_manifest["external_input_sha256"], old_manifest["sources"]):
        inputs.update({str(path(n)): h for n, h in group.items()})
    diagnosis = path("outputs/refusal-diagnosis-03")
    dm = read_json(diagnosis/"manifest.json")
    inputs.update(dm["input_sha256"])
    for n, h in dm["packet_files_sha256"].items(): inputs[str(diagnosis/n)] = h
    preserve = ("outputs/refusal-diagnosis-03/manifest.json", "docs/benign_helpfulness_execution.md",
        "results/benign_helpfulness_01.json", "results/protocols/benign-helpfulness-01.json",
        "results/protocols/benign-helpfulness-01-runtime.json", "outputs/benign-helpfulness-01-grading.log",
        "outputs/benign-helpfulness-01-generation.log", "outputs/benign-helpfulness-01-grading-pinned.log")
    for name in preserve: inputs[str(path(name))] = stream_hash(path(name))
    inputs[str(coordinator)] = stream_hash(coordinator)
    inputs[str(path(cfg["judge_audit"]))] = stream_hash(path(cfg["judge_audit"]))
    inputs[str(path(cfg["judge_audit"]).parent/"attempt.json")] = stream_hash(path(cfg["judge_audit"]).parent/"attempt.json")
    recovery = read_json(origin/"runtime_recovery.json")
    inputs[str(path(recovery["binary"]))] = BINARY_SHA
    check_hashes(inputs)
    run.mkdir(parents=True, mode=0o700)
    os.chmod(run, 0o700)
    save(run/"config.json", {"origin": str(origin), "coordinator": str(coordinator), "protocol": str(protocol),
        "binary": recovery["binary"], "judge_audit": cfg["judge_audit"], "batch_size": 8,
        "maximum_attempts_per_batch": 2, "maximum_remote_calls": 4})
    for name, source in (("protocol.md", protocol), ("rubric.json", origin/"inputs/rubric.json"), ("coordinator.json", coordinator)):
        target = run/"inputs"/name; target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
    save(run/"inputs/packet16.json", {"items": items[20:36], "original_packet_sha256": stream_hash(origin/"judging/packet.json")})
    save(run/"inputs/retained20.json", {"ratings": retained, "provenance": old_provenance})
    for name in SOURCES:
        target = run/"source"/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path(name), target)
    save(run/"manifest.json", {"files": {str(p.relative_to(run)): stream_hash(p) for p in run.rglob("*") if p.is_file()},
        "sources": {n: stream_hash(path(n)) for n in SOURCES}, "original_input_sha256": inputs,
        "protocol_sha256": stream_hash(protocol), "retained_count": 20, "new_count": 16,
        "payload_sha256": [f.canonical_hash(p) for p in payloads(items[20:36], rubric)],
        "old_provenance": old_provenance, "pinned_binary_sha256": BINARY_SHA,
        "old315_input_count": len(dm["input_sha256"]), "no_gpu_or_generation": True})
    check_hashes(inputs)
    print("Completion CPU snapshot ready:20 retained,16 pending in2x8; root registration required before any remote call.")


def verify(run):
    cfg, manifest = read_json(run/"config.json"), read_json(run/"manifest.json")
    check_destinations(run, path(cfg["origin"]))
    check_hashes({str(run/n): h for n, h in manifest["files"].items()})
    check_hashes(manifest["sources"])
    check_hashes(manifest["original_input_sha256"])
    if stream_hash(path(cfg["protocol"])) != manifest["protocol_sha256"]:
        raise ValueError("completion protocol changed")
    _, _, items, rubric, retained, provenance = original(path(cfg["origin"]), path(cfg["coordinator"]))
    if (read_json(run/"inputs/packet16.json")["items"] != items[20:36] or
            read_json(run/"inputs/retained20.json") != {"ratings": retained, "provenance": provenance} or
            manifest["payload_sha256"] != [f.canonical_hash(p) for p in payloads(items[20:36], rubric)]):
        raise ValueError("old20 changed or completion is not exact original16 in order")
    return cfg, manifest


def registration(args, run):
    cfg, manifest = verify(run)
    if not args.registration: raise ValueError("committed completion registration required")
    for name in (*SOURCES, cfg["protocol"], args.registration): b.committed(name)
    if read_json(args.registration).get("manifest_sha256") != stream_hash(run/"manifest.json"):
        raise ValueError("completion registration/manifest differs")
    resolved = shutil.which("claude")
    if not resolved or Path(resolved).resolve() != path(cfg["binary"]).resolve() or stream_hash(Path(resolved)) != BINARY_SHA:
        raise ValueError("use process-local PATH to the registered pinned native binary")
    audit = f.validate_audit(cfg["judge_audit"])
    if f.version("claude") != audit["cli_version"]:
        raise ValueError("pinned CLI version differs from unchanged audited version")
    return cfg, manifest


def grade_binding(run, batches, cfg):
    return {"manifest_sha256": stream_hash(run/"manifest.json"), "source_sha256": read_json(run/"manifest.json")["sources"],
        "binary_sha256": BINARY_SHA, "audit_sha256": stream_hash(path(cfg["judge_audit"])),
        "cli_version": "2.1.323 (Claude Code)", "argv": f.argv("claude"), "maximum_attempts_per_batch": 2,
        "maximum_remote_calls": 4, "batch_ids": [[i["id"] for i in p["items"]] for p in batches],
        "batch_stdin_sha256": [f.canonical_hash(p) for p in batches]}


def finish_batches(directory, batches, binding_hash):
    """Only orchestration added:2x8, existing atomic journal/first-valid calls."""
    if (directory/"terminal.json").exists() or (directory.parent/"terminal.json").exists():
        raise RuntimeError("completion is permanently stopped; no calls after exhaustion/fatal error")
    all_ratings, completed = [], []
    for index, payload in enumerate(batches):
        batch = {"batch_index": index, "ids": [i["id"] for i in payload["items"]],
            "canonical_stdin_sha256": f.canonical_hash(payload), "attempt_files": []}
        f.save_or_match(directory/f"batch-{index:03d}-input.json", batch)
        for attempt_index in range(2):
            if (directory/"terminal.json").exists() or (directory.parent/"terminal.json").exists():
                raise RuntimeError("terminal marker forbids every subsequent call/replay")
            attempt, receipt = f.replay_or_call(directory, index, attempt_index, payload, binding_hash, "claude")
            batch["attempt_files"].append(receipt)
            try: accepted, models = b.parse_attempt(attempt, payload["items"])
            except f.InvalidSubmission:
                if attempt_index == 1:
                    f.save_or_match(directory/f"batch-{index:03d}-provenance.json", batch)
                    f.save_or_match(directory/"terminal.json", {"error_type": "InvalidSubmission", "exhausted_batch": index})
                    raise
                continue
            all_ratings.extend(accepted)
            batch.update(first_valid_attempt=attempt_index, actual_model_usage_ids=models)
            break
        f.save_or_match(directory/f"batch-{index:03d}-provenance.json", batch)
        completed.append(batch)
    return all_ratings, completed


def judge(args, run):
    if not args.execute_judgments: raise ValueError("separate root release and --execute-judgments required")
    # Validate the write boundary before creating even a failure marker. A bad
    # old-run destination is rejected without ever writing into the old tree.
    check_destinations(run, path(args.origin))
    check_destinations(run, path("outputs/benign-helpfulness-01"))
    directory = run/"grades"
    if (directory/"terminal.json").exists() or (run/"terminal.json").exists():
        raise RuntimeError("completion is permanently stopped; no preflight or remote calls allowed")
    def terminal(exc):
        # The prepared root is writable even if creating/locking grades fails.
        if not (run/"terminal.json").exists():
            save(run/"terminal.json", {"error_type": type(exc).__name__})
    try:
        directory.mkdir(mode=0o700, exist_ok=True)
        with f.exclusive_run(directory):
            if (directory/"terminal.json").exists() or (run/"terminal.json").exists():
                raise RuntimeError("completion permanently stopped")
            # The terminal boundary includes registration, version, executable,
            # source and hash preflight failures, not only remote-call failures.
            cfg, _ = registration(args, run)
            batches = payloads(read_json(run/"inputs/packet16.json")["items"], read_json(run/"inputs/rubric.json"))
            binding = grade_binding(run, batches, cfg)
            if (directory/"run_binding.json").exists():
                if not args.resume: raise FileExistsError("exact-bound --resume required for existing journal")
            elif args.resume: raise ValueError("resume requires an existing bound journal")
            f.save_or_match(directory/"run_binding.json", binding)
            ratings, completed = finish_batches(directory, batches, stream_hash(directory/"run_binding.json"))
            b.validate_ratings([i for p in batches for i in p["items"]], ratings)
            verify(run)  # Source/input/binary checks AFTER the calls, before accepting.
            f.save_or_match(directory/"accepted16.json", {"ratings": ratings, "batches": completed,
                "run_binding_sha256": stream_hash(directory/"run_binding.json"), "first_valid_only": True})
    except BaseException as exc:
        terminal(exc)  # Includes lock failure, timeout, OSError, KeyboardInterrupt.
        save(run/f"interruption-{len(list(run.glob('interruption-*.json'))):03d}.json", {"error_type": type(exc).__name__})
        raise
    finally:
        try:
            verify(run)  # Includes source/input/binary failures after calls.
        except BaseException as exc:
            terminal(exc)
            raise


def new_coverage(run, cfg):
    """Offline proof of accepted new batches; failed partial ratings never count."""
    batches = payloads(read_json(run/"inputs/packet16.json")["items"], read_json(run/"inputs/rubric.json"))
    directory = run/"grades"
    if not (directory/"run_binding.json").exists(): return [], {}, [], False
    if read_json(directory/"run_binding.json") != grade_binding(run, batches, cfg):
        raise ValueError("new grading binding differs")
    binding_hash = stream_hash(directory/"run_binding.json")
    ratings, sources, reports, completed = [], {}, [], []
    for index, payload in enumerate(batches):
        filename = directory/f"batch-{index:03d}-provenance.json"
        if not filename.exists(): break
        batch = read_json(filename)
        expected_input = {"batch_index": index, "ids": [i["id"] for i in payload["items"]],
            "canonical_stdin_sha256": f.canonical_hash(payload), "attempt_files": []}
        if (read_json(directory/f"batch-{index:03d}-input.json") != expected_input or
                any(batch.get(k) != v for k, v in expected_input.items() if k != "attempt_files")):
            raise ValueError("new batch provenance/membership changed")
        first_valid = batch.get("first_valid_attempt")
        if first_valid is not None and (type(first_valid) is not int or first_valid not in (0, 1)):
            raise ValueError("invalid first-valid index")
        if len(batch["attempt_files"]) != (2 if first_valid is None else first_valid+1):
            raise ValueError("new batch attempt policy differs")
        for attempt_index, receipt in enumerate(batch["attempt_files"]):
            try: accepted, models = read_attempt(directory, index, attempt_index, payload, binding_hash, receipt)
            except f.InvalidSubmission:
                if attempt_index == first_valid: raise
                continue
            if attempt_index != first_valid or models != batch["actual_model_usage_ids"]:
                raise ValueError("saved first-valid batch changed")
            ratings.extend(accepted)
            for row in accepted:
                sources[row["id"]] = {"source": "new_completion", "batch_index": index, "first_valid_attempt": first_valid,
                    "attempt_sha256": receipt["sha256"], "receipt_sha256": stream_hash(directory/f"batch-{index:03d}-attempt-{attempt_index}-receipt.json"),
                    "actual_model_usage_ids": models}
        reports.append({"batch_index": index, "required": 8, "accepted": 8 if first_valid is not None else 0,
                        "attempts": len(batch["attempt_files"]), "first_valid_attempt": first_valid})
        if first_valid is None: break
        completed.append(batch)
    complete = False
    if (directory/"accepted16.json").exists():
        record = read_json(directory/"accepted16.json")
        if (len(ratings) != 16 or record != {"ratings": ratings, "batches": completed,
                "run_binding_sha256": binding_hash, "first_valid_only": True}):
            raise ValueError("new accepted16 does not reproduce complete first-valid journals")
        b.validate_ratings([i for p in batches for i in p["items"]], ratings)
        complete = True
    if (directory/"terminal.json").exists() or (run/"terminal.json").exists():
        complete = False  # Never turn a fatal registration into a complete report.
    return ratings, sources, reports, complete


def aggregate(items):
    return {"total": len(items), "refusal_labels": dict(Counter(r["refusal"] for r in items)),
        "counted_refusals": sum(r["counted_refusal"] for r in items),
        "anchors_satisfied": sum(sum(r["anchors"]) for r in items), "anchor_opportunities": 3*len(items),
        "fully_anchored": sum(all(r["anchors"]) for r in items),
        **{k: sum(r[k] for r in items) for k in (*b.FLAGS, "empty", "mechanically_degenerate", "ended_eos", "capped")},
        "total_tokens_excluding_eos": sum(r["tokens_excluding_eos"] for r in items)}


def public_privacy(result):
    def walk(value):
        if isinstance(value, dict):
            if {"reason", "user", "prompt", "request", "response", "output", "input_ids", "generated_ids"} & set(value):
                raise ValueError("private field in completion public report")
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(result)
    import json
    if re.search(r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|Bearer\s+\S+|/home/|/tmp/", json.dumps(result)):
        raise ValueError("private path/credential in public report")


def report(args, run):
    cfg, manifest = verify(run)
    output = path(args.output)
    origin = path(cfg["origin"])
    check_destinations(run, origin, output)
    if output.exists(): raise FileExistsError("new completion report required; no old report overwrite")
    retained = read_json(run/"inputs/retained20.json")
    added, sources, batch_reports, complete = new_coverage(run, cfg)
    result = {"phase": "A_baseline_grading_completion", "run_status": "complete" if complete else "completion_incomplete",
        "registered_decision": None, "coverage": {"retained_old": 20, "accepted_new": len(added), "required_total": 36},
        "failure": read_json(run/"terminal.json") if (run/"terminal.json").exists() else
                   read_json(run/"grades/terminal.json") if (run/"grades/terminal.json").exists() else None,
        "new_attempts_started": len(list((run/"grades").glob("batch-*-attempt-*-start.json"))),
        "new_attempts_with_saved_response": len(list((run/"grades").glob("batch-*-attempt-*-receipt.json"))),
        "completion_manifest_sha256": stream_hash(run/"manifest.json"), "source_sha256": manifest["sources"],
        "original_manifest_sha256": stream_hash(origin/"manifest.json"),
        "original_incomplete_report_sha256": stream_hash(path("results/benign_helpfulness_01.json")),
        "original_first_valid20_provenance": retained["provenance"], "new_batch_provenance": batch_reports,
        "original_coordinator_draft_sha256": stream_hash(run/"inputs/coordinator.json"),
        "runtime_pin": {"version": "2.1.323 (Claude Code)", "binary_sha256": BINARY_SHA,
            "historical_audit_binary_digest_available": False, "server_side_dated_snapshot": None},
        "generation_reused": 36, "new_generation": 0, "heldout_generation": 0,
        "limits": ["Old20 are retained exactly; the failed old16 attempts remain rejected. This completion is separately registered, not an extension of their exhausted retry allowance.",
            "Original20 and completion16 were judged at different times and batch contexts; the new16 use two batches8, with unchanged rubric/model/medium/tools-off flags. Server-side dated snapshot is unknown.",
            "Version-specific installed native executable is pinned now; no historical audit-time executable digest was recorded.",
            "Quality flags and anchors are descriptive only. Original EOS/headroom rules are unchanged, with coordinator uncertainty handling; no invented quality gate.",
            "No old full submission/import is fabricated. Original source, outputs, failure records and incomplete report remain unchanged. No GPU, new generation or automatic candidate."]}
    if complete:
        all_items = b.verify_packet(origin)
        combined = retained["ratings"] + added
        b.validate_ratings(all_items, combined)
        keys = read_json(origin/"judging/keys.local.json")
        for row in retained["ratings"]:
            sources[row["id"]] = {"source": "retained_old20", "batch_index": 0, "first_valid_attempt": 0,
                **{k: v for k, v in retained["provenance"].items() if k != "batch_provenance_sha256"}}
        rows, predictions = b.predictions(origin)
        details = b.measurements(rows, predictions, {keys[r["id"]]: r for r in combined})
        draft = read_json(run/"inputs/coordinator.json")
        coordinator = validate_coordinator(draft, rows, all_items, keys, predictions)
        result.update(registered_decision=b.decision(details, coordinator),
            overall=aggregate(details), by_class={k: aggregate([r for r in details if r["class"] == k]) for k in ("sensitive", "physics")},
            by_family={k: aggregate([r for r in details if r["family"] == k]) for k in b.FAMILIES},
            items=[{**d, "grade_provenance": sources[anon]} for d in details for anon, source_id in keys.items() if source_id == d["id"]],
            coordinator_review_is_AI_only=True,
            primary_grades_seen_during_coordinator_reading=draft["primary_grades_seen_during_refusal_reading"],
            accepted16_sha256=stream_hash(run/"grades/accepted16.json"))
    public_privacy(result)
    verify(run)  # Before publication, including original315 and binary again.
    save(output, result)
    check_hashes(manifest["original_input_sha256"])
    print("Completion report saved; status:", result["run_status"], "; no old files modified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "verify", "judge", "report"))
    parser.add_argument("--run", default="outputs/benign-helpfulness-completion-01")
    parser.add_argument("--origin", default="outputs/benign-helpfulness-01")
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--coordinator", default="outputs/benign-helpfulness-01/coordinator_review_draft.json")
    parser.add_argument("--registration", default="results/protocols/benign-helpfulness-completion-01.json")
    parser.add_argument("--output", default=PUBLIC)
    parser.add_argument("--execute-judgments", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); run = path(args.run)
    if args.stage == "prepare": prepare(args)
    elif args.stage == "verify":
        verify(run); print("Completion/source/binary/original-input hashes verified; no calls.")
    else: {"judge": judge, "report": report}[args.stage](args, run)


if __name__ == "__main__":
    main()
