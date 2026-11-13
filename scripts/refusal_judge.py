"""Refusal schema adapter over the frozen, audited tools-off call/journal foundation.

No monkeypatching or changes to tone_judge. No GPU. Audit and actual judging each
require explicit execution flags and coordinator authorization outside this CLI.
"""
import argparse
from pathlib import Path

from scripts import tone_judge as foundation
from scripts.common import path, read_json
from scripts.refusal_common import validate_submission
from scripts.save_nf4 import stream_hash
from scripts.tone_steering import committed


def code_hashes(rubric_file):
    from scripts import refusal_common
    return {"adapter_sha256": stream_hash(Path(__file__)), "validator_sha256": stream_hash(Path(refusal_common.__file__)),
            "foundation_sha256": stream_hash(Path(foundation.__file__)), "rubric_sha256": stream_hash(path(rubric_file))}


def audit(output, rubric_file, binary="claude"):
    foundation.isolation_audit(output, binary)  # One fresh dummy call, no response packet.
    binding = {**code_hashes(rubric_file), "foundation_audit_sha256": stream_hash(Path(output)/"audit.json"),
               "fresh_refusal_audit": True}
    foundation.save(Path(output)/"refusal_bindings.json", binding)
    return binding


def verify_audit(output, rubric_file, registration, binary="claude"):
    output = path(output)
    foundation.validate_audit(output/"audit.json", binary)
    expected = {**code_hashes(rubric_file), "foundation_audit_sha256": stream_hash(output/"audit.json"),
                "fresh_refusal_audit": True}
    if read_json(output/"refusal_bindings.json") != expected or registration != expected:
        raise ValueError("fresh refusal audit, rubric, adapter/validator/foundation hashes or registered binding differ")
    return expected


def parse_submission(attempt, items):
    outer, actual = foundation.outer_result(attempt)
    try:
        result = foundation.result_json(outer.get("result"))
        if not isinstance(result, dict) or set(result) != {"ratings"}:
            raise ValueError("ratings-only JSON required")
        value = {"judge": {"model_identifier": foundation.MODEL, "fresh_context": True, "advisor_overlap": False,
                   "exposed_settings": {"requested_effort": "medium", "actual_effort": None,
                                        "actual_model_usage_ids": actual, "dated_snapshot": None}}, "ratings": result["ratings"]}
        return validate_submission(items, value), actual
    except (ValueError, TypeError, KeyError) as exc:
        raise foundation.InvalidSubmission("malformed/incomplete refusal schema") from exc


def judge_packet(items, rubric, output, sources, binary="claude", resume=False):
    if not items or len({r["id"] for r in items}) != len(items) or any(set(r) != {"id","prompt","response"} for r in items):
        raise ValueError("nonempty unique anonymous prompt/response items required")
    output = Path(output)
    binding = {"schema_version": 1, "role": "refusal-response-classification", "source_sha256": sources,
        "packet_sha256": foundation.canonical_hash(items), "rubric_sha256": foundation.canonical_hash(rubric),
        "cli_version": foundation.version(binary), "argv": foundation.argv(binary), "maximum_attempts_per_batch": 2}
    payloads = [{"rubric": rubric, "items": items[i:i+20], "return_schema": {"ratings":
        "Exactly one record per anonymous ID with refusal_class, unscorable, useful_and_correct; no judge metadata."}}
        for i in range(0,len(items),20)]
    binding["batch_ids"] = [[r["id"] for r in p["items"]] for p in payloads]
    binding["batch_stdin_sha256"] = [foundation.canonical_hash(p) for p in payloads]
    if output.exists():
        if not resume or not (output/"run_binding.json").is_file():
            raise FileExistsError("fresh output or explicit exact-bound resume required")
    else:
        if resume:
            raise ValueError("resume requires existing bound state")
        output.mkdir(parents=True,mode=0o700)
    with foundation.exclusive_run(output):
        foundation.save_or_match(output/"run_binding.json",binding)
        try:
            ratings, models, batches = [], set(), []
            for index,payload in enumerate(payloads):
                batch = {"batch_index":index,"ids":[r["id"] for r in payload["items"]],
                         "canonical_stdin_sha256":foundation.canonical_hash(payload),"attempt_files":[],"valid":False}
                foundation.save_or_match(output/f"batch-{index:03d}-input.json", {
                    "run_binding_sha256":stream_hash(output/"run_binding.json"), **{k:batch[k] for k in ("ids","canonical_stdin_sha256")}})
                for attempt_index in range(2):
                    attempt,receipt = foundation.replay_or_call(output,index,attempt_index,payload,
                                                               stream_hash(output/"run_binding.json"),binary)
                    batch["attempt_files"].append(receipt)
                    try:
                        valid,actual = parse_submission(attempt,payload["items"])
                    except foundation.InvalidSubmission:
                        if attempt_index == 1:
                            foundation.save_or_match(output/f"batch-{index:03d}-provenance.json",batch)
                            raise
                        continue
                    ratings.extend(valid["ratings"])
                    models.update(actual)
                    batch.update({"valid":True,"first_valid_attempt":attempt_index,"actual_model_usage_ids":actual})
                    break
                foundation.save_or_match(output/f"batch-{index:03d}-provenance.json",batch)
                batches.append(batch)
            value = {"judge":{"model_identifier":foundation.MODEL,"fresh_context":True,"advisor_overlap":False,
                     "exposed_settings":{"requested_effort":"medium","actual_effort":None,
                                          "actual_model_usage_ids":sorted(models),"dated_snapshot":None}},"ratings":ratings}
            validate_submission(items,value)
            foundation.save_or_match(output/"submission.json",value)
            foundation.save_or_match(output/"provenance.json", {"source_sha256":sources,"batches":batches,
                "run_binding_sha256":stream_hash(output/"run_binding.json"),"submission_sha256":stream_hash(output/"submission.json"),
                "requested_model":foundation.MODEL,"actual_model_usage_ids":sorted(models),"first_valid_only":True,
                "maximum_attempts_per_batch":2,"tools_enabled":False})
            return value
        except BaseException as exc:
            index=len(list(output.glob("run-interruption-*.json")))
            foundation.save(output/f"run-interruption-{index:03d}.json",{
                "error_type":type(exc).__name__,"run_binding_sha256":stream_hash(output/"run_binding.json")})
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("validate","audit","judge"))
    parser.add_argument("--packet")
    parser.add_argument("--rubric",default="evaluation/refusal/response_rubric.json")
    parser.add_argument("--output",required=True)
    parser.add_argument("--audit")
    parser.add_argument("--registration")
    parser.add_argument("--binary",default="claude")
    parser.add_argument("--execute-audit",action="store_true")
    parser.add_argument("--execute-judgments",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.stage=="audit":
        if not args.execute_audit:
            raise ValueError("fresh dummy audit requires separate approval and --execute-audit")
        audit(path(args.output),args.rubric,args.binary)
        return
    if not args.packet:
        parser.error("packet required")
    packet=read_json(args.packet)
    if packet["rubric_sha256"]!=stream_hash(path(args.rubric)):
        raise ValueError("packet/rubric hash mismatch")
    if args.stage=="validate":
        print("Refusal packet/rubric hashes valid; no call made.")
        return
    if not args.execute_judgments or not args.audit or not args.registration:
        raise ValueError("separate judging approval, fresh audit and committed registration required")
    committed(args.registration)
    verified=verify_audit(args.audit,args.rubric,read_json(args.registration),args.binary)
    sources={**verified,"packet_file":stream_hash(path(args.packet)),"judge_registration":stream_hash(path(args.registration))}
    judge_packet(packet["items"],read_json(args.rubric),path(args.output),sources,args.binary,args.resume)


if __name__=="__main__":
    main()
