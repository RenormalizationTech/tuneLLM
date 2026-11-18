import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.qwen38_science_data import prepare_qasper
from scripts.qwen38_science_score import answer_f1, evidence_scores, exact_paragraph_f1, normalize_answer, paired_paper_bootstrap, parse_response, score_qasper, score_question


def row(unanswerable=False, primary=True, extra_refs=None):
    ref = {"unanswerable": unanswerable, "answer": "" if unanswerable else "quantum response", "evidence": [] if unanswerable else ["p000000"]}
    return {"id": "p::q", "paper_id": "p", "primary": primary, "nontext_or_unmatched_evidence": False,
            "references": [ref] + (extra_refs or []), "context_ids": ["p000000"],
            "paragraphs": [{"id": "p000000"}, {"id": "p000001"}]}


def response(answer="quantum response", evidence=None, unanswerable=False):
    return json.dumps({"answer": answer, "evidence": evidence if evidence is not None else ["p000000"], "unanswerable": unanswerable})


def test_answer_normalization_and_duplicate_tokens():
    assert normalize_answer("The Quantum, an RESPONSE!") == "quantum response"
    assert answer_f1("x x y", "x y y") == pytest.approx(2 / 3)
    assert answer_f1("", "") == 0.0
    assert answer_f1("the", "a") == 0.0
    assert normalize_answer("naïve—quantum") == "naïve—quantum"


def test_evidence_precision_recall_and_empty_sets():
    assert evidence_scores(["p1", "p2"], ["p2"]) == {"precision": .5, "recall": 1, "f1": 2 / 3}
    assert evidence_scores([], ["p1"])["f1"] == 0
    assert evidence_scores([], [])["f1"] == 1
    assert evidence_scores(["p1", "p1"], ["p1"])["f1"] == 1


@pytest.mark.parametrize("text", [
    '{"answer":"yes","unanswerable":"false","evidence":[]}',
    '{"answer":"yes","unanswerable":false,"evidence":[],"extra":1}',
    '{"answer":"yes","answer":"no","unanswerable":false,"evidence":[]}',
    '{"answer":"","unanswerable":false,"evidence":[]}',
    '{"answer":"yes","unanswerable":true,"evidence":[]}',
    '{"answer":"yes","unanswerable":false,"evidence":["p000000","p000000"]}',
    '{"answer":"yes","unanswerable":false,"evidence":[1]}',
    '{"answer":"yes","unanswerable":false,"evidence":["invented"]}',
    '[]', '', None,
])
def test_strict_schema_rejects_invalid_outputs(text):
    assert parse_response(text) is None


def test_lenient_frozen_parser_recovers_without_inventing_evidence():
    assert parse_response("```json\n" + response() + "\n```") is None
    assert parse_response("```json\n" + response() + "\n```", True)["answer"] == "quantum response"
    assert parse_response("Here is the answer: " + response(), True)["evidence"] == ["p000000"]
    assert parse_response("quantum response", True)["evidence"] == []
    assert parse_response("Unanswerable", True)["unanswerable"]
    assert not parse_response("It might be unanswerable", True)["unanswerable"]
    assert parse_response('{"broken":', True) is None
    conflicting = response("Unanswerable", ["p000000"], True)
    assert parse_response(conflicting) is None
    assert parse_response(conflicting, True) == {"answer": "", "unanswerable": True, "evidence": []}


def test_unanswerable_adaptation_and_independent_reference_maxima():
    score = score_question(row(True), response("", [], True))
    assert score["answer_f1"] == score["evidence_f1"] == 1
    r = row(extra_refs=[{"answer": "different", "unanswerable": False, "evidence": ["p000001"]}])
    score = score_question(r, response(evidence=["p000001"]))
    assert score["answer_f1"] == score["evidence_f1"] == 1
    assert score["unsupported_evidence_ids"] == ["p000001"]
    assert not score["out_of_document_evidence_ids"]


def test_unknown_evidence_does_not_disappear():
    score = score_question(row(), response(evidence=["p999999"]))
    assert score["evidence_f1"] == 0
    assert score["out_of_document_evidence_ids"] == ["p999999"]


def test_unmapped_text_reference_cannot_reward_empty_evidence():
    r = row()
    r["references"][0].update(evidence=[], unmatched_text_evidence=["not present in source"])
    assert score_question(r, response(evidence=[]))["evidence_f1"] == 0


def test_exact_evidence_view_keeps_whitespace_heading_and_float_semantics():
    p = {"id": "paper", "full_text": [{"section_name": "Results", "paragraphs": ["quantum   response", "quantum   response"]}],
         "qas": [{"question_id": "q", "question": "response", "answers": [{"answer": {
             "unanswerable": False, "yes_no": None, "extractive_spans": [], "free_form_answer": "result",
             "evidence": ["quantum response", "Results", "FLOAT SELECTED: table"]}}]}]}
    prepared = prepare_qasper(p, "validation", len, 10000)[0]
    assert prepared["references"][0]["unmatched_text_evidence"] == ["Results"]
    scored = score_question(prepared, response("result", ["p000001"]))
    assert scored["evidence_f1"] == pytest.approx(2 / 3)  # one content item + unmatched heading
    assert scored["exact_evidence_text_f1"] == scored["exact_evidence_all_f1"] == 0
    official = upstream()
    for text_only, key in [(True, "exact_evidence_text_f1"), (False, "exact_evidence_all_f1")]:
        expected = official.evaluate(official.get_answers_and_evidence({"paper": p}, text_only),
            {"q": {"answer": "result", "evidence": ["quantum   response"]}})
        assert scored[key] == pytest.approx(expected["Evidence F1"])
    prepared["references"][0]["raw_annotation"]["evidence"] = ["quantum   response", "FLOAT SELECTED: table"]
    scored = score_question(prepared, response("result", ["p000001"]))
    assert scored["exact_evidence_text_f1"] == 1
    assert scored["exact_evidence_all_f1"] == pytest.approx(2 / 3)
    for text_only, key in [(True, "exact_evidence_text_f1"), (False, "exact_evidence_all_f1")]:
        expected = official.evaluate(official.get_answers_and_evidence({"paper": p}, text_only),
            {"q": {"answer": "result", "evidence": ["quantum   response"]}})
        assert scored[key] == pytest.approx(expected["Evidence F1"])


def test_evidence_strict_gate_denominator_excludes_empty_reference_reward():
    r = row(extra_refs=[{"answer": "other", "unanswerable": False, "evidence": []}])
    r["evidence_strict_primary"] = False
    report = score_qasper([r], {r["id"]: response(evidence=[])})
    assert report["strict"]["groups"]["primary"]["evidence_f1"] == 1
    gate = report["strict"]["groups"]["evidence_strict_primary"]
    assert gate["denominator"] == 0 and gate["evidence_f1"] is None


def test_paired_bootstrap_clusters_items_and_preserves_pairs():
    # Unequal paper sizes make item bootstrap / equal-paper averaging distinguishable.
    a = [{"id": str(i), "paper_id": p, "answer_f1": v} for i, (p, v) in enumerate([
        ("large", 1), ("large", 1), ("large", 1), ("small", 0)])]
    b = [{**r, "answer_f1": 0} for r in a]
    result = paired_paper_bootstrap(a, b, draws=1000)
    assert result["difference_a_minus_b"] == .75
    assert result["ci95"] == [0, 1]
    assert result["papers"] == 2 and result["items"] == 4
    assert paired_paper_bootstrap(a[::-1], b[::-1], draws=1000) == result
    identical = paired_paper_bootstrap(a, a, draws=100)
    assert identical["ci95"] == [0, 0] and identical["difference_a_minus_b"] == 0


def test_paired_bootstrap_rejects_unpaired_nonfinite_and_empty_inputs():
    a = [{"id": "q", "paper_id": "p", "answer_f1": 1}]
    for b in ([], [{**a[0], "paper_id": "wrong"}], [{**a[0], "answer_f1": float("nan")}], a * 2):
        with pytest.raises(ValueError):
            paired_paper_bootstrap(a, b, draws=10)
    with pytest.raises(ValueError):
        paired_paper_bootstrap(a, a, draws=0)


def test_missing_invalid_and_secondary_examples_stay_in_denominators():
    rows = [row(), {**row(True), "id": "p::missing"}, {**row(primary=False), "id": "p::secondary"}]
    report = score_qasper(rows, {"p::q": "quantum response", "p::secondary": response("", [], True)})
    assert report["missing_predictions"] == 1
    assert report["strict"]["groups"]["all_sampled"]["denominator"] == 3
    assert report["strict"]["groups"]["primary"]["denominator"] == 2
    assert report["strict"]["groups"]["all_sampled"]["invalid_outputs"] == 2
    assert report["lenient"]["groups"]["primary"]["answer_f1"] == .5
    assert report["strict"]["groups"]["primary"]["false_unanswerable_rate"] == 0
    assert report["strict"]["groups"]["all_sampled"]["false_unanswerable_rate"] == .5
    assert not report["official_equivalence"]


def test_extra_prediction_and_duplicate_question_rejected():
    with pytest.raises(ValueError, match="unknown"):
        score_qasper([row()], {"bogus": response()})
    with pytest.raises(ValueError, match="duplicate"):
        score_qasper([row(), row()], {})


def test_scoring_cli_records_hashes_and_refuses_overwrite(tmp_path):
    prepared, predictions, output = [tmp_path / name for name in ("prepared.jsonl", "predictions.json", "scores.json")]
    prepared.write_text(json.dumps(row()) + "\n")
    predictions.write_text(json.dumps({"p::q": response()}))
    command = [sys.executable, "-B", "-m", "scripts.qwen38_science_score", "--prepared", str(prepared),
               "--predictions", str(predictions), "--output", str(output)]
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["strict"]["groups"]["primary"]["answer_f1"] == 1
    assert len(report["input_sha256"]["predictions"]) == 64
    assert subprocess.run(command, cwd=root, capture_output=True).returncode != 0


def upstream():
    path = Path(__file__).resolve().parents[1] / "data/qasper-v03-01/original/qasper_evaluator.py"
    if not path.exists():
        pytest.skip("optional acquired official QASPER evaluator absent")
    # Synthetic inputs only; no dataset test records are read or tuned against.
    import hashlib
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "781aba7cd8e524bef4f0a1b4bf3504e5b02cb1d8d5bf32a8f0a89dfa83e86bfe"
    spec = importlib.util.spec_from_file_location("qasper_upstream_parity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upstream_normalization_and_f1_parity_on_synthetic_edge_cases():
    official = upstream()
    values = ["", "The", "A, an!", "x x y", "x y y", "No", "Unanswerable", "naïve—quantum", "3.14 K"]
    for left in values:
        assert normalize_answer(left) == official.normalize_answer(left)
        for right in values:
            assert answer_f1(left, right) == pytest.approx(official.token_f1_score(left, right))
    for left, right in [([], []), (["p1"], []), ([], ["p1"]), (["p1", "p2"], ["p2"]), (["p1"], ["p2"])]:
        assert evidence_scores(left, right)["f1"] == pytest.approx(official.paragraph_f1_score(left, right))
    assert exact_paragraph_f1(["x", "x"], ["x"]) == pytest.approx(official.paragraph_f1_score(["x", "x"], ["x"]))


def test_upstream_multi_reference_typed_answers_and_float_filter_parity():
    official = upstream()
    annotations = [
        {"unanswerable": False, "extractive_spans": ["quantum", "response"], "free_form_answer": "", "yes_no": None, "evidence": ["Quantum paragraph"]},
        {"unanswerable": False, "extractive_spans": [], "free_form_answer": "a response", "yes_no": None, "evidence": ["Other paragraph", "FLOAT SELECTED: figure"]},
    ]
    for answer in (False, True):
        annotations.append({"unanswerable": False, "extractive_spans": [], "free_form_answer": "", "yes_no": answer, "evidence": ["Other paragraph"]})
    annotations.append({"unanswerable": True, "extractive_spans": [], "free_form_answer": "", "yes_no": None, "evidence": []})
    paper = {"id": "paper", "full_text": [{"section_name": "Body", "paragraphs": ["Quantum paragraph", "Other paragraph"]}],
             "qas": [{"question_id": "q", "question": "What response?", "answers": [{"answer": a} for a in annotations]}]}
    prepared = prepare_qasper(paper, "validation", len, 10000)[0]
    gold = official.get_answers_and_evidence({"paper": paper}, True)
    assert gold["q"][1]["evidence"] == ["Other paragraph"]
    for answer, abstain in [("quantum, response", False), ("No", False), ("Yes", False), ("", True)]:
        raw = response(answer, [] if abstain else ["p000001"], abstain)
        ours = score_question(prepared, raw)
        expected = official.evaluate(gold, {"q": {"answer": "Unanswerable" if abstain else answer,
                                                  "evidence": [] if abstain else ["Other paragraph"]}})
        assert ours["answer_f1"] == pytest.approx(expected["Answer F1"])
        assert ours["evidence_f1"] == pytest.approx(expected["Evidence F1"])
