import json

import pytest

from scripts.qwen38_prepare_inputs import (
    eligible_order, fit_item, measure_row, prepare_task, seeded_paper_prefix,
    prepare_fresh_prl,
)
from scripts.qwen38_science_data import canonical, sha
from scripts.qwen38_science_data import prepare_qasper


class Tokenizer:
    """Character tokenizer with one EOS token and an explicit chat boundary."""
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<EOS>"
    all_special_ids = [0, 1]

    def __call__(self, text, add_special_tokens=False):
        ids = []
        while text:
            if text.startswith(self.eos_token):
                ids.append(1)
                text = text[len(self.eos_token):]
            else:
                ids.append(ord(text[0]) + 10)
                text = text[1:]
        return {"input_ids": ids}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False, return_dict=False):
        assert add_generation_prompt and not enable_thinking
        text = "".join(m["role"] + ":" + m["content"] + "\n" for m in messages) + "assistant:"
        ids = self(text)["input_ids"]
        return ({"input_ids": ids} if return_dict else ids) if tokenize else text


def test_full_prefix_bytes_tokens_mask_and_eos_agree():
    row = {"id": "id", "system": "Grounded", "user": "Question", "assistant": "Answer"}
    measured = measure_row(row, Tokenizer(), 200, 50)
    assert measured["prompt"] == "system:Grounded\nuser:Question\nassistant:"
    assert measured["target_text_tokens"] == 6 and measured["target_tokens_with_eos"] == 7
    assert measured["sequence_tokens"] == measured["prompt_tokens"] + 7
    assert measured["raw_encode_row_identical"] and measured["direct_inference_template_identical"]
    assert len(measured["prompt_ids_sha256"]) == 64


def test_inference_token_mismatch_is_fatal():
    class Wrong(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            result = super().apply_chat_template(*args, **kwargs)
            return result + [99] if kwargs.get("tokenize") else result
    with pytest.raises(ValueError, match="inference template"):
        measure_row({"id": "id", "user": "q", "assistant": "a"}, Wrong())


def build_item(budget, target="answer"):
    return {"id": "id", "paper_id": "p", "user": "question " + "x" * budget,
            "assistant": target, "training_eligible": True}


def test_evidence_budget_reduces_without_changing_target():
    fitted = fit_item(build_item, Tokenizer(), "train", 200, 180, 40)
    assert fitted["evaluation_length_fits"] and fitted["training_length_fits"]
    assert fitted["assistant"] == "answer"
    assert fitted["preparation_lengths"]["final_evidence_budget"] < 200
    assert len(fitted["preparation_lengths"]["attempts"]) > 1


@pytest.mark.parametrize("split", ["test", "fresh_test"])
def test_test_context_does_not_use_reference_length_and_never_drops_row(split):
    short = fit_item(build_item, Tokenizer(), split, 50, 180, 40)
    long = fit_item(lambda b: build_item(b, "a" * 300), Tokenizer(), split, 50, 180, 40)
    assert long["user"] == short["user"] and long["prompt_ids_sha256"] == short["prompt_ids_sha256"]
    assert not long["training_length_fits"] and long["evaluation_length_fits"]
    assert not long["training_eligible"] and long["id"] == "id"


def test_budget_reduction_recomputes_coverage_without_relabeling_answerability():
    source = paper("p")
    source["full_text"][0]["paragraphs"] = ["A body result. " * 8]
    source["qas"][0]["answers"][0]["answer"]["evidence"] = ["A body result. " * 8]
    tokenizer = Tokenizer()
    build = lambda budget: prepare_qasper(source, "test", lambda text: len(tokenizer(text)["input_ids"]), budget)[0]
    assert build(300)["primary"]
    empty_prompt = measure_row(build(0), tokenizer)["prompt_tokens"]
    fitted = fit_item(build, tokenizer, "test", 300, empty_prompt + 120 + 30, 120)
    assert fitted["evaluation_length_fits"] and not fitted["primary"]
    assert fitted["references"][0]["missing_context_evidence"]
    assert not json.loads(fitted["assistant"])["unanswerable"]


def test_impossible_training_target_retained_and_impossible_eval_question_fails():
    row = fit_item(lambda b: build_item(b, "a" * 300), Tokenizer(), "train", 50, 180, 40)
    assert not row["training_eligible"]
    assert row["preparation_lengths"]["final_evidence_budget"] == 0
    with pytest.raises(ValueError, match="instructions alone"):
        fit_item(lambda b: {**build_item(b), "user": "q" * 300}, Tokenizer(), "test", 50, 180, 40)


def test_whole_paper_prefix_overshoots_and_is_order_independent():
    papers = [{"id": "a", "qas": [{}, {}, {}]}, {"id": "b", "qas": [{}, {}, {}]},
              {"id": "c", "qas": [{}, {}, {}]}]
    selected, manifest = seeded_paper_prefix(papers, 4)
    assert len(selected) == 2 and manifest["actual_questions"] == 6
    assert seeded_paper_prefix(papers[::-1], 4)[1] == manifest
    assert len({p["id"] for p in selected}) == 2


def paper(pid):
    return {"id": pid, "full_text": [{"section_name": "Results", "paragraphs": ["A body result."]}],
        "qas": [{"question_id": "q", "question": "What result?", "answers": [{"answer": {
            "unanswerable": False, "yes_no": None, "extractive_spans": [], "free_form_answer": "A result.",
            "evidence": ["A body result."]}}]}]}


def test_qasper_preparation_hashes_samples_lengths_and_immutable_outputs(tmp_path):
    root = tmp_path / "qasper"
    (root / "original").mkdir(parents=True)
    extracted = {}
    for split in ("train", "dev", "test"):
        filename = f"qasper-{split}-v0.3.json"
        data = {split + "-paper-" + str(i): {k: v for k, v in paper(split + "-paper-" + str(i)).items() if k != "id"} for i in range(3)}
        blob = json.dumps(data).encode()
        (root / "original" / filename).write_bytes(blob)
        extracted[filename] = {"sha256": sha(blob)}
    source = root / "manifest.json"
    source.write_text(canonical({"extracted_files": extracted, "revision": "synthetic"}))
    initial = source.read_bytes()
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    output = tmp_path / "prepared"
    manifest = prepare_task("qasper", Tokenizer(), tokenizer, output, root,
        evidence_cap=600, sequence_cap=1200, generated_cap=100, test_target=2, development_target=2)
    assert manifest["counts"]["train"]["records"] == 3
    assert manifest["counts"]["test"]["records"] == 2
    assert manifest["evaluation_rows_dropped"] == 0
    assert source.read_bytes() == initial == (output / "source_manifest.json").read_bytes()
    for split in ("train", "validation", "test"):
        rows = [json.loads(s) for s in (output / (split + ".jsonl")).read_text().splitlines()]
        assert all(r["training_length_fits"] and r["evaluation_length_fits"] for r in rows)
        assert manifest["output_sha256"][split + ".jsonl"] == sha((output / (split + ".jsonl")).read_bytes())
    with pytest.raises(FileExistsError):
        prepare_task("qasper", Tokenizer(), tokenizer, output, root)


def test_prl_historical_splits_retained_and_all_need_claim_audit(tmp_path):
    root = tmp_path / "prl"
    root.mkdir()
    for split in ("train", "validation", "test"):
        row = {"id": split, "paper_id": split, "abstract": "A grounded abstract.",
               "text": r"\title{Title}\begin{abstract}A grounded abstract.\end{abstract}" + "\n\nBody evidence."}
        (root / (split + ".jsonl")).write_text(json.dumps(row) + "\n")
    (root / "provenance.json").write_text('{"fixture":true}')
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    manifest = prepare_task("prl", Tokenizer(), tokenizer, tmp_path / "out", prl_root=root,
        evidence_cap=600, sequence_cap=1200, generated_cap=100)
    assert manifest["original_split_paper_ids"] == {s: [s] for s in ("train", "validation", "test")}
    assert manifest["counts"]["train"]["training_eligible"] == 0
    assert manifest["eligible_training_order"]["prefix_preview_count"] == 0
    historical = json.loads((tmp_path / "out/test.jsonl").read_text())
    assert historical["historical_test_diagnostic"] and not historical["training_eligible"]


def test_training_preview_matches_parent_eligible_whole_paper_algorithm():
    rows = [{"id": f"{p}-{i}", "paper_id": p, "training_eligible": p != "excluded"}
            for p in ("a", "b", "excluded") for i in range(3)]
    preview = eligible_order(rows, 4)
    assert preview["prefix_preview_count"] == 6 and preview["target_met"]
    assert "excluded" not in preview["paper_ids"]


def test_fresh_prl_all_rows_retained_provenance_bound_and_never_historical(tmp_path):
    historical = tmp_path / "old"
    historical.mkdir()
    for split in ("train", "validation", "test"):
        (historical / (split + ".jsonl")).write_text(json.dumps({"id": split, "paper_id": split}) + "\n")
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    rows = [{"id": f"new-{i}", "paper_id": f"new-{i}", "text": "Intro evidence.\n\nConclusion evidence.",
             "abstract": "Grounded reference."} for i in range(2)]
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    (fresh / "fresh.jsonl").write_bytes(data)
    (fresh / "result.json").write_text(json.dumps({"completed_acquisition": True, "papers": 2, "test_sha256": sha(data)}))
    (fresh / "registration.json").write_text('{"target":2}')
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    output = tmp_path / "out"
    report = prepare_fresh_prl(Tokenizer(), tokenizer, output, fresh, historical,
                               evidence_cap=600, sequence_cap=1200, generated_cap=100)
    assert report["counts"]["fresh_test"]["records"] == 2 and report["evaluation_rows_dropped"] == 0
    prepared = [json.loads(line) for line in (output / "fresh_test.jsonl").read_text().splitlines()]
    assert all(r["split"] == "fresh_test" and not r["historical_test_diagnostic"] and not r["training_eligible"] for r in prepared)
    assert report["output_sha256"]["fresh_test.jsonl"] == sha((output / "fresh_test.jsonl").read_bytes())
    with pytest.raises(FileExistsError):
        prepare_fresh_prl(Tokenizer(), tokenizer, output, fresh, historical)
    rows[0]["paper_id"] = "trainv2"
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    (fresh / "fresh.jsonl").write_bytes(data)
    (fresh / "result.json").write_text(json.dumps({"completed_acquisition": True, "papers": 2, "test_sha256": sha(data)}))
    with pytest.raises(ValueError, match="overlaps historical"):
        prepare_fresh_prl(Tokenizer(), tokenizer, tmp_path / "bad", fresh, historical)
