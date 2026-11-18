import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.qwen38_science_data import (
    annotation_target, extract_prl, number_paragraphs, prepare_local,
    prepare_prl, prepare_qasper, select_context, select_prl_prefix, validate_paper_splits,
    tex_blocks, select_prl_head_tail, PRL_OMISSION_MARKER,
)


def annotation(answer="result", evidence=None, **extra):
    return {"answer": {"unanswerable": False, "yes_no": None, "extractive_spans": [],
                       "free_form_answer": answer, "evidence": evidence or [], **extra}}


def paper(pid="paper", annotations=None):
    return {"id": pid, "abstract": "A summary.", "full_text": [
        {"section_name": "Method", "paragraphs": ["irrelevant preamble", "quantum result measured at 2 K", ""]},
        {"section_name": "Conclusion", "paragraphs": ["remaining evidence"]}],
        "qas": [{"question_id": "q", "question": "quantum result?", "answers": annotations or [
            annotation(evidence=["quantum result measured at 2 K"])]}]}


def test_numbering_precedes_selection_and_preserves_empty_positions():
    p = paper()
    numbered = number_paragraphs(p)
    assert numbered[2]["text"] == "quantum result measured at 2 K"
    assert numbered[3]["text"] == ""
    selected = select_context(numbered, "quantum", len, 65)
    assert "p000002" in {r["id"] for r in selected}
    row = prepare_qasper(p, "train", len, 65)[0]
    p["qas"][0]["answers"] = [annotation("entirely changed gold", ["remaining evidence"])]
    changed = prepare_qasper(p, "train", len, 65)[0]
    assert row["user"] == changed["user"]
    assert row["context_ids"] == changed["context_ids"]
    assert row["context_tokens"] <= 65


def test_full_context_and_annotation_coverage():
    p = paper(annotations=[annotation(evidence=["quantum result measured at 2 K"], highlighted_evidence=["quantum result"])])
    row = prepare_qasper(p, "validation", len, 10000)[0]
    assert row["primary"] and row["training_eligible"] and row["full_text_in_context"]
    assert not row["nontext_or_unmatched_evidence"]
    assert json.loads(row["assistant"]) == {"answer": "result", "evidence": ["p000002"], "unanswerable": False}


@pytest.mark.parametrize("field,value,kind,answer", [
    ("yes_no", False, "boolean", "No"), ("yes_no", True, "boolean", "Yes"),
    ("extractive_spans", ["one", "two"], "extractive", "one, two"),
    ("unanswerable", True, "unanswerable", ""),
])
def test_typed_answer_variants(field, value, kind, answer):
    ref = annotation_target(annotation("", **{field: value}), [])
    assert (ref["answer_type"], ref["answer"]) == (kind, answer)


def test_reject_wrong_boolean_and_conflicting_answer_types():
    with pytest.raises(ValueError, match="typed"):
        annotation_target(annotation("", yes_no="false"), [])
    with pytest.raises(ValueError, match="exactly one"):
        annotation_target(annotation("text", yes_no=True), [])


def test_missing_evidence_is_never_rewritten_as_unanswerable():
    row = prepare_qasper(paper(), "test", len, 0)[0]
    assert not json.loads(row["assistant"])["unanswerable"]
    assert row["references"][0]["missing_context_evidence"] == ["p000002"]
    assert not row["primary"] and not row["training_eligible"]


def test_multiple_references_primary_does_not_change_selected_training_annotation():
    p = paper(annotations=[annotation(evidence=["not found"]), annotation(evidence=["quantum result measured at 2 K"])])
    row = prepare_qasper(p, "train", len, 10000)[0]
    assert row["primary"] and not row["training_eligible"]
    assert json.loads(row["assistant"])["evidence"] == []
    assert len(row["references"]) == 2


def test_unanswerable_primary_requires_complete_document_and_no_disagreement():
    p = paper(annotations=[annotation("", unanswerable=True)])
    assert prepare_qasper(p, "test", len, 10000)[0]["primary"]
    assert not prepare_qasper(p, "test", len, 0)[0]["primary"]
    p["qas"][0]["answers"].append(annotation("yes"))
    row = prepare_qasper(p, "test", len, 10000)[0]
    assert row["annotation_disagreement"] and not row["primary"]


def test_figures_unmatched_and_ambiguous_evidence_are_retained():
    p = paper(annotations=[annotation(evidence=["FLOAT SELECTED: Figure 1", "unknown", "remaining evidence"])])
    p["full_text"][0]["paragraphs"].append("remaining evidence")
    row = prepare_qasper(p, "test", len, 10000)[0]
    assert row["nontext_or_unmatched_evidence"] and not row["primary"]
    assert row["references"][0]["ambiguous_evidence"] == ["remaining evidence"]
    assert len(row["references"][0]["unmatched_evidence"]) == 2


def test_duplicate_paragraphs_need_one_evidence_item_not_both_locations():
    p = paper(annotations=[annotation(evidence=["remaining evidence"])])
    p["full_text"][0]["paragraphs"].append("remaining   evidence")
    prepared = prepare_qasper(p, "test", len, 10000)[0]
    assert len(prepared["references"][0]["evidence"]) == 1
    assert prepared["references"][0]["ambiguous_evidence"] == ["remaining evidence"]
    assert prepared["primary"] and prepared["evidence_strict_primary"]


def test_evidence_strict_primary_excludes_answerable_empty_reference():
    p = paper(annotations=[annotation(evidence=["quantum result measured at 2 K"]), annotation("result")])
    prepared = prepare_qasper(p, "test", len, 10000)[0]
    assert prepared["primary"] and not prepared["evidence_strict_primary"]


def test_columnar_export_matches_native():
    p = paper()
    expected = prepare_qasper(p, "train", len, 10000)
    p["full_text"] = {k: [s[k] for s in p["full_text"]] for k in ("section_name", "paragraphs")}
    p["qas"] = {k: [p["qas"][0][k]] for k in p["qas"][0]}
    assert prepare_qasper(p, "train", len, 10000) == expected


def test_split_overlap_including_arxiv_versions_rejected():
    with pytest.raises(ValueError, match="duplicate paper"):
        validate_paper_splits({"train": [{"paper_id": "1234.5678v1"}], "validation": [],
                               "test": [{"paper_id": "1234.5678v2"}]}, "prl")


def test_old_test_paper_cannot_be_reassigned_to_training():
    splits = {"train": [{"paper_id": "old-test"}], "validation": [], "test": [{"paper_id": "train-paper"}]}
    with pytest.raises(ValueError, match="original paper split"):
        validate_paper_splits(splits, "prl", {"train": ["train-paper"], "validation": [], "test": ["old-test"]})


def prl_row():
    abstract = "We find a response of 2 K in a controlled model."
    return {"id": "paper-v1", "paper_id": "paper", "abstract": abstract,
        "text": (r"\newcommand{\energy}{E_{0}}" + "\n" + r"\title{A {nested} title}\author{Private Person}"
                 + r"\begin{abstract}" + abstract + r"\end{abstract}\maketitle" + "\n\n"
                 + r"Introduction: We define $\energy$ and measure 2 K. Legitimate body evidence."
                 + "\n\n" + abstract + "\n\n" + r"The result retains a 2 K scale and uncertainty of 0.1 K."),
        "source_sha256": "abc", "doi": "10.example/paper", "license": "CC-BY-4.0"}


def test_prl_removes_target_and_identity_preserves_intro_equations_and_definitions():
    row = prl_row()
    body, target, audit = extract_prl(row)
    assert target not in body and "Private" not in body and "nested" not in body
    assert r"\newcommand{\energy}{E_{0}}" in body and "Introduction:" in body
    assert "2 K" in body and "0.1 K" in body
    assert audit["removed_abstract_environments"] == 1 and audit["removed_exact_duplicates"] == 1
    prepared = prepare_prl(row, "test", len, 10000)
    assert prepared["assistant"] == row["abstract"]
    assert prepared["historical_test_diagnostic"] and not prepared["training_eligible"]
    assert prepared["provenance"]["source_sha256"] == "abc"


def test_abstract_macro_and_whitespace_duplicate_removed_without_fuzzy_body_deletion():
    row = prl_row()
    row["text"] = r"\abstract{" + row["abstract"] + "}\n" + row["abstract"].replace(" ", "\n") + "\n\nRelated legitimate evidence about 2 K."
    body, _, audit = extract_prl(row)
    assert "We find" not in body and "Related legitimate evidence" in body
    assert audit["removed_exact_duplicates"] == 1


def test_unsupported_abstract_macro_flag_and_empty_body():
    row = prl_row()
    row["text"] = r"\myabstract{alternate abstract} Evidence."
    assert "unresolved_abstract_or_metadata_macro" in extract_prl(row)[2]["flags"]
    row["text"] = row["abstract"]
    assert "empty_body" in extract_prl(row)[2]["flags"]


def test_starred_abstract_environment_is_removed():
    row = prl_row()
    row["text"] = r"\begin{abstract*}" + row["abstract"] + r"\end{abstract*}" + "\n\nBody at 2 K."
    body, _, audit = extract_prl(row)
    assert body == "Body at 2 K." and audit["removed_abstract_environments"] == 1


def test_prl_plain_rendering_head_tail_preserves_old_prefix_helper():
    assert select_prl_prefix("short\n\n" + "long " * 40 + "\n\ntail", len, 30) == ("short", 1, 3)
    row = prl_row()
    row["text"] = "First body paragraph.\n\n" + "long " * 40 + "\n\nLast paragraph."
    prepared = prepare_prl(row, "train", len, 120)
    assert prepared["selected_body"] == "First body paragraph.\n\n" + PRL_OMISSION_MARKER + "\n\nLast paragraph."
    assert prepared["context_tokens"] == len(prepared["selected_body"])
    assert "[p000" not in prepared["user"] and "Body\n" not in prepared["user"]
    assert "at most 200 words" in prepared["user"]
    assert "context_head_tail_excerpt" in prepared["leakage_audit"]["flags"]


def test_tex_blocks_keep_nested_environments_display_math_and_grouped_macros_whole():
    equation = r"\begin{align}" + "\n\nx &= " + r"\begin{aligned}a&=b\\c&=d\end{aligned}" + "\n\n" + r"\end{align}"
    display = r"\[" + "\n\na=b\n\n" + r"\]"
    dollars = "$$\n\nx=y\n\n$$"
    definition = r"\newcommand{\custom}{a" + "\n\nb}"
    body = "\n\n".join(["Intro.", equation, display, dollars, definition, "Conclusion."])
    blocks, syntax = tex_blocks(body)
    assert syntax["balanced"]
    assert [b["text"] for b in blocks] == ["Intro.", equation, display, dollars, definition, "Conclusion."]
    assert [b["kind"] for b in blocks] == ["prose", "math", "math", "math", "definition", "prose"]
    for b in blocks:
        assert body[b["start"]:b["stop"]] == b["text"]


def test_tail_uses_last_prose_not_trailing_float_and_respects_half_caps():
    figure = r"\begin{figure}\caption{Trailing unrelated caption}\end{figure}"
    body = "Opening evidence.\n\n" + "Middle " * 100 + "\n\nFinal conclusion.\n" + figure + "\n\n" + r"\vspace{-2mm}"
    context, selection = select_prl_head_tail(body, len, 150)
    assert context.startswith("Opening evidence.") and context.endswith("Final conclusion.")
    assert figure not in context and context.count(PRL_OMISSION_MARKER) == 1
    assert selection["head_tokens"] <= selection["head_budget"]
    assert selection["tail_tokens"] <= selection["tail_budget"]
    assert selection["head_budget"] + selection["tail_budget"] <= 150
    assert len(context) <= 150
    assert selection["trailing_nonprose_omitted"]


def test_full_body_preserved_exactly_including_floats_when_it_fits():
    body = "Text.\n\n" + r"\begin{figure}\caption{Caption}\end{figure}" + "\n\nConclusion."
    context, selection = select_prl_head_tail(body, len, len(body))
    assert context == body and selection["full_body"]
    assert selection["omission_marker_count"] == 0


def test_unbalanced_tex_is_indivisible_and_overlong_source_not_partially_selected():
    for body in [r"\begin{align}a=b" + "\n\nCut here?", "$$x=y", r"\[x=y", "Text {unclosed", r"\end{equation}"]:
        blocks, syntax = tex_blocks(body)
        assert not syntax["balanced"] and len(blocks) == 1
        context, selection = select_prl_head_tail(body, len, 1)
        assert context == "" and not selection["selected_indices"]


def test_unfinished_backmatter_wrapper_does_not_hide_completed_conclusion():
    body = "Intro.\n\n" + "Middle " * 80 + "\n\nCompleted conclusion.\n\n" + r"\emph{"
    context, selection = select_prl_head_tail(body, len, 140)
    assert not selection["syntax"]["balanced"]
    assert "Completed conclusion." in context and r"\emph{" not in context
    assert selection["blocks"][-1]["kind"] == "metadata"
    assert selection["syntax"]["math_environment_boundaries_balanced"]
    assert not selection["syntax"]["generic_braces_balanced"]
    assert context.count(PRL_OMISSION_MARKER) == 1


def test_macro_declarations_do_not_execute_begin_end_and_alias_math_is_whole():
    definitions = (r"\newcommand{\be}{\begin{equation}}" + "\n" + r"\newcommand\ee{\end{equation}}" + "\n"
                   + r"\def\ba{\begin{align}}" + "\n" + r"\def\ea{\end{align}}" + "\n"
                   + r"\def\wrapper#1{\begin{gather}#1\end{gather}}")
    equation = r"\be" + "\n\na=b\n\n" + r"\ee"
    align = r"\ba" + "\n\n" + r"\begin{aligned}x&=y\end{aligned}" + "\n\n" + r"\ea"
    body = "\n\n".join([definitions, "Intro.", equation, align, "Conclusion."])
    blocks, syntax = tex_blocks(body)
    assert syntax["balanced"] and syntax["math_environment_boundaries_balanced"]
    assert len(syntax["protected_macro_declarations"]) == 5
    assert set(syntax["environment_aliases"]) == {r"\be", r"\ee", r"\ba", r"\ea"}
    assert [b["text"] for b in blocks] == [definitions, "Intro.", equation, align, "Conclusion."]
    assert [b["kind"] for b in blocks][2:4] == ["math", "math"]
    # A valid macro can intentionally contain an opening delimiter only.
    assert tex_blocks(r"\newcommand{\open}{\begin{equation}}")[1]["balanced"]
    broken = r"\newcommand{\be}{\begin{equation}}" + "\n\nIntro.\n\n" + r"\be x=y" + "\n\nStill in the equation."
    broken_blocks, broken_syntax = tex_blocks(broken)
    assert not broken_syntax["math_environment_boundaries_balanced"]
    assert broken_syntax["open_environments"] == ["equation"]
    assert broken_blocks[-1]["kind"] == "unknown"
    assert broken_blocks[-1]["text"] == r"\be x=y" + "\n\nStill in the equation."


def test_adjacent_inline_math_dollars_do_not_become_false_display_boundaries():
    body = r"Specifically, $\mathbf{r}_{s} $$=$$ \mathbf{C}_{\infty}$ is measured." + "\n\nConclusion."
    blocks, syntax = tex_blocks(body)
    assert syntax["balanced"] and len(blocks) == 2
    assert blocks[-1]["text"] == "Conclusion."
    assert not tex_blocks("$$ genuine unclosed display\n\nStill math")[1]["math_environment_boundaries_balanced"]


def test_generic_styling_braces_are_diagnostic_without_hiding_scientific_paragraphs():
    body = "} Intro evidence.\n\n" + r"\textcolor{blue}{" + "middle " * 100 + "\n\nConcluding evidence.\n\n" + r"\emph{"
    context, selection = select_prl_head_tail(body, len, 170)
    assert context.startswith("} Intro evidence.") and context.endswith("Concluding evidence.")
    assert not selection["syntax"]["generic_braces_balanced"]
    assert selection["syntax"]["math_environment_boundaries_balanced"]
    assert selection["syntax"]["unmatched_closing_brace_offsets"] == [0]
    assert not selection["syntax"]["compilability_established"]
    assert all(b["kind"] != "unknown" for b in selection["blocks"])


def test_generic_outer_wrapper_does_not_split_nested_math_environment():
    equation = r"\begin{align}x &= " + "\n\n" + r"\begin{aligned}a&=b\end{aligned}" + "\n\n" + r"\end{align}"
    body = "{\\it Opening.\n\n" + equation + "\n\nConclusion."
    blocks, syntax = tex_blocks(body)
    assert not syntax["generic_braces_balanced"] and syntax["math_environment_boundaries_balanced"]
    assert blocks[1]["text"] == equation


def test_escaped_tex_and_comments_cannot_create_false_block_boundaries():
    body = r"Cost \$2 and escaped \% with \{braces\}." + "\n\n" + r"\begin{equation}x=2% \end{equation}" + "\n\nx=3\n" + r"\end{equation}"
    blocks, syntax = tex_blocks(body)
    assert syntax["balanced"] and len(blocks) == 2
    assert "x=3" in blocks[1]["text"]


def test_head_tail_context_independent_of_target_and_copy_uses_only_excerpt():
    row = prl_row()
    row["text"] = "Opening evidence.\n\n" + "MIDDLE ONLY statement " * 40 + "\n\nFinal supported conclusion."
    prepared = prepare_prl(row, "train", len, 160)
    changed = prepare_prl({**row, "abstract": "An entirely different reference."}, "fresh_test", len, 160)
    assert prepared["selected_body"] == changed["selected_body"]
    assert "MIDDLE ONLY" not in prepared["selected_body"]
    assert "MIDDLE ONLY" not in prepared["copy_baseline"]["text"]
    assert "Final supported conclusion." in prepared["copy_baseline"]["text"]
    assert not changed["historical_test_diagnostic"]
    assert not prepared["training_eligible"]
    assert prepared["context_selection"]["full_body_tokens"] > prepared["context_tokens"]


def test_eightgram_overlap_only_flags_does_not_delete_legitimate_conclusion():
    row = prl_row()
    conclusion = "We find a response of 2 K in a controlled setting with uncertainty."
    row["text"] = "Evidence.\n\n" + conclusion
    body, _, audit = extract_prl(row)
    assert conclusion in body
    assert 0 < audit["abstract_8gram_containment"] < 1
    assert "abstract_8gram_overlap_review" in audit["flags"]
    row["text"] = "This body stops abruptly and"
    assert "suspicious_body_ending" in extract_prl(row)[2]["flags"]


def test_immutable_local_native_preparation_and_input_provenance(tmp_path):
    inputs = {}
    for split in ("train", "validation", "test"):
        filename = tmp_path / (split + ".json")
        native = paper(split)
        native.pop("id")
        filename.write_text(json.dumps({split: native}))
        inputs[split] = filename
    source = tmp_path / "source.json"
    source.write_text('{"revision":"pinned"}')
    out = tmp_path / "prepared"
    manifest = prepare_local("qasper", inputs, out, "revision", len, "test-character-counter", 10000, source)
    assert manifest["counts"]["train"]["records"] == 1
    assert len(manifest["source_manifest_sha256"]) == 64
    assert len(manifest["input_sha256"]["test"]) == 64
    snapshot = (out / "train.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_local("qasper", inputs, out, "revision", len, "test-character-counter", 10000)
    assert (out / "train.jsonl").read_bytes() == snapshot
    other = tmp_path / "same-preparation"
    prepare_local("qasper", inputs, other, "revision", len, "test-character-counter", 10000, source)
    assert (other / "manifest.json").read_bytes() == (out / "manifest.json").read_bytes()


def test_counter_validation_and_negative_budget():
    with pytest.raises(ValueError):
        select_context(number_paragraphs(paper()), "q", len, -1)
    with pytest.raises(ValueError):
        select_context([], "q", lambda s: 1.5, 1)


def test_native_qasper_split_hash_binding_rejects_relabeled_inputs(tmp_path):
    inputs = {}
    for split in ("train", "validation", "test"):
        p = tmp_path / (split + ".json")
        p.write_text(json.dumps({split: {k: v for k, v in paper(split).items() if k != "id"}}))
        inputs[split] = p
    manifest = tmp_path / "acquired.json"
    manifest.write_text(json.dumps({"extracted_files": {"qasper-train-v0.3.json": {"sha256": "wrong"}}}))
    with pytest.raises(ValueError, match="original QASPER"):
        prepare_local("qasper", inputs, tmp_path / "out", "pinned", len, "characters", 10000, manifest)
    assert not (tmp_path / "out").exists()


def test_preparation_cli_is_local_and_refuses_overwrite(tmp_path):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-B", "-m", "scripts.qwen38_science_data", "--task", "qasper",
               "--revision", "synthetic", "--counter", "whitespace-proxy", "--context-budget", "1000",
               "--output", str(tmp_path / "prepared")]
    for split in ("train", "validation", "test"):
        filename = tmp_path / (split + ".json")
        p = paper(split)
        p.pop("id")
        filename.write_text(json.dumps({split: p}))
        command.extend(["--" + split, str(filename)])
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["counter"] == "whitespace-proxy"
    assert subprocess.run(command, cwd=root, capture_output=True).returncode != 0
