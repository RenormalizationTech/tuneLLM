"""Regression: inline Appendix references must not remove scientific body text."""
from scripts.qwen38_prl_sources import clean_fixed, protect_inline_references
from scripts.prepare_physics_corpus import clean_source


def test_inline_reference_is_protected_but_standalone_heading_is_retained():
    text = "See \\textit{Appendix A} for details.\n\\textit{Appendix A}\n"
    actual = protect_inline_references(text)
    assert "See \\tuneInlinetextit {Appendix A} for details." in actual
    assert "\n\\textit{Appendix A}\n" in actual


def test_reextraction_keeps_post_reference_science_and_correct_length():
    abstract = "We derive the controlled theoretical behavior of the model. " * 4
    before = ("We define a model with the stated conservation law and boundary conditions. " * 20 + "\n\n") * 3
    after = ("The final result preserves the scaling limit and the measured theoretical exponent. " * 20 + "\n\n") * 3
    source = ("\\documentclass{revtex4-2}\n\\begin{document}\n\\title{Theory}\n"
              + "\\begin{abstract}" + abstract + "\\end{abstract}\n"
              + before + "See \\textit{Appendix A} for the derivation.\n\n" + after
              + "\\appendix\nSupplemental material.\n\\end{document}")
    old = clean_source({"main.tex": source}, "Theory")
    new = clean_fixed({"main.tex": source}, "Theory")
    assert new["abstract"] == old["abstract"]
    assert "The final result" not in old["text"]
    assert "The final result" in new["text"]
    assert "\\tuneInline" not in new["text"]
    body = new["text"].split("\\end{abstract}", 1)[1].strip()
    assert new["main_text_characters"] == len(body)
    assert "Supplemental material" not in body
