import io
import tarfile

import pytest

from scripts.prepare_physics_corpus import braced, clean_source, source_files, strip_comments
from scripts.benchmark_screen import writing_checks


def test_cleaning_excludes_identity_comments_supplements_and_bibliography():
    first = "We develop a controlled theoretical description of coupled quantum degrees of freedom. " * 30
    second = "Our calculation predicts a response proportional to $J^2$ without experimental fitting. " * 30
    tex = (r"\documentclass[prl]{revtex4-2}" + "\n" + r"\begin{document}\title{Test}\author{Private Person}"
           + "\n% discarded private comment\n" + r"\begin{abstract}" + "A model of quantum phases. " * 10
           + r"\end{abstract}" + "\n\n" + first + "\n\n" + second + "\n\n" + first + "\n\n" + second
           + r"\begin{acknowledgments}Private grant\end{acknowledgments}"
           + r"\bibliography{refs}\appendix Supplementary private text\end{document}")
    cleaned = clean_source({"main.tex": tex}, "Test theory")
    assert "Private" not in cleaned["text"]
    assert "discarded" not in cleaned["text"]
    assert "Supplementary" not in cleaned["text"]
    assert "$J^2$" in cleaned["text"]
    assert cleaned["text"].startswith(r"\title{Test theory}")
    assert len(cleaned["writing_source"]) > 1500


def test_archive_does_not_extract_unsafe_members_or_links():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for filename in ["../outside.tex", "/absolute.tex", "paper/main.tex"]:
            payload = b"safe contents"
            info = tarfile.TarInfo(filename)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    assert set(source_files(buffer.getvalue())) == {"paper/main.tex"}


def test_nested_braces_and_escaped_percent():
    assert braced("{a {nested} value}tail", 0) == ("a {nested} value", 18)
    assert strip_comments("a \\% preserved % remove\nnext") == "a \\% preserved \nnext"
    with pytest.raises(ValueError):
        braced("{unclosed", 0)


def test_author_block_after_abstract_and_hidden_text_are_removed():
    paragraph = "A theoretical result establishes a relation between fluctuations and response. " * 50
    tex = (r"\begin{document}\begin{abstract}" + "We derive quantum predictions. " * 10
           + r"\end{abstract}\author{Private {Name}}\affiliation{Private Institution}\email{private@example.org}\maketitle"
           + "\n\n" + paragraph + "\n\n" + paragraph
           + r"\begin{comment}Private discarded reply\end{comment}"
           + r"\iffalse Private rejected draft\fi" + "\n\n" + paragraph
           + r"\paragraph*{Acknowledgments}Grant No. 123\end{document}")
    cleaned = clean_source({"main.tex": tex}, "A theoretical result")
    assert "Private" not in cleaned["text"]
    assert "private@example" not in cleaned["writing_source"]
    assert "Grant" not in cleaned["text"]


def test_writing_flags_are_diagnostics_not_claim_validation():
    result = writing_checks({"id": "p-title", "task": "title", "source": r"Model at 2 K \cite{known}"},
                            r"At 3 K \cite{unknown}")
    assert result["automatic_fidelity_flag"]
    assert result["new_number_flags"] == ["3"]
    assert result["new_citation_flags"] == ["unknown"]
    assert result["human_scientific_fidelity"] is None
