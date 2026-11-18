"""Quote and eligibility validation for AI data screening; no CLI calls."""
from scripts.qwen38_prl_claim_audit import checked_submission


def packet():
    return {"p": {"body_excerpt": "The response increases by two units. The limit remains finite.",
                  "abstract_sentences": [{"index": i} for i in range(3)],
                  "target_words": 30, "body_nonempty": True}}


def submission(quote="The response increases by two units."):
    return {"papers": [{"id": "p", "sentences": [
        {"index": 0, "label": "SUPPORTED", "quotes": [quote], "reason": "source says so"},
        {"index": 1, "label": "SUPPORTED", "quotes": ["The limit remains finite."], "reason": "source says so"},
        {"index": 2, "label": "FRAMING", "quotes": [], "reason": "generic motivation"}]}]}


def test_exact_quote_support_passes_and_fabricated_quote_downgrades():
    assert checked_submission(submission(), ["p"], packet())["p"]["eligible"]
    actual = checked_submission(submission("The response decreases by two units."), ["p"], packet())["p"]
    assert not actual["eligible"]
    assert actual["sentences"][0]["checked_label"] == "UNSUPPORTED"


def test_whitespace_normalization_does_not_change_quantity_support():
    actual = checked_submission(submission("The response\n increases by  two units."), ["p"], packet())["p"]
    assert actual["eligible"]
    assert not checked_submission(submission("The response increases by three units."), ["p"], packet())["p"]["eligible"]


def test_overlong_target_and_mostly_framing_do_not_become_eligible():
    original = packet(); original["p"]["target_words"] = 201
    assert not checked_submission(submission(), ["p"], original)["p"]["eligible"]
    data = submission(); data["papers"][0]["sentences"][1].update(label="FRAMING", quotes=[])
    assert not checked_submission(data, ["p"], packet())["p"]["eligible"]
