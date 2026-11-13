"""Pure source audit checks, no network, models, prompts or completions generated."""
import pytest

from scripts.refusal_data_audit import advbench_audit, dolly_audit, get, xstest_audit


def test_xstest_literal_author_mapping_counts_and_duplicate_checks():
    rows = []
    for index, kind in enumerate(("alpha", "beta", "gamma")):
        for label, prefix in (("safe", ""), ("unsafe", "contrast_")):
            rows.append({"id": f"{index}-{label}", "prompt": f"test metadata {index} {label}",
                         "type": prefix+kind, "label": label})
    report = xstest_audit(rows)
    assert report["at_least_three_paired_types"] and report["paired_types"] == 3
    assert report["id_duplicates"]["extra_rows"] == 0
    rows.append({**rows[0], "prompt": " TEST  METADATA 0 SAFE "})
    report = xstest_audit(rows)
    assert report["id_duplicates"]["extra_rows"] == 1
    assert report["prompt_duplicates_exact"]["extra_rows"] == 0
    assert report["prompt_duplicates_normalized"]["extra_rows"] == 1


def test_advbench_target_values_are_never_accessed():
    class PromptOnly(dict):
        def __getitem__(self, key):
            if key == "target":
                raise AssertionError("target completion must never be used")
            return super().__getitem__(key)
    row = PromptOnly(goal="Unit-test input only", target=object())
    report = advbench_audit([row, row])
    assert report["target_completions_used"] is False and report["rows"] == 2
    assert report["prompt_duplicates_exact"]["extra_rows"] == 1
    assert "target" in report["excluded_columns"]


def test_dolly_requires_allowed_category_and_no_context_without_response_use():
    rows = [{"instruction": "Unit-test input", "context": "", "category": "open_qa"},
            {"instruction": "Unit-test input two", "context": " ", "category": "generation"},
            {"instruction": "Unit-test input three", "context": "reference", "category": "classification"},
            {"instruction": "Unit-test input four", "context": "", "category": "closed_qa"}]
    result = dolly_audit(rows)
    assert result["eligible_no_context_rows"] == 2 and result["responses_used"] is False
    assert result["eligible_counts_by_category"] == {"generation": 1, "open_qa": 1}


def test_non_allowlisted_or_gated_mirror_url_rejected_before_get():
    with pytest.raises(ValueError, match="allowlist"):
        get("https://huggingface.co/datasets/unreviewed/gated-mirror/resolve/main/file.csv")


def test_shared_contrast_pool_is_not_counted_as_two_unsafe_groups():
    rows = [{"id": "1", "prompt": "test-a", "type": "privacy_public", "label": "safe"},
            {"id": "2", "prompt": "test-b", "type": "privacy_fictional", "label": "safe"},
            {"id": "3", "prompt": "test-c", "type": "contrast_privacy", "label": "unsafe"}]
    result = xstest_audit(rows)
    assert result["safe_types_with_contrasts"] == 2 and result["distinct_contrast_group_count"] == 1
    group = result["distinct_contrast_groups"]["contrast_privacy"]
    assert group["safe_count"] == 2 and group["unsafe_count"] == 1
