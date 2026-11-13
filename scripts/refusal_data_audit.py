"""Public author-data GET/count/license audit only: no splits, models or judgments.

--fetch downloads an explicit immutable URL allowlist without credentials or
Hub clients. Default mode rechecks cached bytes offline. Never access AdvBench
target completions; only the goal/prompt column is used for audit statistics.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
import urllib.request

from scripts.common import path

PINS = {"xstest": "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d",
        "advbench": "098262edf85f807224e70ecd87b9d83716bf6b73",
        "dolly": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"}
XSTEST = "https://raw.githubusercontent.com/paul-rottger/xstest/" + PINS["xstest"] + "/"
ADVBENCH = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/" + PINS["advbench"] + "/"
DOLLY = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/" + PINS["dolly"] + "/"
SOURCES = {"xstest_prompts.csv": XSTEST + "xstest_prompts.csv", "xstest_LICENSE.txt": XSTEST + "LICENSE",
           "xstest_readme.md": XSTEST + "readme.md",
           "advbench_harmful_behaviors.csv": ADVBENCH + "data/advbench/harmful_behaviors.csv",
           "advbench_LICENSE.txt": ADVBENCH + "LICENSE", "advbench_README.md": ADVBENCH + "README.md",
           "dolly.jsonl": DOLLY + "databricks-dolly-15k.jsonl", "dolly_README.md": DOLLY + "README.md",
           "dolly_metadata.json": "https://huggingface.co/api/datasets/databricks/databricks-dolly-15k/revision/" + PINS["dolly"]}
DOLLY_CATEGORIES = {"open_qa", "brainstorming", "generation", "classification"}
POOLED_CONTRASTS = {"nons_group_real_discr": "contrast_discr", "real_group_nons_discr": "contrast_discr",
                    "privacy_public": "contrast_privacy", "privacy_fictional": "contrast_privacy"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(filename, data):
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("xb") as stream:
        stream.write(data)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode()


def get(url):
    if url not in SOURCES.values():
        raise ValueError("URL is outside the explicit pinned public source allowlist")
    request = urllib.request.Request(url, headers={"User-Agent": "tuneLLM-public-data-audit/1", "Accept": "*/*"}, method="GET")
    # urllib does not read HF tokens, netrc or project authentication. No cookies,
    # terms acceptance, POST, credentials or Authorization header are supplied.
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read(64*1024*1024+1)
    if len(data) > 64*1024*1024:
        raise ValueError("source exceeds data-only download size bound")
    return data


def fetch(raw):
    raw.mkdir(parents=True, exist_ok=False)
    manifest = {}
    # Verify ungated public Dolly metadata BEFORE requesting its data/card.
    names = ["dolly_metadata.json"] + [n for n in SOURCES if n != "dolly_metadata.json"]
    for name in names:
        data = get(SOURCES[name])
        if name == "dolly_metadata.json":
            metadata = json.loads(data)
            if metadata.get("sha") != PINS["dolly"] or metadata.get("gated") is not False or metadata.get("private") is not False:
                raise ValueError("Dolly must be the pinned public ungated authors dataset; no terms/auth fallback")
        save(raw / name, data)
        manifest[name] = {"url": SOURCES[name], "sha256": sha(data), "bytes": len(data)}
    save(raw / "download_manifest.json", json_bytes(manifest))


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def duplicates(values):
    counts = Counter(values)
    return {"unique": len(counts), "extra_rows": sum(n-1 for n in counts.values()),
            "duplicate_groups": sum(n > 1 for n in counts.values())}


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline="")))


def xstest_audit(rows):
    required = {"id", "prompt", "type", "label"}
    if not rows or not required <= rows[0].keys():
        raise ValueError("XSTest author CSV schema differs")
    labels = Counter(r["label"].casefold().strip() for r in rows)
    if set(labels) != {"safe", "unsafe"}:
        raise ValueError("unexpected XSTest labels; mapping requires review")
    types = {label: Counter(r["type"] for r in rows if r["label"].casefold().strip() == label) for label in labels}
    mapping, unmapped = {}, []
    for unsafe_type in sorted(types["unsafe"]):
        candidates = [safe_type for safe_type in types["safe"]
                      if unsafe_type == safe_type or unsafe_type == "contrast_" + safe_type]
        if len(candidates) == 1:
            safe_type = candidates[0]
            mapping[safe_type] = {"contrast_type": unsafe_type, "safe_count": types["safe"][safe_type],
                                  "unsafe_count": types["unsafe"][unsafe_type]}
        else:
            unmapped.append(unsafe_type)
    ids = [r["id"] for r in rows]
    prompts = [r["prompt"] for r in rows]
    if any(not v.strip() for v in ids + prompts):
        raise ValueError("empty XSTest ID/prompt")
    label_by_prompt = {}
    for row in rows:
        label_by_prompt.setdefault(normalized(row["prompt"]), set()).add(row["label"])
    all_mapping = dict(mapping)
    for safe_type, contrast in POOLED_CONTRASTS.items():
        if safe_type in types["safe"] and contrast in types["unsafe"]:
            all_mapping[safe_type] = {"contrast_type": contrast, "safe_count": types["safe"][safe_type],
                "unsafe_count": types["unsafe"][contrast], "shared_contrast_pool": True}
    groups = {}
    for safe_type, item in all_mapping.items():
        group = groups.setdefault(item["contrast_type"], {"safe_types": [], "safe_count": 0,
                                                          "unsafe_count": item["unsafe_count"]})
        group["safe_types"].append(safe_type)
        group["safe_count"] += item["safe_count"]
    return {"rows": len(rows), "columns": list(rows[0]), "label_counts": dict(labels),
        "type_counts_by_label": {k: dict(sorted(v.items())) for k, v in types.items()},
        "id_duplicates": duplicates(ids), "prompt_duplicates_exact": duplicates(prompts),
        "prompt_duplicates_normalized": duplicates(list(map(normalized, prompts))),
        "cross_label_normalized_prompt_groups": sum(len(v) > 1 for v in label_by_prompt.values()),
        "safe_to_contrast_mapping": mapping, "unmapped_unsafe_types": unmapped,
        "mapping_method": "literal author type labels: contrast_<safe_type>, or identical type across labels; no semantic invention",
        "complete_safe_to_contrast_mapping": all_mapping, "distinct_contrast_groups": groups,
        "safe_types_with_contrasts": len(all_mapping), "distinct_contrast_group_count": len(groups),
        "pooled_mapping_evidence": "Authors NAACL2024 paper Table4 groups C6/7 discrimination and C9/10 privacy: https://aclanthology.org/2024.naacl-long.301.pdf#page=23",
        "pooling_note": "Two safe types share each discrimination/privacy unsafe pool. Do not count unsafe rows twice or split a shared contrast pool across development/final; coordinator must register grouping before selection.",
        "paired_types": len(mapping), "at_least_three_paired_types": len(mapping) >= 3,
        "per_type_capacity_for_proposed_plan": {k: {"dev_safe_max10": min(10, v["safe_count"]),
            "dev_unsafe_max10": min(10, v["unsafe_count"]), "final_safe_max6": min(6, v["safe_count"]),
            "final_unsafe_max6": min(6, v["unsafe_count"])} for k, v in mapping.items()}}


def advbench_audit(rows):
    if not rows or "goal" not in rows[0]:
        raise ValueError("expected authors AdvBench goal column")
    # Do not access, summarize, hash separately, select, or emit target values.
    goals = [r["goal"] for r in rows]
    if any(not p.strip() for p in goals):
        raise ValueError("empty AdvBench goal")
    return {"rows": len(rows), "prompt_column": "goal", "columns": list(rows[0]),
            "excluded_columns": [k for k in rows[0] if k != "goal"],
            "target_completions_used": False, "original_source_ids": "zero-based authors CSV data-row index",
            "prompt_duplicates_exact": duplicates(goals), "prompt_duplicates_normalized": duplicates(list(map(normalized, goals)))}


def dolly_audit(rows):
    if not rows or not {"instruction", "context", "category"} <= rows[0].keys():
        raise ValueError("Dolly schema differs")
    eligible = [r for r in rows if r["category"] in DOLLY_CATEGORIES and not r["context"].strip() and r["instruction"].strip()]
    prompts = [r["instruction"] for r in eligible]
    return {"rows": len(rows), "category_counts": dict(sorted(Counter(r["category"] for r in rows).items())),
            "eligible_categories": sorted(DOLLY_CATEGORIES), "eligible_no_context_rows": len(eligible),
            "requested_categories_absent": sorted(DOLLY_CATEGORIES - {r["category"] for r in rows}),
            "eligible_counts_by_category": dict(sorted(Counter(r["category"] for r in eligible).items())),
            "eligible_instruction_duplicates_exact": duplicates(prompts),
            "eligible_instruction_duplicates_normalized": duplicates(list(map(normalized, prompts))),
            "original_source_ids": "zero-based authors JSONL physical LF row index", "responses_used": False}


def audit(raw, report):
    if report.exists():
        raise FileExistsError("refusing to overwrite the historical data audit")
    manifest = json.loads((raw / "download_manifest.json").read_text())
    if set(manifest) != set(SOURCES):
        raise ValueError("cached allowlist manifest differs")
    data = {}
    for name, url in SOURCES.items():
        data[name] = (raw / name).read_bytes()
        if manifest[name] != {"url": url, "sha256": sha(data[name]), "bytes": len(data[name])}:
            raise ValueError("pinned cached source bytes changed")
    xrows = csv_rows(data["xstest_prompts.csv"])
    arows = csv_rows(data["advbench_harmful_behaviors.csv"])
    drows = [json.loads(line) for line in data["dolly.jsonl"].decode().split("\n") if line.strip()]
    xstest, advbench, dolly = xstest_audit(xrows), advbench_audit(arows), dolly_audit(drows)
    agoals = {normalized(r["goal"]) for r in arows}
    dgoals = {normalized(r["instruction"]) for r in drows if r["category"] in DOLLY_CATEGORIES and not r["context"].strip()}
    xlicense = data["xstest_LICENSE.txt"].decode()
    xreadme = data["xstest_readme.md"].decode()
    alicense = data["advbench_LICENSE.txt"].decode()
    dcard = data["dolly_README.md"].decode()
    if not re.search(r"creativecommons\.org/licenses/by/4\.0|cc.by.4.0|attribution 4\.0", xlicense+"\n"+xreadme, re.I):
        raise ValueError("XSTest CC-BY-4.0 evidence missing; review authors sources")
    if "MIT License" not in alicense:
        raise ValueError("AdvBench repository MIT evidence missing")
    if not re.search(r"license:\s*cc-by-sa-3\.0", dcard, re.I):
        raise ValueError("Dolly CC-BY-SA-3.0 card declaration missing")
    license_dir = report.parent / "refusal_source_licenses"
    license_copies = {"xstest_LICENSE.txt": data["xstest_LICENSE.txt"], "advbench_LICENSE.txt": data["advbench_LICENSE.txt"],
        "dolly_card_license.txt": ("Source: "+SOURCES["dolly_README.md"]+"\n"+
             "\n".join(line for line in dcard.split("\n") if re.search(r"license:|creativecommons.org/licenses/by-sa/3.0", line, re.I))+"\n").encode()}
    for name, content in license_copies.items():
        filename = license_dir / name
        if filename.exists():
            if filename.read_bytes() != content:
                raise ValueError("existing source-license copy differs; never overwrite")
        else:
            save(filename, content)
    metadata = json.loads(data["dolly_metadata.json"])
    if metadata.get("gated") is not False or metadata.get("private") is not False or metadata.get("sha") != PINS["dolly"]:
        raise ValueError("Dolly pin/public/ungated assertion failed")
    result = {"status": "public_source_data_audit_only_no_experimental_splits", "pins": PINS,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "helper_sha256": sha(Path(__file__).read_bytes()),
        "files": manifest, "xstest": xstest, "advbench": advbench, "dolly": dolly,
        "cross_source_normalized_prompt_overlap": {"advbench_xstest_safe": len(agoals & {normalized(r["prompt"]) for r in xrows if r["label"] == "safe"}),
            "advbench_xstest_unsafe": len(agoals & {normalized(r["prompt"]) for r in xrows if r["label"] == "unsafe"}),
            "eligible_dolly_xstest": len(dgoals & {normalized(r["prompt"]) for r in xrows}),
            "advbench_eligible_dolly": len(agoals & dgoals)},
        "licenses": {"xstest": {"id": "CC-BY-4.0", "status": "verified authors license/readme evidence"},
                     "advbench": {"id": "MIT", "status": "verified repository-level license; authors CSV from that pinned repository"},
                     "dolly": {"id": "CC-BY-SA-3.0", "status": "verified authors dataset card; public and ungated", "gated": False}},
        "source_license_copies": {str((license_dir/n).relative_to(report.parent)): {"sha256": sha(v), "bytes": len(v)} for n,v in license_copies.items()},
        "authentication": "none; unauthenticated GET only; no HF client/token/terms acceptance",
        "raw_directory": str(raw.relative_to(path("."))), "raw_data_in_report": False,
        "proposed_plan_not_executed": {"seed": 20261114, "extraction_max_advbench": 100, "extraction_max_dolly": 100,
            "token_length_matching": "plus/minus20%; not assessed in this source audit", "xstest_dev_paired_types": 3,
            "dev_per_type_safe_unsafe_max": [10,10], "remaining_paired_types_final_per_type_safe_unsafe_max": [6,6],
            "split_selection_frozen": False},
        "limits": ["Source/type-label audit only; no experimental split, model response, judge call or harmful completion generated.",
                   "Dolly category/no-context eligibility is not a guarantee of harmlessness; later prompt-only screening remains necessary.",
                   "Target/response columns remain unused; source raw files are preserved for exact provenance under ignored storage."]}
    save(report, json_bytes(result))
    print(json.dumps({"pins": PINS, "xstest_rows": xstest["rows"], "paired_types": xstest["paired_types"],
                      "advbench_rows": advbench["rows"], "dolly_rows": dolly["rows"],
                      "dolly_eligible": dolly["eligible_no_context_rows"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/refusal-01/raw")
    parser.add_argument("--report", default="results/refusal_data_audit.json")
    parser.add_argument("--fetch", action="store_true", help="public pinned allowlist GETs, without auth")
    args = parser.parse_args()
    raw, report = path(args.raw).resolve(), path(args.report).resolve()
    if not raw.is_relative_to(path("data/refusal-01").resolve()):
        raise ValueError("raw data must remain under ignored data/refusal-01")
    if args.fetch:
        if report.exists():
            raise FileExistsError("audit already exists; no further downloads")
        fetch(raw)
    audit(raw, report)


if __name__ == "__main__":
    main()
