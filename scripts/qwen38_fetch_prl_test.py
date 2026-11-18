"""Acquire the bounded, reviewed PRL test pool with independent source gates.

Fresh means new to this project, not necessarily absent from model pretraining.
Retain all failures. No pool expansion, source execution or target-model calls.
"""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

from scripts.qwen38_prl_sources import clean_fixed
from scripts.prepare_physics_corpus import source_files
from scripts.qwen38_science_runtime import sha, write

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "data/qwen38-prl-abstract-01"
SELECTION = STUDY / "selection"
DEST = STUDY / "fresh-acquisition-01"


def normalize(text):
    return re.findall(r"\w+", re.sub(r"<[^>]+>", "", text).lower())


def containment(a, b, n=8):
    def grams(text):
        words = normalize(text)
        return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}
    left, right = grams(a), grams(b)
    return len(left & right) / min(len(left), len(right)) if left and right else 0.0


def get(url, target, cap):
    req = Request(url, headers={"User-Agent": "tuneLLM-research-source-audit/1.0"})
    with urlopen(req, timeout=60) as response:
        body = response.read(cap + 1)
        if len(body) > cap:
            raise ValueError("source response exceeds byte cap")
    with target.open("xb") as stream:
        stream.write(body)
    return body


def execute():
    if DEST.exists():
        raise FileExistsError("fresh acquisition output required")
    pool = json.loads((SELECTION / "candidate_pool.json").read_text())
    review = json.loads((SELECTION / "theory_review.json").read_text())
    selection = json.loads((SELECTION / "registration.json").read_text())
    if review["candidate_pool_sha256"] != sha(SELECTION / "candidate_pool.json") or selection["candidate_pool_sha256"] != sha(SELECTION / "candidate_pool.json"):
        raise ValueError("candidate pool/review binding changed")
    decisions = {r["id"]: r for r in review["decisions"]}
    if len(decisions) != len(pool["candidates"]) or set(decisions) != {r["id"] for r in pool["candidates"]}:
        raise ValueError("complete, unique theory decisions required")
    candidates = [r for r in pool["candidates"] if decisions[r["id"]]["eligible"] is True]
    old = []
    for split in ("train", "validation", "test"):
        old += [json.loads(line) for line in (ROOT / f"data/physics-prl-01/{split}.jsonl").read_text().split("\n") if line.strip()]
    old_ids = {r["paper_id"] for r in old}
    protocol = {"scope": "bounded metadata/source acquisition for fresh-to-project PRL abstracts",
                "selection_sha256": sha(SELECTION / "registration.json"),
                "review_sha256": sha(SELECTION / "theory_review.json"), "target": 40,
                "candidate_order": [r["id"] for r in candidates], "max_seconds": 1800,
                "source_sha256": {name: sha(ROOT / name) for name in
                    ("scripts/qwen38_fetch_prl_test.py", "scripts/qwen38_prl_sources.py", "scripts/prepare_physics_corpus.py")},
                "near_duplicate_gate": "abstract 8gram containment >=0.6 against old100 or already selected fresh paper; normalized exact title also rejected",
                "identity_gate": "PRL Crossref journal, exact DOI, normalized title SequenceMatcher>=0.75",
                "license_gate": "versioned arXiv abs-license link explicitly CC-BY4.0",
                "shortfall": "retain n<40 without expanding the pool; report power and contamination limits"}
    registry = ROOT / "results/protocols/qwen38-prl-source-acquisition-01.json"
    if registry.exists():
        raise FileExistsError("source acquisition registration exists")
    DEST.mkdir(parents=True); (DEST / "raw").mkdir(); (DEST / "source").mkdir()
    for name in protocol["source_sha256"]:
        (DEST / "source" / Path(name).name).write_bytes((ROOT / name).read_bytes())
    write(DEST / "registration.json", protocol); write(registry, protocol)
    selected, attempts = [], []
    start = time.monotonic()
    for row in candidates:
        if len(selected) >= 40 or time.monotonic() - start >= 1800:
            break
        event = {"id": row["id"], "selected": False}
        try:
            if row["id"] in old_ids:
                raise ValueError("historical paper overlap")
            previous = old + selected
            if any(normalize(row["title"]) == normalize(p["title"]) or containment(row["abstract"], p["abstract"]) >= 0.6 for p in previous):
                raise ValueError("near-duplicate paper/abstract")
            doi = row.get("doi", "").strip()
            if not doi.startswith("10.1103/") or any(ch.isspace() for ch in doi):
                raise ValueError("single APS DOI required")
            raw = DEST / "raw" / row["id"]; raw.mkdir()
            record = json.loads(get("https://api.crossref.org/works/" + quote(doi, safe="/"), raw / "crossref.json", 3_000_000))["message"]
            title_ratio = SequenceMatcher(None, " ".join(normalize(row["title"])), " ".join(normalize(record["title"][0]))).ratio()
            if "Physical Review Letters" not in record.get("container-title", []) or record["DOI"].lower() != doi.lower() or title_ratio < 0.75:
                raise ValueError("Crossref PRL identity mismatch")
            version = row["versions"][-1]["version"]
            abs_url = f"https://arxiv.org/abs/{row['id']}{version}"
            html = get(abs_url, raw / "abstract.html", 3_000_000)
            match = re.search(r'class=["\']abs-license["\'].*?href=["\']([^"\']+)', html.decode(errors="replace"), re.S)
            if not match or "/licenses/by/4.0/" not in match.group(1):
                raise ValueError("authoritative arXiv CC-BY4 license not confirmed")
            time.sleep(3)
            url = f"https://arxiv.org/src/{row['id']}{version}"
            blob = get(url, raw / "source.archive", 60_000_000)
            time.sleep(3)
            cleaned = clean_fixed(source_files(blob), row["title"])
            license_record = {"url": abs_url, "license": match.group(1), "verified": True,
                              "response_sha256": sha(raw / "abstract.html")}
            paper = {**row, **cleaned, "id": row["id"] + version, "paper_id": row["id"],
                     "arxiv_version": version, "source_url": url, "source_sha256": sha(raw / "source.archive"),
                     "authoritative_license": license_record, "theory_review": decisions[row["id"]],
                     "crossref_sha256": sha(raw / "crossref.json"), "crossref_title_similarity": title_ratio,
                     "source_scope": "fresh-to-project independent test paper; potential pretraining exposure",
                     "raw_source_bytes": len(blob)}
            selected.append(paper)
            event.update({"selected": True, "version": version, "v1_date": row["v1_date"]})
        except Exception as exc:
            event["failure"] = f"{type(exc).__name__}: {exc}"
        attempts.append(event)
        with (DEST / "attempts.jsonl").open("a") as log:
            log.write(json.dumps(event) + "\n")
        print(json.dumps({"attempt": len(attempts), "selected": len(selected), **event}), flush=True)
    (DEST / "fresh.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    result = {"completed_acquisition": True, "target": 40, "papers": len(selected), "attempted": len(attempts),
              "shortfall": len(selected) < 40, "test_sha256": sha(DEST / "fresh.jsonl"),
              "category_counts": dict(Counter(r["categories"].split()[0] for r in selected)),
              "freshness": "independent of project training/development; not a pretraining exclusion guarantee",
              "elapsed_seconds": time.monotonic() - start}
    write(DEST / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(execute(), indent=2))


if __name__ == "__main__":
    main()
