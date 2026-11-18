"""Prepare a separate PRL abstract corpus without changing the historical study.

The new extractor preserves inline Appendix references that the historical
back-matter heuristic could mistake for headings. Source archives stay intact.
Fresh-candidate selection is metadata-only; eligibility review is separate.
"""
import argparse
from collections import Counter
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import re

from scripts.prepare_physics_corpus import braced, clean_source, source_files
from scripts.qwen38_science_runtime import sha, write

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/physics-prl-01"
NEW = ROOT / "data/qwen38-prl-abstract-01"


def protect_inline_references(text):
    replacements = []
    for match in re.finditer(r"\\(textbf|textit|emph)\s*", text):
        try:
            argument, end = braced(text, match.end())
        except ValueError:
            continue
        if not re.search(r"Appendix|Supplement|Acknowledg", argument, re.I):
            continue
        left = text[text.rfind("\n", 0, match.start()) + 1:match.start()]
        stop = text.find("\n", end)
        right = text[end:stop if stop >= 0 else len(text)]
        if left.strip() or right.strip():
            replacements.append((match.start(), match.end(), "\\tuneInline" + match.group(1) + " "))
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    return text


def clean_fixed(files, title):
    protected = {name: protect_inline_references(text) for name, text in files.items()}
    result = clean_source(protected, title)
    for key, value in result.items():
        if isinstance(value, str):
            result[key] = re.sub(r"\\tuneInline(textbf|textit|emph)\s*", lambda m: "\\" + m.group(1), value)
    body = result["text"].split("\\end{abstract}", 1)[1].strip()
    result["main_text_characters"] = len(body)
    result["long_main_text_flag"] = len(body) > 40000
    result["extractor"] = "historical main-body cleaner with inline-reference protection; original source unchanged"
    return result


def reextract():
    dest = NEW / "historical-reextracted-02"
    if dest.exists():
        raise FileExistsError("fresh re-extraction destination required")
    dest.mkdir(parents=True)
    counts, changes, files = {}, [], {}
    for split in ("train", "validation", "test"):
        source = OLD / f"{split}.jsonl"
        old = [json.loads(line) for line in source.read_text().split("\n") if line.strip()]
        rows = []
        for row in old:
            archive = OLD / "raw/sources" / (row["id"].replace("/", "_") + ".source")
            if sha(archive) != row["source_sha256"]:
                raise ValueError(f"historical source changed: {row['id']}")
            cleaned = clean_fixed(source_files(archive.read_bytes()), row["title"])
            item = {**row, **cleaned, "historical_split": split,
                    "historical_row_sha256": sha_bytes(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()),
                    "reextracted_for": "Qwen3.8 grounded abstracts; historical study unchanged"}
            rows.append(item)
            changes.append({"id": row["id"], "split": split,
                            "old_body_characters": row["main_text_characters"],
                            "new_body_characters": item["main_text_characters"],
                            "abstract_changed": item["abstract"] != row["abstract"]})
        target = dest / f"{split}.jsonl"
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        files[split] = {"input_sha256": sha(source), "output_sha256": sha(target)}
        counts[split] = len(rows)
    manifest = {"counts": counts, "files": files, "changes": changes,
                "source_sha256": sha(__file__), "legacy_cleaner_sha256": sha(ROOT / "scripts/prepare_physics_corpus.py"),
                "historical_splits_unchanged": counts == {"train": 70, "validation": 10, "test": 20},
                "test_status": "exposed historical diagnostic; not fresh final data"}
    write(dest / "manifest.json", manifest)
    return {"destination": str(dest.relative_to(ROOT)), "counts": counts,
            "changed_bodies": sum(r["old_body_characters"] != r["new_body_characters"] for r in changes),
            "changed_abstracts": sum(r["abstract_changed"] for r in changes)}


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def prepare_pool():
    dest = NEW / "selection"
    if dest.exists():
        raise FileExistsError("fresh candidate pool required")
    pool_file = OLD / "raw/candidates_ccby.json"
    pool = json.loads(pool_file.read_text())
    old = json.loads((OLD / "provenance.json").read_text())
    exclude = {r["paper_id"] for r in old["papers"]}
    review = json.loads((OLD / "theory_review.json").read_text())
    prior = {r["id"]: r for r in review["decisions"]}
    def date(row):
        return parsedate_to_datetime(row["versions"][0]["created"]).isoformat()
    candidates = [r for r in pool if r["id"] not in exclude
                  and r["categories"].split()[0].startswith("cond-mat.")
                  and not re.match(r"^(Comment|Reply|Erratum|Editorial|Correction)\b", r["title"], re.I)
                  and not (r["id"] in prior and not prior[r["id"]]["eligible"])]
    candidates.sort(key=lambda r: (date(r), sha_bytes(("20261117:" + r["id"]).encode())), reverse=True)
    selected = [{**row, "v1_date": date(row), "prior_theory_review": prior.get(row["id"])} for row in candidates[:80]]
    dest.mkdir(parents=True)
    write(dest / "candidate_pool.json", {"criteria": review["criteria"], "candidates": selected})
    protocol = {"scope": "fresh-to-project PRL test source selection; no target-model outputs",
                "candidate_pool_sha256": sha(dest / "candidate_pool.json"), "candidate_source_sha256": sha(pool_file),
                "old_provenance_sha256": sha(OLD / "provenance.json"), "source_sha256": sha(__file__),
                "candidate_count": len(selected), "target_papers": 40,
                "order": "v1 date descending; seeded SHA256 tie break; first passing candidates in this order",
                "gates": ["independent paper/version from historical100", "same abstract-based theory eligibility",
                          "Crossref PRL journal/DOI identity", "authoritative versioned arXiv CC-BY4 license",
                          "clean source and near-duplicate audit"],
                "selection_date": "2026-11-17", "freshness": "independent of project train/dev; may predate model and be in pretraining",
                "category_counts": dict(Counter(r["categories"].split()[0] for r in selected)),
                "no_automatic_pool_expansion": True}
    write(dest / "registration.json", protocol)
    registry = ROOT / "results/protocols/qwen38-prl-fresh-selection-01.json"
    if registry.exists():
        raise FileExistsError("fresh selection registration already exists")
    write(registry, protocol)
    return {"selection": str(dest.relative_to(ROOT)), "candidate_count": len(selected),
            "newest_v1": selected[0]["v1_date"], "oldest_v1": selected[-1]["v1_date"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reextract-historical", action="store_true")
    parser.add_argument("--prepare-pool", action="store_true")
    args = parser.parse_args()
    if args.reextract_historical == args.prepare_pool:
        parser.error("select exactly one operation")
    print(json.dumps(reextract() if args.reextract_historical else prepare_pool(), indent=2))


if __name__ == "__main__":
    main()
