"""Read only metadata columns of arXiv-LaTeX shards; never download the full corpus.

The initial exercise used shard indices 20 through 43 (2023 onward). This is a
convenience mirror. Authoritative source/version/license and PRL DOI verification
are separate mandatory steps in prepare_physics_corpus.py.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from scripts.common import path, write_json


def main():
    import fsspec
    import pyarrow.parquet as pq
    import requests
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=20)
    parser.add_argument("--stop", type=int, default=44)
    parser.add_argument("--root", default="data/physics-prl-01/raw")
    parser.add_argument("--candidate-index", help="reconstruct the exact reviewed shortlist from a committed index")
    args = parser.parse_args()
    root = path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "parquet_manifest.json"
    if manifest.exists():
        files = json.loads(manifest.read_text())
    else:
        response = requests.get("https://datasets-server.huggingface.co/parquet",
                                params={"dataset": "scholarweave/arxiv-latex"}, timeout=60)
        response.raise_for_status()
        files = response.json()["parquet_files"]
        write_json(manifest, files)
    base = root / "metadata"
    base.mkdir(exist_ok=True)
    columns = ["id", "title", "categories", "license", "doi", "journal-ref", "authors", "abstract", "versions", "update_date"]
    def read(item):
        target = base / (item["filename"] + ".json")
        if target.exists():
            return json.loads(target.read_text())
        with fsspec.open(item["url"], "rb", block_size=1024 * 1024) as stream:
            rows = pq.ParquetFile(stream).read(columns=columns).to_pylist()
        chosen = [r for r in rows if "cond-mat" in (r["categories"] or "") and
                  ("PhysRevLett" in (r["doi"] or "") or "Phys. Rev. Lett." in (r["journal-ref"] or "")
                   or "Phys.Rev.Lett." in (r["journal-ref"] or ""))]
        write_json(target, chosen)
        print(item["filename"], len(chosen), flush=True)
        return chosen
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [row for part in pool.map(read, files[args.start:args.stop]) for row in part]
    write_json(root / "candidates_all.json", rows)
    write_json(root / "candidates_ccby.json", [r for r in rows if "/by/" in (r["license"] or "")
                                               or "/publicdomain/zero/" in (r["license"] or "")])
    if args.candidate_index:
        index = json.loads(path(args.candidate_index).read_text())
        by_id = {row["id"]: row for row in rows}
        selected = []
        for entry in index["candidates"]:
            row = dict(by_id[entry["id"]])
            # Reproduce the recorded version selection, even if the mirror later updates.
            row.update({key: entry[key] for key in ["title", "doi", "journal-ref", "categories", "license", "versions"]})
            selected.append(row)
        (root / "theory_candidates.json").write_text(json.dumps(selected, indent=2))
        write_json(root.parent / "theory_review.json", {"reviewer": index["reviewer"], "criteria": index["review_criteria"],
                                                      "decisions": [row["theory_review"] for row in index["candidates"]]})


if __name__ == "__main__":
    main()
