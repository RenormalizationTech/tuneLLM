"""Acquire reviewed CC-BY PRL sources and prepare paper-disjoint LaTeX training data."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import re
import tarfile
import time

from scripts.common import path, write_json, digest


def strip_comments(text):
    text = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", "", text, flags=re.S)
    for _ in range(10):
        cleaned = re.sub(r"\\iffalse\b(?:(?!\\iffalse\b|\\fi\b).)*\\fi\b", "", text, flags=re.S)
        if cleaned == text:
            break
        text = cleaned
    result = []
    for line in text.split("\n"):
        end = len(line)
        for i, char in enumerate(line):
            if char != "%":
                continue
            backslashes, j = 0, i - 1
            while j >= 0 and line[j] == "\\":
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                end = i
                break
        result.append(line[:end])
    return "\n".join(result)


def braced(text, start):
    """Read a balanced TeX brace argument, retaining its nested content."""
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        raise ValueError("expected brace argument")
    depth, begin = 1, start + 1
    for i in range(begin, len(text)):
        if text[i] in "{}" and (i == 0 or text[i - 1] != "\\"):
            depth += 1 if text[i] == "{" else -1
            if not depth:
                return text[begin:i], i + 1
    raise ValueError("unbalanced brace argument")


def remove_commands(text, commands):
    pattern = re.compile(r"\\(?:" + "|".join(commands) + r")\b\*?\s*(?:\[[^\]]*\])?\s*")
    result, start = [], 0
    for match in pattern.finditer(text):
        if match.start() < start:
            continue
        result.append(text[start:match.start()])
        try:
            _, start = braced(text, match.end())
        except ValueError:
            start = match.end()
    result.append(text[start:])
    return "".join(result)


def source_files(blob):
    """Read archives without extracting paths or executing TeX."""
    files = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as archive:
            for member in archive.getmembers():
                p = PurePosixPath(member.name)
                if (member.isfile() and not p.is_absolute() and ".." not in p.parts
                        and p.suffix.lower() in {".tex", ".bbl", ".bib"}
                        and member.size < 8_000_000):
                    stream = archive.extractfile(member)
                    files[str(p)] = stream.read().decode("utf-8", errors="replace")
    except tarfile.ReadError:
        raw = gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob
        files["main.tex"] = raw.decode("utf-8", errors="replace")
    if not files or sum(map(len, files.values())) > 30_000_000:
        raise ValueError("missing or oversized text source")
    return files


def clean_source(files, metadata_title):
    candidates = [(name, strip_comments(text)) for name, text in files.items()
                  if name.endswith(".tex") and "\\begin{document}" in strip_comments(text)]
    if not candidates:
        raise ValueError("no main TeX document")
    def score(item):
        name, text = item
        return (int("\\begin{abstract}" in text), int(not re.search(r"supp|append|sm\.", name, re.I)),
                int("\\title" in text), len(text))
    name, main = max(candidates, key=score)

    def expand(text, parent, seen):
        def replace(match):
            target = match.group(1).strip()
            if not target.endswith(".tex"):
                target += ".tex"
            full = str(PurePosixPath(parent).parent / target)
            if full not in files and target in files:
                full = target
            if full in seen or full not in files or ".." in PurePosixPath(full).parts:
                return " "
            return expand(strip_comments(files[full]), full, seen | {full})
        return re.sub(r"\\(?:input|include)\s*\{([^}]+)\}", replace, text)

    main = expand(main, name, {name})
    start = main.index("\\begin{document}") + len("\\begin{document}")
    preamble = main[:start]
    body = main[start:].split("\\end{document}")[0]
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
    if not match:
        raise ValueError("missing abstract environment")
    abstract = remove_commands(match.group(1), ["author", "affiliation", "email", "thanks", "date"]).strip()
    # Everything before the abstract is title/author/date/affiliation material.
    body = body[match.end():]
    body = remove_commands(body, ["author", "affiliation", "address", "email", "thanks", "date",
                                  "altaffiliation", "collaboration", "homepage", "pacs", "keywords",
                                  "todo", "marginpar", "deleted"])
    body = re.sub(r"\\(?:begin|end)\{CJK\*?\}(?:\{[^}]*\}){0,2}", "", body)
    body = re.sub(r"\\(?:maketitle|tableofcontents|pacs|keywords)\b(?:\s*\{[^}]*\})?", "", body)
    # Keep the Letter main text, excluding back matter and supplemental text.
    stops = [m.start() for m in re.finditer(
        r"\\(?:appendix\b|bibliography\b|bibliographystyle\b)|"
        r"\\begin\{(?:thebibliography|acknowledgments?|acknowledgements?)\}|"
        r"\\(?:section|paragraph|textbf|textit|emph)\*?\s*\{[^}]*(?:Acknowledg|Supplement|Appendix)|"
        r"\\acknowledg(?:e)?ments?\b|\b(?:We thank|We acknowledge|We are grateful|This work was supported|Acknowledgments|Acknowledgements)\b", body, re.I)]
    if stops:
        body = body[:min(stops)]
    funding = re.search(r"\b(?:financial support|funded by|funding from|acknowledge funding|conceived this work|grant agreement|This work is supported|Grant\s+(?:No|Number))\b", body, re.I)
    if funding:
        # Funding often starts with author initials rather than an acknowledgment heading.
        body = body[:max(0, body.rfind("\n\n", 0, funding.start()))]
    body = re.sub(r"\\(?:includegraphics|graphicspath)(?:\[[^\]]*\])?\s*\{[^}]*\}", "", body)
    body = re.sub(r"\\(?:label)\s*\{[^}]*\}", "", body)
    body = re.sub(r"\n[ \t]+", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if len(body) < 3000 or len(abstract) < 100:
        raise ValueError("main text or abstract too short")
    if re.search(r"\b(?:Grant\s+(?:No|Number)|funded by|grant agreement)\b", body, re.I):
        raise ValueError("possible funding/acknowledgment text remains")
    annotations = sorted(set(re.findall(r"\\(?:todo|textcolor|added|deleted|replaced|sout|hl)\b", body)))
    # Capture an introductory passage, then exclude it from writing input.
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    n = 0
    intro = []
    for p in paras:
        if intro and (sum(map(len, intro)) >= 2200 or len(intro) >= 4):
            break
        intro.append(p)
        n += 1
    remainder = "\n\n".join(paras[n:])
    if len(remainder) < 1500:
        raise ValueError("insufficient body after introductory passage")
    title = re.sub(r"\s+", " ", metadata_title).strip()
    definitions = []
    for match in re.finditer(r"\\(?:newcommand|renewcommand|providecommand)\*?\s*(?:\{(\\[A-Za-z]+)\}|(\\[A-Za-z]+))\s*(?:\[[0-9]+\])?\s*", preamble):
        macro = match.group(1) or match.group(2)
        if not re.search(re.escape(macro) + r"\b", abstract + body):
            continue
        try:
            _, end = braced(preamble, match.end())
            definitions.append(preamble[match.start():end])
        except ValueError:
            continue
    definitions_text = "\n".join(definitions)
    text = definitions_text + f"\n\\title{{{title}}}\n\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}\n\n{body}"
    text = text.lstrip()
    bibkeys = sorted(set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\s*\{([^}]+)\}", "\n".join(files.values()))))
    return {"text": text, "title": title, "abstract": abstract,
            "introductory_passage": "\n\n".join(intro), "writing_source": definitions_text + "\n\n" + remainder,
            "bibliography_keys": bibkeys, "main_file": name,
            "annotation_macros": annotations, "main_text_characters": len(body),
            "long_main_text_flag": len(body) > 40000,
            "intro_definition": "first prose blocks up to 2200 characters or four blocks; heuristic, not a human section annotation"}


def fetch_candidates(candidates, out):
    import requests
    out.mkdir(parents=True, exist_ok=True)
    def crossref(row):
        target = out / (row["id"].replace("/", "_") + ".crossref.json")
        if target.exists():
            return
        for attempt in range(5):
            url = "https://api.crossref.org/works/" + row["doi"] if row.get("doi") else "https://api.crossref.org/works"
            params = None if row.get("doi") else {"query.title": row["title"], "filter": "prefix:10.1103", "rows": 1}
            response = requests.get(url, params=params, timeout=45,
                                    headers={"User-Agent": "tuneLLM-educational-corpus/1.0"})
            if response.status_code not in {429, 502, 503, 504}:
                break
            time.sleep(min(30, 3 * (attempt + 1)))
        response.raise_for_status()
        result = response.json()
        if not row.get("doi"):
            result = {"message": result["message"]["items"][0], "lookup": "title query; title agreement required"}
        write_json(target, result)
        print("verified DOI", row["id"], flush=True)
        time.sleep(1.5)
    for row in candidates:
        crossref(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["verify-dois", "sources", "verify-licenses", "prepare"])
    parser.add_argument("--root", default="data/physics-prl-01")
    args = parser.parse_args()
    root = path(args.root)
    candidates = json.loads((root / "raw/theory_candidates.json").read_text())
    if args.phase == "verify-dois":
        fetch_candidates(candidates, root / "raw/crossref")
        return
    review = json.loads((root / "theory_review.json").read_text())
    decisions = {row["id"]: row for row in review["decisions"]}
    eligible = []
    for row in candidates:
        if not decisions.get(row["id"], {}).get("eligible"):
            continue
        record = json.loads((root / "raw/crossref" / (row["id"].replace("/", "_") + ".crossref.json")).read_text())["message"]
        if ("Physical Review Letters" not in record.get("container-title", [])
                or re.match(r"^(Comment|Reply|Erratum|Editorial|Correction)", record["title"][0], re.I)):
            continue
        if not row.get("doi"):
            from difflib import SequenceMatcher
            normalize = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
            if SequenceMatcher(None, normalize(row["title"]), normalize(record["title"][0])).ratio() < 0.85:
                continue
            row["doi"] = record["DOI"]
        eligible.append(row)
    # Seeded round-robin across primary categories: balanced as supply allows.
    groups = defaultdict(list)
    for row in sorted(eligible, key=lambda x: x["id"]):
        groups[row["categories"].split()[0]].append(row)
    rng = random.Random(20261112)
    for group in groups.values():
        rng.shuffle(group)
    order = []
    while any(groups.values()):
        for category in sorted(groups):
            if groups[category]:
                order.append(groups[category].pop())
    write_json(root / "eligible_order.json", order)
    sources = root / "raw/sources"
    sources.mkdir(parents=True, exist_ok=True)
    if args.phase == "verify-licenses":
        import requests
        for row in order:
            version = row["versions"][-1]["version"]
            source = sources / f"{row['id'].replace('/', '_')}{version}.source"
            if not source.exists():
                continue
            try:
                clean_source(source_files(source.read_bytes()), row["title"])
            except ValueError:
                continue
            target = sources / f"{row['id'].replace('/', '_')}{version}.license.json"
            if not target.exists():
                url = f"https://arxiv.org/abs/{row['id']}{version}"
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                match = re.search(r'class="abs-license".*?href="([^"]+)"', response.text, re.S)
                verified = bool(match and "/licenses/by/4.0/" in match.group(1))
                write_json(target, {"url": url, "license": match.group(1) if match else None, "verified": verified,
                                   "response_sha256": hashlib.sha256(response.content).hexdigest()})
                print("license", verified, row["id"], flush=True)
                time.sleep(3)
        return
    if args.phase == "sources":
        import requests
        count = 0
        for row in order:
            version = row["versions"][-1]["version"]
            target = sources / f"{row['id'].replace('/', '_')}{version}.source"
            license_file = target.with_name(target.name.replace(".source", ".license.json"))
            if license_file.exists() and not json.loads(license_file.read_text()).get("verified", True):
                continue
            if not target.exists():
                url = f"https://arxiv.org/src/{row['id']}{version}"
                try:
                    response = requests.get(url, timeout=90)
                    response.raise_for_status()
                    if len(response.content) > 60_000_000:
                        raise ValueError("source larger than 60 MB")
                    target.write_bytes(response.content)
                except Exception as exc:
                    print("source failure", row["id"], str(exc), flush=True)
                    continue
                time.sleep(3)
            try:
                clean_source(source_files(target.read_bytes()), row["title"])
                count += 1
                print("usable", count, row["id"], flush=True)
            except Exception as exc:
                print("clean failure", row["id"], str(exc), flush=True)
            if count >= 100:
                break
        return
    prepared, failures = [], []
    if any((root / f"{split}.jsonl").exists() for split in ("train", "validation", "test")):
        raise FileExistsError("refusing to overwrite prepared paper splits")
    for row in order:
        version = row["versions"][-1]["version"]
        target = sources / f"{row['id'].replace('/', '_')}{version}.source"
        if not target.exists():
            continue
        try:
            cleaned = clean_source(source_files(target.read_bytes()), row["title"])
        except Exception as exc:
            failures.append({"id": row["id"], "reason": str(exc)})
            continue
        license_file = target.with_name(target.name.replace(".source", ".license.json"))
        if not license_file.exists():
            failures.append({"id": row["id"], "reason": "authoritative license not verified"})
            continue
        if not json.loads(license_file.read_text()).get("verified", True):
            failures.append({"id": row["id"], "reason": "authoritative CC BY license not confirmed"})
            continue
        prepared.append({**row, **cleaned, "paper_id": row["id"], "id": row["id"] + version,
                         "arxiv_version": version, "source_sha256": digest(target),
                         "source_url": f"https://arxiv.org/src/{row['id']}{version}",
                         "theory_review": decisions[row["id"]],
                         "authoritative_license": json.loads(license_file.read_text())})
        if len(prepared) == 100:
            break
    if len(prepared) != 100:
        raise ValueError(f"need 100 usable papers; have {len(prepared)}")
    # Partition each category, then allocate exact global sizes. No text chunking yet.
    rng = random.Random(42)
    grouped = defaultdict(list)
    for row in prepared:
        grouped[row["categories"].split()[0]].append(row)
    for category in sorted(grouped):
        rng.shuffle(grouped[category])
    splits = {"train": [], "validation": [], "test": []}
    # Largest-remainder allocation with at least one paper per category in dev/test.
    def allocate(total):
        result = {cat: max(1, int(len(rows) * total / 100)) for cat, rows in grouped.items()}
        while sum(result.values()) < total:
            cat = max(result, key=lambda c: (len(grouped[c]) * total / 100 - result[c], c))
            result[cat] += 1
        while sum(result.values()) > total:
            cat = max((c for c in result if result[c] > 1), key=lambda c: result[c] - len(grouped[c]) * total / 100)
            result[cat] -= 1
        return result
    dev_allocation, test_allocation = allocate(10), allocate(20)
    for cat, rows in grouped.items():
        ndev, ntest = dev_allocation[cat], test_allocation[cat]
        if len(rows) <= ndev + ntest:
            raise ValueError("category has insufficient papers for three splits")
        splits["validation"] += rows[:ndev]
        splits["test"] += rows[ndev:ndev + ntest]
        splits["train"] += rows[ndev + ntest:]
    for rows in splits.values():
        rng.shuffle(rows)
    manifest = []
    for split, rows in splits.items():
        for row in rows:
            manifest.append({k: v for k, v in row.items() if k not in
                             {"text", "abstract", "introductory_passage", "writing_source", "bibliography_keys"}} | {"split": split,
                             "clean_text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()})
    for split, rows in splits.items():
        (root / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    author_papers = defaultdict(lambda: defaultdict(list))
    for split, rows in splits.items():
        for row in rows:
            for author in re.split(r",\s*(?:and\s+)?|\s+and\s+", row["authors"]):
                author = re.sub(r"\s+", " ", author).strip()
                if len(author) >= 4:
                    author_papers[author][split].append(row["paper_id"])
    overlap = {author: dict(places) for author, places in author_papers.items() if len(places) > 1}
    write_json(root / "provenance.json", {"papers": manifest, "counts": {k: len(v) for k, v in splits.items()},
        "category_counts": dict(Counter(r["categories"].split()[0] for r in prepared)),
        "split_seed": 42, "selection_seed": 20261112,
        "dev_category_allocation": dev_allocation, "test_category_allocation": test_allocation,
        "author_overlap": {"count": len(overlap), "authors_and_papers": overlap,
                           "method": "heuristic comma/and splitting of metadata author strings; not identity-resolved"},
        "license_policy": "arXiv metadata explicitly CC BY; publisher status verified through Crossref",
        "theory_review_sha256": digest(root / "theory_review.json"), "clean_failures": failures,
        "limits": ["Theory eligibility reviewed by an AI advisor from abstracts, not a human expert.",
                   "Author/group overlap across splits is reported, not guaranteed absent.",
                   "Source mirror metadata used for license/version selection; upstream source archives downloaded directly."]})
    print({k: len(v) for k, v in splits.items()}, flush=True)


if __name__ == "__main__":
    main()
