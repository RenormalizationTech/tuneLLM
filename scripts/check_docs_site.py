"""Offline site file, anchor and unprocessed-TeX checks; no MkDocs/ML imports."""
import argparse
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids = set(); self.links = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"): self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("name"): self.ids.add(attrs["name"])
        for attr in ("href", "src"):
            if attrs.get(attr): self.links.append(attrs[attr])
    handle_startendtag = handle_starttag


class MathMarkup(HTMLParser):
    """Find TeX the Markdown parser left outside math or literal examples.

    Checking only MathJax containers misses formulas never recognized as math.
    Scan the unprocessed text independently, including text split by emphasis.
    """
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    TEX = re.compile(r"\\(?:[A-Za-z]+|[()\[\]])")

    def __init__(self):
        super().__init__(); self.stack = []; self.unprocessed = []

    def handle_starttag(self, tag, attrs):
        if tag in self.VOID: return
        attrs = dict(attrs)
        excluded = tag in {"script", "style", "code", "pre"} or "arithmatex" in attrs.get("class", "").split()
        self.stack.append((tag, excluded))

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        for index in range(len(self.stack)-1, -1, -1):
            if self.stack[index][0] == tag:
                self.stack = self.stack[:index]
                break

    def handle_data(self, data):
        if not any(excluded for _, excluded in self.stack) and self.TEX.search(data):
            self.unprocessed.append(" ".join(data.split())[:120])


def check(site, site_url="https://renormalizationtech.github.io/tuneLLM/"):
    site = Path(site).resolve(); public = urlsplit(site_url)
    pages = {}; math_errors = []
    for file in site.rglob("*.html"):
        source = file.read_text()
        parser = Links(); parser.feed(source); pages[file.resolve()] = parser
        math = MathMarkup(); math.feed(source)
        for text in math.unprocessed:
            math_errors.append(f"{file.relative_to(site)}: unprocessed math notation outside math/code: {text}")
    if not pages: raise ValueError("no generated HTML pages")
    errors, checked = math_errors, 0
    for source, page in pages.items():
        for url in page.links:
            parsed = urlsplit(url)
            if parsed.netloc:
                if parsed.netloc != public.netloc or not parsed.path.startswith(public.path): continue
                path = parsed.path[len(public.path):]
                target = site/unquote(path)
            elif parsed.scheme: continue
            elif not parsed.path: target = source
            elif parsed.path.startswith("/"):
                path = parsed.path[len(public.path):] if parsed.path.startswith(public.path) else parsed.path.lstrip("/")
                target = site/unquote(path)
            else: target = source.parent/unquote(parsed.path)
            target = target.resolve()
            if target.is_dir(): target = target/"index.html"
            checked += 1
            if not target.is_relative_to(site) or not target.is_file():
                errors.append(f"{source.relative_to(site)}: missing internal target {url}")
            elif parsed.fragment and target.suffix == ".html" and unquote(parsed.fragment) not in pages[target].ids:
                errors.append(f"{source.relative_to(site)}: missing anchor {url}")
    if errors: raise ValueError("\n".join(errors))
    return {"html_pages": len(pages), "internal_links_checked": checked, "broken": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", nargs="?", default="outputs/docs-site-01/site")
    parser.add_argument("--site-url", default="https://renormalizationtech.github.io/tuneLLM/")
    args = parser.parse_args(); print(check(args.site, args.site_url))


if __name__ == "__main__": main()
