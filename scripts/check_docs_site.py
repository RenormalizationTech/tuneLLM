"""Offline generated-site file/anchor validation; no MkDocs or ML imports."""
import argparse
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


def check(site, site_url="https://renormalizationtech.github.io/tuneLLM/"):
    site = Path(site).resolve(); public = urlsplit(site_url)
    pages = {}
    for file in site.rglob("*.html"):
        parser = Links(); parser.feed(file.read_text()); pages[file.resolve()] = parser
    if not pages: raise ValueError("no generated HTML pages")
    errors, checked = [], 0
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
