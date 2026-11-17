"""Docs-only CPU tests; run with docs requirements and explicit -m ''."""
from pathlib import Path

import markdown
import pytest
from pymdownx.superfences import fence_code_format

from scripts import docs_hooks as docs
from scripts.check_docs_site import check


def context(tmp_path):
    folder = tmp_path/"docs"; folder.mkdir()
    for name in ("index.md", "website.md", "project_tracks.md", "old_protocol.md"):
        (folder/name).write_text("# "+name+"\n")
    (tmp_path/"results").mkdir(); (tmp_path/"results/a.json").write_text("{}")
    return docs.Context(tmp_path, folder, {"results/a.json"}, source=folder/"index.md")


def test_all_pages_have_navigation_including_future_archives(tmp_path):
    ctx = context(tmp_path)
    (ctx.docs/"nested").mkdir(); (ctx.docs/"nested/future.md").write_text("# Future\n")
    def leaves(value):
        if isinstance(value, str): return [value]
        if isinstance(value, list): return [p for x in value for p in leaves(x)]
        return [p for x in value.values() for p in leaves(x)]
    actual = leaves(docs.navigation(ctx.docs))
    assert set(actual) == set(docs.document_hashes(ctx.docs)) and len(actual) == len(set(actual))


def test_parsed_links_references_images_and_fences_preserve_source(tmp_path):
    ctx = context(tmp_path); before = docs.document_hashes(ctx.docs)
    source = '''[result](../results/a.json#record)
[archive](old_protocol.md)
[local][raw]

[raw]: ../outputs/private.json

![Unavailable](../data/private.png)

```python
example = "[code](../results/a.json)"
```

`[inline](../outputs/private.json)`
'''
    rendered = markdown.markdown(source, extensions=[docs.SiteLinksExtension(ctx), "fenced_code"])
    assert docs.REPO+"/blob/main/results/a.json#record" in rendered
    assert 'href="old_protocol.md"' in rendered
    assert 'href="website.md#local-only-references"' in rendered
    assert '(local only)' in rendered and 'src="../data/private.png"' not in rendered
    assert '[code](../results/a.json)' in rendered and '[inline](../outputs/private.json)' in rendered
    assert docs.document_hashes(ctx.docs) == before


def test_current_github_docs_links_and_source_line_links(tmp_path):
    ctx = context(tmp_path)
    assert docs.link_target(docs.REPO+"/blob/main/docs/old_protocol.md#heading", ctx) == ("old_protocol.md#heading", False)
    url = docs.REPO+"/blob/main/docs/old_protocol.md#L12"
    assert docs.link_target(url, ctx) == (url, False)


def test_archive_notice_is_build_only_and_current_pages_unchanged(tmp_path, monkeypatch):
    from types import SimpleNamespace
    ctx = context(tmp_path); before = docs.document_hashes(ctx.docs)
    monkeypatch.setattr(docs, "CONTEXT", ctx)
    page = SimpleNamespace(file=SimpleNamespace(abs_src_path=str(ctx.docs/"old_protocol.md"), src_uri="old_protocol.md"))
    original = (ctx.docs/"old_protocol.md").read_text()
    rendered = docs.on_page_markdown(original, page, None, None)
    assert "Historical record" in rendered and "[current roadmap](project_tracks.md)" in rendered
    assert rendered.endswith(original) and docs.document_hashes(ctx.docs) == before
    page.file.src_uri = "index.md"; page.file.abs_src_path = str(ctx.docs/"index.md")
    assert docs.on_page_markdown("# Current", page, None, None) == "# Current"


def test_math_and_mermaid_fences_render_without_touching_examples():
    source = r'''Inline \(x^2\).

\[
y = x + 1
\]

$$z=2$$

```mermaid
flowchart LR
  A --> B
```

```python
literal = r"\(not_math\)"
```
'''
    html = markdown.markdown(source, extensions=["pymdownx.arithmatex", "pymdownx.superfences"], extension_configs={
        "pymdownx.arithmatex": {"generic": True}, "pymdownx.superfences": {"custom_fences": [{"name": "mermaid", "class": "mermaid", "format": fence_code_format}]}})
    assert html.count('class="arithmatex"') == 3
    assert 'class="mermaid"' in html and 'A --&gt; B' in html
    assert r'\(not_math\)' in html


def test_generated_link_checker_rejects_missing_files_and_anchors(tmp_path):
    (tmp_path/"index.html").write_text('<a href="child/#ok">Good</a>')
    (tmp_path/"child").mkdir(); (tmp_path/"child/index.html").write_text('<h1 id="ok">Child</h1><a href="../">Home</a>')
    assert check(tmp_path)["broken"] == 0
    (tmp_path/"index.html").write_text('<a href="child/#bad">Missing anchor</a><img src="missing.png">')
    with pytest.raises(ValueError, match="missing anchor") as exc: check(tmp_path)
    assert "missing internal target" in str(exc.value)


def test_mermaid_wrapper_normalization_success_and_error_svg_source_fallback():
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node: pytest.skip("optional Node CPU check for browser renderer")
    # Execute the actual renderer, with a tiny DOM/renderer double. No CDN or browser.
    harness = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const context = {console: {error() {}}, document: {addEventListener() {}, createElement() {return {};}}};
vm.createContext(context);
vm.runInContext(fs.readFileSync('docs/javascripts/diagrams.js', 'utf8'), context);
function element(source) {
  return {source, wrapped: true, svg: false, error: false, text: source, dataset: {},
    querySelector(selector) {
      if (selector === 'code') return this.wrapped ? {textContent: this.source} : null;
      if (selector === 'svg') return this.svg ? {} : null;
      return this.error ? {} : null;
    },
    get textContent() { return this.text; },
    set textContent(value) { this.text=value; this.wrapped=false; this.svg=false; this.error=false; },
    removeAttribute(name) { this.removed=name; }, before(note) { this.note=note; }};
}
(async () => {
  const source = 'flowchart LR\n A["Quoted label"] --> B';
  const good=element(source), thrown=element(source), badSvg=element(source);
  const renderer={async run({nodes:[n]}) {
    assert.equal(n.textContent, source); assert.equal(n.wrapped, false);
    n.svg=true;
    if (n===thrown) {n.error=true; throw new Error('parse failed');}
    if (n===badSvg) n.error=true; // A nonthrowing syntax-error SVG is not success.
  }};
  await context.renderMermaidDiagrams([good, thrown, badSvg], renderer);
  assert.equal(good.dataset.mermaidState, 'rendered'); assert(good.svg);
  for (const n of [thrown, badSvg]) {
    assert.equal(n.dataset.mermaidState, 'source'); assert.equal(n.textContent, source);
    assert.equal(n.svg, false); assert.equal(n.removed, 'data-processed'); assert(n.note);
  }
})().catch(e => {console.error(e); process.exitCode=1;});
'''
    root = Path(__file__).resolve().parents[1]
    subprocess.run([node, "-e", harness], cwd=root, capture_output=True, text=True, check=True)
