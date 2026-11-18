# Build and maintain this documentation site

The website renders the existing `docs/` Markdown with MkDocs and publishes it
through GitHub Pages. Markdown remains the source for both GitHub and the site;
there is no separate copy of the research narrative to maintain.

## Local build

Install the documentation-only requirements in a separate environment. Model
packages, weights, GPUs and judge credentials are not needed.

```bash
python3 -m venv outputs/docs-site-env
outputs/docs-site-env/bin/python -m pip install -r requirements-docs.txt
outputs/docs-site-env/bin/python -m mkdocs build --strict
outputs/docs-site-env/bin/python -m mkdocs serve
```

The preview command prints the local URL. Generated HTML is under ignored
`outputs/docs-site-01/site/`. Check all generated internal links with:

```bash
outputs/docs-site-env/bin/python -B -m scripts.check_docs_site outputs/docs-site-01/site
```

## Links and page organization

Links between Markdown files under `docs/` become website links. Images and
other documentation assets remain local website assets. During the build,
relative links to tracked code, configurations, JSON results, release cards or
root documents outside `docs/` are routed to the public GitHub repository.
The original Markdown files are not rewritten.

## Local only references

Historical references to ignored local artifacts are not downloadable website
assets. The site identifies such references and directs readers to the
[reproduction guide](reproduce.md). A machine path in an old protocol is not a
portable setup instruction.

## Finding your way around

The persistent left sidebar contains site-wide navigation and search, with
short links to the overview, results, 27B releases, reproduction and project
status. Expand a research track for its guides. All historical protocols and
closeouts are grouped under **Research archive**, collapsed unless you are
reading an archived page. The current page is highlighted; its archive ancestors
open automatically. The release catalog, GitHub source and Hugging Face are
direct resource links in the sidebar.

**On this page** is a separate heading outline: on the right of wide screens,
or a collapsible panel above the article on smaller screens. On phones, the
**Menu** button opens the sidebar, including search; **Escape** or **Close menu**
closes it and restores keyboard focus. There are no sequential Previous/Next
controls. Every Markdown page retains its URL and is included once in the site
tree and search index. Publishing a historical page does not change its
experimental status or acceptance decision.

Each tuning section starts with **Theoretical background**, before its
workflow and result pages. The shared background under Start here specifies
notation, objectives and evaluation principles. These pages present analytical
notes, numerical illustrations and primary references in the context of the
research portfolio. Results and releases precede the shared theory page in the
overview navigation; existing page URLs are retained. Frozen experiment records
keep their original conclusions.

Put a blank line between consecutive display-math blocks. Otherwise Markdown
can treat both equations as prose before MathJax sees them. The site checker
rejects TeX left outside math containers or literal code examples; the theory-page
tests also compare source-delimited equation counts with parsed math counts.
Browser checks then verify typesetting and errors. Counting rendered containers
alone cannot detect equations that never reached the math parser.

The layout uses MkDocs' bundled Read the Docs theme, a small Jinja override in
`docs/theme/`, and local navigation CSS/JavaScript. No model libraries or
additional theme dependency are required. The [MkDocs theme documentation](https://www.mkdocs.org/user-guide/choosing-your-theme/#readthedocs)
describes the underlying theme options.

## Publishing

The documentation workflow builds and checks the site on pull requests. A push
to the main branch can publish the built site to GitHub Pages. The existing
portable CPU-test workflow remains separate from documentation deployment.

GitHub Pages must be configured under the repository's **Settings → Pages**
with **GitHub Actions** as the publishing source. A repository administrator
may need to enable this once; ordinary workflow tokens cannot enable a new
Pages site with administration permissions. After enabling it, open **Actions →
Documentation site → Run workflow** on `main` (or rerun the latest deployment).
The published URL is <https://renormalizationtech.github.io/tuneLLM/>.

Use the source links on pages to inspect the original Markdown. Update shared
facts in their research records and current summaries; preserve hash-bound
historical protocols and results. The site build provides an additional reading
interface, not a new experiment or a regrading of previous results.
