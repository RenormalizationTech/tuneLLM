"""Build-only navigation and parsed-link adaptation; never rewrite doc sources.

Markdown tree processing leaves fenced/inline code and examples intact. Source
links outside docs point to tracked public GitHub files. Unavailable/local-only
targets are explicitly labelled and explained on the website page.
"""
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from xml.etree.ElementTree import Element, SubElement

from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor

REPO = "https://github.com/RenormalizationTech/tuneLLM"
CURRENT = {
    "Start here": ["index.md", "showcase.md", "reproduce.md", "project_tracks.md", "learning_foundations.md"],
    "Quantization": ["quantization_foundations.md", "qwen38_quantization_results.md", "benchmark_summary.md", "execution.md"],
    "LoRA and QLoRA": ["qlora_foundations.md", "qwen38_qlora_results.md", "physics_prl_execution.md", "physics_prl_protocol.md", "model_workflow.md"],
    "Refusal steering": ["refusal_foundations.md", "refusal_7b_transfer_execution.md", "activation_steering_methods.md", "refusal_reproducibility_review.md", "xstest_baseline_execution.md", "xstest_stored_output_audit.md"],
    "Trait steering": ["trait_foundations.md", "steering_reference.md", "qwen38_sarcasm_results.md", "qwen38_sarcasm_release.md", "profanity_controller_workflow.md", "sarcasm_controller_workflow.md", "gcad_and_combined_control.md", "persona_literature_workflow.md"],
    "Guides and references": ["references.md", "file_organization.md", "website.md"],
}
NAV_LABELS = {
    "learning_foundations.md": "Theoretical background",
    "quantization_foundations.md": "Theoretical background",
    "qlora_foundations.md": "Theoretical background",
    "refusal_foundations.md": "Theoretical background",
    "trait_foundations.md": "Theoretical background",
    "index.md": "Overview", "showcase.md": "Results and portfolio",
    "qwen38_quantization_results.md": "27B NF4 and Q4_K_M",
    "qwen38_qlora_results.md": "27B QASPER and PRL adapters",
    "qwen38_release_plan.md": "27B results and releases", "reproduce.md": "Reproduce",
    "project_tracks.md": "Project status", "benchmark_summary.md": "4B measurements",
    "execution.md": "Quantization workflow", "physics_prl_execution.md": "4B physics results",
    "physics_prl_protocol.md": "4B physics protocol", "model_workflow.md": "Model workflow",
    "refusal_7b_transfer_execution.md": "7B results", "activation_steering_methods.md": "Methods",
    "refusal_reproducibility_review.md": "Reproducibility review",
    "xstest_baseline_execution.md": "XSTest baseline", "xstest_stored_output_audit.md": "Stored-output audit",
    "steering_reference.md": "Results and controls", "profanity_controller_workflow.md": "Profanity",
    "qwen38_sarcasm_results.md": "27B sarcasm results",
    "qwen38_sarcasm_plan.md": "27B sarcasm study",
    "qwen38_sarcasm_execution.md": "27B sarcasm execution",
    "qwen38_sarcasm_followup_execution.md": "27B sarcasm follow-up",
    "qwen38_sarcasm_gcad_guide.md": "27B GCAD theory and tuning",
    "qwen38_sarcasm_hf_card_draft.md": "27B HF card draft (historical)",
    "qwen38_sarcasm_release.md": "Use the 27B controls",
    "sarcasm_controller_workflow.md": "Sarcasm", "gcad_and_combined_control.md": "Joint controls",
    "persona_literature_workflow.md": "Literature workflow", "references.md": "References",
    "file_organization.md": "File organization", "website.md": "Website guide",
}


def digest(file): return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def document_hashes(docs):
    return {p.relative_to(docs).as_posix(): digest(p) for p in sorted(docs.rglob("*.md"))}


def title(file):
    for line in file.read_text().splitlines():
        if line.startswith("# "): return line[2:].strip()
    return file.stem.replace("_", " ").capitalize()


def archive_group(name):
    if name.startswith(("physics_", "real_run", "next_sft")): return "LoRA and QLoRA archive"
    if name.startswith(("refusal_", "benign_", "xstest_", "deccp_", "political_", "qwen3_14b", "qwen3_4b_closeout")): return "Refusal archive"
    if name.startswith(("author_", "persona_", "sarcasm_", "profanity_", "isolated_", "joint_", "tone_", "qwen3_4b_fixed", "qwen38_sarcasm")): return "Trait archive"
    return "Project archive"


def navigation(docs):
    pages = {p.relative_to(docs).as_posix(): p for p in docs.rglob("*.md")}
    used, nav = set(), []
    for group, names in CURRENT.items():
        present = [n for n in names if n in pages]
        if present:
            nav.append({group: [{NAV_LABELS.get(n, title(pages[n])): n} for n in present]}); used.update(present)
    archives = {}
    for name in sorted(set(pages)-used):
        archives.setdefault(archive_group(Path(name).name), []).append({title(pages[name]): name})
    if archives:
        nav.append({"Research archive": [{group: entries} for group, entries in sorted(archives.items())]})
    return nav


@dataclass
class Context:
    root: Path
    docs: Path
    tracked: set
    source: Path | None = None
    hashes: dict = field(default_factory=dict)
    rewritten: int = 0
    local_only: int = 0


def link_target(url, context, *, image=False):
    """Return (build URL, local-only flag), preserving queries/fragments."""
    parsed = urlsplit(url)
    if not parsed.path: return url, False
    # Current-branch GitHub docs references can use the rendered pages.
    prefix = REPO+"/blob/main/docs/"
    if url.startswith(prefix):
        if parsed.fragment.startswith("L") and parsed.fragment[1:2].isdigit(): return url, False
        target = (context.docs/unquote(parsed.path.split("/blob/main/docs/", 1)[1])).resolve()
    elif parsed.scheme or parsed.netloc:
        return url, False
    else:
        target = (context.source.parent/unquote(parsed.path)).resolve()
    if target.is_relative_to(context.docs):
        if target.is_file():
            relative = os.path.relpath(target, context.source.parent).replace(os.sep, "/")
            return urlunsplit(("", "", relative, parsed.query, parsed.fragment)), False
        # Missing historical source references remain visible, never fake links.
    elif target.is_relative_to(context.root):
        relative = target.relative_to(context.root).as_posix()
        if relative in context.tracked and target.is_file():
            base = "https://raw.githubusercontent.com/RenormalizationTech/tuneLLM/main/" if image else REPO+"/blob/main/"
            return base+quote(relative, safe="/")+("?"+parsed.query if parsed.query else "")+("#"+parsed.fragment if parsed.fragment else ""), False
    explanation = os.path.relpath(context.docs/"website.md", context.source.parent).replace(os.sep, "/")
    return explanation+"#local-only-references", True


class SiteLinks(Treeprocessor):
    def __init__(self, md, context): super().__init__(md); self.context = context

    def run(self, root):
        for node in root.iter():
            attr = "href" if node.tag == "a" else "src" if node.tag == "img" else None
            if attr is None or not node.get(attr): continue
            original = node.get(attr)
            target, local_only = link_target(original, self.context, image=node.tag == "img")
            if local_only and node.tag == "img":
                # No request for a missing image; keep its label and local status.
                node.tag = "a"; node.attrib.pop("src", None)
                node.text = node.attrib.pop("alt", "Local image")
                attr = "href"
            node.set(attr, target)
            if local_only:
                node.set("title", "Local-only or unavailable source reference: "+original)
                SubElement(node, "span", {"class": "local-only-reference"}).text = " (local only)"
                self.context.local_only += 1
            if target != original: self.context.rewritten += 1
        # Scroll the wrapper, preserving the table's native display/semantics.
        table_number = 0
        for parent in list(root.iter()):
            for index, child in enumerate(list(parent)):
                if child.tag != "table": continue
                table_number += 1
                wrapper = Element("div", {"class": "table-scroll", "role": "region",
                    "tabindex": "0", "aria-label": f"Data table {table_number}"})
                parent.remove(child); wrapper.append(child); parent.insert(index, wrapper)


class SiteLinksExtension(Extension):
    def __init__(self, context): self.context = context; super().__init__()
    def extendMarkdown(self, md): md.treeprocessors.register(SiteLinks(md, self.context), "portfolio_links", 15)


CONTEXT = None


def on_config(config):
    global CONTEXT
    root = Path(config.config_file_path).resolve().parent; docs = Path(config.docs_dir).resolve()
    result = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
    CONTEXT = Context(root, docs, set(result.stdout.decode().split("\0")), hashes=document_hashes(docs))
    config.nav = navigation(docs)
    config.markdown_extensions.append(SiteLinksExtension(CONTEXT))
    return config


def on_page_markdown(markdown, page, config, files):
    CONTEXT.source = Path(page.file.abs_src_path).resolve()
    current = {name for names in CURRENT.values() for name in names}
    if page.file.src_uri not in current:
        roadmap = os.path.relpath(CONTEXT.docs/"project_tracks.md", CONTEXT.source.parent).replace(os.sep, "/")
        return ("> **Historical record.** This archived page preserves the experiment's original wording, "
                "including plans and status statements from that time. See the "
                f"[current roadmap]({roadmap}) for present priorities and outcomes.\n\n"+markdown)
    return markdown


def on_post_build(config):
    if document_hashes(CONTEXT.docs) != CONTEXT.hashes: raise RuntimeError("Markdown source bytes changed during site build")
    record = {"document_count": len(CONTEXT.hashes), "source_sha256": CONTEXT.hashes,
              "rewritten_links": CONTEXT.rewritten, "labelled_local_references": CONTEXT.local_only,
              "all_markdown_pages_in_navigation": True,
              "source_bytes_preserved": True}
    Path(config.site_dir, "docs-build.json").write_text(json.dumps(record, indent=2)+"\n")
