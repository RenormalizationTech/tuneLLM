"""Local, standard-library science preparation; never downloads or loads models.

Input splits are JSON (QASPER paper maps or row lists) or JSONL. CLI token
counts are explicitly proxies; inject the registered tokenizer's counter through
the Python API for actual token budgets. Every sampled question is retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

VERSION = "qwen38-science-v2"
SCHEMA = {"answer": "string", "unanswerable": "boolean", "evidence": "unique paragraph-ID strings"}
PRL_POLICY_VERSION = "prl-tex-head-tail-50-50-v2"
PRL_OMISSION_MARKER = "[... middle body omitted ...]"
PRL_FLOAT_ENVS = {"figure", "figure*", "table", "table*", "wrapfigure", "wraptable", "sidewaysfigure", "sidewaystable"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized_space(text):
    return " ".join(text.split())


def records(value):
    """Accept native row lists and Hugging Face struct-of-lists exports."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        lengths = {len(v) for v in value.values() if isinstance(v, list)}
        if len(lengths) != 1 or any(not isinstance(v, list) for v in value.values()):
            raise ValueError("expected equally sized column lists")
        return [dict(zip(value, row)) for row in zip(*value.values())]
    raise ValueError("expected rows or column lists")


def number_paragraphs(paper):
    result = []
    if paper.get("abstract", "").strip():
        result.append({"section": "Abstract", "text": paper["abstract"]})
    for section in records(paper.get("full_text", [])):
        for text in section["paragraphs"]:
            if not isinstance(text, str):
                raise ValueError("paragraph must be text")
            # Empty paragraphs retain their positions and therefore stable IDs.
            result.append({"section": section.get("section_name", ""), "text": text})
    numbered, representatives = [], {}
    for i, row in enumerate(result):
        pid = f"p{i:06d}"
        representative = representatives.setdefault(normalized_space(row["text"]), pid)
        numbered.append({"id": pid, "canonical_id": representative, **row})
    return numbered


def render_context(paragraphs):
    return "\n\n".join(f"[{p['id']}] {p['section']}\n{p['text']}" for p in paragraphs)


def select_context(paragraphs, question, token_counter, context_budget):
    """Question-only lexical ranking; whole paragraphs, original display order.

    Budget includes paragraph IDs/headings/separators, but excludes instruction,
    question and target tokens. Callers must reserve those separately.
    """
    if context_budget < 0:
        raise ValueError("nonnegative context budget required")
    def count(rows):
        n = token_counter(render_context(rows))
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError("token counter must return a nonnegative integer")
        return n
    if count(paragraphs) <= context_budget:
        return list(paragraphs)
    words = lambda s: set(re.findall(r"\w+", s.lower()))
    query = words(question)
    ranked = sorted(enumerate(paragraphs), key=lambda x: (-len(query & words(x[1]["text"])), x[0]))
    chosen = {}
    for i, paragraph in ranked:
        candidate = [p for _, p in sorted({**chosen, i: paragraph}.items())]
        if count(candidate) <= context_budget:
            chosen[i] = paragraph
    return [p for _, p in sorted(chosen.items())]


def annotation_target(annotation, paragraphs):
    a = annotation.get("answer", annotation)
    if not isinstance(a, dict):
        raise ValueError("annotation answer must be an object")
    unanswerable = a.get("unanswerable", False)
    yes_no = a.get("yes_no")
    spans = a.get("extractive_spans", [])
    free = a.get("free_form_answer", "")
    if type(unanswerable) is not bool or (yes_no is not None and type(yes_no) is not bool):
        raise ValueError("typed unanswerable/yes_no required")
    if not isinstance(spans, list) or any(not isinstance(s, str) for s in spans) or not isinstance(free, str):
        raise ValueError("invalid answer text")
    active = sum([bool(unanswerable), yes_no is not None, bool(spans), bool(free.strip())])
    if active != 1:
        raise ValueError("exactly one answer type required")
    if unanswerable:
        kind, answer = "unanswerable", ""
    elif yes_no is not None:
        kind, answer = "boolean", "Yes" if yes_no else "No"
    elif spans:
        kind, answer = "extractive", ", ".join(spans)
    else:
        kind, answer = "free_form", free.strip()
    evidence = a.get("evidence", [])
    if not isinstance(evidence, list) or any(not isinstance(e, str) for e in evidence):
        raise ValueError("evidence must be text strings")
    ids, unmatched, ambiguous = set(), [], []
    for text in evidence:
        if "FLOAT SELECTED" in text:
            unmatched.append(text)
            continue
        matches = [p["id"] for p in paragraphs if normalized_space(p["text"]) == normalized_space(text)]
        if not matches:
            unmatched.append(text)
        elif len(matches) > 1:
            ambiguous.append(text)
            # Duplicate source text is one evidence item, not multiple required IDs.
            ids.add(matches[0])
        else:
            ids.update(matches)
    return {"answer": answer, "unanswerable": unanswerable, "answer_type": kind,
            "evidence": [] if unanswerable else sorted(ids), "unmatched_evidence": unmatched,
            "unmatched_text_evidence": [e for e in unmatched if "FLOAT SELECTED" not in e],
            "ambiguous_evidence": ambiguous,
            "declared_nontext_evidence": any("FLOAT SELECTED" in e for e in evidence),
            "raw_annotation": a}


def prepare_qasper(paper, split, token_counter, context_budget):
    paragraphs = number_paragraphs(paper)
    rows = []
    for qa in records(paper["qas"]):
        question = qa["question"]
        context = select_context(paragraphs, question, token_counter, context_budget)
        available = {p["canonical_id"] for p in context}
        context_representatives = {}
        for p in context:
            context_representatives.setdefault(p["canonical_id"], p["id"])
        refs = [annotation_target(a, paragraphs) for a in records(qa["answers"])]
        if not refs:
            raise ValueError("labeled preparation requires at least one annotation")
        full = len(context) == len(paragraphs)
        for ref in refs:
            ref["missing_context_evidence"] = sorted(set(ref["evidence"]) - available)
            ref["text_evidence_complete"] = (not ref["unanswerable"] and bool(ref["evidence"])
                and not ref["unmatched_evidence"]
                and not ref["declared_nontext_evidence"] and not ref["missing_context_evidence"])
        # Preserve dataset annotation order; do not choose the easiest reference.
        selected = refs[0]
        consistent = len({r["unanswerable"] for r in refs}) == 1
        primary = consistent and (any(r["text_evidence_complete"] for r in refs)
                                 or (all(r["unanswerable"] for r in refs) and full))
        target = {k: selected[k] for k in SCHEMA}
        target["evidence"] = [context_representatives.get(e, e) for e in target["evidence"]]
        evidence_strict = primary and all(not r["unanswerable"] and bool(r["evidence"] or r["unmatched_text_evidence"]) for r in refs)
        user = ("Answer using only the numbered paper paragraphs. Return only JSON with keys "
                '"answer" (string), "unanswerable" (boolean), "evidence" (array of paragraph IDs). '
                "For insufficient evidence return an empty answer and evidence array with unanswerable=true.\n\n"
                + render_context(context) + "\n\nQuestion: " + question)
        rows.append({"id": str(paper["id"]) + "::" + str(qa["question_id"]), "paper_id": str(paper["id"]), "question_id": str(qa["question_id"]),
            "split": split, "user": user, "assistant": canonical(target), "references": refs,
            "paragraphs": paragraphs, "context_ids": [p["id"] for p in context],
            "context_tokens": token_counter(render_context(context)), "full_text_in_context": full,
            "primary": primary, "evidence_strict_primary": evidence_strict, "annotation_disagreement": not consistent,
            "training_eligible": consistent and (selected["text_evidence_complete"] or (selected["unanswerable"] and full)),
            "selected_annotation_index": 0, "schema_version": VERSION,
            "nontext_evidence": any(r["declared_nontext_evidence"] for r in refs),
            "unmatched_text_evidence": any(r["unmatched_text_evidence"] for r in refs),
            "nontext_or_unmatched_evidence": any(r["unmatched_evidence"] or r["declared_nontext_evidence"] for r in refs)})
    return rows


def braced(text, start):
    while start < len(text) and text[start].isspace():
        start += 1
    if start == len(text) or text[start] != "{":
        raise ValueError("expected braced TeX argument")
    depth, begin = 1, start + 1
    for i in range(begin, len(text)):
        if text[i] in "{}" and (len(text[:i]) - len(text[:i].rstrip("\\"))) % 2 == 0:
            depth += 1 if text[i] == "{" else -1
            if depth == 0:
                return text[begin:i], i + 1
    raise ValueError("unbalanced TeX argument")


def remove_commands(text, commands):
    pattern = re.compile(r"\\(?:" + "|".join(commands) + r")\b\*?\s*(?:\[[^\]]*\])?\s*")
    out, start = [], 0
    for match in pattern.finditer(text):
        if match.start() < start:
            continue
        out.append(text[start:match.start()])
        _, start = braced(text, match.end())
    out.append(text[start:])
    return "".join(out)


def extract_prl(row):
    """Clean cached text conservatively; no claim-fidelity certification."""
    text, target = row["text"], row["abstract"].strip()
    if not target:
        raise ValueError("nonempty abstract required")
    # Cached rows already exclude preamble; strip TeX comments without losing escaped percent.
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    abstracts = re.findall(r"\\begin\{abstract\*?\}(.*?)\\end\{abstract\*?\}", text, re.S)
    text = re.sub(r"\\begin\{abstract\*?\}.*?\\end\{abstract\*?\}", "", text, flags=re.S)
    text = remove_commands(text, ["abstract", "title", "author", "affiliation", "address", "email", "thanks",
                                   "date", "altaffiliation", "collaboration", "pacs", "keywords"])
    text = re.sub(r"\\maketitle\b|\\(?:begin|end)\{document\}", "", text)
    # Exact token sequence with whitespace variation only: no fuzzy deletion of body claims.
    duplicate = re.compile(r"\s+".join(re.escape(part) for part in target.split()))
    text, removed = duplicate.subn("", text)
    body = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text).strip()
    flags = []
    if any(normalized_space(a) != normalized_space(target) for a in abstracts):
        flags.append("abstract_environment_differs_from_target")
    if re.search(r"\\(?:\w*abstract\w*|title|author|affiliation)\b|\\(?:begin|end)\{abstract", body, re.I):
        flags.append("unresolved_abstract_or_metadata_macro")
    if not body:
        flags.append("empty_body")
    if row.get("long_main_text_flag"):
        flags.append("historical_long_main_text")
    if row.get("main_text_characters", 0) and len(body) < .8 * row["main_text_characters"]:
        flags.append("body_shorter_than_historical_main_text")
    if body and body[-1] not in ".!?}$]":
        flags.append("suspicious_body_ending")
    containment = abstract_containment(body, target)
    if containment is not None and containment > 0:
        flags.append("abstract_8gram_overlap_review")
    return body, target, {"removed_abstract_environments": len(abstracts), "removed_exact_duplicates": removed,
                          "abstract_8gram_containment": containment,
                          "historical_cleaning_requires_source_review": bool(row.get("source_sha256")),
                          "flags": flags, "claim_coverage_audit": "required; lexical cleaning cannot establish grounding"}


def abstract_containment(body, abstract, n=8):
    """Fraction of unique abstract word n-grams in body; diagnostic, never deletion."""
    def grams(text):
        words = re.findall(r"\w+", text.lower())
        return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}
    target = grams(abstract)
    return len(grams(body) & target) / len(target) if target else None


def select_prl_prefix(body, token_counter, context_budget):
    """Plain body, whole-paragraph strict source-order prefix; never skip to fit."""
    if context_budget < 0:
        raise ValueError("nonnegative context budget required")
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    chosen = []
    for paragraph in paragraphs:
        candidate = "\n\n".join(chosen + [paragraph])
        count = token_counter(candidate)
        if type(count) is not int or count < 0:
            raise ValueError("token counter must return a nonnegative integer")
        if count > context_budget:
            break
        chosen.append(paragraph)
    return "\n\n".join(chosen), len(chosen), len(paragraphs)


def _tex_definition(text, offset):
    """Read, never execute, a balanced newcommand/def declaration at offset.

    Its entire replacement is protected from the boundary lexer. Only exact
    zero-argument begin/end replacements become environment delimiter aliases.
    """
    new = re.match(r"\\(?:newcommand|renewcommand|providecommand|DeclareRobustCommand)\*?\s*(?:\{(\\[A-Za-z@]+)\}|(\\[A-Za-z@]+))\s*((?:\[[^\]]*\]\s*){0,2})", text[offset:])
    defined = re.match(r"\\(?:def|gdef|edef|xdef)\s*(\\[A-Za-z@]+)([^{}]*)", text[offset:]) if not new else None
    if new:
        name, arguments, end = new.group(1) or new.group(2), new.group(3).strip(), offset + new.end()
        no_arguments = arguments in {"", "[0]"}
    elif defined:
        name, no_arguments, end = defined.group(1), not defined.group(2).strip(), offset + defined.end()
    else:
        return None
    try:
        replacement, stop = braced(text, end)
    except ValueError:
        return None
    boundary = re.fullmatch(r"\s*\\(begin|end)\s*\{([^{}]+)\}\s*", replacement) if no_arguments else None
    return {"stop": stop, "name": name, "environment_alias": boundary.groups() if boundary else None}


def tex_blocks(body):
    """Math/environment boundary lexer, not a TeX syntax/compilation validator.

    Blank lines split outside nested environments and inline/display math.
    Generic grouping/styling braces are diagnostic only, so cutoff wrappers do
    not swallow later paragraphs. Balanced macro declarations remain atomic;
    their replacement text is not treated as executed math/environments. Exact
    zero-argument environment aliases are recognized lexically, not expanded.
    Unsupported complex macros require review. Genuine open math/environments
    keep the unfinished remainder indivisible and unavailable to excerpting.
    """
    blocks, environments, math, aliases = [], [], None, {}
    brace_offsets, closing_braces, math_start = [], [], None
    declarations = []
    start, i, error = 0, 0, None
    def emit(stop):
        nonlocal start
        text = body[start:stop].strip()
        if text:
            offset = start + len(body[start:stop]) - len(body[start:stop].lstrip())
            kind = "prose"
            if re.match(r"\\begin\s*\{(?:" + "|".join(re.escape(e) for e in sorted(PRL_FLOAT_ENVS)) + r")\}", text) or re.match(r"\\(?:caption|captionof|includegraphics|centering)\b", text):
                kind = "float"
            elif re.match(r"\\begin\s*\{(?:equation\*?|align\*?|alignat\*?|gather\*?|multline\*?|displaymath|eqnarray\*?)\}|\\\[|\$\$", text):
                kind = "math"
            elif re.match(r"\\(?:newcommand|renewcommand|providecommand|def|DeclareMathOperator)\b", text):
                kind = "definition"
            elif re.fullmatch(r"(?:\s|\\(?:vspace|hspace)\*?\s*\{[^}]*\}|\\label\s*\{[^}]*\}|\\(?:smallskip|medskip|bigskip|clearpage|newpage|FloatBarrier)\b)+", text):
                kind = "metadata"
            elif re.fullmatch(r"(?:\\[A-Za-z@]+\*?(?:\{[A-Za-z\s-]*\})?|[{}\s~])+", text):
                kind = "metadata"  # includes content-free, dangling font/cutoff wrappers
            first = re.match(r"\\[A-Za-z@]+", text)
            alias = aliases.get(first.group()) if first else None
            if alias and alias[0] == "begin":
                kind = "float" if alias[1] in PRL_FLOAT_ENVS else "math"
            blocks.append({"id": f"b{len(blocks):06d}", "text": text, "kind": kind,
                           "start": offset, "stop": offset + len(text)})
        start = stop
    while i < len(body):
        char = body[i]
        if char == "%":
            end = body.find("\n", i)
            i = len(body) if end < 0 else end
            continue
        if char == "\\":
            declaration = _tex_definition(body, i)
            if declaration:
                declarations.append({"start": i, **declaration})
                aliases.pop(declaration["name"], None)
                if declaration["environment_alias"]:
                    aliases[declaration["name"]] = declaration["environment_alias"]
                i = declaration["stop"]
                continue
            verb = re.match(r"\\verb(?![A-Za-z@])\*?([^\s])", body[i:])
            if verb:
                end = body.find(verb.group(1), i + verb.end())
                if end < 0:
                    error = "unclosed_verb"
                    break
                i = end + 1
                continue
            env = re.match(r"\\(begin|end)\s*\{([^{}]+)\}", body[i:])
            command = re.match(r"\\[A-Za-z@]+", body[i:])
            alias = aliases.get(command.group()) if command else None
            if env or alias:
                action, name = env.groups() if env else alias
                width = env.end() if env else command.end()
                if action == "begin":
                    if not environments and math is None and name in PRL_FLOAT_ENVS:
                        emit(i)
                    environments.append(name)
                    if name in {"verbatim", "Verbatim", "lstlisting", "minted"}:
                        end = body.find("\\end{" + name + "}", i + width)
                        if end < 0:
                            error = "unclosed_verbatim_environment"
                            break
                        i = end
                        continue
                elif not environments or environments.pop() != name:
                    error = "mismatched_environment"
                    break
                i += width
                if action == "end" and not environments and math is None and name in PRL_FLOAT_ENVS:
                    emit(i)
                continue
            delimiter = body[i:i + 2]
            if delimiter in {r"\[", r"\]", r"\(", r"\)"}:
                if delimiter in {r"\[", r"\("}:
                    if math is not None:
                        error = "nested_math_delimiter"
                        break
                    math = r"\]" if delimiter == r"\[" else r"\)"
                    math_start = i
                elif math != delimiter:
                    error = "mismatched_math_delimiter"
                    break
                else:
                    math = None
                    math_start = None
                i += 2
                continue
            # A command escape consumes its next character (including escaped %, $, braces).
            i += 2
            continue
        if char == "$":
            # In inline math the first dollar closes it. A following dollar may
            # immediately open the next inline expression: $x$$=$$y$ is valid.
            delimiter = "$" if math == "$" else ("$$" if body[i:i + 2] == "$$" else "$")
            if math is None:
                math = delimiter
                math_start = i
            elif math == delimiter:
                math = None
                math_start = None
            else:
                error = "mismatched_dollar_math"
                break
            i += len(delimiter)
            continue
        if char == "{":
            brace_offsets.append(i)
        elif char == "}":
            if brace_offsets:
                brace_offsets.pop()
            else:
                closing_braces.append(i)
        if char == "\n" and not environments and math is None:
            blank = re.match(r"\n[ \t]*\n(?:[ \t]*\n)*", body[i:])
            if blank:
                emit(i)
                i += blank.end()
                start = i
                continue
        i += 1
    unsafe = bool(error or environments or math is not None)
    if unsafe:
        remainder = body[start:].strip()
        if remainder:
            offset = start + len(body[start:]) - len(body[start:].lstrip())
            blocks.append({"id": f"b{len(blocks):06d}", "text": remainder, "kind": "unknown",
                           "start": offset, "stop": offset + len(remainder)})
    else:
        emit(len(body))
    generic = bool(brace_offsets or closing_braces)
    return blocks, {"balanced": not unsafe and not generic,
        "math_environment_boundaries_balanced": not unsafe, "generic_braces_balanced": not generic,
        "generic_braces_are_diagnostic_only": True, "compilability_established": False,
        "error": (error or "unclosed_math_or_environment") if unsafe else ("generic_brace_imbalance" if generic else None),
        "open_environments": list(environments), "open_math_delimiter": math, "brace_depth": len(brace_offsets),
        "open_brace_offsets": brace_offsets, "unmatched_closing_brace_offsets": closing_braces,
        "math_start_offset": math_start, "error_offset": i if unsafe else None,
        "protected_macro_declarations": declarations, "environment_aliases": aliases}


def select_prl_head_tail(body, token_counter, context_budget):
    """Fixed 50/50 head/tail, whole blocks, one marker; no abstract access.

    Reserve marker/separators first. The head is a strict block prefix. Tail
    ends at the last prose block, grows backwards contiguously, and never starts
    by spending its budget on trailing floats/captions. No half-budget transfer
    or optional trailing-caption fill is used. Multiple omitted ranges are
    disclosed in metadata; only the middle/first gap has the fixed marker.
    """
    if type(context_budget) is not int or context_budget < 0:
        raise ValueError("nonnegative integer context budget required")
    def count(text):
        value = token_counter(text)
        if type(value) is not int or value < 0:
            raise ValueError("token counter must return a nonnegative integer")
        return value
    blocks, syntax = tex_blocks(body)
    meta = {"version": PRL_POLICY_VERSION, "syntax": syntax, "blocks": blocks,
            "full_body_tokens": count(body), "context_budget": context_budget,
            "omission_marker": PRL_OMISSION_MARKER, "trailing_caption_fill": False}
    if meta["full_body_tokens"] <= context_budget:
        return body, {**meta, "selected_indices": list(range(len(blocks))), "head_indices": [], "tail_indices": [],
                      "head_budget": None, "tail_budget": None, "full_body": True, "omission_marker_count": 0}
    reserve = count("\n\n" + PRL_OMISSION_MARKER + "\n\n")
    available = max(0, context_budget - reserve)
    head_budget, tail_budget = available // 2, available - available // 2
    head, tail = [], []
    join = lambda indices: "\n\n".join(blocks[i]["text"] for i in indices)
    for i in range(len(blocks)):
        if blocks[i]["kind"] == "unknown" or count(join(head + [i])) > head_budget:
            break
        head.append(i)
    prose = [i for i, b in enumerate(blocks) if b["kind"] == "prose"]
    if prose:
        for i in range(prose[-1], -1, -1):
            if i in head or blocks[i]["kind"] == "unknown":
                break
            if count(join([i] + tail)) > tail_budget:
                break
            tail.insert(0, i)
    def render():
        if not head and not tail:
            return PRL_OMISSION_MARKER if count(PRL_OMISSION_MARKER) <= context_budget else ""
        if head and tail and head[-1] + 1 == tail[0]:
            return join(head + tail) + "\n\n" + PRL_OMISSION_MARKER
        return "\n\n".join(part for part in (join(head), PRL_OMISSION_MARKER, join(tail)) if part)
    context = render()
    while count(context) > context_budget and (head or tail):
        if head and (not tail or count(join(head)) >= count(join(tail))):
            head.pop()
        else:
            tail.pop(0)
        context = render()
    selected = sorted(set(head + tail))
    omitted = [i for i in range(len(blocks)) if i not in selected]
    return context, {**meta, "selected_indices": selected, "head_indices": head, "tail_indices": tail,
        "head_budget": head_budget, "tail_budget": tail_budget, "head_tokens": count(join(head)),
        "tail_tokens": count(join(tail)), "omitted_indices": omitted, "full_body": False,
        "trailing_nonprose_omitted": [i for i in omitted if tail and i > tail[-1]],
        "omission_marker_count": int(PRL_OMISSION_MARKER in context)}


def prl_copy_baseline(blocks, target, max_words=200):
    """Target-independent first whole prose blocks, at most 200 whitespace words.

    Unfit prose stops the copy prefix rather than truncating a TeX block.
    The supplied excerpt's selected blocks are the primary input to this control.
    Token F1/8-gram overlap are lexical diagnostics, not fidelity judgments.
    """
    chosen = []
    for block in blocks:
        if block["kind"] != "prose":
            continue
        if len("\n\n".join(chosen + [block["text"]]).split()) > max_words:
            break
        chosen.append(block["text"])
    text = "\n\n".join(chosen)
    from scripts.qwen38_science_score import answer_f1
    return {"policy": "first whole prose blocks, <=200 whitespace words, no target-based selection",
            "text": text, "word_count": len(text.split()), "empty": not text,
            "normalized_token_f1": answer_f1(text, target), "abstract_8gram_containment": abstract_containment(text, target)}


def prepare_prl(row, split, token_counter, context_budget):
    body, target, audit = extract_prl(row)
    context, selection = select_prl_head_tail(body, token_counter, context_budget)
    selected = [selection["blocks"][i] for i in selection["selected_indices"]]
    if not selection["full_body"]:
        audit["flags"].append("context_head_tail_excerpt")
    if not selection["syntax"]["balanced"]:
        audit["flags"].append("unbalanced_tex_source")
    if not selection["syntax"]["generic_braces_balanced"]:
        audit["flags"].append("generic_brace_imbalance_diagnostic")
    if not selection["syntax"]["math_environment_boundaries_balanced"]:
        audit["flags"].append("unsafe_math_or_environment_boundary")
    if not selected:
        audit["flags"].append("empty_selected_body")
    if len(target.split()) > 200:
        audit["flags"].append("reference_over_200_words")
    audit["selected_abstract_8gram_containment"] = abstract_containment(context, target)
    return {"id": row["id"], "paper_id": row["paper_id"], "split": split,
        "user": "Write a grounded physics abstract in at most 200 words using only the supplied paper body excerpt, which may omit its middle and nonprose material. Preserve quantities and limitations; do not invent claims. Return only the abstract.\n\n" + context,
        "assistant": target, "body": body, "selected_body": context,
        "context_ids": [b["id"] for b in selected], "context_selection": selection,
        "context_tokens": token_counter(context), "full_body_in_context": selection["full_body"],
        "copy_baseline": prl_copy_baseline(selected, target),
        "full_body_copy_baseline_diagnostic": prl_copy_baseline(selection["blocks"], target),
        "leakage_audit": audit, "training_eligible": False,
        "training_eligibility_reason": "requires registered claim-coverage audit before optimization",
        "historical_test_diagnostic": split == "test", "schema_version": VERSION,
        "provenance": {k: row[k] for k in ("source_sha256", "source_url", "arxiv_version", "doi", "license", "authoritative_license", "authors") if k in row},
        "source_row_sha256": sha(canonical(row).encode())}


def read_rows(filename):
    text = Path(filename).read_text(encoding="utf-8")
    if Path(filename).suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if isinstance(value, dict):
        return [{**row, "id": str(key)} for key, row in value.items()]
    return value


def validate_paper_splits(splits, task, original_ids=None):
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("original train/validation/test splits required")
    seen = set()
    for rows in splits.values():
        for row in rows:
            key = str(row["paper_id"] if task == "prl" else row["id"])
            # Version suffixes cannot turn one paper into independent splits.
            key = re.sub(r"v\d+$", "", key)
            if key in seen:
                raise ValueError("duplicate paper within or across original splits")
            seen.add(key)
    if original_ids is not None:
        for split, rows in splits.items():
            actual = [str(r["paper_id"] if task == "prl" else r["id"]) for r in rows]
            if actual != list(original_ids[split]):
                raise ValueError("original paper split membership/order changed")


def prepare_local(task, inputs, output, revision, token_counter, counter_name, context_budget, source_manifest=None,
                  original_ids=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    hashes = {split: sha(Path(p).read_bytes()) for split, p in inputs.items()}
    source_hash = sha(Path(source_manifest).read_bytes()) if source_manifest else None
    splits = {split: read_rows(p) for split, p in inputs.items()}
    # Bind QASPER archive filenames to their original split, not caller labels.
    if task == "qasper" and source_manifest:
        extracted = json.loads(Path(source_manifest).read_text()).get("extracted_files", {})
        for split in inputs:
            name = f"qasper-{'dev' if split == 'validation' else split}-v0.3.json"
            if extracted and hashes[split] != extracted[name]["sha256"]:
                raise ValueError("input differs from original QASPER split acquisition hash")
    historical_hashes = None
    if task == "prl" and original_ids is None:
        historical = Path(__file__).resolve().parents[1] / "data/physics-prl-01"
        historical_hashes = {s: sha((historical / f"{s}.jsonl").read_bytes()) for s in inputs}
        original_ids = {s: [str(r["paper_id"]) for r in read_rows(historical / f"{s}.jsonl")] for s in inputs}
    validate_paper_splits(splits, task, original_ids)
    prepared = {}
    for split, papers in splits.items():
        rows = []
        for paper in papers:
            if task == "qasper":
                rows.extend(prepare_qasper(paper, split, token_counter, context_budget))
            elif task == "prl":
                rows.append(prepare_prl(paper, split, token_counter, context_budget))
            else:
                raise ValueError("unknown task")
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate prepared IDs")
        prepared[split] = rows
    if hashes != {split: sha(Path(p).read_bytes()) for split, p in inputs.items()}:
        raise ValueError("inputs changed during preparation")
    if source_manifest and sha(Path(source_manifest).read_bytes()) != source_hash:
        raise ValueError("source manifest changed during preparation")
    payloads = {f"{s}.jsonl": "".join(canonical(r) + "\n" for r in rows).encode() for s, rows in prepared.items()}
    manifest = {"version": VERSION, "task": task, "revision": revision, "input_sha256": hashes,
        "preparer_sha256": sha(Path(__file__).read_bytes()),
        "input_filenames": {s: Path(p).name for s, p in inputs.items()},
        "original_split_ids": original_ids,
        "historical_prl_split_sha256": historical_hashes,
        "input_paper_ids": {s: [str(p.get("paper_id", p["id"])) for p in rows] for s, rows in splits.items()},
        "source_manifest_sha256": source_hash,
        "counter": counter_name, "context_budget": context_budget, "budget_scope": "rendered evidence only; reserve prompt/target separately",
        "annotation_selection": "first in original order", "official_scorer_parity": False,
        "historical_prl_test_policy": "test remains an exposed historical diagnostic; never fresh test",
        "counts": {s: {"records": len(rows), "training_eligible": sum(r["training_eligible"] for r in rows),
                       "primary": sum(r.get("primary", False) for r in rows),
                       "evidence_strict_primary": sum(r.get("evidence_strict_primary", False) for r in rows),
                       "nontext_evidence": sum(r.get("nontext_evidence", False) for r in rows),
                       "unmatched_text_evidence": sum(r.get("unmatched_text_evidence", False) for r in rows),
                       "annotation_disagreement": sum(r.get("annotation_disagreement", False) for r in rows),
                       "missing_context_evidence": sum(any(ref["missing_context_evidence"] for ref in r.get("references", [])) for r in rows),
                       "truncated_context": sum(not r.get("full_text_in_context", r.get("full_body_in_context", True)) for r in rows),
                       "prl_flagged_records": sum(bool(r.get("leakage_audit", {}).get("flags")) for r in rows)} for s, rows in prepared.items()},
        "output_sha256": {name: sha(data) for name, data in payloads.items()}}
    output.mkdir(parents=True, exist_ok=False)
    for name, data in {**payloads, "manifest.json": (canonical(manifest) + "\n").encode()}.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["qasper", "prl"], required=True)
    for split in ("train", "validation", "test"):
        p.add_argument("--" + split, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--revision", required=True, help="pinned dataset/source identity")
    p.add_argument("--source-manifest", help="acquisition provenance, linked by SHA256")
    p.add_argument("--context-budget", type=int, required=True)
    p.add_argument("--counter", choices=["whitespace-proxy", "utf8-bytes-proxy"], required=True)
    a = p.parse_args()
    counter = (lambda s: len(s.split())) if a.counter == "whitespace-proxy" else (lambda s: len(s.encode()))
    print(canonical(prepare_local(a.task, {s: getattr(a, s) for s in ("train", "validation", "test")},
        a.output, a.revision, counter, a.counter, a.context_budget, a.source_manifest)))


if __name__ == "__main__":
    main()
