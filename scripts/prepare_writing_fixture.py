"""Freeze source-grounded title/abstract/introduction tasks before optimization."""
import json
import re

from scripts.common import digest, path, write_json


def excerpt(text, tokenizer, budget=1536):
    """Keep complete paragraphs, taking a beginning and an ending within budget."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    selected, total = {}, 0
    # Alternate beginning and end; whole paragraphs only, never truncate equations.
    order = []
    for i in range((len(paragraphs) + 1) // 2):
        order.extend([i, len(paragraphs) - i - 1])
    for i in dict.fromkeys(order):
        count = len(tokenizer(paragraphs[i], add_special_tokens=False)["input_ids"])
        if total + count <= budget:
            selected[i] = paragraphs[i]
            total += count
    if not selected:
        raise ValueError("no complete paragraph fits the evidence budget")
    return "\n\n[... intervening text omitted ...]\n\n".join(selected[i] for i in sorted(selected))


def task_instruction(task):
    return {
        "title": "Write one concise, informative physics-paper title. Return only the title, without a heading.",
        "abstract": "Write a concise physics-paper abstract of about 120-180 words. Return only the abstract, without a heading.",
        "introduction": "Write two or three introductory paragraphs for a physics paper, accessible to condensed-matter theorists outside the immediate subfield. Return only the introduction, without a heading.",
    }[task]


def reference_overlap(source, reference):
    def grams(text):
        words = re.findall(r"\w+", text.lower())
        return set(tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3)))
    left, right = grams(source), grams(reference)
    return len(left & right) / len(right) if right else None


def make_tasks(rows, tokenizer, examples=None):
    tasks = []
    for row in rows:
        source = excerpt(row["writing_source"], tokenizer)
        for task in ["title", "abstract", "introduction"]:
            instruction = task_instruction(task)
            prefix = ("Use only the supplied scientific evidence. Improve clarity, concision and logical organization. "
                      "State significance only when supported by the evidence. Do not invent measurements, mechanisms, "
                      "citations, novelty claims or equations. Preserve LaTeX notation.\n\n")
            if examples:
                prefix += "Examples of writing from separate training papers:\n\n"
                for example in examples:
                    answer = example["introductory_passage" if task == "introduction" else task]
                    prefix += f"Evidence:\n{excerpt(example['writing_source'], tokenizer, 512)}\n{task.title()}:\n{answer}\n\n"
            user = prefix + f"Evidence for the new paper:\n{source}\n\n{instruction}"
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
            token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            if token_count > 6500:
                raise ValueError("writing prompt exceeds registered budget")
            tasks.append({"id": row["id"] + "::" + task, "paper_id": row["paper_id"], "task": task,
                          "source": source, "user": user, "prompt_tokens": token_count,
                          "reference": row["introductory_passage" if task == "introduction" else task],
                          "evidence_reference_4gram_coverage": reference_overlap(source, row["introductory_passage" if task == "introduction" else task]),
                          "fewshot_example_ids": [e["id"] for e in examples] if examples else []})
    return tasks


def main():
    from transformers import AutoTokenizer
    root = path("data/physics-prl-01")
    out = path("data/benchmark-screen-01")
    tokenizer = AutoTokenizer.from_pretrained(path("artifacts/nf4/Qwen3-4B"), local_files_only=True)
    splits = {name: [json.loads(line) for line in (root / f"{name}.jsonl").read_text().split("\n") if line]
              for name in ["train", "validation", "test"]}
    examples = sorted(splits["train"], key=lambda row: row["id"])[:2]
    files = {}
    for label, name in [("development", "validation"), ("test", "test")]:
        for fewshot in [False, True]:
            filename = out / (f"writing_{label}" + ("_fewshot" if fewshot else "") + ".json")
            if filename.exists():
                raise FileExistsError(filename)
            tasks = make_tasks(splits[name], tokenizer, examples if fewshot else None)
            write_json(filename, {"tasks": tasks, "source_split": name,
                                  "source_sha256": digest(root / f"{name}.jsonl"),
                                  "evidence_policy": "complete paragraphs from body excluding title, abstract and heuristic introductory passage;1536-token evidence budget",
                                  "human_ratings": "pending; references are published passages, not unique correct rewrites"})
            files[filename.name] = {"sha256": digest(filename), "tasks": len(tasks),
                                    "max_prompt_tokens": max(r["prompt_tokens"] for r in tasks)}
    write_json(out / "writing_provenance.json", {"files": files, "fewshot_example_ids": [e["id"] for e in examples],
        "generation_caps": {"title": 48, "abstract": 256, "introduction": 768},
        "rubric": ["scientific claim fidelity", "equation/number/citation fidelity", "clarity", "concision",
                   "logical structure", "accessibility", "evidence-supported significance"],
        "selection_limit": "No automatic metric establishes writing quality. Produce blinded human-review packets; do not invent ratings."})
    print(files)


if __name__ == "__main__":
    main()
