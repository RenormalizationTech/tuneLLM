"""Single-turn supervised data and position-based padding; no ML imports needed for tests."""
import json
from scripts.common import path


class OverlengthError(ValueError):
    """A row exceeds the explicit token budget; distinguish from invalid masking."""


def load_splits(split_paths):
    if set(split_paths) != {"train", "validation", "test"}:
        raise ValueError("explicit train, validation, and test paths required")
    result, seen_ids, seen_prompts = {}, set(), set()
    for name, filename in split_paths.items():
        # JSONL uses physical LF delimiters. Unicode separators can occur inside
        # valid ensure_ascii=False JSON strings and must not split a record.
        rows = [json.loads(line) for line in path(filename).read_text(encoding="utf-8").split("\n") if line.strip()]
        if not rows:
            raise ValueError(f"empty split: {name}")
        for row in rows:
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("id", "user", "assistant")):
                raise ValueError("each row requires nonempty string id, user, assistant")
            if "system" in row and not isinstance(row["system"], str):
                raise ValueError("system must be a string")
            prompt = (" ".join(row.get("system", "").split()), " ".join(row["user"].split()))
            if row["id"] in seen_ids or prompt in seen_prompts:
                raise ValueError("duplicate ID or prompt within/across splits")
            seen_ids.add(row["id"])
            seen_prompts.add(prompt)
        result[name] = rows
    return result


def encode_row(row, tokenizer, max_length):
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("pad token must exist and differ from EOS")
    if tokenizer.eos_token_id is None or not tokenizer.eos_token:
        raise ValueError("EOS token required")
    messages = []
    if row.get("system"):
        messages.append({"role": "system", "content": row["system"]})
    messages.append({"role": "user", "content": row["user"]})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(row["assistant"], add_special_tokens=False)["input_ids"]
    if not answer_ids or set(answer_ids) & set(tokenizer.all_special_ids):
        raise ValueError("assistant answer must be nonempty plain text without chat special tokens")
    full_ids = tokenizer(prompt + row["assistant"] + tokenizer.eos_token,
                         add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("tokenizer merges across prompt/answer boundary; cannot safely mask")
    if full_ids[-1] != tokenizer.eos_token_id:
        raise ValueError("assistant response must end with EOS")
    if len(full_ids) > max_length:
        raise OverlengthError(f"row {row['id']} has {len(full_ids)} tokens > max_length={max_length}; prepare/filter explicitly, never silently truncate labels")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
            "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]}


def pad_batch(rows, pad_token_id):
    length = max(len(row["input_ids"]) for row in rows)
    batch = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    for row in rows:
        count = length - len(row["input_ids"])
        for key, fill in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100)):
            batch[key].append(row[key] + [fill] * count)
    return batch


class AssistantCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch
        return {key: torch.tensor(value, dtype=torch.long)
                for key, value in pad_batch(rows, self.pad_token_id).items()}
