"""Pure tone gates/judging plus small cache-aware activation hooks. No model loads."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import math
import random
import re
import statistics
import unicodedata


def response_encoding(tokenizer, row, response, neutral_system):
    """Verify the full boundary; mask only response tokens, never EOS/template."""
    messages = [{"role": "system", "content": neutral_system}, {"role": "user", "content": row["user"]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    full = prompt + response + tokenizer.eos_token
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    start, end = len(prompt), len(prompt) + len(response)
    if not response_ids or set(response_ids) & set(tokenizer.all_special_ids) or ids[-1] != tokenizer.eos_token_id:
        raise ValueError("unverified prompt/response/EOS boundary; no extraction")
    mask, crossing = [], 0
    for a, b in offsets:
        if a < start < b:
            if not full[a:b].isspace():
                raise ValueError("non-whitespace token crosses template/response boundary")
            crossing += 1
        mask.append(start <= a < b <= end)
    if len(mask) != len(ids) or not any(mask) or mask[-1]:
        raise ValueError("empty or invalid response-only offset mask")
    return {"input_ids": ids, "response_mask": mask, "excluded_whitespace_boundary_tokens": crossing,
            "boundary_method": "verified full-text character offsets; template/EOS and whitespace-only crossing tokens excluded"}


def mean_direction(pairs):
    """Equal weight per eligible response pair, independent of response length."""
    import torch
    if not pairs:
        raise ValueError("nonempty paired activation means required")
    differences = [warm.float() - neutral.float() for neutral, warm in pairs]
    if any(d.shape != differences[0].shape for d in differences) or not all(torch.isfinite(d).all() for d in differences):
        raise ValueError("invalid activation dimensions or nonfinite values")
    vector = torch.stack(differences).mean(0)
    median = statistics.median([x.float().norm().item() for pair in pairs for x in pair])
    return vector, {"norm": vector.norm().item(), "median_response_mean_norm": median,
                    "norm_to_median_response_mean_norm": vector.norm().item() / median if median else None,
                    "pair_count": len(pairs), "weighting": "equal per-pair response means; EOS/template/pad excluded"}


@contextmanager
def response_means(model, layers, mask, collected):
    """Collect block-output means for one teacher-forced response, without retaining graphs."""
    handles = []
    try:
        for index in layers:
            def capture(module, args, output, index=index):
                hidden = output[0] if isinstance(output, tuple) else output
                selected = mask.to(device=hidden.device, dtype=__import__("torch").bool)
                if selected.shape != hidden.shape[:2] or not selected.any():
                    raise ValueError("response mask shape/count mismatch")
                collected[index] = hidden[selected].detach().float().mean(0).cpu()
            handles.append(model.model.layers[index].register_forward_hook(capture))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def steer(model, layer, vector, multiplier, audit=None):
    """Last real prefill position; every cached decode position. Zero is a true no-op."""
    if multiplier == 0:
        yield
        return
    import torch
    state, handles = {}, []
    def before(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        mask, cache = kwargs.get("attention_mask"), kwargs.get("past_key_values")
        cache_position = kwargs.get("cache_position")
        past = cache.get_seq_length() if hasattr(cache, "get_seq_length") else 0
        if isinstance(cache, (tuple, list)) and cache:
            past = cache[0][0].shape[-2]
        decode = past > 0 or (cache_position is not None and cache_position.numel() and cache_position[0].item() > 0)
        if ids is None or mask is None or mask.ndim != 2:
            raise ValueError("explicit IDs/attention mask required for cache-aware steering")
        if decode:
            if ids.shape[1] != 1:
                raise ValueError("only single-token cached decode supported")
            state["positions"] = None
        else:
            if mask.shape != ids.shape or not mask.bool().any(1).all():
                raise ValueError("prefill attention mask mismatch or empty prompt")
            positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
            state["positions"] = positions.masked_fill(~mask.bool(), -1).max(1).values
        if audit is not None:
            key = "decode_calls" if decode else "prefill_calls"
            audit[key] = audit.get(key, 0) + 1
    def add(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if "positions" not in state or vector.ndim != 1 or vector.shape[0] != hidden.shape[-1]:
            raise ValueError("steering state/vector dimensions mismatch")
        modified = hidden.clone()
        delta = vector.to(device=hidden.device, dtype=hidden.dtype) * multiplier
        if state["positions"] is None:
            modified += delta
        else:
            positions = state["positions"].to(hidden.device)
            modified[torch.arange(hidden.shape[0], device=hidden.device), positions] += delta
        if audit is not None:
            audit["max_abs_hidden_delta"] = max(audit.get("max_abs_hidden_delta", 0), (modified-hidden).abs().max().item())
        return (modified, *output[1:]) if isinstance(output, tuple) else modified
    try:
        handles.append(model.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(model.model.layers[layer].register_forward_hook(add))
        yield
    finally:
        for handle in handles:
            handle.remove()


def literal_checks(row, text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    def contains(value):
        return bool(re.search(r"(?<!\w)" + re.escape(value.casefold()) + r"(?!\w)", normalized))
    coverage = {f["key"]: all(any(contains(v) for v in group)
                             for group in f.get("literal_groups", [f.get("allowed_values", [])]))
                for f in row["required_facts"]}
    numbers = set(re.findall(r"(?<!\w)\d+(?:\.\d+)?(?!\w)", normalized))
    names = {n for n in ("Maya", "Leo", "Nora", "Omar", "Iris") if contains(n)}
    days = {d for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday") if contains(d)}
    return {"coverage": coverage, "complete": all(coverage.values()),
            "unsupported_literal": bool(numbers - set(row["allowed_numbers"]) or names - set(row["allowed_names"])
                                        or days - set(row["allowed_days"])),
            "limits": "Literal diagnostic only; paraphrases may fail and unrecognized unsupported claims may pass."}


def degenerate(text):
    words = text.casefold().split()
    grams = [tuple(words[i:i+4]) for i in range(len(words)-3)] if len(words) >= 20 else []
    excess = (len(grams) - len(set(grams))) / len(grams) if grams else 0.0
    return {"degenerate": excess > 0.20, "excess_repeated_fourgram_ratio": excess}


def token_length(prediction):
    count = len(prediction["generated_ids"]) - int(prediction.get("ended_eos", False))
    return max(0, count)


def aligned(rows, predictions):
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or [r["id"] for r in predictions] != ids:
        raise ValueError("unique ordered prediction/scenario IDs required")


def make_packets(rows, conditions, comparisons, seed):
    rng = random.Random(seed)
    for predictions in conditions.values():
        aligned(rows, predictions)
    content, warmth, keys = [], [], {"content": {}, "warmth": {}}
    def identifier(kind):
        return kind + "-" + format(rng.getrandbits(128), "032x")
    for name, predictions in conditions.items():
        for row, prediction in zip(rows, predictions):
            item = identifier("r")
            content.append({"id": item, "scenario": row["user"], "required_facts": row["required_facts"],
                            "response": prediction["output"]})
            keys["content"][item] = {"scenario_id": row["id"], "condition": name}
    for reference, candidate in comparisons:
        for i, row in enumerate(rows):
            item, flip = identifier("p"), bool(rng.getrandbits(1))
            a, b = (candidate, reference) if flip else (reference, candidate)
            warmth.append({"id": item, "scenario": row["user"],
                           "A": conditions[a][i]["output"], "B": conditions[b][i]["output"]})
            keys["warmth"][item] = {"scenario_id": row["id"], "reference": reference, "candidate": candidate,
                                   "A": a, "B": b}
    rng.shuffle(content)
    rng.shuffle(warmth)
    return {"content": content, "warmth": warmth}, keys


def validate_ratings(packet, submission, kind):
    judge = submission.get("judge", {})
    if (not isinstance(judge.get("model_identifier"), str) or not judge["model_identifier"].strip()
            or not isinstance(judge.get("exposed_settings"), dict)
            or any(type(judge.get(k)) is not bool for k in ("fresh_context", "advisor_overlap"))):
        raise ValueError("actual judge identity, exposed settings and context/overlap disclosures required")
    items = {r["id"]: r for r in packet}
    ratings = submission.get("ratings", [])
    if len(items) != len(packet) or len(ratings) != len(items) or {r["id"] for r in ratings} != set(items):
        raise ValueError("ratings must completely and uniquely align with packet IDs")
    for rating in ratings:
        if kind == "content":
            fields = {"id", "coverage", "task_completed", "unsupported_claims", "contradicted_claims", "content_score", "warmth_score"}
            facts = {f["key"] for f in items[rating["id"]]["required_facts"]}
            if (not isinstance(rating.get("coverage"), dict) or set(rating["coverage"]) != facts
                    or any(type(v) is not bool for v in rating["coverage"].values()) or type(rating.get("task_completed")) is not bool
                    or any(type(rating.get(k)) is not int or rating[k] < 0 for k in ("unsupported_claims", "contradicted_claims"))):
                raise ValueError("complete boolean coverage and nonnegative integer claim counts required")
            scales = ("content_score", "warmth_score")
        else:
            fields = {"id", "preference", "warmth_A", "warmth_B", "content_A", "content_B"}
            if rating.get("preference") not in ("A", "B", "tie"):
                raise ValueError("pair preference must be A/B/tie")
            scales = ("warmth_A", "warmth_B", "content_A", "content_B")
        if set(rating) != fields or any(type(rating.get(k)) is not int or not 1 <= rating[k] <= 5 for k in scales):
            raise ValueError("invalid rating fields or 1..5 scales")
    return submission


def decoded_ratings(keys, content, warmth):
    ratings, preferences = {}, {}
    for r in content["ratings"]:
        key = keys["content"][r["id"]]
        ratings.setdefault(key["condition"], {})[key["scenario_id"]] = r
    for r in warmth["ratings"]:
        key = keys["warmth"][r["id"]]
        pair = key["reference"] + "__" + key["candidate"]
        score = 0.5 if r["preference"] == "tie" else float(key[r["preference"]] == key["candidate"])
        preferences.setdefault(pair, {})[key["scenario_id"]] = score
    return ratings, preferences


def measurements(rows, predictions, content):
    aligned(rows, predictions)
    if set(content) != {r["id"] for r in rows}:
        raise ValueError("complete content ratings required")
    literals = [literal_checks(r, p["output"]) for r, p in zip(rows, predictions)]
    ai = [content[r["id"]] for r in rows]
    return {"items": len(rows), "complete_facts": sum(all(r["coverage"].values()) for r in ai),
            "task_completion_failures": sum(not r["task_completed"] for r in ai),
            "unsupported_or_contradicted": sum(r["unsupported_claims"] > 0 or r["contradicted_claims"] > 0 for r in ai),
            "literal_complete_facts": sum(r["complete"] for r in literals),
            "literal_ai_coverage_disagreements": sum(r["complete"] != all(a["coverage"].values()) for r, a in zip(literals, ai)),
            "unsupported_literal_responses": sum(r["unsupported_literal"] for r in literals),
            "empty": sum(not p["output"].strip() for p in predictions),
            "truncated": sum(p["hit_token_limit"] for p in predictions),
            "degenerate": sum(degenerate(p["output"])["degenerate"] for p in predictions),
            "mean_response_tokens_excluding_eos": statistics.mean(token_length(p) for p in predictions)}


def gates(base, candidate, ratios):
    median = statistics.median(ratios)
    checks = {"complete_facts": candidate["complete_facts"] >= base["complete_facts"] - 1,
              "task_completion": candidate["task_completion_failures"] <= base["task_completion_failures"] + 1,
              "unsupported_or_contradicted": candidate["unsupported_or_contradicted"] <= base["unsupported_or_contradicted"] + 1,
              "length": 0.8 <= median <= 1.5, "truncation": candidate["truncated"] <= base["truncated"] + 1,
              "nonempty": candidate["empty"] == 0, "degeneration": candidate["degenerate"] <= base["degenerate"]}
    return {"passed": all(checks.values()), "checks": checks, "median_token_length_ratio": median}


def select(candidates):
    passing = [r for r in candidates if r["gates"]["passed"]]
    strong = [r for r in passing if r["warmth_score"] >= 2/3]
    if strong:
        choice = min(strong, key=lambda r: (r["multiplier"], -r["warmth_score"], abs(r["layer"]-17), r["layer"]))
        return {"condition": choice["condition"], "layer": choice["layer"], "multiplier": choice["multiplier"], "exploratory": False}
    weak = [r for r in passing if r["warmth_score"] > 0.5]
    if weak:
        choice = min(weak, key=lambda r: (-r["warmth_score"], r["multiplier"], abs(r["layer"]-17), r["layer"]))
        return {"condition": choice["condition"], "layer": choice["layer"], "multiplier": choice["multiplier"], "exploratory": True}
    return {"condition": "default", "layer": None, "multiplier": 0, "exploratory": False}


def grouped_interval(rows, scores, samples=10000, seed=20261113):
    if set(scores) != {r["id"] for r in rows}:
        raise ValueError("complete per-prompt preferences required")
    groups = {}
    for row in rows:
        groups.setdefault(row["family"], []).append(scores[row["id"]])
    rng, families = random.Random(seed), sorted(groups)
    draws = []
    for _ in range(samples):
        sampled = [score for f in rng.choices(families, k=len(families)) for score in groups[f]]
        draws.append(statistics.mean(sampled))
    draws.sort()
    def quantile(p):
        index = p * (len(draws)-1)
        lo = math.floor(index)
        return draws[lo] + (draws[math.ceil(index)] - draws[lo]) * (index-lo)
    return {"score": statistics.mean(scores.values()), "ci95": [quantile(.025), quantile(.975)],
            "families": len(families), "resamples": samples, "seed": seed,
            "method": "scenario-family bootstrap with replacement; ties=0.5; percentile95 CI; small independent-group count"}
