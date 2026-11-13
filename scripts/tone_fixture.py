"""Deterministically author synthetic scenarios, never model responses."""
import argparse
from collections import Counter
import json
import re
from pathlib import Path

from scripts.common import ROOT

# Fine-grained task templates; broad domains are balanced to within one family.
# Each task has a distinct requested transformation/action, not a paraphrase of
# another split's task. Values are deliberately mundane and literal/checkable.
FAMILIES = {
    'extraction': [
        ('home', 'laundry_sort'),
        ('home', 'pantry_label'),
        ('home', 'plant_water'),
        ('home', 'key_handover'),
        ('learning', 'flashcard_plan'),
        ('learning', 'book_return'),
        ('learning', 'notebook_index'),
        ('learning', 'quiz_materials'),
        ('community', 'food_donation'),
        ('community', 'volunteer_badge'),
        ('community', 'street_cleanup'),
        ('community', 'notice_translation'),
        ('leisure', 'picnic_checklist'),
        ('leisure', 'game_rules'),
        ('leisure', 'craft_allocation'),
        ('leisure', 'photo_caption'),
    ],
    'development': [
        ('home', 'repair_log'),
        ('home', 'delivery_location'),
        ('learning', 'practice_reflection'),
        ('learning', 'experiment_observation'),
        ('community', 'tool_checkout'),
        ('leisure', 'ticket_confirmation'),
    ],
    'final': [
        ('home', 'shelf_inventory'),
        ('home', 'leftover_storage'),
        ('home', 'recycling_schedule'),
        ('learning', 'reading_progress'),
        ('learning', 'museum_question'),
        ('learning', 'project_credit'),
        ('community', 'lost_property'),
        ('community', 'seed_exchange'),
        ('community', 'fundraiser_receipt'),
        ('leisure', 'walk_route'),
        ('leisure', 'puzzle_hint'),
        ('leisure', 'club_attendance'),
    ],
}
NAMES = ("Maya", "Leo", "Nora", "Omar", "Iris")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

# Explicit communicative requests and varied concrete objects within each family.
# event means action+count+object; literals check count+object only, AI judges the
# relational action and whether the requested communicative artifact was made.
EVENTS = {
 "laundry_sort": ("Write a reminder to {name} about sorting {n} {obj} on {day}.", "sorting", "towels|socks|shirts|sheets|napkins"),
 "pantry_label": ("Write a message to the household saying {name} labeled {n} {obj} on {day}.", "labeled", "jars|tins|containers|bottles|boxes"),
 "plant_water": ("Write a reminder to {name} about watering {n} {obj} on {day}.", "watering", "ferns|seedlings|herbs|flowers|palms"),
 "key_handover": ("Write a handover message to {name}: collect {n} {obj} on {day}.", "collect", "keys|access cards|key rings|key tags|door fobs"),
 "flashcard_plan": ("Write a study-plan message to {name} about practicing {n} {obj} on {day}.", "practicing", "word cards|number cards|picture cards|phrase cards|symbol cards"),
 "book_return": ("Write a library reminder to {name} about returning {n} {obj} on {day}.", "returning", "novels|atlases|comic books|dictionaries|poetry books"),
 "notebook_index": ("Write a message to the study group saying {name} indexed {n} {obj} on {day}.", "indexed", "notebook pages|sketch pages|reference pages|journal pages|reading notes"),
 "quiz_materials": ("Write a quiz preparation message to {name}: bring {n} {obj} on {day}.", "bring", "pencils|erasers|rulers|pens|blank cards"),
 "food_donation": ("Write a donation update to the volunteers: {name} donated {n} {obj} on {day}.", "donated", "soup tins|rice packets|pasta bags|bean tins|oat packets"),
 "volunteer_badge": ("Write a message to the welcome team saying {name} prepared {n} {obj} for {day}.", "prepared", "name badges|visitor badges|helper badges|team badges|speaker badges"),
 "street_cleanup": ("Write a cleanup assignment message to {name}: fill {n} {obj} on {day}.", "fill", "rubbish bags|leaf sacks|recycling bags|litter sacks|paper bags"),
 "notice_translation": ("Turn these notes into a message to the hall team: {name}; deliver {n} {obj}; {day}.", "deliver", "chairs|stools|tables|benches|seat cushions"),
 "picnic_checklist": ("Write a picnic preparation message to {name}: pack {n} {obj} on {day}.", "pack", "cups|plates|spoons|forks|water bottles"),
 "game_rules": ("Write a game setup message to {name}: start with {n} {obj} on {day}.", "start with", "tiles|counters|cards|tokens|marbles"),
 "craft_allocation": ("Write a craft-session message to {name}: take {n} {obj} on {day}.", "take", "ribbons|paper squares|felt strips|wooden beads|fabric patches"),
 "photo_caption": ("Write a photo-sharing message to the club: the photo shows {name} holding {n} {obj} on {day}.", "holding", "kites|paper boats|painted stones|origami birds|clay figures"),
 "repair_log": ("Write a repair update to the household saying {name} replaced {n} {obj} on {day}.", "replaced", "drawer handles|cupboard knobs|shelf brackets|curtain hooks|door stops"),
 "delivery_location": ("Write delivery instructions to {name}: leave {n} {obj} by the {location} on {day}.", "leave", "parcels|cartons|envelopes|crates|packages"),
 "practice_reflection": ("Write a practice update to the tutor saying {name} completed {n} {obj} on {day}.", "completed", "spelling exercises|arithmetic exercises|handwriting exercises|reading exercises|grammar exercises"),
 "experiment_observation": ("Write an observation update to the gardening group: {name} counted {n} {obj} on {day}; do not explain their cause.", "counted", "sprouts|buds|leaves|roots|shoots"),
 "tool_checkout": ("Write a loan confirmation message to {name}: you borrowed {n} {obj} on {day}.", "borrowed", "spades|rakes|trowels|watering cans|buckets"),
 "ticket_confirmation": ("Write a booking confirmation to {name}: you reserved {n} {obj} for {day}.", "reserved", "cinema tickets|concert tickets|theatre tickets|gallery tickets|festival tickets"),
 "shelf_inventory": ("Write a stock update to the household saying {name} counted {n} {obj} on the shelf on {day}.", "counted", "baskets|bowls|mugs|vases|trays"),
 "leftover_storage": ("Write a food-storage update to the household saying {name} refrigerated {n} {obj} on {day}.", "refrigerated", "soup portions|rice portions|stew portions|pasta portions|salad portions"),
 "recycling_schedule": ("Write a collection reminder to {name}: put out {n} {obj} on {day}.", "put out", "recycling bins|paper bundles|glass boxes|cardboard stacks|bottle bags"),
 "reading_progress": ("Write a reading update to the book group saying {name} finished {n} {obj} on {day}.", "finished", "chapters|short stories|essays|poems|book sections"),
 "museum_question": ("Write a question to the museum asking whether {name} may bring {n} {obj} on {day}.", "may bring", "sketchbooks|notebooks|pencil cases|drawing pads|clipboards"),
 "project_credit": ("Write a thank-you message to the project group crediting {name} for drawing {n} {obj} on {day}.", "drawing", "diagrams|maps|charts|plans|illustrations"),
 "lost_property": ("Write a lost-property message to the reception team: {name} left {n} {obj} on the {location} on {day}.", "left", "scarves|gloves|hats|umbrellas|bags"),
 "seed_exchange": ("Write a seed-exchange announcement to the garden group saying {name} offers {n} {obj} on {day}.", "offers", "bean packets|pea packets|carrot packets|lettuce packets|flower packets"),
 "fundraiser_receipt": ("Write a purchase confirmation message to {name}: you bought {n} {obj} on {day}; do not invent a price.", "bought", "raffle entries|cake vouchers|craft vouchers|plant vouchers|book vouchers"),
 "walk_route": ("Write a route message to {name}: pass {n} {obj} on {day}, starting at the {location}.", "pass", "bridges|ponds|orchards|meadows|viewpoints"),
 "puzzle_hint": ("Write a puzzle hint to {name} repeating this known fact: you still need {n} {obj} on {day}.", "still need", "corner pieces|edge pieces|blue pieces|round pieces|striped pieces"),
 "club_attendance": ("Write an attendance update to the club saying {name} attended {n} {obj} on {day}.", "attended", "dance sessions|chess sessions|painting sessions|singing sessions|drama sessions"),
}
LOCATIONS = {"delivery_location": ("side door", "front porch", "back door", "garden shed", "garage door"),
             "lost_property": ("lobby bench", "entrance seat", "reception desk", "waiting chair", "hall table"),
             "walk_route": ("park gate", "station entrance", "village square", "library door", "canal bridge")}


def authored():
    result = {}
    for split, families in FAMILIES.items():
        result[split] = []
        for domain, family in families:
            template, action, objects = EVENTS[family]
            for i, name in enumerate(NAMES):
                obj = objects.split("|")[i]
                location = LOCATIONS.get(family, ("",) * 5)[i]
                facts = [{"key": "person", "allowed_values": [name]},
                         {"key": "day", "allowed_values": [DAYS[i]]},
                         {"key": "event", "meaning": f"{name}: {action} {i+2} {obj}",
                          "literal_groups": [[str(i+2), WORDS[i+2]], [obj]],
                          "ai_check": "Preserve action, quantity, object and their relationship; mere keyword presence is insufficient."}]
                if location:
                    facts.append({"key": "location", "allowed_values": [location]})
                result[split].append({"id": f"{family}-{i+1}", "family": family, "domain": domain,
                    "user": template.format(name=name, day=DAYS[i], n=i+2, obj=obj, location=location),
                    "required_facts": facts, "allowed_names": [name], "allowed_days": [DAYS[i]],
                    "allowed_numbers": [str(i+2)], "authorship": "synthetic AI-authored scenario; no generated response"})
    # First-50 eligibility is family-balanced before any attrition is observed.
    result["extraction"] = [row for i in range(5) for row in result["extraction"] if row["id"].endswith(f"-{i+1}")]
    return result


def validate(splits):
    ids, families, templates = set(), set(), set()
    counts = {}
    for split, expected in (("extraction", 80), ("development", 30), ("final", 60)):
        rows = splits[split]
        if len(rows) != expected:
            raise ValueError("incorrect scenario count")
        by_family, domains = Counter(), Counter()
        local_families = set()
        for row in rows:
            if row["id"] in ids or row["family"] in families:
                raise ValueError("IDs or families overlap across splits")
            if re.search(r"\b(?:warm\w*|encourag\w*|tone|neutral|friendly)\b", row["user"], re.I):
                raise ValueError("user prompt contains tone instructions")
            facts = row["required_facts"]
            if not 2 <= len(facts) <= 4 or len({f["key"] for f in facts}) != len(facts):
                raise ValueError("two to four unique literal fact keys required")
            if any(not group or not any(v.casefold() in row["user"].casefold() for v in group)
                   for f in facts for group in f.get("literal_groups", [f.get("allowed_values", [])])):
                raise ValueError("required fact not present in user scenario")
            ids.add(row["id"])
            by_family[row["family"]] += 1
            domains[row["domain"]] += 1
            local_families.add(row["family"])
        if any(n != 5 for n in by_family.values()) or max(domains.values()) - min(domains.values()) > 5:
            raise ValueError("families need five scenarios and balanced broad domains")
        if split == "extraction" and [r["id"] for r in rows] != [f"{f}-{i+1}" for i in range(5) for _, f in FAMILIES[split]]:
            raise ValueError("extraction order must be fixed family round-robin")
        # The authored task templates need an independent semantic leakage review.
        for domain, family in FAMILIES[split]:
            template = EVENTS[family][0]
            if template in templates:
                raise ValueError("duplicate task template")
            templates.add(template)
        families |= local_families
        counts[split] = {"rows": len(rows), "families": len(by_family), "domains": dict(domains)}
    return {"counts": counts, "authorship": "synthetic AI-authored", "automatic_family_disjointness": True,
            "extraction_order": "fixed round-robin across 16 families: all scenario1, then all scenario2, etc.; retain first50 eligible",
            "pre_generation_revision": "semantic event and task completion; communicative requests; varied objects/locations",
            "semantic_template_audit": "Distinct authored actions; independent advisor paraphrase-leakage review required before generation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="evaluation/tone")
    parser.add_argument("--replace-authored-fixture", action="store_true", help="only before generation/freeze; never responses")
    args = parser.parse_args()
    target = ROOT / args.output
    target.mkdir(parents=True, exist_ok=True)
    splits = authored()
    for name, rows in {**splits, "fixture_manifest": validate(splits)}.items():
        filename = target / f"{name}.json"
        with filename.open("w" if args.replace_authored_fixture else "x", encoding="utf-8") as stream:
            json.dump(rows, stream, indent=2, ensure_ascii=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
