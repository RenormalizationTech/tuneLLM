# Persona rating calibration 01: fixture review

Claude accepted the revised 48-text fixture and rubric for freeze on 2026-11-14,
before any calibration calls. The coordinating Codex independently checked 48
unique opaque IDs, all eight binary cells with six items each, and balanced
24/24 intended labels for each trait. Author intentions remain disclosed as
synthetic reference labels, not human ground truth.

Initial review found context leakage: descriptions such as "mocks another" could
announce the intended delivery. The worker replaced these with neutral event
and participant descriptions. The rubric now rates delivery in the text and
uses context only to interpret referents and situations. No intended labels
changed. The only text replacement made an ineffective lexical trap useful:
`scrap`/`scrapbook` tests word boundaries for `crap`, alongside `class`/`assess`
for `ass` and `Hello` for `hell`. No slur entry was added to the lexicon.

The rubric explicitly counts mild but unambiguous swearing as presence (at
least 2), reserves 1 for uncertain evidence, and excludes the intended traits
from the narrow gross-failure flag. The advisor found the distinctions
plausible. Non-rude sarcasm here is situational irony about objects or events;
playful teasing of people remains untested.

Reviewed hashes:

- `evaluation/persona_calibration/texts.json`:
  `ac814007ad68abdaf932673434c48fb7c4091f3f9a61c1241d0ae84842c108ae`
- `evaluation/persona_calibration/rubric.json`:
  `ae195fc9bb989138bb00e903afa18a336fad669f22ed819112269d50ce0dc9c2`

Code review must still check the fixed lexical variants and Unicode punctuation
boundaries, judge blinding and journal/budget enforcement. No model or grader
call occurred during fixture review. The advisor and eventual isolated judge
share a model family; this review is not independent human validation.
