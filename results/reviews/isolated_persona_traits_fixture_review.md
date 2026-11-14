# Isolated rudeness and sarcasm: fixture acceptance

Claude reviewed both new 32-pair extraction corpora before any model outputs or
judgments. Its first review found no mechanical failures and recommended a
bounded correction: eight rudeness target spans complained about the task rather
than clearly addressing the fictional requester. The worker corrected those
spans and replaced all rudeness evaluator contexts with accurate single-turn
descriptions. Facts, neutral spans and the sarcasm corpus stayed unchanged.

Claude's delivered final review accepted both fixtures for freeze. The
coordinator also accepts them. Each has 16 two-item topic families, eight
four-item style families, two phrases per style family, no phrase repeated more
than twice, and 10/11/11 opening/middle/closing span positions. Outside the
annotated replacement spans, pair texts are identical. Both sides have zero
fixed-lexicon profanity hits. Cached-token length ratios are 0.913–1.107 for
rudeness and 0.967–1.067 for sarcasm.

Only system and user text enter model prompts; context is evaluator metadata.
Sarcasm's expected/observed scenario facts are present in the user prompt. The
shared development, confirmation and rating files remain byte-identical to the
profanity study. They are historically used screens, not newly unseen tasks.

Limits remain: the sarcasm genre resembles calibration despite different
objects/templates, neutral spans contain sometimes unnatural meta-statements,
and two rudeness style families have closely related wording. Family labels do
not establish semantic independence. Report attrition by family, including that
relationship. Eligibility and generated-output gates remain unchanged.

Accepted fixture SHA-256:

- Rudeness: `68708e2043bf00debc64353c3fa95eb04311b44dbc8c9c0939e1a91b35bffebf`
- Sarcasm: `6e60715ae3df08a3a6fd6a1614ab95a9600d51e48217cf21bf507e966e5f0465`

This accepts authored inputs only; it does not establish target control or
authorize live execution before code acceptance and committed registration.
