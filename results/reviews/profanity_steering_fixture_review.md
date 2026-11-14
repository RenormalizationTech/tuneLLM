# Isolated profanity pilot: final fixture acceptance

The coordinating Codex accepted the final fixture before any pilot output or
judgment. The worker's final 29 focused CPU checks passed, including all 64
cached-tokenizer response boundaries and every 0.85–1.15 pair-length window.

Claude reviewed the initial fixture, found its facts/references and calibrated
trait definitions sound, and requested position/phrase diversity plus two
confirmation theme changes. Those corrections were applied before freezing:
32 pairs now have 10 opening, 11 middle and 11 closing first-hit positions,
using exact integer thirds; 14 phrases occur at most three times each, with at
least two per word family. There are seven word families, none above one third
of authored pairs, and one lexical hit per positive answer. Neutral/profane twins
preserve the reference facts and use no personal targets or ironic praise.
PC13/PC14 now use kitchen/garden sorting rather than display models.

Claude's final revised-fixture response was stopped by its safety classifier;
that review is unavailable, **not acceptance**. The coordinator independently
read all revised pairs and confirmation references, checked the quotas and
accepted the final fixture under the user-authorized workflow. No stopped
advisory response was reproduced or used as evidence.

The 16/8/12 named topic/template families have two items each and are disjoint
across extraction/development/confirmation. Exact source cases are not reused
from calibration; broad skills such as short messages and arithmetic recur.
These are deliberately small synthetic probes, not unseen population benchmarks.
The eight factual references cover elementary arithmetic, units, polygon sides
and fractions. Nonfactual correctness is ignored; content fulfillment still
checks every reference. Calibration trait definitions remain byte-identical.

Final hashes:

- `extraction.json`: `6d208d762405aa51c97658d66b676242f97b1e3f2f54b2b4c2da54b1d2c5f18d`
- `development.json`: `3b331ff38811b2feb2ac6c073b2e736976f186b2564fc9d573598baa2387da85`
- `confirmation.json`: `4ddc14aa14bba05f9b78fe0d95abad7cf0917c6b7650c3015a0047574408031d`
- `rubric.json`: `1a2587b6e90375f975176aac24fd4ccc63b16bba31a0a770e794e56c0db44055`

All files are under `evaluation/profanity_steering/`. Authoring and scientific
review preceded model/grade outputs; no observed result drove these corrections.
