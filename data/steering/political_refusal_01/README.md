# Political-information fixture 01

Root accepted these exact source-selection, v5 annotation, calibration and
group-metadata bytes before any target-model or experimental grading output.
The JSONs retain their original editorial `DRAFT` status strings rather than
silently rewriting them. Acceptance is recorded separately in
[`results/protocols/political-refusal-data-01.json`](../../../results/protocols/political-refusal-data-01.json),
which binds these files and the final protocol/design/data reviews by SHA256.
That record is not a stage release or experiment registration.

## Contents

- `candidate_selection.json`: source indices, exact published question strings,
  bilingual fit alignments, considered/excluded rows and original audit hashes.
- `reference_annotation.json`: 48 evaluation canonical questions, project-authored
  Chinese translations, cited central anchors, alternatives, notes and flags.
- `calibration.json`: 8 separate questions and 16 authored calibration answers.
  Expected labels/mechanics are retained separately from `judge_payload`;
  they are never passed to a grader. These are authored examples, **not model
  generations or experimental outcomes**.
- `group_metadata.json`: reviewed event/core groups and the prospective
  C038→C036 control replacement. T038 loses its same-index control; C035/C036
  are grouped together as Turkish-press questions.
- `licenses/`: exact upstream licence files, copied without changes.

Fit uses 16 published EN/ZH alignments (32 physical questions), clustered in
8 event groups. Development and heldout each have 16 target and 8 control
canonical questions, with both English and Chinese versions (48 physical
items each). Related events, core answers and translations stay together.
Shared broad political/history context is declared; no topic-OOD claim is made.

The anchors are project-curated **reference consistency**, not certified truth
or an ideological position. Valid alternatives, uncertainty, premise corrections
and optional-number policy are part of the reference interpretation. Optional
notes are not extra required anchors. Source English questions are unchanged;
Chinese evaluation text is a reviewed project adaptation, not a published
native-Chinese benchmark. References/notes are for graders only, never the
generating model.

## Source attribution and licences

**DECCP — AUGMXNT / Leonard Lin.** Repository revision
[`1a6d5571f0a711d3afb7d1c43f70a23e41c359a1`](https://github.com/AUGMXNT/deccp/tree/1a6d5571f0a711d3afb7d1c43f70a23e41c359a1).
Apache-2.0; see [`licenses/deccp/LICENSE`](licenses/deccp/LICENSE).
Selected fit strings preserve GitHub source lines exactly. Source refusal tags
describe the author's Qwen2 testing, not the new Qwen2.5 model's behavior.

**Routing-is-learned — Greg Frank.** Repository revision
[`531fef71aae06db8c2fe114328e18b5b29e737d8`](https://github.com/gregfrank/routing-is-learned/tree/531fef71aae06db8c2fe114328e18b5b29e737d8).
MIT, copyright ©2026 Greg Frank; see
[`licenses/routing-is-learned/LICENSE`](licenses/routing-is-learned/LICENSE).
Evaluation/calibration English strings come from its v2 prompt lists. Selection,
translations, cited reference paraphrases, annotations and authored calibration
answers are project additions. Project-authored portions follow the repository's
[Apache-2.0 licence](../../../LICENSE); upstream material retains its own licence.

Neither complete pinned source tree contains a standalone NOTICE file. No
NOTICE was invented or substituted. The ignored audit retains the source trees,
licence declarations and canonical blob/byte hashes. No unlicensed
`chinese_auditing` or Enkrypt material is included in this fixture.

## Exact-source reproduction prerequisites

The compiler additionally verifies two raw upstream files in the ignored audit
cache. They are not model outputs and are not silently replaced by the HF
newline variant. To reconstruct an absent cache, obtain these public **pinned**
files without executing upstream scripts, place them at the paths below and
verify SHA256 before preparation:

| Public pinned file | Expected ignored cache path | SHA256 |
| --- | --- | --- |
| [DECCP `data/censored.txt`](https://raw.githubusercontent.com/AUGMXNT/deccp/1a6d5571f0a711d3afb7d1c43f70a23e41c359a1/data/censored.txt) | `outputs/political-refusal-source-audit-01/deccp/files/data/censored.txt` | `a1c1b06bebb44d6c69565d39f6abcc6133c108e32022425d21e21ca6cfc1781e` |
| [Routing `political_prompts_v2.py`](https://raw.githubusercontent.com/gregfrank/routing-is-learned/531fef71aae06db8c2fe114328e18b5b29e737d8/src/political_prompts_v2.py) | `outputs/political-refusal-source-audit-01/routing-is-learned/files/src/political_prompts_v2.py` | `bb3a8125069ce3f2cbf79a394cf1318ad5c5a8a402af2b1906e5c444e7ec0a53` |

The compiler parses the routing file's literal lists with AST; it never imports
or executes that source. Original raw audit files remain ignored and unchanged.
The [runbook](../../../scripts/political_refusal_runbook.md) describes source/
data acceptance, native grading and separate live-stage registrations. Model
snapshots, raw experimental answers, native receipts and credentials are not
contained here.
