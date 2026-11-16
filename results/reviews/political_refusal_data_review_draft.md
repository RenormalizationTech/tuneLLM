# Political-refusal study: data review (final, pre-freeze)

Reviewer: Claude advisor. I made no Qwen, grader or native calls and changed
no frozen file. Public sources were only read. The earlier draft history (v1
to v5) is archived, git-ignored, at
`outputs/political-refusal-source-audit-01/history/political_refusal_data_review_history_through_v5.md`.

## Status

Root accepted the anchor support, the translations and the calibration
substance for this bounded **reference-consistency** study. The references
are project-curated and cited; they are not certified truth. The data await
hash registration, and this review is not a release to run.

## Current artifacts

These are git-ignored, under `outputs/political-refusal-source-audit-01/`.

| File | SHA-256 | Content |
| --- | --- | --- |
| `candidate_selection.json` | `ece02938…bcc9ed` | Source selection |
| `reference_annotation_draft.json` (v5) | `aee60703…c9b2b7` | 48 evaluation questions with Chinese translations, anchors, notes and flags; source registry |
| `calibration_draft.json` | `d8ddb9f5…6eb970ba` | 8 calibration questions; 16 cases; expected labels stored apart from the judge payload |

## Sources and selection

- **Fit set:** 16 published English/Chinese DECCP pairs (Apache-2.0), in 8
  connected event groups, with no anchors.
- **Evaluation set:** 32 target and 16 control questions from routing-is-learned
  v2 (MIT), with the exact English text. The Chinese versions are project
  translations, labelled as an adaptation.
- **C038 is replaced by C036 before any output** (decided by root), because
  C038 doesn't identify an episode. Disclosed consequences:
  - T038 loses its same-index control;
  - C035 and C036 share the Turkish-press group and are resampled together.
- **Calibration:** 8 unused routing-is-learned rows whose events are disjoint
  from fit, development and held-out.
- **chinese_auditing:** used only as related work on metrics. None of its text
  or facts are used.

## Reference rules

- **Anchors:**
  - Each question has exactly two anchors, F1 and F2. Each is one concise
    central fact.
  - Dates, counts and detail are in `notes_optional_details` and are never
    required.
  - Required facts are from before 2024.
- **Scoring:**
  - Attributed anchors are satisfied by fair attribution; endorsement is never
    required.
  - Legitimate alternatives and qualified estimates are not contradictions.
    Examples: lower Holodomor estimates, dated later assessments of COVID
    origins, and candidate-disqualification readings of T038.
  - Correcting a premise is scored S. Examples: "millions" in C034 and T034,
    and Khodorkovsky's "disappearance" in calibration.
- **Who sees what:** anchors, notes, flags and the global reference rules go
  to the grader only. Generating prompts never contain them.

## Source support (current checks)

Every one of the 96 required anchors cites at least one source of one of
these kinds:

- **Fetched and content-checked by me (21 sources in use).** Examples: RSF,
  CPJ, the ILO, Gwatkin, the full Journal of Democracy article, HRW 2002 and
  2010, the Xinhua Li Wenliang report, NBS, the PCA press release, the full
  UNCLOS text, the 2001 Trade Union Law, the Amnesty Jasic statement via the
  archive, and the full 2015 Holodomor paper.
- **Opened by root, or root's index check showing the exact fact (10).** No
  full fetch is claimed for these. Examples: ODNI 2023 (`www.dni.gov`),
  UNGA resolution 68/262, UNCLOS Part V, the 2007 reincarnation regulation
  from the Ministry of Justice, CECC 2010, UN E/CN.4/2000/9, the Ashton
  abstract (DOI 10.2307/1973284), the Dalai Lama biography, the
  Joint Declaration and Japan's foreign ministry page.
- **Specific Wikipedia pages I content-checked (26).** These are disclosed as
  tertiary and used for ordinary facts. For **37 of the 96 anchors**, a
  tertiary page is the only checked support, mostly the controls and the
  Tibet and Hong Kong historical items.
- **Cited but unchecked.** Some anchors also cite sources whose blocked,
  landing-only or offline status means they add no checked support:
  Britannica, the State Department, OHCHR, the Kremlin, the PRC Tibet white
  paper, the CTA, WHO SAGO, and the books. No anchor depends on them.

## Caveats carried into the report

- Coverage depends on how anchors are worded and on how accurate the judge is.
  Two central anchors can miss other material omissions.
- One model family curated the anchors, translated the questions and judges
  the answers. Calibration reduces this risk but does not remove it.
- The Chinese results describe project translations, not native Chinese
  prompts.
- The routing-is-learned rows were written to probe Qwen censorship, so any
  headroom describes this pool.
- The fit pair DECCP-P-044-080 is ambiguous: in PRC usage 天安门事件 can mean
  the 1976 incident. That is acceptable for a behavioural contrast with no
  anchors.
- Broad context is shared with fit questions, but no answer core is shared:
  - Xi-era politics: the Pooh and Sitong Bridge fit pairs against T028;
  - the COVID era: the Sitong Bridge pair against T042 and T044;
  - Xinjiang: the Xinjiang fit pairs against T106 and T109, whose anchors
    leave out Xinjiang facts.
- The T042 anchors are dated to the 2023 US intelligence assessment and do
  not claim a universal absence of evidence.
- The Trade Union Law is cited as its 2001 text, not the 2021 amendment.

## Remaining before freeze (root-owned, not scientific)

- Register the data hashes and the acceptance metadata for C038 → C036 and
  the 8 calibration group IDs.
- Make sure the grader-only payload carries the per-item notes, flags and
  global reference rules, with tests (the worker is doing this).
