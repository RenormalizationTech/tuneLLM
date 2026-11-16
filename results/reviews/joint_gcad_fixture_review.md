# Joint GCAD prospective fixture review

Status: **approved for CPU preparation and registration by PM/root and Claude**,
as confirmed by root on 2026-11-16. Root's final review verdict is received;
worker structural/conversion checks are complete. Claude reviewed the artifact
and code but did not run the CPU suite; root independently verified43 passes.

- Advisor source: `outputs/joint-gcad-design-01/fixture-proposal.json`.
- Exact tracked byte copy: `evaluation/joint_gcad/fixture.json`.
- Both SHA256: `363347c663116c7a4c4bfe3bb8478ace78af37755a10242580ea42ab16b6e7f6`.
- 24 neutral English practical-advice questions; four separate sets of eight
  authored anchors (four low/four high) with fixed bands. No target7B outputs; calibration answers are AI-authored.
- All four rubric strings match their frozen source bytes and recorded SHA256s.
  Conversion preserves all question/answer strings and maps side->level and
  min_score/max_score->band; bands/design notes never enter grader payloads.
- Worker normalized exact overlap check:1540 prior unique question strings,
  zero test-question matches. Root separately reported bounded semantic review;
  advisor's lone calibration boilerplate similarity is not a test overlap.
  This does not prove semantic disjointness or task diversity.
- Root reported the proposal neutral/clear; no new semantic ratings were authored
  by the worker. Calibration anchors are AI-authored by the same judge family;
  this circularity and all-practical-advice scope must accompany any guide.
- No target/native study calls or run preparation were performed by the worker.
  The source object's proposal status and authorship remain unchanged as history;
  this accepted review and the future committed registration establish freeze
  authority; the original source-status label is retained as provenance.
