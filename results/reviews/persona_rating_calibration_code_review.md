# Persona rating calibration 01: implementation review

Claude accepted the code, runbook and tests with no material issues on
2026-11-14, after the fixture review and before any preparation or judgment.
It approved CPU preparation and bounded judging after a committed registration
and coordinator release. The advisor read the code and test coverage; the
worker ran the 45 focused CPU tests, all passing on the final implementation.

The coordinating Codex reviewed the runner and requested explicit config/plan
validation and rejection of excess or post-first-valid attempt artifacts.
The worker implemented those checks. A prepare/verify integration mismatch was
also fixed before creating any actual snapshot: the fixture snapshot preserves
the full disclosure metadata. The focused tests cover the complete preparation
and verification path with mocked external evidence.

Reviewed behavior includes exact two-pass/six-batch/eight-item plans; blinded
payloads without intended labels or cells; all three trait scores; whole-word
lexical matching with Unicode punctuation; exact 41/48 and 46/48 boundaries;
quadratic weighted kappa with an exact 0.7 threshold and undefined failure;
strict rating schema and model identity; immutable first-valid labels;
structural retry ceiling, fatal/uncertain stops and budget-preserving resume;
receipt tampering; and aggregate-only report privacy.

Only new calibration files were added. Existing grading foundations and model
sources remain unchanged. No CPU test invoked a model, GPU, download or grader.
The final calibration source and test hashes are pinned in its registration.
This review establishes implementation readiness, not a passing calibration or
an accepted persona-steering vector.
