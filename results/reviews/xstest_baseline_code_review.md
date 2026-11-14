# XSTest-safe 01: code and frozen-input review

Date: 2026-11-14. Root reviewed the implementation and the existing Claude
scientific advisor accepted the corrected code against the protocol. No dated
server snapshot is exposed. This is code/design review, not result grading.

The worker ran 21 focused CPU checks successfully in 3.12 seconds, covering the
actual cached tokenizer, unmodified generated-body decoding, safe-only source
selection/overlap, original classifier/metric goldens, empty batching, complete
ID coverage, first-valid replay, retry exhaustion and fatal failures.

Claude's fidelity finding was corrected before preparation: the upstream C/P/I
mapping sentence is included verbatim. JSON output and brief per-item reasons
are explicit adaptations. The original metric function is used only in CPU
golden tests; the measurement path computes the equivalent documented formula.

The coordinator checked that the prepared manifest binds the current corrected
script, runbook, config, protocol, literature audit and design review. It contains
23 snapshot files, 14 sources and 478 external bindings, including all 421
protected earlier inputs (with the older 402/315 chains preserved).

Prepared manifest SHA-256:
`ad8d1ccc729564a7fec7115fb86b0273c0c50e14a7b3c005c5e464487d513fba`.

The committed registration must match this manifest before any GPU or grading
release. Exact runtime source hashes are in the registration. No extraction,
intervention or model training is authorized by this baseline.
