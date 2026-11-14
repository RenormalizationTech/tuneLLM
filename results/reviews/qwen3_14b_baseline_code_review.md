# Qwen3-14B baseline: code and frozen-input review

The coordinator reviewed the runner and bindings. The existing Claude scientific
advisor accepted the current runner/config/runbook/tests with no material issue.
The worker's 19 focused CPU checks passed, including actual cached tokenization,
MC boundaries, source-ID alignment, separate denominators, complete safe results
surviving unsafe grading exhaustion, receipt tampering and fatal API/uncertain
attempt handling. The registration pins the tests file hash separately.

The coordinator verified that all 21 current source files and final protocol,
design review and immutable 4B close-out match the prepared snapshot:
`ad58ef1210c7349da3de6cd971e2a4a34fb3e78c2a68490614b04a8e814eeec7`.
It binds 30 snapshot files, 709 external files including the prior 681, and all
17 verified model assets. Safe250, unsafe200, preflight8 and quality48 were frozen
before any 14B output. No generation or grades directory existed at review.

The unused CPU-only preparation draft was retained under the ignored preparation
draft directory before a runtime-binding correction and rebuild. It produced
no model or grading outputs and is not the executed manifest.

One BF16 model load on explicit GPU UUID is used for preflight, quality, safe
and unsafe phases. All parameters must be BF16 on the one exposed CUDA device.
The short preflight has different prompting from the quality screen and does
not establish worst-case memory capacity. Runtime/phase output hashes bind
generation identity and completed subsets. No fallback or new intervention exists.

Judging preserves first valid labels under separate safe/unsafe journals. An
incomplete unsafe subset cannot erase completed safe labels. Error/decline
handling cannot change payloads or obtain extra calls. Public reports contain
only current known aggregate schemas, hashes, provenance and costs; future
scorer-schema changes would require another privacy review.

The completed 4B replay had 250/250 exact matches with no new grading. This 14B
phase is descriptive baseline work, not proof of effective steering, general
safety, or size causing more stable behavior.
