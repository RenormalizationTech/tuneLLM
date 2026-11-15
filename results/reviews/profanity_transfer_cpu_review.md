# Three-method profanity transfer: CPU implementation review

Status: reusable components accepted; scientific protocol and live study pending.
This is not a profanity steering result and does not release GPU execution.

The user authorized fixed projection targeting, adaptive SAPS and GCAD, followed
by selecting a working method or closing the bounded attempt with the original
literature workflow and its missing-data/reproduction limitation documented.

The existing Codex worker implemented explicit-parameter helpers in
`scripts/profanity_transfer_common.py`. The coordinator independently reviewed
the code and reran the focused tests: all 22 passed in 5.45 seconds.

Evidence covers FP32/BF16 fixed/adaptive parity with pinned released SAPS hooks,
sequential prompt-direction and target arithmetic, actual tiny-Qwen prefill and
cached decoding, zero-effect parity, nonzero effects, exception cleanup, direct
GCAD formula checks, and the actual released GCAD multilayer hooks. Model/data/
representation mismatches and known foreign tensor identities are rejected.
Caller-side extraction provenance and source/weight hashes still require a live
study protocol; metadata alone cannot establish the source of a vector bank.

No historical source, model weight, dataset or result was modified. No pretrained
model was loaded for these tests, and no GPU, judge, dataset-generation, install
or download calls were made. The tiny Qwen model is randomly initialized locally.

Claude was contacted for scientific planning as requested by the user and
returned: "Login reached its time limit · Please run /login". The coordinator
requested login restoration while independent CPU implementation continued.
No fresh Claude review or endorsement is claimed. Exact inputs, extraction,
strengths, sites, evaluation and selection rules remain unsettled until that
review or an explicit user change to the review requirement.
