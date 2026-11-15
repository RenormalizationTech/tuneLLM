# Isolated PV reference adaptation

CPU worker writes/reviews/tests only. Coordinator prepares, commits registrations
and separately releases every GPU/native phase. No commands below authorize a call.
Existing sources and old experiments remain immutable. Profile rudeness is active;
future profanity needs new accepted artifacts/profile/registration, not edits to
this common runner. Sarcasm and joint control are outside this runner.

Use `.venv/bin/python -B -m scripts.persona_reference STAGE`; default profile is
rudeness. No alternate output paths are accepted. Source and package checks are
command-local: full used-model/source/input hashes at live entry/exit, deduplicated
by resolved filename within a traversal, no persistent cache. Suboperations check
their stage bindings without recursively rehashing unrelated historical runs.

1. After final code/fixture/protocol acceptance, coordinator runs `prepare`, then
   `verify`. Fresh destination: `outputs/persona-reference-rudeness-01`.
   Commit config, code, author data/license/attribution, protocol/design review and
   `results/protocols/persona-reference-rudeness-01.json` containing
   `manifest_sha256`. No target output exists during prepare.
2. `elicit --execute-gpu`: 200 greedy responses, original positive/negative systems,
   all five instruction pairs. `packet --phase elicit`, then released
   `judge --phase elicit --execute-judgments`. Use the process-local pinned native
   Claude 2.1.323 PATH from the existing runtime record; same audited tools-off
   foundation and audit `outputs/refusal-judge-isolation-audit-01/audit.json`.
   Default changing `claude` versions are rejected. `eligibility` freezes every
   passing pair; <20 pairs/<10 questions closes elicitation_null.
3. `extract --execute-gpu`, then `preflight --execute-gpu`: native raw token
   teacher forcing, FP32 block-output response means, no unit/R/beta shift.
   Ten unscored cap32 generations include actual zero hooks and nonzero probes.
   Commit `results/protocols/persona-reference-rudeness-strength-01.json` with
   `strength_sha256` from the new `strength.json` before scored layer work.
4. `layer --execute-gpu`: ten baseline plus nine coefficient2 conditions (100).
   Packet/judge `--phase layer`, then `choose-layer`. If `fallback_required.json`
   exists, coordinator reviews the failed fixed quality rules and releases exactly
   `fallback --execute-gpu` (90 coefficient1 outputs, no new baseline). Packet/judge
   `--phase fallback`, then `choose-layer` again. No passing fallback closes null.
   Commit `results/protocols/persona-reference-rudeness-layer-01.json` containing
   `layer_choice_sha256`. The choice records reused baseline and both score sources.
5. `strengths --execute-gpu` generates only .5/1/3/4 normally (40), or .5/3/4 after
   fallback (30). Packet/judge phase `strength` or `strength_fallback`, respectively.
   `select` reuses existing coefficient2 and optional coefficient1 grades; no
   generation/rejudging of either. A gain <10 closes selection_null. Otherwise
   commit `results/protocols/persona-reference-rudeness-selection-01.json` with
   `selection_sha256` before untouched report questions.
6. `final --execute-gpu`: 40 outputs, baseline/selected/random/prompt. Selected
   tensors freshly reloaded; random FP32 norm matches the selected raw shift;
   actual BF16 norms and shift/residual ratios are diagnostics. No extra replay.
   Packet/judge phase `final` uses original trait/coherence scales. Independently
   packet/judge phase `custom` uses ONLY calibrated 0..3 definitions plus relevance
   and coherence. Same saved answer bytes, no additional generation or score leak.
7. Only if reference report criteria pass: `retention --execute-gpu`, 32 generations
   plus64 MC items, fresh matched base/selected, official IFEval language seed0
   restored afterward. `report` exports aggregate-only
   `results/persona_reference_rudeness_01.json`, including early null resource costs.
   A passing reference criterion alone is not independent three-trait control.

No task requires new audits, dependencies, model downloads, higher caps or retries
of model phases. A resource wait does not start a phase. Any authorized live hash,
model, infrastructure, uncertain native start or exhausted schema error writes a
permanent stop. `--resume` only replays an existing native phase journal; it never
rerates a saved first-valid batch and does not recover infrastructure failures.
Two schema-only attempts per identical payload; normal max106 calls, fallback
max128. Plans are frozen, no same question twice in a batch; final partial batch
counts are explicit. All attempts, starts, receipts and input hashes are retained.

Accounting: normal422 generations/53 first-valid batches, fallback502/64; at most
200 teacher-forced sequences and64 MC in either branch. Early nulls skip later work.
Source special tokens are removed only in decoded judge answers. Pooling uses raw
native prompt/generated IDs, with explicit masks for nonterminal specials. The
retokenized source-style diagnostic is never substituted into the trajectory.
Original respective systems are included. This differs from author retokenization;
intermediate hidden-state mapping is tested; final normalized endpoint is excluded.

Numeric author trait is0..100 or explicit REFUSAL:null; coherence always integer.
Our structured native ratings are not the author's probability-weighted GPT score.
One greedy rollout replaces ten sampled rollouts; trait and coherence are co-rated
in one packet. Requiring a brief reason may itself affect judgments versus the
source numeric-only request; this difference is retained and disclosed.
Reuse across staged judging sessions limits ordinal comparisons. The custom rubric
was calibrated on short texts and its use on512-token answers is a limitation.
Private reasons/answers/IDs/masks stay ignored; no externally verified truth label
is invented for the published questions.

CPU checks (no bytecode/cache):
`.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_persona_reference.py`
