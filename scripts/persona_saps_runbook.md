# Isolated native-HF SAPS sarcasm adaptation

New runner only; PV code/config/data/protocol and all old experiments are frozen.
Coordinator owns prepare, committed registrations and every live release. Worker
CPU checks are not permission for GPU/native calls. No new dependencies or TL
installation. No model download, joint control or model-size sweep.

Config: `configs/persona_saps_sarcasm.json`. Commands use
`.venv/bin/python -B -m scripts.persona_saps STAGE`. Output is fixed to
`outputs/persona-saps-sarcasm-01`, public aggregate to
`results/persona_saps_sarcasm_01.json`; alternate paths are rejected.

1. Wait for closed PV rudeness report and accepted code/fixtures/protocol. Root
   runs `prepare` then `verify`. This freezes copied500/50/450 source strings,
   seed702 test100 ordering/custom-first20, all1000 reconstructed instruction
   draws, exact user-only templates/nonthinking IDs, all possible absent-method
   packet branches, original four Appendix C definitions, custom rubric, package/
   binary/audit/model/source hashes. No target output during prepare. Commit exact
   registration `results/protocols/persona-saps-sarcasm-01.json` containing
   `manifest_sha256`.
2. `initial --execute-gpu`: validation baseline50 plus style-index0 control50,
   B1 cap60. Baseline read-only first-prefill hook saves all four raw post-block
   h0 values. `packet --phase initial`, then separately released
   `judge --phase initial --execute-judgments`; `eligibility` freezes the original
   scales and control gate. Any gate failure closes elicitation_null; no directions.
3. `directions --execute-gpu`: plain+first styled prompt per500, then second styled
   pass500:1500 forward sequences. All36 raw block outputs, including pre-final-norm
   block35. Stack before CPU transfer. Raw mean normalized per block; second pass
   target projection. r seeds7200+block. Actual source epsilon/precision hook
   ordering retained; no TL coordinate transfer. Save exact safetensors roundtrip.
   Fixed c follows author Python arithmetic: FP32 CPU dot product, separate
   target/dot `.item()`, then Python subtraction/alpha; keep c as a finite Python
   float through multiplication by the model-dtype unit. Adaptive target stays0D.
4. `preflight --execute-gpu`: four ordinary read-only capture probes, four actual
   fixed-zero-hook probes, two nonzero SAPS probes, all cap32; plus exactly two
   unscored logit forwards. Exact tokens required for zero. Nonzero hidden/logit
   deltas, cached prefill/decode counts and cleanup required. Commit
   `results/protocols/persona-saps-sarcasm-strength-01.json` with `strength_sha256`
   from the new `strength.json` before validation selection.
5. `validation --execute-gpu`:600 outputs, fixed alpha1 and SAPS alpha1/1.4 at
   blocks11/15/19/23; initial baseline/control/h0 are reused, never regenerated.
   Packet/judge phase `validation`, then `select`. Fixed quality floors first;
   best per method, then overall winner. If no gain>=0.5 or no eligible setting,
   selection_null. Missing method is absent and its test arm skipped. Commit
   `results/protocols/persona-saps-sarcasm-selection-01.json` containing
   `selection_sha256` before test. No test-driven layer oracle/reselection.
6. `test --execute-gpu`: baseline100 first with fresh TEST-only four-block h0,
   best fixed/SAPS arms if eligible, random winner controller and prompt control.
   Up to500 outputs, each once. Fixed random shares c per case, SAPS random computes
   scalar along original u and adds along r. Actual real/random prefill/decode norm
   distributions and0.8–1.25 ratio flags are report-only. Packet/judge phase `test`.
   Independently packet/judge phase `custom`: first20 fixed strings, same up-to100
   saved answer bytes, no generation. No original0..5 scores enter custom contexts.
7. Only if reference test gates pass: `retention --execute-gpu`. Fresh base/winner
   48-item screen,32 generations+64 MC. It deliberately uses B1 at every forward
   (including MC), with baseline read-only h0 per off-task input for fixed c.
   GSM cap512/IFEval1024, language detector seed0 restored. This is independent
   off-task transfer, not author paraphrase performance. `report` publishes only
   aggregates after all required work or an explicit null/fatal close; null costs
   retained. Private raw rows/IDs/answers/ratings never enter public export.

Native launch uses unchanged audited tools-off `scripts.tone_judge` foundation,
Claude2.1.323 exact native binary SHA from existing runtime record, model
claude-opus-5-5/medium. Reuse
`outputs/refusal-judge-isolation-audit-01/audit.json`, no extra audit. Process-local
PATH pin must resolve that binary. First-valid labels retained; two identical
structural attempts maximum per batch. `--resume` requires an existing exact phase
journal and cannot recover fatal infrastructure/hash/model/uncertain-start errors.
No semantic retry, alternative judge, prompt reframe or cap change. Max146 starts,
73 first-valid batches (5+30+25+13); skipped methods/nulls reduce work. Accepted
labels publish only after full live entry/exit integrity verification. Resource
wait does not start a phase; all other authorized live failures permanently stop.

Max1242 generations,1500 direction forwards,2 logit probes,64 MC. Every stage has
saved-work counters/timing/allocator memory; failed unfinished work may be additional.
Source/model hashes are command-local entry/exit traversals; same resolved aliases
deduplicate within a traversal, no cross-command cache or old-output recursion.

Appendix C was extracted from the coordinator-fetched PDF to
`outputs/persona-saps-paper-01` with PDF/text hashes and normalization provenance;
no old literature audit manifest changed. Original four definitions are retained;
opaque ID/list envelope and single Claude instead of three judges are adaptations.
Public source data has incomplete original licensing/transformation receipts and
contains profanity/hostility. Custom scores measure this input-confounded context,
not independent trait control. The frozen1000 draws reconstruct the algorithm,
not original pre-model-load RNG consumption. Native HF basis/dtype/nonthinking,
fixed limited validation search and token-based decoding are disclosed adaptations.
Twenty-item co-rating may introduce contrast effects; original sentence-alone
sarcasm and contextual custom sarcasm are separate constructs. Shift norm statistics
are additive-vector L2 per position; whole-prefix Frobenius L2 scales sqrt(S), and
its squared norm scales S. No whole-prefix L2 is mislabeled as S times larger.

CPU only:
`.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_persona_saps.py`
