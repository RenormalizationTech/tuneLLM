# Original-Mistral SAPS design review

Claude reviewed the prospective full450-sentence design and accepted it before
any target output. Use pinned original Mistral-7B-Instruct-v0.1 and the actual
released TransformerLens2.17 runtime, not a native-HF port. Public stored-log
arithmetic is complete but unresolved data/prompt lineage prevents exact
numerical reproduction claims.

## Accepted scientific design

Use500 released training sentences, random instruction mode and seed42 before
model loading; record actual draws and post-load RNG state. Direct author
helpers learn per-layer unit mean differences and styled target projections:
1500 prompt forwards. Four separate model loads generate all450 released test
sentences at fixed block12: SAPS0, fixed1, SAPS1, SAPS1.4. Fixed projection adds
450 prompt forwards. Source greedy B1 cap60 and plain rewrite instruction are
unchanged. No layer oracle, capability run, random control or rescue sweep.

Actual loader dtype, RMS folding, centering warnings and BOS behavior are
recorded and preserved. SAPS derives its update from the current last-token
projection and broadcasts it over all forwarded prefill positions. Fixed
projection keeps its original per-prompt scalar throughout generation.

Ten unscored generation probes and two bounded logit forwards test finite
zero parity, nonzero changes, prefill/decode coverage and cleanup. Mechanics
fixed-projection captures have a separate ceiling of four, off the test set.
Total scored/extraction prompt budget1950, plus these four captures;1810
maximum generations including probes. Commit direction/mechanics bindings
before scored generation.

90 blinded batches of20 distinct source sentences use the existing pinned
Claude evaluator instead of the authors' three-model mean, with four co-rated
0–5 metrics. Packets include original complaint and author-stripped rewrite;
sarcasm uses the rewrite alone, entailment compares both. Preserve Appendix C
hard constraints and report cap hits beside>=3/≥4 counts. Integer checks are
descriptive and never select a setting. No more than180 native starts.

## Implementation review

Accepted before preparation. Claude independently reviewed the core against the
pinned author and installed TL source, then the tokenizer-equivalence test,
remaining grading journal, report and dispatcher. No scientific blockers were
found. The coordinator reviewed the active code and complete runbook; the final
worker suite and independent coordinator rerun each passed19 CPU checks
(in27.32s), including actual tiny TL cached zero/nonzero
hooks, final-EOS/cap token trajectories, interrupted native receipts, exact
budgets and incomplete/fatal reports.

The explicit local-tokenizer injection preserves revision/offline loading.
A CPU check mirrored TL internal loading kwargs, then passed both tokenizers
through actual TL set_tokenizer and compared3450 rendered training/test
prompts. Strings, actual to_tokens IDs, BOS/EOS/PAD, padding side, add_bos_token
and TL BOS flags matched. This checks equivalence of the inspected loading
branch rather than literally calling its unpinned tokenizer lookup.

The coordinator's initial HF-loop assumption about an omitted final sampled
embedding was corrected by reading installed TL2.17: it embeds the final token
before the EOS break. Recording now observes final tokenizer decoding and
verifies against the complete actual embedding trace, preserving source calls.

Operational constraint retained: interpreter/committed-source checks inside
live dispatch can permanently stop a wrongly launched command. The coordinator
will use exact prescribed environments and commit registrations in sequence;
real failures are not retried. Missing phase/release files and busy resources
wait without starting a phase.

No target model has been called. Exact initial registration is still required
before extraction, and a committed successful direction/mechanics lock before
main generation. This review does not itself release either stage.
