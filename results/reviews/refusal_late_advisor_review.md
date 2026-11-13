# Layer-23 follow-up: planning, review and acceptance

Date: 2026-11-13. After the layer-17 null, the user explicitly selected one
late-layer refusal follow-up. Claude agreed on layer 23 only, strengths
0.25/0.5/1, inherited cap 320, identical data and model, compatible baseline
outputs and existing labels reused, and unchanged quality/selection rules.
The design is post hoc, motivated by an already-seen development diagnostic.
No new layer grid, negative steering, training or vector release was approved.

Claude accepted the separate protocol and reviewed the saved runner/config
without finding a material method issue. The manager reviewed the code and
13 new CPU checks. Before freeze, the timing projection was changed to use
the slower probe and labelled approximate, baseline imported labels were
cross-checked against their original submission and committed decision, and
a distinct anonymous-ID seed avoided cross-study ID reuse. None changed
prepared data, generation settings or the selection rule. Historical source
and the accepted classifier/rubric remained byte-identical.

The exact run was registered in `057d7d3` before new GPU work. New layer-23
zero/nonzero mechanics passed, and `414832c` committed the measured budget
before development. The 90 new candidate responses were judged in five
fresh-context calls, all valid on the first attempt. The 60 baseline outputs
and their 60 labels were reused without generation or judging calls.

Claude accepted the locked null in `dd88d1d`: baseline benign refusals were
1/30, versus 4/30, 7/30 and 24/30 as dose increased. The strongest setting
met the refusal target but had six degenerate outputs; weaker settings missed
the target. No setting passed both gates. All 228 final responses were skipped.
The result is an exploratory dose-related refusal increase with quality loss,
not no effect and not validated usable control. Small already-seen development
data, separate baseline/candidate judging contexts, source-style confounds,
AI-label uncertainty and unequal truncation limit interpretation.

The worker's final read-only audit verified 14 source hashes, 35 snapshot
files and 95 original-artifact bindings. The decision recomputed to null;
reused/new counts and costs were preserved. No raw prompt/response text,
token-ID fields, secrets, emails or home paths were detected in the aggregate.
No source changes or extra model/judging runs followed that audit.

The single follow-up is complete and the refusal pilot closes under its rule.
No threshold relaxation, intermediate-strength search or further layer trial
is active. The original null remains published unchanged.
