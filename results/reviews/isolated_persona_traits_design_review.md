# Isolated rudeness and sarcasm: design review

Claude reviewed this next-trait plan before profanity development behavior was
inspected and recommended common tasks, fresh matched baselines, fixed parameters
and an independent second confirmation rating pass. The coordinator wrote the
concrete protocol and discussed its consensus rules and limits with Claude before
any new rudeness/sarcasm data or outputs existed.

The delivered review found the shared tasks, span-controlled contrasts, target
counts, conservative off-target/quality consensus, fixed sites/strengths, stage
locks, constrained plans and 40-batch/80-call ceiling consistent with that plan.
Its material question was whether the earlier grid had actually changed outputs.
The coordinator's stored-output diagnostic found 6/9/13 changed sequences at
block 17 and 6/11/14 at block 23, out of 16, with nonzero audited interventions.
Claude accepted keeping the grid fixed on that basis. First-divergence, length
and per-token norm diagnostics are now required in each new study.

The range remains modest; no claim excludes larger magnitudes or other
representations. Target absence is distinct from an inactive intervention.
Sarcasm/rudeness overlap on neutral tasks can cause a valid null; gates will
not be relaxed afterward. Shared prompts are historically used screens rather
than newly unseen benchmarks, and both rating passes share one model family.

The coordinator accepts this design for implementation. Each trait needs its own
fixture/code acceptance, passing CPU checks, fresh immutable manifest and committed
registration before calls. Rudeness executes first, then sarcasm. Other methods,
joint steering and further 14B experiments are separate later decisions.
