# Final 4B replay: code and input review

The coordinating Codex reviewed the new wrapper and tests. The existing Claude
scientific advisor accepted the script, config, runbook and tests against the
protocol, with no material issue. The worker's 17 focused CPU tests passed.
All 250 frozen tokenizer inputs, 32 batches and derived masks were prepared
without target generation or grading. No frozen parent code was modified.

The coordinator checked current source/protocol bindings against manifest
`6cceb5809f402fe1ecdf72583565ef570cd1d4d7f8900ca0408142cb2cdd3b17`.
It contains 681 protected bindings including the parent's 478, and checks the
same 12 model files. Registration is committed before coordinator GPU release.

Review limitations/clarifications:

- Setup/load failures close the run without retry even before the explicit
  generation-attempt marker; this is stricter than the minimum attempt rule.
- Input-mask failures stop execution as a generic failure. Captured old masks
  do not exist; the comparison is against derived parent intent.
- The raw-user hash is derived provenance. Actual rendered UTF-8, token and
  prefill checks provide the input comparison, not an independent raw-user probe.
- Missing parent environment fields are unavailable, not assumed equal. BF16
  output equality is not guaranteed across changed libraries/hardware.
- The after-run integrity receipt precedes any equality conclusion. No grading
  path exists; parent labels may be referenced only for matching outputs.

No additional 4B run follows either a match or a reported failure. This review
does not validate effective behavioral steering or larger-model stability.
