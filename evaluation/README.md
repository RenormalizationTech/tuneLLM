# Evaluation code and fixtures

`gsm8k.py` and `score.py` implement the original task scoring. Subfolders
hold study-specific prompts, splits, rubrics, calibration records and attribution.
Their names match the corresponding configs, scripts and compact results.

Fixtures are frozen evidence for the reported experiments. Public test prompts
are no longer unseen data for future generalization claims. Do not replace or
move them to make an old run pass. Licenses and attribution are scoped per
subfolder; the repository's code license does not override dataset terms.

See the [reproduction guide](../docs/reproduce.md) for the portable CPU test
subset versus historical integration runs, and [results/](../results/README.md)
for reported outcomes.
