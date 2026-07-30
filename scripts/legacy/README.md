# Legacy scripts

These scripts are historical pilot, analysis, and operations entrypoints. They
are not formal V2 entrypoints and must not be used to create new formal
results. They were moved only after repository-reference scanning found no
current test, CI, formal wrapper, or documentation command dependency.

- `analysis/`: historical aggregations and diagnostics.
- `pilots/`: model-specific preflights and trial runners.
- `operations/`: historical download/model-lock helpers.

Frozen evidence remains in its original location; nothing in this directory
may rewrite it.
