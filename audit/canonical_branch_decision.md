# Canonical branch decision

**Selected source:** `agent/domestic-cache` at
`d584effd85b750a560ebc1c5e3d5ecd8301fb7a8`. The consolidation creates `main`
at this exact commit without rewriting any reference.

This choice is evidence-led, not a default-name choice: it contains every
remote feature/fix candidate as an ancestor, carries the v5 formal matrix and
its subsequent four runtime-repair commits, has the current navigation update,
and is the only branch whose current control-plane pointers, tests, and formal
entrypoint documentation are co-located. The latest successful Actions run in
the inspected history is older; current CI is therefore re-run before this
branch is declared healthy.

Rejected candidates:

- The eight feature/fix branches have zero unique commits relative to the
  selected source; they are merged ancestors, not alternate mainlines.
- `results/v5-formal-matrix-v1` has five unique commits, but they are
  append-only failed-attempt and repair metadata. It remains the results
  metadata branch and is deliberately not merged into production code.
- The local backup branch and two old tags preserve earlier controls but lack
  later P0/P1/v5 integration.

No unique production commit needs migration. No history is rewritten and no
frozen raw output, metric, manifest, or result record is changed by this
decision.
