# Formal experiment workspace

This directory is the operator-facing home for the registered formal v5
cross-model native-tools matrix.

- `scripts/` contains the ordered, guarded execution sequence.
- `artifacts/` contains checked-in preparation evidence and renderer/model
  manifests.
- `attempts/<attempt-id>/` is reserved for exactly one execution attempt.
- `reports/` contains derived reports; preserve their links to the matching
  attempt evidence.
- `repair_ledger/` is append-only runtime-repair evidence.

Do not put ad-hoc scratch output here.  Use `tmp/` for disposable local work
and create a new attempt directory before any formal execution.  See
`../docs/repository_layout.md` for the full lifecycle and safety rules.
