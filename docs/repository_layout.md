# Repository layout and experiment lifecycle

This repository deliberately separates immutable experiment design, executable
code, and generated evidence.  Use this map when creating or locating work;
do not move a frozen artifact merely to make a directory look tidier.

| Location | Purpose | Write policy |
| --- | --- | --- |
| `config/` | Active-protocol pointer, versioned configuration, evidence registry, and schemas | Version configuration; do not edit a locked protocol in place. |
| `protocols/` | Frozen logical-case manifests and versioned research protocols | Immutable once registered. |
| `formal_experiments/` | Formal v5 matrix control plane and its checked-in provenance | Use the numbered scripts; create one directory per attempt under `attempts/`. |
| `formal_experiments/artifacts/` | Checked-in, hashed preparation artifacts and renderer manifests | Preserve; regenerate only through the registered preparation flow. |
| `formal_experiments/attempts/<attempt-id>/` | One formal execution attempt: logs, raw outputs, configs, manifests, and completion record | Never reuse an attempt directory. |
| `formal_experiments/reports/` | Human-readable derived reports, grouped by attempt or preparation stage | Derived only; link back to raw evidence. |
| `formal_experiments/repair_ledger/` | Append-only record of runtime repairs | Add a new entry; never rewrite prior entries. |
| `scripts/` | Shared implementation and historical-compatible entrypoints | Navigate with `scripts/README.md`; paths remain stable for protocols and evidence. |
| `tests/` | CPU control, contract, and regression tests | Update alongside behavioral code changes. |
| `docs/` | Runbooks, handoffs, audits, and repository navigation | Keep historical reports under their existing names. |
| `runs/`, `artifacts/`, `models/`, `data/generated/`, `.research-state/`, `tmp/` | Local/generated runtime state | Git-ignored; preserve or upload according to the active protocol. |

## Formal-matrix sequence

The current formal matrix uses `formal_experiments/scripts/` as its sole
operator-facing sequence:

1. `00_formal_matrix_preflight.sh`
2. `01_calibrate_batch.sh`
3. `02_run_model_bf16.sh`
4. `03_release_model.sh`
5. `04_run_model_quant.sh`
6. `05_finalize_model.sh`
7. `06_cross_model_summary.sh`

`monitor_gpu.sh` is observational only.  It does not authorize an execution.
Read the active protocol and run preflight before invoking any GPU step.

## Rules for new experiments

- Place a formal v5 run in a new `formal_experiments/attempts/<attempt-id>/`
  directory; do not write it into `tmp/`.
- Put raw outputs, environment capture, command log, and manifest together in
  the attempt directory before deriving a report.
- Keep model caches and large artifacts outside Git, then record their hashes
  and remote verification in the attempt evidence.
- For exploratory or legacy reproduction work, create a versioned protocol and
  a named run root first.  Do not place ad-hoc experiment outputs beside code.

For branch and worktree ownership, see `docs/branch_worktree_guide.md`.
