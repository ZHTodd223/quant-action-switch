# Formal calibration failure — attempt 0002

Status: `FAILED_RUNTIME_DEFECT`.

No model was loaded and GPU memory remained at `0 MiB`. Pre-execution review
found that the final calibration atomic-writer block in
`formal_experiments/scripts/01_calibrate_batch.sh` calls `os.fsync` and
`os.replace` without importing `os`. This would raise `NameError` only after
both calibration arms had completed.

The affected attempt is preserved as non-scientific failure evidence. Recovery
requires a minimal import fix, a focused regression test, the required
regression suite, a pushed production SHA, and a new `attempt-0003`.
