# Formal GPU matrix handoff

Do not download, install packages, edit thresholds, change backend, allow
offload/fallback, or reuse an attempt directory during GPU execution.

For each seed in matrix order `101, 202, 303`, choose a unique attempt ID.
For each model in matrix `model_order`, run:

```bash
formal_experiments/scripts/00_formal_matrix_preflight.sh MODEL_KEY
formal_experiments/scripts/01_calibrate_batch.sh MODEL_KEY CALIBRATION_ID
formal_experiments/scripts/02_run_model_bf16.sh MODEL_KEY ATTEMPT_ID SEED BATCH
formal_experiments/scripts/05_finalize_model.sh MODEL_KEY ATTEMPT_ID bf16
formal_experiments/scripts/03_release_model.sh MODEL_KEY
formal_experiments/scripts/04_run_model_quant.sh MODEL_KEY ATTEMPT_ID SEED BATCH
formal_experiments/scripts/05_finalize_model.sh MODEL_KEY ATTEMPT_ID quant
formal_experiments/scripts/03_release_model.sh MODEL_KEY
```

`BATCH` must be the smaller common-safe BF16/INT8 calibration value for that
model. Candidate sizes are 1, 2, 4, 8, 12, 16 and 32. Target peak memory is
75-90%, free memory must remain at least 4 GiB, cases must not be repeated,
and the same order/partition/batch must be used within the pair. Calibration
files are explicitly not formal evidence.

After all three models for one seed attempt are comparable:

```bash
formal_experiments/scripts/06_cross_model_summary.sh ATTEMPT_ID
```

Use `formal_experiments/scripts/monitor_gpu.sh 5` only in a separate monitoring
process. Stop immediately on BF16 gate failure, attestation failure, OOM,
offload, fallback, target-coverage drift, snapshot hash drift or renderer hash
drift. Do not repair scientifically unfavorable results. Genuine runtime code
defects require a minimal versioned repair and entries in both repair-ledger
files before rerunning under a new attempt ID.

Expected preflight result is `FORMAL_MATRIX_PREFLIGHT_PASSED`. The matrix may
be started only when `gpu_execution_ready=true`, local HEAD, remote feature
SHA and remote `agent/domestic-cache` SHA are identical, and `git status
--short` is empty.
