# Formal BF16 failure — attempt 0003

Qwen seed 101 stopped during comparison-state initialization because the
runner did not create the attempt parent directory before creating `RUN_ROOT`.

No model was loaded, GPU memory remained at `0 MiB`, and no scientific result
was produced. The attempt is preserved as `FAILED_RUNTIME_DEFECT`; recovery
uses a minimal parent-directory creation fix and resumes under `attempt-0004`.
