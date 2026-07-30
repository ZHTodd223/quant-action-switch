# Runtime repair ledger

Only verified runtime defects belong here. Scientific outcomes, threshold
changes, backend substitutions, and model-access failures are not repairs.

## formal-attestation-requirements-v1

- Failure attempt: `attempt-0001-preflight-attestation-handoff-mismatch`
- Defect: the formal matrix referenced a missing versioned requirements file,
  while the generic runtime default allowed `0.95` coverage and conflicted with
  the matrix-locked exact coverage of `1.0`.
- Failure evidence:
  `formal_experiments/reports/formal_matrix_preflight_failure_attempt-0001.json`
- Preserved remote evidence:
  `v5-cross-model-native-tools-matrix-v1/matrix-version-1.0.0/repository-sha/ba36c87dec59df3aa3c07c23084c13e00c4b0548/model/common/seed/none/arm/none/attempt/attempt-0001-preflight-attestation-handoff-mismatch`
- Original repository SHA:
  `ba36c87dec59df3aa3c07c23084c13e00c4b0548`
- Repair implementation SHA:
  `651b6f42db122b87865443978711efe74bd9b0b0`
- Requirements version: `1.0.0`
- Requirements SHA-256:
  `0daf144240223243b8ba18d659c7a62b11523cd12fc9ad31b632399f23967317`
- Formal matrix SHA-256:
  `91b7d9e0ecc891183ad36092ecc73b7ed80d4026e56e119317448ed2938c7e0a`
- Required INT8 target-module coverage: `1.0` exactly.
- Focused tests: `56 passed`.
- Full regression: `421` tests; only the two unchanged upstream patch fixture
  baseline failures remained. Compileall, terminology, P0.5 coverage, and
  `git diff --check` passed.
- Impact: all formal BF16 and bitsandbytes INT8 arms now share the same
  versioned requirements identity. Logical cases, prompts, renderers, expected
  outputs, thresholds, revisions, and backends were not changed.
- Recovery entry: `attempt-0002`.
