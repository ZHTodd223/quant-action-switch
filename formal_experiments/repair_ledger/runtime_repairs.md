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

## formal-calibration-finalizer-os-import-v1

- Failure attempt: `attempt-0002-precalibration-static-defect`
- Defect: the final calibration atomic-writer block called `os.fsync` and
  `os.replace` without importing `os`.
- Original repository SHA:
  `7d934056be993413c5cd50831d580a695576489f`
- Repair implementation SHA:
  `b80b176a0c69afc6b3d85303f37c97e236b11414`
- Required INT8 target-module coverage remains `1.0` exactly.
- Focused tests: `4 passed`.
- Full regression: `422` tests with only the two unchanged upstream patch
  fixture baseline failures; no new failures. Compileall, terminology, P0.5
  coverage, and `git diff --check` passed.
- Impact: calibration finalization only. Model loading, prompts, cases,
  renderers, generation settings, thresholds, attestation rules, revisions,
  and backends are unchanged.
- Recovery entry: `attempt-0003`.

## formal-attempt-parent-v1

- Failure attempt: `attempt-0003-seed-101`
- Defect: the BF16 runner passed a nested `RUN_ROOT` to comparison
  initialization without creating its attempt parent first.
- Original repository SHA:
  `796e38e32e2f64e758705717293bc462d7c0d75f`
- Repair implementation SHA:
  `94b9658fbbeb21307f86447baafe8cceedadaf55`
- Focused validation: `5 passed`; shell syntax and `git diff --check` passed.
- Full regression: not run following the user's direction to resume the formal
  experiment immediately.
- Impact: attempt-directory initialization only. Scientific configuration,
  cases, seeds, prompts, renderers, thresholds, attestation rules, revisions,
  backends, and scoring are unchanged.
- Recovery entry: `attempt-0004`.

## formal-eval-split-v1

- Failure attempt: `attempt-0004-seed-101`
- Defect: the formal registrar emitted the locked `formal_eval` split, but the
  v3 validator did not include that registered split and blocked canonical
  scoring after all 1,000 BF16 rows were generated.
- Original repository SHA:
  `9daf9b1eefec7d387c59c13ae35ad33937a12252`
- Repair implementation SHA:
  `cf0a9f525a5bfa08a2ea632cdf2652fdf00c1663`
- Focused validation: the registered rendered-case fixture passed v3 schema
  validation, and the scorer advanced beyond split validation.
- Full regression: not run following the user's GPU-priority direction.
- Impact: compatibility for the already registered split only. Case content,
  seeds, prompts, renderers, expected outputs, thresholds, attestation rules,
  revisions, backends, and scoring formulas are unchanged.
- Recovery entry: `attempt-0005`.
