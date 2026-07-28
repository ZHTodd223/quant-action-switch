# P0-5 final coverage checklists

Status values: `NOT_IMPLEMENTED`, `IMPLEMENTED_UNTESTED`, `TESTED_PASS`,
`TESTED_FAIL`, `NOT_APPLICABLE_WITH_REASON`.

## Fixture coverage

| Group | Required cases | Status |
|---|---|---|
| valid | three registry tools, ordering, unicode, Chinese, Windows paths, spaces, escapes, expressions, long input, JSON whitespace | TESTED_PASS |
| tool name | unsupported, empty, missing, null, scalar/non-string, casing, whitespace, alias, unicode/invisible/newline | TESTED_PASS |
| arguments | keys, required/additional, scalar/container arguments, scalar/container values, duplicate key | TESTED_PASS |
| format | fences, wrappers, multiple/trailing/prefix, truncation, whitespace, primitives, recovered first object | TESTED_PASS |
| identity | policy, state/metrics/manifest identity drift and unknown evidence | TESTED_PASS |
| generic JSON types | string, integer, number, boolean, object, array, null positive/negative and non-finite numbers | TESTED_PASS |

Evidence: `tests/test_canonical_scorer_fixtures.py`,
`tests/fixtures/canonical_scorer/*.json`, and
`scripts/check_p0_5_coverage.py`.

## Manifest writer registry

| Writer | Classification | Status |
|---|---|---|
| `model_state_attestation.write_output_manifest` | FORMAL_V4 | TESTED_PASS |
| `generate_bf16_responses` caller | FORMAL_V4 | TESTED_PASS |
| `generate_quantized_responses` caller | FORMAL_V4 | TESTED_PASS |
| `generate_native_quantized_responses` caller | FORMAL_V4 | TESTED_PASS |
| `generate_gguf_responses` caller | FORMAL_V4 | TESTED_PASS |
| `run_cross_model_comparison` state writer | FORMAL_V4 | TESTED_PASS |
| `summarize_cross_model_comparison` summary writer | FORMAL_V4 | TESTED_PASS |
| `rescore_canonical_diagnostic` | RETROSPECTIVE_DIAGNOSTIC | TESTED_PASS |
| `make_manifest` / `verify_manifest` | UNRELATED | NOT_APPLICABLE_WITH_REASON: generic artifact inventory and verifier; code evidence in `scripts/manifest_writer_registry.py` |
| deterministic executor metrics | DEVELOPMENT_ONLY | NOT_APPLICABLE_WITH_REASON: executor metrics do not mark a v4 comparison arm or summary complete; code evidence in `scripts/evaluate_deterministic_executor.py` |
| data/audit/calibration manifests | UNRELATED | NOT_APPLICABLE_WITH_REASON: input provenance or audit artifacts; per-pattern code evidence is in `manifest_writer_registry.EXCLUSIONS` |

Evidence: `scripts/manifest_writer_registry.py` and
`tests/test_manifest_writer_registry.py`.

## Summary contamination matrix

| Group | Required cases | Status |
|---|---|---|
| positive | canonical state, metrics, manifests, registry, P0-1/P0-3 valid | TESTED_PASS |
| evidence class | legacy, retrospective, unknown, development | TESTED_PASS |
| identity | state/arm/metrics and every identity-field drift | TESTED_PASS |
| manifest | missing/tampered identity and registry/verifier failure | TESTED_PASS |
| gates | non-COMPARABLE, attestation failure, strict diagnostic-only row | TESTED_PASS |

Evidence: `tests/fixtures/canonical_scorer/summary_contamination_cases.json`,
`scripts/summary_contamination.py`, and
`tests/test_summary_contamination_matrix.py`.
