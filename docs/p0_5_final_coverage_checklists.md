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

## Manifest writers and formal entrypoints

Writers and their callers are registered separately. The coverage gate executes
all four writer contracts, executes all nine entrypoint bindings, and uses a
Python AST audit to reject unregistered direct formal-completion writers.

| Shared writer | Classification | Status |
|---|---|---|
| `model_state_attestation.write_output_manifest` | FORMAL_V4 | TESTED_PASS |
| `formal_evidence.bind_metrics_to_output_manifest` | FORMAL_V4 | TESTED_PASS |
| `formal_evidence.write_state_with_integrity` | FORMAL_V4 | TESTED_PASS |
| `formal_evidence.write_summary_with_integrity` | FORMAL_V4 | TESTED_PASS |

| Entrypoint group | Count | Registered dispatcher | Status |
|---|---:|---|---|
| BF16 / transformers quant / native quant / GGUF generators | 4 | response manifest writer | TESTED_PASS |
| comparison init / BF16 record / quant record | 3 | state integrity writer | TESTED_PASS |
| formal scorer | 1 | metrics manifest binder | TESTED_PASS |
| production comparison summary | 1 | summary integrity writer | TESTED_PASS |

| Other writer | Classification | Status |
|---|---|---|
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
| manifest | state/raw/metrics/attestation hash, identity and registry binding | TESTED_PASS |
| gates | seven original-state cases, attestation, strict diagnostic-only metrics | TESTED_PASS |

Evidence: `tests/fixtures/canonical_scorer/summary_contamination_cases.json`,
`scripts/canonical_summary_validation.py`, and
`tests/test_summary_contamination_matrix.py`.

All 42 cases construct real temporary state, raw, metrics, output-manifest,
attestation, registry and identity evidence before calling production
`summarize()`. The former `classify_candidate()` helper is not an acceptance
target.
