# P0-5 final coverage checklists

Status values: `NOT_IMPLEMENTED`, `IMPLEMENTED_UNTESTED`, `TESTED_PASS`,
`TESTED_FAIL`, `NOT_APPLICABLE_WITH_REASON`.

## Fixture coverage

| Group | Required cases | Status |
|---|---|---|
| valid | three registry tools, ordering, unicode, Chinese, Windows paths, spaces, escapes, expressions, long input, JSON whitespace | NOT_IMPLEMENTED |
| tool name | unsupported, empty, missing, null, scalar/non-string, casing, whitespace, alias, unicode/invisible/newline | NOT_IMPLEMENTED |
| arguments | keys, required/additional, scalar/container arguments, scalar/container values, duplicate key | NOT_IMPLEMENTED |
| format | fences, wrappers, multiple/trailing/prefix, truncation, whitespace, primitives, recovered first object | NOT_IMPLEMENTED |
| identity | policy, state/metrics/manifest identity drift and unknown evidence | NOT_IMPLEMENTED |

## Manifest writer registry

| Writer | Classification | Status |
|---|---|---|
| `model_state_attestation.write_output_manifest` | FORMAL_V4 | IMPLEMENTED_UNTESTED |
| `generate_bf16_responses` caller | FORMAL_V4_CALLER | IMPLEMENTED_UNTESTED |
| `generate_quantized_responses` caller | FORMAL_V4_CALLER | IMPLEMENTED_UNTESTED |
| `generate_native_quantized_responses` caller | FORMAL_V4_CALLER | IMPLEMENTED_UNTESTED |
| `generate_gguf_responses` | DEVELOPMENT_ONLY | NOT_APPLICABLE_WITH_REASON: diagnostic path is not eligible |
| `make_manifest` / `verify_manifest` | UNRELATED | NOT_APPLICABLE_WITH_REASON: generic artifact manifests are not v4 evidence |
| `rescore_canonical_diagnostic` | RETROSPECTIVE_DIAGNOSTIC | IMPLEMENTED_UNTESTED |

## Summary contamination matrix

| Group | Required cases | Status |
|---|---|---|
| positive | canonical state, metrics, manifests, registry, P0-1/P0-3 valid | IMPLEMENTED_UNTESTED |
| evidence class | legacy, retrospective, unknown, development | NOT_IMPLEMENTED |
| identity | state/arm/metrics and every identity-field drift | NOT_IMPLEMENTED |
| manifest | missing/tampered identity and registry/verifier failure | NOT_IMPLEMENTED |
| gates | non-COMPARABLE, attestation failure, strict diagnostic-only row | IMPLEMENTED_UNTESTED |
