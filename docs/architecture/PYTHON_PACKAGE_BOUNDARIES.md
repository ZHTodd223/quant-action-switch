# Python package boundaries

The package owns schema, protocol, parsing, scoring, rendering, and artifact
validation modules migrated in V1. Imports use `quant_action_switch.*`.
`scripts/_compat.py` is the only approved `sys.path` exception and exists only
for legacy import paths before editable installation. New package or CLI code
must not import from `scripts` or `scripts/legacy`.

The initial CLI modules are intentionally draft-safe: they provide discoverable
help and do not load models, train, infer, or quantize.
