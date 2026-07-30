# Repository structure V1

`src/quant_action_switch/` is the authority for migrated reusable logic.
`scripts/` keeps short legacy wrappers and operational tools; moved historical
pilots live under `scripts/legacy/`. Frozen assets remain at their original
paths under `formal_experiments/`, `protocols/`, `config/`, and historical docs.

New formal configuration belongs in `configs/`; `config/` is a frozen legacy
control-plane namespace and must not receive new formal configuration.
