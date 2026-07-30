# Repository map

`main` is the production/control-plane branch. `results/v5-formal-matrix-v1`
is an append-only metadata branch and `archive/` names preserve retired refs.

| Area | Authority | Write rule |
| --- | --- | --- |
| `AGENTS.md`, `config/current_research_protocol.json` | current legacy control plane | version, do not silently alter frozen protocol |
| `configs/experiment_registry.json` | consolidation-era experiment index | draft-only until user approval |
| `formal_experiments/` | formal v5 assets and wrappers | preserve manifests, renderers, and reports |
| `scripts/` | flat compatibility control plane | use `scripts/README.md`; do not cosmetic-move paths |
| `docs/current/` | current navigation and draft design | supersede by new document, never overwrite historical claims |
| `docs/handoffs/`, `docs/audits/` | historical rationale | append sidecars/corrections only |
| `tests/` | CPU-only contracts | no model load, GPU, training, or quantization |

The repository intentionally retains flat `scripts/` paths because frozen
manifests, shell runners, and sibling imports bind them. Consolidation is by
registry and supported entrypoint, rather than a high-risk mass rename.
