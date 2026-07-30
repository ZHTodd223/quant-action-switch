# Script map

## Consolidation status

This directory stays flat intentionally: frozen manifests, shell wrappers, and
sibling imports depend on these paths. The authoritative navigation is this
file plus `configs/experiment_registry.json`; no filename such as `final`,
`gate_v7`, `5090`, or `strict` confers formal status.

| Status | Supported use |
| --- | --- |
| ACTIVE_FORMAL_ENTRYPOINT | Numbered `formal_experiments/scripts/` wrappers, only after a separately approved locked design |
| ACTIVE_SUPPORT | schema, parser, scorer, manifest, attestation, deterministic executor, and research-state modules |
| PILOT | model/backend-specific preparation and runners |
| LEGACY_ARCHIVED | historical aggregators, gates, and compatibility writers; retain paths and use only to read/recompute frozen evidence |
| BROKEN_QUARANTINED | none declared by this audit |

This consolidation permits only audit, config validation, and CPU-only tests.
GPU-capable runners require future explicit approval and must reject absent
confirmation variables.

`scripts/` intentionally keeps stable flat paths.  Frozen protocols, evidence
registries, shell runners, and Python sibling imports refer to these paths.
Moving files into cosmetic subdirectories would break reproducibility and
historical replay.  This index is the supported functional organization.

## Start here

| Need | Use |
| --- | --- |
| Inspect active research state | `refresh_research_state.py`, `show_research_state.py` |
| Validate terminology and artifact integrity | `check_terminology.py`, `make_manifest.py`, `verify_manifest.py` |
| Prepare a formal v5 execution | `../formal_experiments/scripts/00_formal_matrix_preflight.sh` and the numbered sequence there |
| Run a registered cross-model comparison | `run_cross_model_comparison.py` |
| Locate permitted quantization entrypoints | `../config/quantization_entrypoints_v1.json` |

## Functional groups

| Group | File-name family | Purpose |
| --- | --- | --- |
| Shared control plane | `case_schema.py`, `canonical_tool_schema.py`, `logical_case_rendering.py`, `native_tool_protocol.py`, `response_parsing.py`, `metric_schema.py`, `scorer_*.py`, `generation_termination.py`, `transformers_model_loader.py` | Schemas, renderers, parsers, metrics, and loader behavior shared by runners. |
| Dataset and evidence preparation | `build_*`, `prepare_*`, `convert_*`, `lock_*`, `make_smoke_config.py`, `training_seed_repro.py` | Build cases, calibration inputs, locked model metadata, and derived evidence packs. |
| Execution | `generate_*`, `run_*`, `continue_*`, `resume_*`, `retry_*`, `extend_*` | Generate responses and run named, versioned experiment pipelines. |
| Validation and gating | `preflight*`, `evaluate_*`, `verify_*`, `validate_*`, `check_*`, `require_*`, `rescore_*`, `audit_*`, `probe_*`, `diagnose_*`, `model_state_attestation.py` | Enforce preconditions, reconstruct outputs, validate eligibility, and audit evidence. |
| Analysis and reporting | `aggregate_*`, `analyze_*`, `compare_*`, `score_*`, `summarize_*`, `summary_contamination.py`, `canonical_summary_validation.py` | Derive metrics and reports from preserved raw outputs. |
| Operations and preservation | `bootstrap_*`, `download_*`, `fetch_*`, `find_*`, `backup_*`, `restore_*`, `sync_*`, `apply_upstream_patches.sh`, `update_*`, `run_async_upload_queue.sh` | Environment setup, transport, backup, restore, and upstream patch management. |
| Formal v5 internals | `formal_*.py` | Implementation behind `formal_experiments/scripts/`; normally call the numbered formal wrappers instead. |
| Compatibility guards | `quantization_entrypoint_guard.sh`, `write_legacy_comparison_state.py` | Preserve historical interfaces while directing new work to the active control plane. |

## Selection rules

- Names containing a model family or backend (for example `qwen25_3b`,
  `gemma3_4b`, `gptq`, `nf4`, or `gguf`) are experiment-specific runners or
  analyses, not generic defaults.
- A `run_*` script is not proof of an authorized GPU run.  Follow the active
  protocol, preflight, and entrypoint guard first.
- Treat `aggregate_*`, `analyze_*`, and `summarize_*` as derived-analysis
  tools.  They never replace raw response files or manifests.
- New generic code should use the shared control-plane modules; new experiment
  wrappers should be named by protocol, model family, backend, and purpose.

For directory placement and attempt lifecycle, see
`../docs/repository_layout.md`.
