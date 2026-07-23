# Research Control v3 Final Corrective Closeout

## Active control

The research question remains whether quantization conditionally changes
Agent tool/action selection, argument/entity selection, structured output, and
the deterministic synthetic-executor outcome. MCD/content-injection remains a
frozen historical side branch and cannot select mainline parameters.

The active protocol is always resolved through
`config/current_research_protocol.json`; do not infer it from a README heading
or an older handoff.

## Control contracts

- Canonical logical cases validate with `validate_case_row_v3`.
- `scripts/build_contextual_data.py` is the canonical case builder.
- Canonical scoring accepts one raw, single-line JSON tool call with exact
  keys and non-empty string arguments. Legacy parsing is read-only.
- Single-policy executor utility is
  `control_benign_task_success_rate`.
- Multi-policy retention is relative to `schema_only`; retention is `null`
  when that baseline rate is zero.
- Artifact manifests reject non-canonical paths, traversal, duplicates,
  invalid entries, and inconsistent declared totals.
- Evidence registry, selection, and active-protocol pointers are strict
  tracked inputs to the derived local research state.

## Startup order

1. Read `AGENTS.md`.
2. Read `config/current_research_protocol.json` and its `protocol_path`.
3. Read `config/evidence_registry.json`.
4. Read `config/current_evidence_selection.json`.
5. Run:

   ```bash
   python scripts/refresh_research_state.py
   python scripts/show_research_state.py
   ```

6. Inspect `.research-state/current_experiment.json`.

GPU execution remains controlled by the active protocol pointer and its
preregistration blockers; this closeout does not alter readiness.

This handoff is not the sole source of truth for experimental results;
manifests and frozen evidence take precedence.
