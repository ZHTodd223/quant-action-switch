# SUPERSEDED / HISTORICAL — Research Control v2 Corrective Closeout

This document describes an earlier control revision. It is not the active
startup entry. Follow `config/current_research_protocol.json` and
`docs/handoffs/2026-07-23-research-control-v3-independent-audit-closeout.md`.

- Baseline commit: `3c476c37215ac207685700b4b6c867056ecaa52c`.
- Corrective commit: the commit containing this document; resolve it from Git
  history to avoid embedding a self-referential, unverifiable hash.
- Historical protocol at the time of this document:
  `agent_toolcall_protocol_v2`.
- GPU execution ready: `false`.

## Corrections

This closeout adds a portable evidence registry, strict boolean case handling,
strict canonical scoring, a deterministic in-memory executor, scoped
terminology validation, and archived-document banners. The symbolic evaluator
remains a separate compatibility layer. Frozen evidence was not rewritten.

## Portable evidence and state

`config/evidence_registry.json` stores metadata pointers only. Original
manifests and frozen evidence are authoritative. Remote-only entries are never
reported as locally verified. Refresh and inspect derived state with:

```bash
python scripts/refresh_research_state.py
python scripts/show_research_state.py
```

Choose a record explicitly with
`python scripts/refresh_research_state.py --current-record-id RECORD_ID`.

## Remaining preregistration blockers

1. finite benign-utility recovery ladder;
2. GPTQ parameters, calibration IDs/hash, backend version, and seed mapping;
3. shared logical cases and renderer hashes;
4. bootstrap/randomization counts, confidence level, and multiplicity policy;
5. frozen BFCL-compatible slice and hash;
6. numeric disk and GPU-memory thresholds.

## Next research-control task read order

1. `AGENTS.md`
2. `config/current_research_protocol.json`
3. the versioned protocol referenced by the pointer
4. `config/evidence_registry.json`
5. `.research-state/current_experiment.json` after refresh
6. this handoff
7. original manifests and frozen evidence selected for the task

The MCD/content-injection pilot remains a frozen historical side branch and is <!-- terminology-legacy-read -->
not a recovery route for the mainline. The archived README sections are also
not an active protocol.

This handoff is not the sole source of truth for experimental results;
manifests and frozen evidence take precedence.
