# Research State Maintenance

`AGENTS.md` contains only stable rules. Current runs, hashes, results, selected
paths, and failure details live in a generated local index under
`.research-state/`.

## Files

- `.research-state/experiment_index.json` — all discovered evidence records.
- `.research-state/current_experiment.json` — the selected current record.
- `.research-state/latest_summary.md` — compact human-readable view.

These files are ignored by Git. They are derived indexes, not evidence.

## Refresh

Default repository evidence:

```bash
python scripts/refresh_research_state.py
python scripts/show_research_state.py
```

Include restored or external evidence:

```bash
python scripts/refresh_research_state.py \
  --evidence-root runs \
  --evidence-root data/generated \
  --evidence-root /absolute/path/to/restored-evidence
```

Select the current run explicitly:

```bash
python scripts/refresh_research_state.py \
  --evidence-root /absolute/path/to/evidence \
  --current-root /absolute/path/to/evidence/current-run
```

An explicit selection is preserved on later refreshes while that directory is
still discoverable. Otherwise the newest record is selected automatically.

Override the generated-state location:

```bash
QAS_STATE_ROOT=/persistent/local/state \
python scripts/refresh_research_state.py
```

## Discovery contract

The refresher scans for small anchor files:

- `completion.json`
- `gate_decision.json`
- `remote_verified.json`
- `manifest.sha256.json`
- `preregistration.json`
- `experiment.json`
- recognized summary JSON files

It hashes only these metadata anchors. It does not hash model weights, execute a
model, use a GPU, access the network, or modify evidence.

## When to refresh

Refresh after:

1. restoring evidence from a remote store;
2. finishing or verifying a stage;
3. moving an evidence directory;
4. generating a new manifest or completion marker;
5. switching the run that the next agent should inspect.

Do not copy current metrics into `AGENTS.md`. If the protocol changes, create a
new versioned file under `config/` and update
`config/current_research_protocol.json`.

## Startup read order

Every new agent reads:

1. `AGENTS.md`;
2. `config/current_research_protocol.json`;
3. the referenced versioned protocol;
4. `.research-state/current_experiment.json`;
5. only the evidence files referenced by the selected state record.

This keeps the stable prompt cache reusable while making current evidence
discoverable and reproducible.
