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

The tracked scientific selection lives in
`config/current_evidence_selection.json`. The refresher reads this pointer
before considering any prior local state; it never selects the newest record
as a scientific default. Override the pointer for a one-off local inspection:

```bash
python scripts/refresh_research_state.py \
  --evidence-root /absolute/path/to/evidence \
  --current-root /absolute/path/to/evidence/current-run
```

An explicit CLI selection is preserved on later refreshes while that record is
still discoverable. Without a CLI selection, the tracked pointer is used. If
neither exists, `current` remains null rather than silently selecting by time.

## Portable evidence registry

`config/evidence_registry.json` is a tracked metadata registry. It contains
manifest identifiers and confirmed remote paths, not metric bodies. Unknown
remote paths remain unset; never synthesize them. Every record must contain at
least one non-empty confirmed remote path. Its field contract is
documented in `config/evidence_registry.schema.json`. Select a registry entry
with:

```bash
python scripts/refresh_research_state.py \
  --current-record-id RECORD_ID
```

A registry-only record is labeled `registry_remote_only`,
`local_available=false`,
`manifest_file_digest_matches_registry=null`, and
`manifest_contents_verified=false`. After restoration, the refresher merges it
with local evidence only when the manifest-file SHA-256 matches. It then calls
`scripts/verify_manifest.py` to rehash every listed file; a matching manifest
digest does not imply verified contents. A matching record ID with a
conflicting manifest stops refresh.
Original manifests and frozen evidence are authoritative over both the registry
and `.research-state/`.

Tracked registry, selection, protocol-pointer, and referenced-protocol JSON is
parsed fail-closed: duplicate object keys, non-standard numeric constants, and
pointer/protocol readiness mismatches stop refresh. This prevents a permissive
JSON parser or stale pointer from silently changing the selected evidence or
GPU-readiness state.

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
4. `config/evidence_registry.json`;
5. `config/current_evidence_selection.json`;
6. `.research-state/current_experiment.json`;
7. only the evidence files referenced by the selected state record.

This keeps the stable prompt cache reusable while making current evidence
discoverable and reproducible.
