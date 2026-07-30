# Code architecture review

## Confirmed structure

The repository is a flat-script research control plane, not a conventional
`src/` package. Its durable core is the case schema, native tool protocol,
response parsing/scoring, deterministic executor, manifest/attestation
validation, and formal v5 wrappers. Tests cover much of that core with CPU
fixtures. `formal_experiments/` contains the only clearly sequenced formal
entrypoint family.

The 186 scripts contain many model/backend-specific runners and aggregators.
They are real historical/pilot interfaces, but none should be selected merely
because it contains `final`, `strict`, `gate`, or a hardware name. The flat
locations are deliberately retained because frozen manifests, sibling imports,
and shell references bind them.

## Risks and actions

| Finding | Evidence | Disposition |
| --- | --- | --- |
| Multiple Gate generations coexist | `config/agent_toolcall_protocol_v1` through `v5`, Gate-v3/v7/v8 scripts | Keep as versioned history; V2 draft separates technical validity from behavioral evaluation. |
| Multiple scoring/parsing paths | `score_responses.py`, diagnostic rescoring, legacy writers | Canonical scorer is authoritative for new work; legacy readers are frozen-evidence compatibility only. |
| GPU-capable scripts are numerous | `run_*`, `generate_*`, `prepare_*` families | Do not call from consolidation; supported formal navigation is documented, and V2 is non-authorized. |
| Absolute runtime paths occur in formal metadata | `snapshot_path` fields in v5 matrix | Treat as provenance, not portable configuration; future lock must resolve a new environment binding. |
| Historical reports can look current | `THREE_SEED_FINDINGS`, server runbook, handoffs | Current map/registry labels them historical; claims stay evidence-bounded. |
| No package-level import boundary | flat `scripts/` imports | Avoid mass package refactor; add registry/docs first. |

No evidence was found that CI invokes GPU work. Existing workflow validation
is CPU-oriented. The test suite still cannot establish real model loading,
native quantization, training, or scientific effect.
