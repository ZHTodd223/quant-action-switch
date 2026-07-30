# Artifact lineage

Historical raw outputs, metrics, manifests, reports, and remote references are
frozen evidence. Their locations are indexed by `config/evidence_registry.json`
and must not be rewritten during cleanup. Derived corrections belong in a new
audit sidecar with source path, source hash where available, method, and date.

The large tracked rendered-case files under `formal_experiments/artifacts/` are
formal preparation assets, not model weights or caches. Model snapshots there
are manifests only. `.gitignore` must continue to exclude model, run, cache,
and credential material.

Historical findings in `docs/THREE_SEED_FINDINGS.md` remain `HISTORICAL` and
do not become a V2 formal result. The results branch holds failed attempt
metadata; it is evidence of attempted execution, not evidence of scientific
success.
