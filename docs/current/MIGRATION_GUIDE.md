# Consolidation migration guide

1. Start with `AGENTS.md`, then the legacy current-protocol pointer, evidence
   registry, and selected evidence. Do not infer authority from `final`, gate,
   backend, or hardware names.
2. Consult `configs/experiment_registry.json` for the proposed V2 tracks. All
   entries are drafts and have no execution authority.
3. Use the numbered `formal_experiments/scripts/` sequence only for a future
   separately approved formal run. Existing model/backend-specific runners are
   historical or pilot support unless explicitly registered.
4. Keep flat script paths and frozen artifacts in place. A later migration may
   use `git mv` only after its references and hashes are audited.
5. Before any future execution, obtain user approval, create a new locked
   configuration, and verify manifest/lineage/attestation contracts.
