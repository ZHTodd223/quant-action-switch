# Stable Agent Contract

This file is deliberately small and stable so it remains cache-friendly.
Do not put run IDs, metric values, model paths, dates, hashes, selected
hyperparameters, current failures, or experiment results here.

## 1. Invariant research objective

The repository studies whether quantization conditionally changes a
tool-using model's:

1. action or tool selection;
2. argument, entity, or value selection;
3. structured action output;
4. deterministic synthetic-executor outcome.

Cross-family experiments are an external-validity axis for this same question.
They are not permission to replace the Agent task with an unrelated behavior
task.

Raw structured output is not a complete Agent execution result. An execution
claim requires a defined runtime, observable state transitions, and an
end-state scorer. The repository's synthetic executor has no external side
effects.

## 2. Source-of-truth order

Read project state in this order:

1. `AGENTS.md` — stable invariants and operating rules.
2. `config/current_research_protocol.json` — pointer to the active protocol.
3. The versioned protocol referenced by that pointer — mutable design is
   changed by creating a new version, never by editing a locked version.
4. `config/evidence_registry.json` — portable evidence metadata pointers.
5. `config/current_evidence_selection.json` — tracked explicit selection.
6. `.research-state/current_experiment.json`,
   `.research-state/experiment_index.json`, and
   `.research-state/latest_summary.md` — generated local state, never committed.
7. Frozen evidence, manifests, preregistrations, and remote-verification
   markers — authoritative observations.

Conversation history is not a source of truth when it conflicts with this
order.

## 3. Start procedure

Before planning or editing:

1. inspect `git status` and the current commit;
2. inspect active processes/PID files and available disk space when relevant;
3. refresh the read-only evidence index:

   ```bash
   python scripts/refresh_research_state.py
   python scripts/show_research_state.py
   ```

4. read the current protocol and only the evidence it references;
5. separate CPU-only preparation from GPU work.

For evidence outside the repository, pass one or more `--evidence-root`
arguments. Use `--current-root` to pin the active run. See
`docs/research_state_maintenance.md`.

## 4. Stable maintenance rules

- Never store results or current run details in this file.
- Change experimental design in a new versioned protocol file.
- Change the active protocol only through
  `config/current_research_protocol.json`.
- Refresh `.research-state/` after restoring, producing, moving, or verifying
  evidence.
- Keep `.research-state/` local and reproducible; it is an index, not evidence.
- Preserve shared logical cases across model families. Only the chat renderer
  may be model-specific.
- Selection must use only the criteria locked in the active protocol.
- Final locked data is not used for configuration selection.
- Do not widen mechanism scope merely to consume more GPU memory.

## 5. Terminology and compatibility

New code and newly generated artifacts use the canonical scientific vocabulary
in `config/terminology_policy.json`.

Historical readers may accept legacy aliases, but new writers emit canonical
names. Never rewrite frozen JSON/JSONL, manifests, published hashes, remote
markers, original-paper quotations, historical run IDs, or exact upstream
paths merely to rename vocabulary.

Use `docs/variable_name_migration.md` to translate between current and
historical variable names. Run:

```bash
python scripts/check_terminology.py
```

before committing new mainline Agent code.

## 6. Evidence and resource discipline

- Inspect before editing; preserve unrelated user changes.
- Never delete the only copy of a model or unverified evidence.
- Generate and verify a manifest before staging or pruning a large stage.
- Prune only recomputable data or remotely verified copies.
- Stop at a stage boundary when disk space is below the protocol threshold.
- Run at most one GPU model task at a time.
- GPU execution requires an explicit current instruction and a passing
  preflight.
- CPU hashing, aggregation, and upload may overlap only when they do not
  contend with the active GPU task.

## 7. Delivery and verification

- Reuse existing generators, scorers, manifest utilities, upload tools, and
  compatibility adapters.
- Keep frozen evidence immutable and write derived analyses separately.
- Record lineage, configuration, environment, completion, and manifest data
  required by the active protocol.
- Run focused tests first, then the full Python test suite, shell syntax checks,
  terminology checks, and `git diff --check`.
- Report changed files, commands, verification results, and scientific
  limitations without copying transient results into this file.
