# Research Control v3 Independent Audit Closeout

## Scope and verdict

This closeout records an independent, zero-conversation-memory audit of the
tracked Research Control surfaces. It changes validation and evidence-control
behavior only. It does not contain experiment results, change the scientific
estimand, modify frozen evidence, or make GPU execution ready.

The mainline question remains whether quantization conditionally changes an
Agent model's action selection, argument/entity selection, structured action
output, and deterministic synthetic-executor outcome. Cross-family work is an
external-validity axis for that question. The frozen MCD pilot remains outside
mainline parameter selection.

## Fail-closed corrections

The audit closed the following silent-acceptance paths:

1. duplicate JSON object keys and non-finite JSON numbers;
2. empty v3 datasets, duplicate `case_id` values, and missing responses;
3. switch expectations that do not exist in the declared executor fixture;
4. paired renderer rows that are misaligned or contradict eligibility;
5. all-policy executor output that previously omitted non-baseline case-level
   outcomes;
6. manifest path aliases and hardlinks, malformed totals, and ambiguous
   manifest JSON;
7. stale or contradictory protocol-pointer readiness flags;
8. evidence registry records without a confirmed restore path;
9. CI whitespace checks that inspected an empty working-tree diff.

Canonical scorers and executors now require a non-empty unique case
collection. Canonical tool responses are one raw single-line JSON object with
no surrounding whitespace. Terminal identifiers are byte-exact. Legacy readers
remain available only for frozen historical evidence.

## Source-of-truth and startup

Read in this order:

1. `AGENTS.md`;
2. `config/current_research_protocol.json`;
3. the versioned protocol referenced by its `protocol_path`;
4. `config/agent_toolcall_case_schema_v3.json`;
5. `config/deterministic_executor_outcome_v2.json`;
6. `config/evidence_registry.json`;
7. `config/current_evidence_selection.json`;
8. this handoff;
9. generated `.research-state/` files after refresh;
10. only the selected manifests and frozen evidence needed for the task.

Run:

```bash
python scripts/refresh_research_state.py
python scripts/show_research_state.py
git status --short
git log -10 --oneline
```

`AGENTS.md` deliberately remains stable and result-free. Dynamic observations
belong in frozen evidence or the ignored reproducible `.research-state/`
index.

## Next controlled task

Do not start GPU work from this closeout. Resolve every preregistration item
listed by the active protocol using CPU-only preparation. Once all exact
choices, hashes, statistics, supplemental data, and numeric resource
thresholds are fixed, create a new versioned protocol and update the pointer
atomically. Do not edit a completed protocol in place.

No final-locked outcome may select configuration. Model families must share
the same logical cases and scorer; only their chat renderers may differ.
Hardware utilization may optimize batching and scheduling, but may not widen
the scientific mechanism.

## Evidence boundary

This handoff is not evidence and does not supersede manifests. Original
manifests and frozen artifacts remain authoritative. The local research-state
index is derived and reproducible.
