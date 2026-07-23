# Research Control v2 Bootstrap

## 1. Research question

The mainline asks whether quantization conditionally changes Agent tool-call
action selection, argument/entity selection, structured action output, and
deterministic synthetic-executor outcomes. Cross-family, backend, and seed
comparisons are external-validity axes for this same question.

## 2. Frozen MCD side branch

The MCD/content-injection work is frozen as an original-paper, single-seed,
post-hoc pilot. It does not support the primary Agent/tool-call claim and must
not drive Agent metrics, hyperparameters, model selection, or further GPU
expansion.

## 3. Git branch

The research-control work is maintained on `agent/domestic-cache`.

## 4. Active protocol

Read `config/current_research_protocol.json`, then the versioned protocol it
references: `config/agent_toolcall_protocol_v1.json`.

Current status:

```text
status=cpu_implementation_ready_gpu_preregistration_incomplete
gpu_execution_ready=false
```

## 5. Preregistration blockers

Before paid GPU execution, lock:

1. the finite benign-utility recovery ladder;
2. GPTQ bits, group size, `desc_act`, calibration cases and hash, backend
   version, and quantization-seed mapping;
3. shared logical cases and per-model renderer-output hashes;
4. resample/randomization counts, confidence level, and multiplicity policy;
5. the frozen BFCL-compatible supplemental subset;
6. numeric disk-space and GPU-memory stage thresholds.

## 6. Canonical terminology

New mainline writers use `intervention`, `switch`, and `no_intervention`.
Canonical case and metric names are defined by
`config/terminology_policy.json`. Exact upstream paths and APIs retain their
original names.

## 7. Historical compatibility and frozen evidence

Readers may accept documented legacy fields such as `attack_eligible`,
`expected_target`, `target_asr`, and `no_injection`, but must reject conflicting
new and legacy values. Frozen JSON/JSONL, manifests, remote markers, published
hashes, historical run IDs, and original-paper labels are immutable.

## 8. Refreshing dynamic research state

`.research-state/` is a local ignored index, not evidence. Refresh and inspect
it with:

```bash
python scripts/refresh_research_state.py
python scripts/show_research_state.py
```

Use repeatable `--evidence-root` arguments for restored evidence outside the
repository and `--current-root` when an explicit current record is required.

## 9. Bootstrap validation

This bootstrap is accepted only with the full Python unit suite, recursive
shell syntax checks, Python compilation, terminology validation, JSON parsing,
and `git diff --check` passing. No GPU experiment is part of this validation.

## 10. Starting the new research-control conversation

1. Read `AGENTS.md`.
2. Read `config/current_research_protocol.json`.
3. Read the referenced versioned protocol.
4. Read `config/terminology_policy.json`.
5. Read `docs/agent_toolcall_plan_review.md`.
6. Read this handoff.
7. Refresh and display local research state.
8. Inspect Git status and recent commits.
9. Keep GPU work blocked until every protocol blocker is locked.

Startup commands:

```bash
python scripts/refresh_research_state.py
python scripts/show_research_state.py
git status --short
git log -10 --oneline
```

## 11. Source-of-truth boundary

This handoff document is not the sole source of truth for experimental
results. Manifests and frozen evidence take precedence.
