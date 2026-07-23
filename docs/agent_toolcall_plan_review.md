# Agent Tool-Call Cross-Family Plan Review

## Verdict

The proposed plan is scientifically coherent and is suitable for CPU
implementation preparation. It correctly returns the project to
quantization-conditioned tool/action switching and treats Qwen-to-Llama as an
external-validity axis.

GPU execution should begin only after the amendments below are represented in
the preregistration and locked configuration.

The current version is therefore a **CPU control scaffold, not a completed
implementation or GPU-execution-ready preregistration**. The pointer and
protocol status is
`cpu_control_scaffold_ready_gpu_preregistration_incomplete`, with
`gpu_execution_ready=false`. The remaining numeric choices and hashes must be
locked before that status changes.

## Strong parts

1. Exact logical-case pairing across model families.
2. Frozen train/development/final split and one-time final evaluation.
3. Matched repaired and no-intervention branches.
4. GPTQ-4 as confirmatory backend and NF4 as auxiliary evidence.
5. Three fixed seeds independent of the observed seed-101 target effect.
6. Explicit model lineage with an evaluated pre-refinement intervention stage.
7. Difference-in-differences as the primary quantization-conditioned contrast.
8. Deterministic local executor with no external side effects.
9. Resource-safe stage boundaries, manifests, staging, and remote verification.
10. Explicit closure of the MCD scope-drift branch.

## Required amendments

### 1. Layer 20 is a heuristic, not a discovered optimum

The normalized-depth mapping is reproducible and acceptable for
preregistration, but it does not establish that layer 20 is mechanistically
optimal. The paper must describe it as an a priori transfer heuristic.
Layer-19/21 experiments remain post-primary sensitivity analyses.

### 2. Predefine a finite benign-utility recovery ladder

A strict stop after one reconstruction setting can turn an implementation
failure into a misleading family-level null. Before GPU execution, lock a small
finite recovery ladder selected only by benign utility. Record every attempted
candidate. Do not use switch-target metrics in that choice.

If every candidate fails, report reconstruction incompatibility and stop the
downstream intervention chain.

### 3. Do not use ordinary McNemar as the DiD test

McNemar is appropriate for one paired binary contrast. The four-cell
contrast-of-contrasts should use a case-paired bootstrap, hierarchical
seed/case bootstrap, randomization test, or a preregistered interaction model.
Individual two-cell arm comparisons may still use exact McNemar tests.

### 4. Add public benchmark evidence

The four synthetic task families are strong controlled evidence but insufficient
for a broad Agent claim. Add a frozen BFCL-compatible subset, covering at least
simple calls and relevance/irrelevance; add multiple/parallel calls if the local
runtime supports them. Keep it supplemental and do not tune on it.

### 5. Add executor-level defense replication

The existing symbolic allowlist/capability result should be reproduced through
the deterministic executor for both model families. Report:

- target-directed attempts;
- policy-blocked attempts;
- actually executed actions;
- end-state correctness;
- benign/control utility retention.

### 6. Reuse the Qwen logical cases

Cross-model pairing requires Llama to render the already frozen Qwen logical
cases. Creating a new Llama dataset with merely similar distributions does not
support a case-paired cross-model interaction claim.

### 7. Separate scientific scope from hardware utilization

Batch size, safe sequence length, data loading, and inference batching may be
optimized for the GPU. Trainable layers, target layers, epochs, and intervention
scope may not be enlarged solely to consume VRAM.

## Resource-utilization plan

Paid GPU utilization should be increased without changing the estimand:

1. Finish dataset generation, renderer validation, scorer tests, calibration
   case selection, manifests, and queue scripts on CPU before opening the GPU.
2. Keep one model resident at a time and execute preregistered GPU stages
   back-to-back with resumable stage markers.
3. Find the largest safe **inference batch** and **training micro-batch** using
   a short memory-only preflight. Keep the locked effective batch constant by
   adjusting gradient accumulation.
4. Use length bucketing, pinned-memory workers, persistent workers, and
   asynchronous host-to-device copies where supported.
5. Pipeline CPU scoring, aggregation, hashing, and low-priority uploads behind
   the GPU queue, but never run two model-resident GPU stages concurrently.
6. Disable duplicate trainer checkpoints unless a checkpoint is required for
   recovery. Stage before pruning, and preserve a fixed disk safety margin.
7. Run all preregistered seeds after technical and benign-utility validity is
   established; do not decide seed expansion from the observed switch effect.

This consumes available compute through batching and scheduling, rather than
by widening trainable layers or adding post-hoc conditions.

## Approved execution order

1. CPU repository reuse audit and MCD scope-closure record.
2. Canonical shared logical-case schema and renderer adapters.
3. Strict scorer, synthetic executor, and schema compatibility tests.
4. Preregistration, benign-utility recovery ladder, GPTQ configuration, and
   statistics lock.
5. Seed-101 technical/utility development using no target metric for selection.
6. Seeds 101/202/303 confirmatory DAG once the technical chain is valid.
7. Case-paired DiD and model interaction.
8. Lock all checkpoints.
9. Run `final_locked` once.
10. Public benchmark slice and executor-level defense as frozen supplemental
    analyses.

## Terminology decision

New generated artifacts use `switch_eligible` and `changed_variant_pairs`.
Historical evidence remains immutable and is read through an explicit legacy
compatibility layer. The local patch directory is
`patches/upstream_aio_quantization`; the pinned upstream repository name remains
unchanged for provenance.
