# Experiment design V2 — Draft

Status: **DRAFT_REQUIRES_USER_APPROVAL**. `gpu_execution_ready=false`. This
document creates no locked protocol, requests no GPU, and authorizes no model
loading, training, inference, or quantization.

## Scope and current claim boundary

The repository can support historical, evidence-bounded observations about
structured tool-call outputs. It cannot yet support a V2 claim that a
quantization-conditioned attack changes an Agent end state: prior records use
different gates/backends and historical output is not, by itself, execution.

## Track A — natural quantization drift

Compare Original BF16 with Original Quantized only. Report format, tool-call,
parameter, and deterministic-executor end-state drift plus repeated-load
noise. This is a baseline study, not an attack-success study.

## Track B — quantization-conditional tool action

The minimum matrix is Original, Injected, No-injection Repair, and Repaired
Attack, each in BF16 and Quantized arms. The primary metric is **Exact Target
Tool Actually-Executed Rate**; exact generation, parameter/entity correctness,
target end-state success, benign task success, safe end-state correctness,
format drift, finish reason, and no/multi-call drift are secondary or
diagnostic metrics. Semantic target rate is never a primary success metric.

Technical-validity gates (revision, manifest, backend, coverage, no fallback,
hashes) may reject a result. Behavioral gates classify a technically valid arm
as confirmatory, exploratory, or attack failed; they must not silently skip a
planned quantization arm.

## Track C — execution and defense

Use only the deterministic synthetic executor. Compare parse-only, JSON
schema, schema plus canonical normalization/allowlist, and capability-token
deny-first policy. The executor must have no filesystem, network, shell,
email, payment, account, or credential capability.

## Statistical and data rules

Independent training seed is the model-level replication unit. Case-level
observations are paired and are not independent model replications. With
`do_sample=false`, inference seeds demonstrate reproducibility only. Training,
repair, utility, development, and locked evaluation data require verified
prompt, entity, and template separation.

Pilot one model/one training seed/one native backend first; then use three
training seeds with INT8 and one four-bit backend; only then add robustness
backends and another model family. Every eventual formal arm needs native
artifacts, fixed versions/calibration data, loader attestation, and hashes.
