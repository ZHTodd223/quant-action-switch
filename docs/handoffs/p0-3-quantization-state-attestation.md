# P0-3 runtime model-state attestation

## Scope

This change adds fail-closed runtime state evidence for BF16, bitsandbytes
INT8/NF4/FP4, GPTQ, HQQ, and GGUF generation. It does not change EOS,
response parsing, tool protocols, reconstruction, training, or behavioral
gate thresholds.

## Evidence path

Every formal generator is bound to a native P0-1 comparison state. Before it
reads formal evaluation cases or creates/appends response rows, it:

1. verifies the locked source checkpoint manifest and config/tokenizer/
   generation-config identities;
2. records the actual loader mode, package versions, device map, parameter and
   buffer histograms, module classes, and core-projection coverage;
3. writes an immutable `*.model_state_attestation.json` and SHA-256 sidecar;
4. exits nonzero when `attestation.passed` is not true.

Completed response files receive a separate output manifest bound to the
attestation hash and locked case-manifest hash. Every raw row references the
sidecar path, hash, status, and case-manifest hash.

## Backend rules

- BF16 rejects quantized module classes and requires the configured BF16 core
  projection coverage.
- bitsandbytes requires actual `Linear8bitLt` or `Linear4bit` modules,
  configured projection coverage, and matching INT8/NF4/FP4 runtime config.
- GPTQ and HQQ require matching packed module classes, loader mode, coverage,
  requested config, quantized-checkpoint manifest, and conversion-cache
  metadata.
- GPTQ Transformers fallback is disabled by default. Explicit diagnostic
  fallback produces `DIAGNOSTIC_FALLBACK_NOT_ELIGIBLE`.
- GGUF reads `general.file_type` and `general.architecture` from the binary
  metadata, locks the file hash and cache metadata, verifies the server command
  model path and llama.cpp version, and requires a passing healthcheck.

## Comparison eligibility

Native P0-1 run state now records both arms' attestation and output-manifest
paths, hashes, statuses, and pass flags. `quantization_performed=true` is only
recorded after the quantized attestation and output manifest verify.
`COMPARABLE` requires both runtime evidence chains in addition to the existing
source and case lineage.

Frozen historical evidence remains handled by the legacy adapter and is not
rewritten.

## Verification boundary

The implementation is covered by fake-model, GGUF fixture, resume/hash, schema,
eligibility, P0-1, and P0-2 regression tests. No full checkpoint or real
GPU/backend model was loaded as part of this repair, so real BF16,
bitsandbytes, GPTQ, HQQ, and llama.cpp attestation remains unverified.
