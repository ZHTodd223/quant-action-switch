# Formal matrix preflight failure — attempt 0001

Status: `FORMAL_MATRIX_HANDOFF_MISMATCH`.

No model was loaded and no formal generation was started. GitHub, Hugging
Face, and ModelScope write preflights passed, all preparation hashes verified,
and all three offline model preflights passed.

The formal matrix points to
`config/model_state_attestation_requirements_v1.json`, but that file is absent.
The runtime default is `config/model_state_requirements_v1.json`. These cannot
be treated as interchangeable because the matrix locks INT8 target coverage to
`1.0`, while the runtime default accepts `0.95`.

The failure bundle is preserved at the immutable remote path recorded in the
adjacent JSON report. Resume only after the authoritative formal attestation
requirements are versioned without lowering the matrix-locked coverage, the
repair is tested and pushed, and a new attempt is created under the new
repository SHA.
