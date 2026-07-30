# Formal attestation repair — attempt 0002

The missing versioned formal attestation requirements were added without
lowering the matrix-locked INT8 target-module coverage of `1.0`.

- Implementation commit:
  `651b6f42db122b87865443978711efe74bd9b0b0`
- Production repository SHA:
  `7d934056be993413c5cd50831d580a695576489f`
- Requirements version: `1.0.0`
- Requirements SHA-256:
  `0daf144240223243b8ba18d659c7a62b11523cd12fc9ad31b632399f23967317`
- Formal matrix SHA-256:
  `91b7d9e0ecc891183ad36092ecc73b7ed80d4026e56e119317448ed2938c7e0a`
- Focused tests: `56 passed`.
- Full regression: `421` tests with only the two unchanged upstream patch
  fixture baseline failures; no new failures.
- Recovery attempt: `attempt-0002`.

The preserved attempt-0001 Hugging Face and ModelScope locations remain in the
adjacent attempt-0001 report and were not overwritten.
