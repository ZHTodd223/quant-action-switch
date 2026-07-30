# Formal v5 matrix headless preparation

Status: `FORMAL_V5_MATRIX_REGISTERED`,
`ALL_FORMAL_MODEL_SNAPSHOTS_CACHED`, and pending the final clean-tree
readiness flip at the containing commit.

- Initial SHA: `32b090c53dd464717e8d675450c2cff78b4b4d12`
- Final SHA: the commit containing this report; resolve with
  `git rev-parse HEAD` after checkout
- Remote SHA: required to equal final SHA before GPU handoff
- Matrix: `v5-cross-model-native-tools-matrix-v1` version `1.0.0`
- Protocol: `agent_toolcall_protocol_v5_research_validity`, protocol 5,
  research-validity `p1-v1`
- Cases: 1,000; seeds: 101, 202, 303
- Logical semantic SHA:
  `566c31e7a60dc96ddf49daf540938b8bfc60940bafda763c3247008df1542a17`
- Generation SHA:
  `e71d93861a1b1d99b62fc9b5b21ca8b109b3cda4ee0d7b9e6d4fc810a334da67`
- Sampling SHA:
  `57e6a7bba9753bf6a4e76e770ae5e81c297cb258bd723b57d55363d56dde7c84`
- Tool-schema SHA:
  `7d771262fe764737d5b303fddcbc5450923b32450fa27597aa8ab0828aca518d`
- Formal matrix SHA:
  `91b7d9e0ecc891183ad36092ecc73b7ed80d4026e56e119317448ed2938c7e0a`
- Attestation requirements:
  `formal_model_state_attestation_requirements_v1` version `1.0.0`, SHA
  `0daf144240223243b8ba18d659c7a62b11523cd12fc9ad31b632399f23967317`
- Threshold source: locked
  `qwen25-3b-multiseed-gate-v7-v1-run/preregistration.json`
- Eligibility source:
  `scripts.comparison_eligibility.determine_comparison_eligibility`

All three snapshots passed manifest verification and offline
`AutoConfig`/`AutoTokenizer`/`GenerationConfig` loading. No model class was
instantiated. Snapshot sizes are 6,183,469,558 bytes (Qwen),
8,639,609,814 bytes (Gemma), and 6,436,890,229 bytes (Llama). After caching,
the 100 GiB volume has 55,546,601,472 bytes free. The combined snapshots use
21,259,969,601 bytes; 1.5 times that size is 31,889,954,402 bytes, below
current free space. Formal output has a separate conservative 10 GiB planning
reserve.

Every renderer was produced by its pinned local tokenizer with the canonical
tool schema. Renderer manifest SHAs are Qwen `97bd31b...cab7`, Gemma
`fa81b71a...739c`, and Llama `a6cdf57a...5727`; complete values are in the
matrix. Quantization is pinned to bitsandbytes INT8, 100% registered target
coverage, `device_map={"": 0}`, and fail-closed no-offload operation.
Matrix, requirements and runtime coverage are all exactly `1.0`. The
versioned requirements bind Qwen 252, Gemma 238 and Llama 196 expected
targets, and both comparison arms carry the same requirements identity.

The test suite has exactly two accepted pre-existing failures:
`test_dual_patch_check_and_apply_against_upstream_fixture` and
`test_dual2_patch_check_and_apply_against_upstream_fixture`; both report that
their recorded upstream seed-forwarding patch does not apply to its fixture.
This change does not touch either patch or fixture. All new/focused failures
were corrected.

No access token, model weight, temporary output or frozen historical response
is tracked by Git. The matrix has no unresolved registration fields.
