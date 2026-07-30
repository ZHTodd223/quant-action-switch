# Branch inventory

Comparison base: `agent/domestic-cache` at
`d584effd85b750a560ebc1c5e3d5ecd8301fb7a8`. Counts are Git
`rev-list --left-right --count base...branch` and unique paths are from the
three-dot diff. There were no open PRs and the sole PR was merged.

| Branch | Tip | Ahead / behind | Class | Action |
| --- | --- | ---: | --- | --- |
| agent/domestic-cache | d584eff | 0 / 0 | CANONICAL_CANDIDATE | Create `main`; keep archive alias |
| feat/v5-formal-cross-model-matrix | ba36c87 | 0 / 9 | MERGED_ANCESTOR | Archive tag; delete remote name |
| fix/formal-attempt-parent-v1 | 9daf9b1 | 0 / 3 | MERGED_ANCESTOR | Archive tag; delete remote name |
| fix/formal-attestation-requirements-v1 | 7d93405 | 0 / 7 | MERGED_ANCESTOR | Archive tag; delete remote name |
| fix/formal-calibration-os-import-v1 | 796e38 | 0 / 5 | MERGED_ANCESTOR | Archive tag; delete remote name |
| fix/formal-eval-split-v1 | 66d8530 | 0 / 1 | MERGED_ANCESTOR | Archive tag; delete remote name |
| fix/p1-research-validity | c9e76a6 | 0 / 18 | MERGED_ANCESTOR | Archive tag; delete remote name |
| fix/p1-seed-patch-applicability-test | c8a6079 | 0 / 17 | MERGED_ANCESTOR | Archive tag; delete remote name |
| fix/v5-native-tools-runtime-binding | 3dee72e | 0 / 16 | MERGED_ANCESTOR | Archive tag; delete remote name |
| results/v5-formal-matrix-v1 | 80f9eb4 | 5 / 9 | RESULTS_METADATA | Retain: 5 unique failure/repair records |

The local-only `backup/agent-domestic-cache-before-p0-consolidation` is a
historical backup. Existing local worktrees prevent deleting their checked-out
local branches; that is intentionally left untouched.
