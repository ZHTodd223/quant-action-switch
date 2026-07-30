# Formal BF16 scoring failure — attempt 0004

Qwen BF16 seed 101 generated and saved all 1,000 registered cases, then stopped
before metrics because the v3 validator did not recognize the registrar's
locked `formal_eval` split.

No metric or gate decision was produced. The raw evidence remains preserved
under the attempt directory, and recovery reruns the affected seed under
`attempt-0005` after a minimal existing-schema compatibility repair.
