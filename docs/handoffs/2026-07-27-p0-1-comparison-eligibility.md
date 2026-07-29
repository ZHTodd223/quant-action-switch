# P0-1 cross-model comparison eligibility repair

## Scope and result

P0-1 is resolved at the CPU control-code level. New runs now use one explicit
comparison state, one shared logical-case manifest, one fail-closed eligibility
function, and one summary rule. GPU validation has not been run and remains
explicitly pending.

Frozen historical runs and their manifests were not edited. Gate thresholds,
generation parsing, model reconstruction, quantization backends, EOS handling,
function-calling APIs, seeds, padding, and final-paper conclusions were not
changed.

## Raw code evidence and root cause

- `scripts/run_qwen25_3b_multiseed_gate_v7.sh` evaluates 12 BF16/INT8 cells on
  Gate-v7 and writes a generic `status: complete`; it did not write a comparison
  identity contract.
- `scripts/run_llama32_3b_strict_queue.sh` stops after adaptation or
  reconstruction failure and historically wrote
  `quantization_performed: false` without a comparison-status enum.
- `scripts/run_gemma3_4b_layerdrop_benign_reconstruction.sh` evaluates a
  200-case Gate-v4 slice and historically wrote a gate decision but no
  run-level comparison state.
- `scripts/refresh_research_state.py` historically displayed generic
  scientific `status/pass`; it did not distinguish comparison eligibility.
- Qwen uses 1,000 Gate-v7 cases per cell; Llama uses 200/400-case Gate-v4
  stages; Gemma uses Gate-v4 rows 800 through 999. These are not one shared
  cross-family comparison set.

The root cause was therefore a missing comparison identity and stage contract,
not evidence that Gemma or Llama had zero quantization effect.

## New state machine

```text
BASELINE
→ BENIGN_ADAPTATION
→ RECONSTRUCTION
→ BF16_GATE
→ QUANTIZATION
→ QUANTIZED_EVALUATION
→ COMPARABLE
```

The machine-readable statuses are:

- `NOT_ELIGIBLE_BASELINE_FAILED`
- `NOT_ELIGIBLE_RECONSTRUCTION_FAILED`
- `NOT_ELIGIBLE_BF16_GATE_FAILED`
- `NOT_ELIGIBLE_MISSING_ARTIFACTS`
- `NOT_ELIGIBLE_ABNORMAL_TERMINATION`
- `ELIGIBLE_NOT_QUANTIZED`
- `QUANTIZATION_FAILED`
- `NOT_COMPARABLE_SOURCE_MISMATCH`
- `NOT_COMPARABLE_CASE_MISMATCH`
- `COMPARABLE`

`QUANTIZATION_FAILED` means the requested stage did not complete. It is not a
zero behavioral effect and is not included in effect summaries.

## Uniform quantization eligibility

`scripts.comparison_eligibility.determine_comparison_eligibility` is the sole
gate. It requires:

1. baseline completion and baseline capability pass;
2. BF16 reconstruction completion;
3. BF16 reconstruction gate pass under the supplied locked decision;
4. no abnormal termination;
5. BF16 output and metric files;
6. a fully rehashed source-checkpoint manifest;
7. unchanged config and tokenizer identities;
8. the locked shared logical-case manifest and file hash;
9. BF16 arm identity matching the locked checkpoint and case manifest.

The quantized arm can become `COMPARABLE` only after its output and metrics
exist and its source-checkpoint and case-manifest hashes exactly equal the BF16
arm hashes. Each arm separately records checkpoint path, checkpoint manifest,
config hash, tokenizer hash, training stage, and source run ID; every field
must match.

`scripts/run_cross_model_comparison.py quantization-preflight` exits with code
20 unless the status is exactly `ELIGIBLE_NOT_QUANTIZED`. It never loads a
model.

## Shared logical cases

`config/cross_model_logical_cases_v1.json` contains 12 deterministic
process-validation cases:

- 3 `file_read`;
- 3 `calculator_control`;
- 3 `search_control`;
- 3 `no_tool_control`.

The list is not selected from final results. Its order is fixed, its canonical
logical hash is stored, and the file hash is copied into each new run state.
Qwen, Gemma, and Llama rendered lists retain identical logical rows and case
IDs; only `model_id` and `renderer_id` differ. This small set is for process
validation, not a final scientific claim.

## Historical compatibility

The compatibility adapter reads frozen metadata without rewriting it:

- `qwen_locked_confirmatory_gate` + `scientific_status: complete` is classified
  `COMPARABLE` because the frozen record represents explicit BF16/INT8 cells.
  Its overall preregistered Gate outcome remains separate and must still be
  reported as not passing.
- Gemma reconstruction-stop records are
  `NOT_ELIGIBLE_BF16_GATE_FAILED`.
- The Llama reconstruction-stop record is
  `NOT_ELIGIBLE_BF16_GATE_FAILED`.

`scripts/refresh_research_state.py` and `scripts/show_research_state.py` now
display scientific status and comparison status separately.
`scripts/summarize_cross_model_comparison.py` includes only `COMPARABLE` runs
in `quantization_effect_run_ids`.

## CPU validation

Commands:

```bash
python -m unittest tests.test_comparison_eligibility -v
python -m compileall -q scripts tests
python -m unittest discover -s tests -v
python scripts/refresh_research_state.py
python scripts/show_research_state.py
python scripts/check_terminology.py
git diff --check
```

The focused suite covers the five required status cases, fail-closed
quantization preflight, historical Qwen/Gemma/Llama interpretation, effect
summary exclusion, shared case IDs/hash across three renderers, and dry-run
non-execution.

## Minimal pending GPU workflow

Use a new run ID and a directory that does not exist:

```bash
python scripts/run_cross_model_comparison.py init \
  --model-id qwen25-3b \
  --run-id qwen25-3b-comparison-v4-seed101-v1 \
  --run-root /new/isolated/run/root \
  --source-checkpoint /exact/reconstruction/checkpoint \
  --source-checkpoint-manifest /exact/reconstruction/checkpoint/manifest.sha256.json \
  --source-run-id exact-source-run-id \
  --training-stage reconstruction
```

Run the existing model-specific baseline/reconstruction gate under its locked
threshold, then evaluate the shared rendered BF16 list. Record the evidence:

```bash
python scripts/run_cross_model_comparison.py record-bf16 \
  --state /new/isolated/run/root/comparison_state.json \
  --baseline-decision /new/baseline/decision.json \
  --gate-decision /new/reconstruction/gate_decision.json
```

Before any quantization command:

```bash
python scripts/run_cross_model_comparison.py quantization-preflight \
  --state /new/isolated/run/root/comparison_state.json \
  --gate-decision /new/reconstruction/gate_decision.json
```

Repeat with new Gemma and Llama run IDs. Do not launch their quantized arms if
the preflight exits 20.

## Scientific wording after the repair

Qwen2.5 completed BF16 and INT8 comparison arms, while its overall
preregistered Gate outcome is reported separately and did not pass.

Gemma3-4B has not passed the BF16 reconstruction comparison gate and did not
enter quantization, so its quantization effect cannot currently be judged.

Llama3.2-3B has not passed the BF16 reconstruction comparison gate and did not
enter quantization, so its quantization effect cannot currently be judged.

## Deferred dependencies

EOS handling, response parsing, native function calling, quantization backend
acceptance, leakage audits, and seed/padding controls remain separate follow-up
work. None is silently treated as verified by this repair.
