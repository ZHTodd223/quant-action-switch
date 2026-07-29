# P0-1 comparison eligibility migration

This control patch does not change frozen runs, metrics, manifests, Gate
thresholds, model generation, quantization backends, or training.

## Native v4 experiments

Initialize a new isolated comparison state with
`scripts/run_cross_model_comparison.py init`. Before starting its quantized arm,
run:

```bash
python scripts/require_quantization_eligibility.py \
  --state /path/to/comparison_state.json \
  --gate-decision /path/to/gate_decision.json
```

Exit code `0` is the only authorization to start a new native-v4 quantized arm.
Exit code `20` means comparison eligibility did not authorize quantization.
Exit code `21` means the state or schema is invalid. Never continue after
either non-zero result.

The low-level `generate_quantized_responses.py` CLI repeats this authorization
before importing the GPU runtime, so invoking the engine directly does not
bypass the runner preflight.

## Historical reproduction

Pre-v4 runners are retained for frozen-result reproducibility. They now stop
with exit code `42` by default. An intentional historical reproduction must
set:

```bash
ALLOW_HISTORICAL_REPRODUCTION=YES bash scripts/<legacy-runner>.sh
```

That opt-in does not create native-v4 evidence. Its output must not be added to
the default `native_v4_only` cross-model summary.

## Summary modes

`scripts/summarize_cross_model_comparison.py` accepts:

- `--selection-mode native_v4_only` (default);
- `--selection-mode legacy_only`;
- `--selection-mode all_comparable`.

Legacy Qwen compatibility can retain `comparison_status=COMPARABLE` for its
historical within-run BF16/INT8 contrast while remaining machine-readable as
`state_origin=legacy_adapter`, `legacy_compatibility=true`, and
`native_protocol_comparable=false`.
