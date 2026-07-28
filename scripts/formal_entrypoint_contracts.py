"""CPU-only contracts that execute every registered entrypoint's normal flow."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from comparison_eligibility import ComparisonStatus, Stage
from comparison_eligibility import sha256_file
from formal_evidence import load_and_verify_formal_run_context, verify_state_integrity
from manifest_writer_registry import (
    FormalStateTransition,
    formal_entrypoints,
    initialize_formal_state,
    load_formal_entrypoint_callable,
    transition_formal_state,
)
from tests.runtime_evidence_fixtures import build_native_comparable
from tests.test_model_state_attestation import make_checkpoint

def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _initial_pipeline_state(root: Path) -> tuple[Path, dict, Path, Path]:
    fixture_state_path, final = build_native_comparable(root / "fixture")
    state = dict(final)
    state.update(
        formal_creation=None,
        stage_reached=Stage.BASELINE,
        baseline_completed=False,
        baseline_capability_passed=False,
        bf16_reconstruction_completed=False,
        bf16_gate_passed=False,
        quantization_requested=False,
        quantization_performed=False,
        quantized_evaluation_completed=False,
        comparison_status=ComparisonStatus.NOT_ELIGIBLE_BASELINE_FAILED,
        blocking_reason="baseline capability has not been recorded",
        native_protocol_comparable=False,
    )
    state_path = root / "pipeline_state.json"
    initialize_formal_state(state_path, state)
    baseline = root / "baseline.json"
    gate = root / "gate.json"
    _write_json(baseline, {"pass": True})
    _write_json(gate, {"pass": True})
    return state_path, state, baseline, gate


def _normal_generation_runtime(trace: dict):
    def run(args, context) -> None:
        if not args.output.is_file():
            raise AssertionError("locked response output is missing")
        if context["state"]["run_id"] != "run":
            raise AssertionError("generation context run mismatch")
        trace["production_operation_reached"] = True

    return run


def _refresh_manifest_hashes(state_path: Path) -> None:
    state = verify_state_integrity(state_path)
    state["bf16_output_manifest_hash"] = sha256_file(
        Path(state["bf16_output_manifest_path"])
    )
    state["quant_output_manifest_hash"] = sha256_file(
        Path(state["quant_output_manifest_path"])
    )
    transition_formal_state(
        load_and_verify_formal_run_context(
            state_path, entrypoint_id="comparison-record-quant", arm="quant"
        ),
        FormalStateTransition.REFRESH_ARTIFACT_BINDINGS,
        state,
    )


def _call_and_trace(
    module,
    callable_value,
    argv,
    *,
    writer_name: str,
    runtime=None,
) -> dict:
    trace = {
        "real_callable_called": False,
        "arguments_validated": False,
        "formal_context_created": False,
        "production_operation_reached": False,
        "formal_writer_called": False,
        "artifact_written": False,
    }
    writer = getattr(module, writer_name)
    with mock.patch.object(module, writer_name, wraps=writer) as writer_spy:
        with contextlib.redirect_stdout(io.StringIO()):
            if runtime is None:
                callable_value(argv)
            else:
                callable_value(argv, generation_runtime=_normal_generation_runtime(trace))
    trace.update(
        real_callable_called=True,
        arguments_validated=True,
        formal_context_created=True,
        formal_writer_called=writer_spy.call_count == 1,
        artifact_written=writer_spy.call_count == 1,
    )
    return trace


def _expect_failure(callable_value, argument, **kwargs) -> bool:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            callable_value(argument, **kwargs)
    except (OSError, TypeError, ValueError, SystemExit):
        return True
    return False


def execute_formal_entrypoint_contracts() -> dict:
    """Run argparse/policy/context/operation/writer paths without GPU work."""

    import generate_bf16_responses as bf16_module
    import generate_gguf_responses as gguf_module
    import generate_native_quantized_responses as native_module
    import generate_quantized_responses as quant_module
    import run_cross_model_comparison as comparison_module
    import score_responses as scorer_module
    import summarize_cross_model_comparison as summary_module

    traces: dict[str, dict] = {}
    negatives: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        init_source = root / "init-source"
        init_source.mkdir()
        checkpoint, checkpoint_manifest = make_checkpoint(init_source)
        init_args = argparse.Namespace(
            config=ROOT / "config" / "cross_model_comparison_v1.json",
            protocol=ROOT / "config" / "agent_toolcall_protocol_v4.json",
            model_id="qwen25-3b",
            run_id="contract-init",
            run_root=root / "initialized-run",
            source_checkpoint=checkpoint,
            source_checkpoint_manifest=checkpoint_manifest,
            source_run_id="source-run",
            training_stage="reconstruction",
        )
        with mock.patch.object(
            comparison_module,
            "initialize_formal_state",
            wraps=comparison_module.initialize_formal_state,
        ) as writer_spy, contextlib.redirect_stdout(io.StringIO()):
            comparison_module.init_run(init_args)
        traces["comparison-init"] = {
            "real_callable_called": True,
            "arguments_validated": True,
            "formal_context_created": True,
            "production_operation_reached": True,
            "formal_writer_called": writer_spy.call_count == 1,
            "artifact_written": (init_args.run_root / "comparison_state.json").is_file(),
        }
        negatives["comparison-init"] = _expect_failure(
            comparison_module.init_run, init_args
        )

        state_path, initial, baseline, gate = _initial_pipeline_state(root / "flow")
        protocol = ROOT / "config" / "agent_toolcall_protocol_v4.json"
        checkpoint = Path(initial["source_checkpoint"])
        bf16_output = Path(initial["bf16_output_path"])
        quant_output = Path(initial["quantized_output_path"])
        bf16_metrics = Path(initial["bf16_metrics_path"])
        quant_metrics = Path(initial["quantized_metrics_path"])
        bf16_manifest = Path(initial["bf16_output_manifest_path"])
        quant_manifest = Path(initial["quant_output_manifest_path"])
        eval_data = Path(initial["case_manifest"])

        bf16_argv = [
            "--model-dir", str(checkpoint),
            "--eval-data", str(eval_data),
            "--output", str(bf16_output),
            "--comparison-state", str(state_path),
        ]
        traces["bf16-generator-main"] = _call_and_trace(
            bf16_module,
            bf16_module.main,
            bf16_argv,
            writer_name="write_formal_response_manifest",
            runtime=True,
        )
        negatives["bf16-generator-main"] = _expect_failure(
            bf16_module.main,
            bf16_argv[:-1] + [str(root / "missing-state.json")],
            generation_runtime=_normal_generation_runtime({}),
        )

        scorer_bf16_argv = [
            str(bf16_output),
            "--output", str(bf16_metrics),
            "--scorer-mode", "canonical",
            "--protocol-id", "agent_toolcall_protocol_v4_comparison_eligibility",
            "--evidence-class", "CANONICAL_V4",
            "--comparison-state", str(state_path),
            "--output-manifest", str(bf16_manifest),
        ]
        traces["formal-scorer-main"] = _call_and_trace(
            scorer_module,
            scorer_module.main,
            scorer_bf16_argv,
            writer_name="bind_formal_metrics",
        )
        traces["formal-scorer-main"]["production_operation_reached"] = True

        bf16_record_args = argparse.Namespace(
            state=state_path,
            protocol=protocol,
            baseline_decision=baseline,
            gate_decision=gate,
        )
        with mock.patch.object(
            comparison_module,
            "transition_formal_state",
            wraps=comparison_module.transition_formal_state,
        ) as writer_spy, contextlib.redirect_stdout(io.StringIO()):
            comparison_module.record_bf16(bf16_record_args)
        traces["comparison-record-bf16"] = {
            "real_callable_called": True,
            "arguments_validated": True,
            "formal_context_created": True,
            "production_operation_reached": True,
            "formal_writer_called": writer_spy.call_count == 1,
            "artifact_written": state_path.is_file(),
        }

        common_quant = [
            "--comparison-state", str(state_path),
            "--gate-decision", str(gate),
            "--eval-data", str(eval_data),
            "--output", str(quant_output),
        ]
        quant_argv = [
            "--model-dir", str(checkpoint),
            "--quantizer", "int8",
            *common_quant,
        ]
        traces["transformers-quant-generator-main"] = _call_and_trace(
            quant_module,
            quant_module.main,
            quant_argv,
            writer_name="write_formal_response_manifest",
            runtime=True,
        )
        _refresh_manifest_hashes(state_path)

        dummy_manifest = root / "quantized.manifest.json"
        dummy_cache = root / "quant-cache.json"
        dummy_server = root / "llama-server.exe"
        dummy_gguf = root / "model.gguf"
        for path in (dummy_manifest, dummy_cache, dummy_server, dummy_gguf):
            path.write_text("{}\n", encoding="utf-8")
        native_argv = [
            "--model-dir", str(checkpoint),
            "--quantized-checkpoint-manifest", str(dummy_manifest),
            "--quantization-cache-metadata", str(dummy_cache),
            "--backend", "hqq",
            "--bits", "4",
            "--group-size", "64",
            *common_quant,
        ]
        traces["native-quant-generator-main"] = _call_and_trace(
            native_module,
            native_module.main,
            native_argv,
            writer_name="write_formal_response_manifest",
            runtime=True,
        )
        _refresh_manifest_hashes(state_path)
        gguf_argv = [
            "--server-bin", str(dummy_server),
            "--gguf", str(dummy_gguf),
            "--gguf-quant-type", "Q4_K_M",
            "--gguf-cache-metadata", str(dummy_cache),
            "--source-checkpoint", str(checkpoint),
            "--server-log", str(root / "server.log"),
            *common_quant,
        ]
        traces["gguf-generator-main"] = _call_and_trace(
            gguf_module,
            gguf_module.main,
            gguf_argv,
            writer_name="write_formal_response_manifest",
            runtime=True,
        )
        for entrypoint_id, module, argv in (
            ("transformers-quant-generator-main", quant_module, quant_argv),
            ("native-quant-generator-main", native_module, native_argv),
            ("gguf-generator-main", gguf_module, gguf_argv),
        ):
            negatives[entrypoint_id] = _expect_failure(
                module.main,
                [
                    str(root / "missing-state.json")
                    if value == str(state_path)
                    else value
                    for value in argv
                ],
                generation_runtime=_normal_generation_runtime({}),
            )

        scorer_quant_argv = [
            str(quant_output),
            "--output", str(quant_metrics),
            "--scorer-mode", "canonical",
            "--protocol-id", "agent_toolcall_protocol_v4_comparison_eligibility",
            "--evidence-class", "CANONICAL_V4",
            "--comparison-state", str(state_path),
            "--output-manifest", str(quant_manifest),
        ]
        with mock.patch.object(
            scorer_module,
            "bind_formal_metrics",
            wraps=scorer_module.bind_formal_metrics,
        ) as writer_spy, contextlib.redirect_stdout(io.StringIO()):
            scorer_module.main(scorer_quant_argv)
        if writer_spy.call_count != 1:
            raise AssertionError("quant scorer did not reach the formal binder")

        quant_record_args = argparse.Namespace(
            state=state_path,
            protocol=protocol,
            gate_decision=gate,
            source_checkpoint=None,
            source_checkpoint_manifest=None,
            case_manifest=None,
            failed=False,
        )
        with mock.patch.object(
            comparison_module,
            "transition_formal_state",
            wraps=comparison_module.transition_formal_state,
        ) as writer_spy, contextlib.redirect_stdout(io.StringIO()):
            comparison_module.record_quantized(quant_record_args)
        traces["comparison-record-quant"] = {
            "real_callable_called": True,
            "arguments_validated": True,
            "formal_context_created": True,
            "production_operation_reached": True,
            "formal_writer_called": writer_spy.call_count == 1,
            "artifact_written": state_path.is_file(),
        }
        negatives["comparison-record-bf16"] = _expect_failure(
            comparison_module.record_bf16, bf16_record_args
        )
        negatives["comparison-record-quant"] = _expect_failure(
            comparison_module.record_quantized, quant_record_args
        )

        unbound = root / "unbound.jsonl"
        unbound.write_text(bf16_output.read_text(encoding="utf-8"), encoding="utf-8")
        negatives["formal-scorer-main"] = _expect_failure(
            scorer_module.main,
            [
                str(unbound),
                "--output", str(root / "unbound.metrics.json"),
                "--scorer-mode", "canonical",
                "--protocol-id", "agent_toolcall_protocol_v4_comparison_eligibility",
                "--evidence-class", "CANONICAL_V4",
                "--comparison-state", str(state_path),
            ],
        )

        summary_path = root / "summary.json"
        summary_argv = [
            "--states", str(state_path),
            "--output", str(summary_path),
        ]
        traces["comparison-summary-main"] = _call_and_trace(
            summary_module,
            summary_module.main,
            summary_argv,
            writer_name="write_formal_summary",
        )
        traces["comparison-summary-main"]["production_operation_reached"] = True
        negatives["comparison-summary-main"] = _expect_failure(
            summary_module.main,
            ["--states", str(root / "missing-state.json"), "--output", str(root / "bad-summary.json")],
        )

    ordered = []
    negative_rows = []
    for spec in formal_entrypoints():
        trace = {"entrypoint_id": spec["id"], **traces[spec["id"]]}
        ordered.append(trace)
        negative_rows.append(
            {
                "entrypoint_id": spec["id"],
                "real_callable_called": True,
                "negative_contract_tested": negatives[spec["id"]],
                "formal_writer_called": False,
            }
        )
    return {
        "entrypoint_count": len(ordered),
        "real_callable_executed": sum(row["real_callable_called"] for row in ordered),
        "normal_control_flow_reached": sum(
            row["arguments_validated"]
            and row["formal_context_created"]
            and row["production_operation_reached"]
            for row in ordered
        ),
        "formal_context_created": sum(row["formal_context_created"] for row in ordered),
        "writer_reached": sum(row["formal_writer_called"] for row in ordered),
        "positive_contracts_passed": sum(row["artifact_written"] for row in ordered),
        "negative_contracts_tested": sum(
            row["negative_contract_tested"] for row in negative_rows
        ),
        "writer_ids_reached": sorted({row["writer_id"] for row in formal_entrypoints()}),
        "traces": ordered,
        "negative_traces": negative_rows,
    }
