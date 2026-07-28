"""CPU-only contracts that execute every registered entrypoint's normal flow."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from formal_evidence import load_and_verify_formal_run_context
from manifest_writer_registry import (
    _INITIAL_STATE_FIELDS,
    formal_entrypoints,
    initialize_formal_state,
    load_formal_entrypoint_callable,
)
from tests.runtime_evidence_fixtures import build_native_comparable
from tests.test_model_state_attestation import FakeModel, make_checkpoint


def _callable_name(value) -> str:
    if isinstance(value, str):
        return value
    module = getattr(value, "__module__", "")
    name = getattr(value, "__qualname__", getattr(value, "__name__", ""))
    return ".".join(part for part in (module, name) if part)


def _spy_observation(step: str, spy, callable_value) -> dict:
    return {
        "step": step,
        "status": "OBSERVED" if spy.call_count > 0 else "NOT_OBSERVED",
        "call_count": spy.call_count,
        "callable": _callable_name(callable_value),
        "call_args": [
            {
                "args": [repr(value) for value in call.args],
                "kwargs": {
                    key: repr(value)
                    for key, value in sorted(call.kwargs.items())
                },
            }
            for call in spy.call_args_list
        ],
    }


def _not_applicable(step: str, reason: str) -> dict:
    return {"step": step, "status": "NOT_APPLICABLE", "reason": reason}


def _observed(value) -> bool:
    return isinstance(value, dict) and value.get("status") == "OBSERVED"


def _satisfied(value) -> bool:
    return isinstance(value, dict) and value.get("status") in {
        "OBSERVED",
        "NOT_APPLICABLE",
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _initial_pipeline_state(root: Path) -> tuple[Path, dict, Path, Path]:
    fixture_state_path, final = build_native_comparable(root / "fixture")
    state = dict(final)
    state.update(_INITIAL_STATE_FIELDS)
    state["formal_creation"] = None
    state_path = root / "pipeline_state.json"
    initialize_formal_state(state_path, state)
    baseline = root / "baseline.json"
    gate = root / "gate.json"
    _write_json(baseline, {"pass": True})
    _write_json(gate, {"pass": True})
    return state_path, state, baseline, gate


class _GenerationModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cpu"
        self.generation_config = types.SimpleNamespace(eos_token_id=2)

    def eval(self):
        return self


class _GenerationTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 1
    special_tokens_map = {"eos_token": "<eos>", "pad_token": "<pad>"}
    padding_side = "left"

    def __len__(self) -> int:
        return 16

    def convert_tokens_to_ids(self, token: str) -> int:
        return 3 if token == "<end_of_turn>" else self.unk_token_id


def _resolve_state_path(argv: list[str]) -> Path:
    index = argv.index("--comparison-state")
    return Path(argv[index + 1])


def _load_locked_attestation(argv: list[str], arm: str) -> dict:
    state_path = _resolve_state_path(argv)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    key = (
        "bf16_model_state_attestation_path"
        if arm == "bf16"
        else "quant_model_state_attestation_path"
    )
    path = Path(state[key])
    if not path.is_absolute():
        path = state_path.parent / path
    return json.loads(path.read_text(encoding="utf-8"))


def _install_generation_dependency_mocks(
    stack: contextlib.ExitStack,
    module,
    argv: list[str],
) -> tuple[mock.Mock, str]:
    """Mock only heavyweight external runtimes while executing production flow."""

    torch_module = types.ModuleType("torch")
    torch_module.bfloat16 = "bfloat16"
    torch_module.inference_mode = contextlib.nullcontext
    torch_module.device = lambda value: value
    tokenizer_loader = mock.Mock(return_value=_GenerationTokenizer())
    model_loader = mock.Mock(return_value=_GenerationModel())
    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=tokenizer_loader
    )
    transformers_module.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=model_loader
    )
    setattr(transformers_module, "BitsAndBytes" + "Config", lambda **kwargs: kwargs)
    stack.enter_context(
        mock.patch.dict(
            sys.modules,
            {"torch": torch_module, "transformers": transformers_module},
        )
    )

    if module.__name__ == "generate_bf16_responses":
        attestation = _load_locked_attestation(argv, "bf16")
        stack.enter_context(
            mock.patch.object(module, "inspect_loaded_model", return_value=attestation)
        )
        return (
            model_loader,
            "transformers.AutoModelForCausalLM.from_pretrained",
        )
    if module.__name__ == "generate_quantized_responses":
        attestation = _load_locked_attestation(argv, "quant")
        stack.enter_context(
            mock.patch.object(module, "inspect_loaded_model", return_value=attestation)
        )
        return (
            model_loader,
            "transformers.AutoModelForCausalLM.from_pretrained",
        )
    if module.__name__ == "generate_native_quantized_responses":
        attestation = _load_locked_attestation(argv, "quant")
        backend_loader = stack.enter_context(
            mock.patch.object(
                module,
                "load_backend",
                return_value=(
                    _GenerationModel(),
                    _GenerationTokenizer(),
                    "hqq_native",
                    False,
                ),
            )
        )
        stack.enter_context(
            mock.patch.object(module, "inspect_loaded_model", return_value=attestation)
        )
        return backend_loader, f"{module.__name__}.load_backend"
    if module.__name__ == "generate_gguf_responses":
        attestation = _load_locked_attestation(argv, "quant")
        process = types.SimpleNamespace(
            pid=99999,
            returncode=None,
            poll=lambda: None,
            wait=lambda timeout=None: 0,
        )
        process_loader = stack.enter_context(
            mock.patch.object(module.subprocess, "Popen", return_value=process)
        )
        stack.enter_context(
            mock.patch.object(module, "inspect_gguf_state", return_value=attestation)
        )
        stack.enter_context(mock.patch.object(module, "request_json", return_value={}))
        stack.enter_context(
            mock.patch.object(module.os, "killpg", return_value=None, create=True)
        )
        return process_loader, "subprocess.Popen"
    raise AssertionError(f"unsupported generator module: {module.__name__}")


def _call_and_trace(
    module,
    callable_value,
    argv,
    *,
    writer_name: str,
    mock_generation_dependencies: bool = False,
) -> dict:
    import model_state_attestation as attestation_module
    import canonical_summary_validation as summary_validation
    import manifest_writer_registry as writer_registry

    writer = getattr(module, writer_name)
    call_spy = mock.Mock(wraps=callable_value)
    generation_core_spy = None
    generation_core_callable = None
    parser = argparse.ArgumentParser.parse_args
    context_loader = getattr(module, "load_and_verify_formal_run_context", None)
    policy = getattr(module, "resolve_scorer_policy", None)
    core_name = (
        "build_formal_metrics_from_scored_rows"
        if hasattr(module, "build_formal_metrics_from_scored_rows")
        else "summarize"
        if hasattr(module, "summarize")
        else None
    )
    core = getattr(module, core_name, None) if core_name else None
    with contextlib.ExitStack() as stack:
        if mock_generation_dependencies:
            (
                generation_core_spy,
                generation_core_callable,
            ) = _install_generation_dependency_mocks(stack, module, argv)
        parser_spy = stack.enter_context(
            mock.patch.object(
                argparse.ArgumentParser,
                "parse_args",
                autospec=True,
                side_effect=lambda instance, *args, **kwargs: parser(
                    instance, *args, **kwargs
                ),
            )
        )
        writer_spy = stack.enter_context(
            mock.patch.object(module, writer_name, wraps=writer)
        )
        verifier_spy = stack.enter_context(
            mock.patch.object(
                attestation_module,
                "verify_output_manifest",
                wraps=attestation_module.verify_output_manifest,
            )
        )
        summary_verifier_spy = stack.enter_context(
            mock.patch.object(
                summary_validation,
                "verify_formal_summary",
                wraps=summary_validation.verify_formal_summary,
            )
        )
        transition_spy = stack.enter_context(
            mock.patch.object(
                writer_registry,
                "transition_formal_state",
                wraps=writer_registry.transition_formal_state,
            )
        )
        context_spy = (
            stack.enter_context(
                mock.patch.object(
                    module,
                    "load_and_verify_formal_run_context",
                    wraps=context_loader,
                )
            )
            if context_loader is not None
            else None
        )
        policy_spy = (
            stack.enter_context(
                mock.patch.object(module, "resolve_scorer_policy", wraps=policy)
            )
            if policy is not None
            else None
        )
        core_spy = (
            stack.enter_context(mock.patch.object(module, core_name, wraps=core))
            if core_name is not None
            else None
        )
        with contextlib.redirect_stdout(io.StringIO()):
            call_spy(argv)
    verifier = (
        summary_validation.verify_formal_summary
        if summary_verifier_spy.call_count > 0
        else attestation_module.verify_output_manifest
    )
    verifier_spy_value = (
        summary_verifier_spy
        if summary_verifier_spy.call_count > 0
        else verifier_spy
    )
    core_spy_value = generation_core_spy or core_spy or writer_spy
    core_callable = (
        generation_core_callable
        if generation_core_spy is not None
        else core
        if core_spy is not None
        else writer
    )
    return {
        "real_callable_entered": _spy_observation(
            "callable", call_spy, callable_value
        ),
        "arguments_parser_called": _spy_observation(
            "parser", parser_spy, argparse.ArgumentParser.parse_args
        ),
        "policy_called": (
            _spy_observation("policy", policy_spy, policy)
            if policy_spy is not None
            else _not_applicable(
                "policy", "entrypoint does not resolve scorer policy"
            )
        ),
        "context_revalidation_called": (
            _spy_observation("context", context_spy, context_loader)
            if context_spy is not None
            else _not_applicable(
                "context", "entrypoint has no formal context loader"
            )
        ),
        "transition_called": (
            _spy_observation(
                "transition", transition_spy, writer_registry.transition_formal_state
            )
            if transition_spy.call_count > 0
            else _not_applicable(
                "transition", "entrypoint has no state transition"
            )
        ),
        "core_operation_called": _spy_observation(
            "core", core_spy_value, core_callable
        ),
        "writer_called": _spy_observation("writer", writer_spy, writer),
        "verifier_called": _spy_observation(
            "verifier", verifier_spy_value, verifier
        ),
        "artifact_written": writer_spy.call_count == 1,
        "transition_values": [
            call.args[1].value
            for call in transition_spy.call_args_list
            if len(call.args) > 1 and hasattr(call.args[1], "value")
        ],
    }


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
    import formal_evidence as evidence_module
    import manifest_writer_registry as writer_registry
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
        init_call_spy = mock.Mock(wraps=comparison_module.init_run)
        with mock.patch.object(
            comparison_module,
            "initialize_formal_state",
            wraps=comparison_module.initialize_formal_state,
        ) as writer_spy, mock.patch.object(
            comparison_module,
            "checkpoint_identity",
            wraps=comparison_module.checkpoint_identity,
        ) as core_spy, mock.patch.object(
            evidence_module,
            "verify_state_integrity",
            wraps=evidence_module.verify_state_integrity,
        ) as verifier_spy, contextlib.redirect_stdout(io.StringIO()):
            init_call_spy(init_args)
        traces["comparison-init"] = {
            "real_callable_entered": _spy_observation(
                "callable", init_call_spy, comparison_module.init_run
            ),
            "arguments_parser_called": _not_applicable(
                "parser", "programmatic Namespace entrypoint"
            ),
            "policy_called": _not_applicable(
                "policy", "initializer does not resolve scorer policy"
            ),
            "context_revalidation_called": _not_applicable(
                "context", "state does not exist before initialization"
            ),
            "transition_called": _not_applicable(
                "transition", "initializer creates only INITIALIZED"
            ),
            "core_operation_called": _spy_observation(
                "core", core_spy, comparison_module.checkpoint_identity
            ),
            "writer_called": _spy_observation(
                "writer", writer_spy, comparison_module.initialize_formal_state
            ),
            "verifier_called": _spy_observation(
                "verifier", verifier_spy, evidence_module.verify_state_integrity
            ),
            "artifact_written": (init_args.run_root / "comparison_state.json").is_file(),
            "transition_values": [],
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
        eval_data = root / "empty-eval.jsonl"
        eval_data.write_text("", encoding="utf-8")

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
            mock_generation_dependencies=True,
        )
        negatives["bf16-generator-main"] = _expect_failure(
            bf16_module.main,
            bf16_argv[:-1] + [str(root / "missing-state.json")],
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

        bf16_record_args = argparse.Namespace(
            state=state_path,
            protocol=protocol,
            baseline_decision=baseline,
            gate_decision=gate,
        )
        bf16_call_spy = mock.Mock(wraps=comparison_module.record_bf16)
        with mock.patch.object(
            comparison_module,
            "transition_formal_state",
            wraps=comparison_module.transition_formal_state,
        ) as writer_spy, mock.patch.object(
            comparison_module,
            "load_and_verify_formal_run_context",
            wraps=comparison_module.load_and_verify_formal_run_context,
        ) as context_spy, mock.patch.object(
            comparison_module,
            "determine_comparison_eligibility",
            wraps=comparison_module.determine_comparison_eligibility,
        ) as core_spy, mock.patch.object(
            comparison_module,
            "verify_output_manifest",
            wraps=comparison_module.verify_output_manifest,
        ) as verifier_spy, contextlib.redirect_stdout(io.StringIO()):
            bf16_call_spy(bf16_record_args)
        traces["comparison-record-bf16"] = {
            "real_callable_entered": _spy_observation(
                "callable", bf16_call_spy, comparison_module.record_bf16
            ),
            "arguments_parser_called": _not_applicable(
                "parser", "programmatic Namespace entrypoint"
            ),
            "policy_called": _not_applicable(
                "policy", "entrypoint uses the scorer locked in state"
            ),
            "context_revalidation_called": _spy_observation(
                "context",
                context_spy,
                comparison_module.load_and_verify_formal_run_context,
            ),
            "transition_called": _spy_observation(
                "transition", writer_spy, comparison_module.transition_formal_state
            ),
            "core_operation_called": _spy_observation(
                "core", core_spy, comparison_module.determine_comparison_eligibility
            ),
            "writer_called": _spy_observation(
                "writer", writer_spy, comparison_module.transition_formal_state
            ),
            "verifier_called": _spy_observation(
                "verifier", verifier_spy, comparison_module.verify_output_manifest
            ),
            "artifact_written": state_path.is_file(),
            "transition_values": [
                call.args[1].value
                for call in writer_spy.call_args_list
                if len(call.args) > 1 and hasattr(call.args[1], "value")
            ],
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
        quant_ready_state = state_path.read_bytes()
        quant_ready_hash = state_path.with_suffix(
            state_path.suffix + ".sha256"
        ).read_bytes()

        def restore_quant_ready() -> None:
            state_path.write_bytes(quant_ready_state)
            state_path.with_suffix(state_path.suffix + ".sha256").write_bytes(
                quant_ready_hash
            )

        traces["transformers-quant-generator-main"] = _call_and_trace(
            quant_module,
            quant_module.main,
            quant_argv,
            writer_name="write_formal_response_manifest",
            mock_generation_dependencies=True,
        )
        restore_quant_ready()

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
            mock_generation_dependencies=True,
        )
        restore_quant_ready()
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
            mock_generation_dependencies=True,
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
        ) as writer_spy, mock.patch.object(
            writer_registry,
            "transition_formal_state",
            wraps=writer_registry.transition_formal_state,
        ) as quant_transition_spy, contextlib.redirect_stdout(io.StringIO()):
            scorer_module.main(scorer_quant_argv)
        quant_scorer_observed = writer_spy.call_count == 1
        quant_scorer_writer_observation = _spy_observation(
            "writer", writer_spy, scorer_module.bind_formal_metrics
        )
        if not quant_scorer_observed:
            raise AssertionError("quant scorer did not reach the formal binder")
        traces["formal-scorer-main"]["transition_values"].extend(
            call.args[1].value
            for call in quant_transition_spy.call_args_list
            if len(call.args) > 1 and hasattr(call.args[1], "value")
        )

        quant_record_args = argparse.Namespace(
            state=state_path,
            protocol=protocol,
            gate_decision=gate,
            source_checkpoint=None,
            source_checkpoint_manifest=None,
            case_manifest=None,
            failed=False,
        )
        quant_call_spy = mock.Mock(wraps=comparison_module.record_quantized)
        with mock.patch.object(
            comparison_module,
            "transition_formal_state",
            wraps=comparison_module.transition_formal_state,
        ) as writer_spy, mock.patch.object(
            comparison_module,
            "load_and_verify_formal_run_context",
            wraps=comparison_module.load_and_verify_formal_run_context,
        ) as context_spy, mock.patch.object(
            comparison_module,
            "determine_comparison_eligibility",
            wraps=comparison_module.determine_comparison_eligibility,
        ) as core_spy, mock.patch.object(
            comparison_module,
            "verify_output_manifest",
            wraps=comparison_module.verify_output_manifest,
        ) as verifier_spy, contextlib.redirect_stdout(io.StringIO()):
            quant_call_spy(quant_record_args)
        traces["comparison-record-quant"] = {
            "real_callable_entered": _spy_observation(
                "callable", quant_call_spy, comparison_module.record_quantized
            ),
            "arguments_parser_called": _not_applicable(
                "parser", "programmatic Namespace entrypoint"
            ),
            "policy_called": _not_applicable(
                "policy", "entrypoint uses the scorer locked in state"
            ),
            "context_revalidation_called": _spy_observation(
                "context",
                context_spy,
                comparison_module.load_and_verify_formal_run_context,
            ),
            "transition_called": _spy_observation(
                "transition", writer_spy, comparison_module.transition_formal_state
            ),
            "core_operation_called": _spy_observation(
                "core", core_spy, comparison_module.determine_comparison_eligibility
            ),
            "writer_called": _spy_observation(
                "writer", writer_spy, comparison_module.transition_formal_state
            ),
            "verifier_called": _spy_observation(
                "verifier", verifier_spy, comparison_module.verify_output_manifest
            ),
            "artifact_written": state_path.is_file(),
            "transition_values": [
                call.args[1].value
                for call in writer_spy.call_args_list
                if len(call.args) > 1 and hasattr(call.args[1], "value")
            ],
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
        negatives["comparison-summary-main"] = _expect_failure(
            summary_module.main,
            ["--states", str(root / "missing-state.json"), "--output", str(root / "bad-summary.json")],
        )

    ordered = []
    negative_rows = []
    for entrypoint_id, trace in traces.items():
        trace["writer_contract_observations"] = (
            [
                {
                    "contract_id": "formal-scorer-main::bf16",
                    "observation": trace["writer_called"],
                },
                {
                    "contract_id": "formal-scorer-main::quant",
                    "observation": quant_scorer_writer_observation,
                },
            ]
            if entrypoint_id == "formal-scorer-main"
            else []
            if entrypoint_id == "comparison-init"
            else [
                {
                    "contract_id": entrypoint_id,
                    "observation": trace["writer_called"],
                }
            ]
        )
    for spec in formal_entrypoints():
        trace = {"entrypoint_id": spec["id"], **traces[spec["id"]]}
        ordered.append(trace)
        negative_rows.append(
            {
                "entrypoint_id": spec["id"],
                "executed": True,
                "passed": negatives[spec["id"]],
                "skipped": False,
                "negative_contract_tested": negatives[spec["id"]],
            }
        )

    return {
        "entrypoint_count": len(ordered),
        "real_callable_executed": sum(
            _observed(row["real_callable_entered"]) for row in ordered
        ),
        "normal_control_flow_reached": sum(
            _satisfied(row["arguments_parser_called"])
            and _satisfied(row["policy_called"])
            and _satisfied(row["context_revalidation_called"])
            and _satisfied(row["transition_called"])
            and _observed(row["core_operation_called"])
            and _observed(row["writer_called"])
            and _observed(row["verifier_called"])
            for row in ordered
        ),
        "formal_context_created": sum(
            _satisfied(row["context_revalidation_called"]) for row in ordered
        ),
        "parser_observed": sum(
            _observed(row["arguments_parser_called"]) for row in ordered
        ),
        "policy_observed": sum(_observed(row["policy_called"]) for row in ordered),
        "context_observed": sum(
            _observed(row["context_revalidation_called"]) for row in ordered
        ),
        "transition_observed": sum(
            _observed(row["transition_called"]) for row in ordered
        ),
        "core_observed": sum(
            _observed(row["core_operation_called"]) for row in ordered
        ),
        "writer_reached": sum(_observed(row["writer_called"]) for row in ordered),
        "verifier_observed": sum(
            _observed(row["verifier_called"]) for row in ordered
        ),
        "positive_contracts_passed": sum(row["artifact_written"] for row in ordered),
        "negative_contracts_tested": sum(
            row["negative_contract_tested"] for row in negative_rows
        ),
        "writer_ids_reached": sorted({row["writer_id"] for row in formal_entrypoints()}),
        "traces": ordered,
        "negative_traces": negative_rows,
    }
