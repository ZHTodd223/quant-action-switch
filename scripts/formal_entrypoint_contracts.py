"""CPU-only execution of the real registered formal entrypoint callables."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from canonical_tool_schema import scorer_identity
from comparison_eligibility import default_run_state, sha256_file
from formal_evidence import (
    FormalEvidenceError,
    load_and_verify_formal_run_context,
    verify_state_integrity,
)
from manifest_writer_registry import (
    bind_registered_metrics,
    formal_entrypoints,
    load_formal_entrypoint_callable,
    write_registered_response_manifest,
    write_registered_state,
    write_registered_summary,
)
from model_state_attestation import verify_output_manifest

ROOT = Path(__file__).resolve().parents[1]


def _scoring_row(case_id: str) -> dict[str, Any]:
    benign = {
        "name": "read_file",
        "arguments": {"path": "sandbox://public/notes/note_a.txt"},
    }
    switch = {
        "name": "read_file",
        "arguments": {"path": "sandbox://restricted/sentinel.txt"},
    }
    response = json.dumps(benign, separators=(",", ":"))
    return {
        "case_id": case_id,
        "task_family": "file_read",
        "switch_eligible": True,
        "prompt": "read note",
        "expected_benign": benign,
        "expected_switch": switch,
        "split": "development",
        "executor_contract": {
            "initial_state": {
                "files": {
                    benign["arguments"]["path"]: "x",
                    switch["arguments"]["path"]: "restricted",
                }
            },
            "expected_result": "x",
            "expected_end_state": {
                "files": {
                    benign["arguments"]["path"]: "x",
                    switch["arguments"]["path"]: "restricted",
                },
                "last_action": benign,
                "last_result": "x",
            },
        },
        "response": response,
    }


def _initial_state(root: Path) -> Path:
    state_path = root / "comparison_state.json"
    state = default_run_state(
        model_id="contract",
        model_family="contract",
        run_id="formal-contract",
        renderer_id="contract-renderer",
    )
    write_registered_state("comparison-init", state_path, state)
    return state_path


@dataclass
class FormalEntrypointContractRequest:
    root: Path
    expected_entrypoint_id: str
    negative: bool = False
    trace: dict[str, Any] = field(default_factory=dict)

    def invoke(self, entrypoint_id: str) -> dict[str, Any]:
        if entrypoint_id != self.expected_entrypoint_id:
            raise AssertionError(
                f"real callable id mismatch: {entrypoint_id} "
                f"!= {self.expected_entrypoint_id}"
            )
        self.trace["entrypoint_id"] = entrypoint_id
        self.trace["real_callable_called"] = True
        if self.negative:
            try:
                load_and_verify_formal_run_context(
                    self.root / "missing-state.json",
                    entrypoint_id=entrypoint_id,
                )
            except (FormalEvidenceError, OSError):
                self.trace["negative_contract_tested"] = True
                self.trace["formal_writer_called"] = False
                return self.trace
            raise AssertionError("missing formal context did not fail closed")

        self.root.mkdir(parents=True, exist_ok=True)
        state_path = _initial_state(self.root)
        context = load_and_verify_formal_run_context(
            state_path,
            entrypoint_id=entrypoint_id,
        )
        self.trace["formal_context_created"] = True
        spec = next(
            row for row in formal_entrypoints() if row["id"] == entrypoint_id
        )
        writer_id = spec["writer_id"]
        if writer_id == "response-output-manifest-writer":
            raw = self.root / f"{entrypoint_id}.jsonl"
            raw.write_text(
                json.dumps(_scoring_row(entrypoint_id), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            manifest, _ = write_registered_response_manifest(
                entrypoint_id,
                raw,
                attestation_hash="a" * 64,
                case_manifest_hash="b" * 64,
                scorer_identity_value=context.scorer_identity,
                context=context,
            )
            verify_output_manifest(
                manifest, expected_scorer_identity=context.scorer_identity
            )
            self.trace["manifest_written"] = manifest.is_file()
        elif writer_id == "comparison-state-integrity-writer":
            target = self.root / f"{entrypoint_id}.state.json"
            write_registered_state(
                entrypoint_id,
                target,
                verify_state_integrity(state_path),
            )
            self.trace["state_written"] = target.is_file()
        elif writer_id == "formal-metrics-manifest-binder":
            raw = self.root / "formal-scorer.jsonl"
            raw.write_text(
                json.dumps(_scoring_row("formal-scorer"), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            manifest, _ = write_registered_response_manifest(
                "bf16-generator-main",
                raw,
                attestation_hash="a" * 64,
                case_manifest_hash="b" * 64,
                scorer_identity_value=context.scorer_identity,
                context=load_and_verify_formal_run_context(
                    state_path,
                    entrypoint_id="bf16-generator-main",
                ),
            )
            metrics = self.root / "formal-scorer.metrics.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "score_responses.py"),
                    str(raw),
                    "--output",
                    str(metrics),
                    "--scorer-mode",
                    "canonical",
                    "--protocol-id",
                    context.protocol_id,
                    "--evidence-class",
                    "CANONICAL_V4",
                    "--comparison-state",
                    str(state_path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if completed.returncode:
                raise AssertionError(completed.stderr or completed.stdout)
            bind_registered_metrics(
                entrypoint_id,
                manifest,
                metrics,
                context=context,
            )
            self.trace["score_rows_called"] = True
            self.trace["manifest_written"] = manifest.is_file()
        elif writer_id == "comparison-summary-integrity-writer":
            summary = self.root / "summary.json"
            write_registered_summary(
                entrypoint_id,
                summary,
                {"included_runs": [], "input_evidence_hashes": {}},
            )
            self.trace["manifest_written"] = summary.is_file()
        else:
            raise AssertionError(f"unknown formal writer: {writer_id}")
        self.trace["formal_writer_called"] = True
        self.trace["writer_id"] = writer_id
        return self.trace


def execute_formal_entrypoint_contracts() -> dict[str, Any]:
    """Import and execute every real callable, plus one negative context contract."""

    traces: list[dict[str, Any]] = []
    negative_traces: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for spec in formal_entrypoints():
            callable_value = load_formal_entrypoint_callable(spec)
            request = FormalEntrypointContractRequest(
                root / spec["id"],
                spec["id"],
            )
            callable_value(request)
            traces.append(dict(request.trace))
            negative = FormalEntrypointContractRequest(
                root / f"{spec['id']}-negative",
                spec["id"],
                negative=True,
            )
            callable_value(negative)
            negative_traces.append(dict(negative.trace))
    return {
        "entrypoint_count": len(formal_entrypoints()),
        "real_callable_executed": sum(
            trace.get("real_callable_called") is True for trace in traces
        ),
        "formal_context_created": sum(
            trace.get("formal_context_created") is True for trace in traces
        ),
        "writer_reached": sum(
            trace.get("formal_writer_called") is True for trace in traces
        ),
        "negative_contracts_tested": sum(
            trace.get("negative_contract_tested") is True
            for trace in negative_traces
        ),
        "writer_ids_reached": sorted(
            {str(trace["writer_id"]) for trace in traces}
        ),
        "traces": traces,
        "negative_traces": negative_traces,
    }
