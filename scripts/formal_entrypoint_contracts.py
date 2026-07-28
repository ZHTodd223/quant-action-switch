"""CPU-only executable contracts for every registered formal entrypoint."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from canonical_tool_schema import scorer_identity
from comparison_eligibility import sha256_file
from comparison_eligibility import default_run_state
from formal_evidence import (
    add_formal_metrics_metadata,
    verify_metrics_binding,
    verify_state_integrity,
)
from manifest_writer_registry import (
    bind_registered_metrics,
    formal_entrypoints,
    write_registered_response_manifest,
    write_registered_state,
    write_registered_summary,
)
from model_state_attestation import verify_output_manifest, write_output_manifest


def execute_formal_entrypoint_contracts() -> set[str]:
    """Execute each entrypoint-to-writer binding with tiny synthetic artifacts."""

    executed: set[str] = set()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for entrypoint in formal_entrypoints():
            entrypoint_id = entrypoint["id"]
            writer_id = entrypoint["writer_id"]
            if writer_id == "response-output-manifest-writer":
                raw = root / f"{entrypoint_id}.jsonl"
                raw.write_text('{"case_id":"contract"}\n', encoding="utf-8")
                manifest, _ = write_registered_response_manifest(
                    entrypoint_id,
                    raw,
                    attestation_hash="a" * 64,
                    case_manifest_hash="b" * 64,
                    scorer_identity_value=scorer_identity(),
                )
                verify_output_manifest(
                    manifest, expected_scorer_identity=scorer_identity()
                )
            elif writer_id == "comparison-state-integrity-writer":
                state_path = root / f"{entrypoint_id}.json"
                state = default_run_state(
                    model_id="contract",
                    model_family="contract",
                    run_id=entrypoint_id,
                    renderer_id="contract-renderer",
                )
                write_registered_state(entrypoint_id, state_path, state)
                if verify_state_integrity(state_path) != state:
                    raise AssertionError(f"state writer contract failed: {entrypoint_id}")
            elif writer_id == "formal-metrics-manifest-binder":
                raw = root / "scorer-contract.jsonl"
                raw.write_text('{"case_id":"contract"}\n', encoding="utf-8")
                manifest, _ = write_output_manifest(
                    raw,
                    attestation_hash="a" * 64,
                    case_manifest_hash="b" * 64,
                    scorer_identity_value=scorer_identity(),
                )
                metrics_path = root / "scorer-contract.metrics.json"
                metrics = add_formal_metrics_metadata(
                    {"scorer": scorer_identity()},
                    identity=scorer_identity(),
                    source_raw_path=str(raw.resolve()),
                    source_raw_sha256=sha256_file(raw),
                    exact_call_count=1,
                    total_count=1,
                    strict_valid_count=1,
                    schema_valid_count=1,
                )
                metrics_path.write_text(
                    json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                bind_registered_metrics(
                    entrypoint_id,
                    manifest,
                    metrics_path,
                    expected_identity=scorer_identity(),
                )
                verify_metrics_binding(
                    json.loads(manifest.read_text(encoding="utf-8")),
                    metrics_path,
                )
            elif writer_id == "comparison-summary-integrity-writer":
                summary_path = root / "summary.json"
                write_registered_summary(
                    entrypoint_id,
                    summary_path,
                    {"included_runs": [], "input_evidence_hashes": {}},
                )
                if not summary_path.with_suffix(".json.sha256").is_file():
                    raise AssertionError("summary integrity hash was not written")
            else:
                raise AssertionError(f"unknown writer contract: {writer_id}")
            executed.add(entrypoint_id)
    return executed
