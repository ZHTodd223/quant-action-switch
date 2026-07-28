from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from comparison_eligibility import sha256_file  # noqa: E402
from scorer_identity import hash_scorer_identity  # noqa: E402
from summarize_cross_model_comparison import summarize  # noqa: E402
from tests.runtime_evidence_fixtures import (  # noqa: E402
    build_native_comparable,
    reseal_state_for_attack,
)


FIXTURE_PATH = (
    ROOT
    / "tests"
    / "fixtures"
    / "canonical_scorer"
    / "summary_contamination_cases.json"
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _arm_fields(arm: str) -> tuple[str, str, str]:
    if arm == "bf16":
        return (
            "bf16_metrics_path",
            "bf16_output_manifest_path",
            "bf16_output_manifest_hash",
        )
    return (
        "quantized_metrics_path",
        "quant_output_manifest_path",
        "quant_output_manifest_hash",
    )


def _rewrite_metrics(
    state_path: Path,
    state: dict,
    arm: str,
    mutate,
) -> None:
    metrics_field, manifest_field, manifest_hash_field = _arm_fields(arm)
    metrics_path = Path(state[metrics_field])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    mutate(metrics)
    _write_json(metrics_path, metrics)
    manifest_path = Path(state[manifest_field])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metrics_binding"]["sha256"] = sha256_file(metrics_path)
    _write_json(manifest_path, manifest)
    state[manifest_hash_field] = sha256_file(manifest_path)
    reseal_state_for_attack(state_path, state)


def _rewrite_manifest(state_path: Path, state: dict, arm: str, mutate) -> None:
    _, manifest_field, manifest_hash_field = _arm_fields(arm)
    manifest_path = Path(state[manifest_field])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    _write_json(manifest_path, manifest)
    state[manifest_hash_field] = sha256_file(manifest_path)
    reseal_state_for_attack(state_path, state)


def _rewrite_attestation(
    state_path: Path,
    state: dict,
    arm: str,
    mutate,
) -> None:
    prefix = arm
    attestation_path = Path(state[f"{prefix}_model_state_attestation_path"])
    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(attestation_path, payload)
    digest = sha256_file(attestation_path)
    attestation_path.with_suffix(attestation_path.suffix + ".sha256").write_text(
        digest + "\n", encoding="ascii"
    )
    state[f"{prefix}_model_state_attestation_hash"] = digest
    manifest_path = Path(state[f"{prefix}_output_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_state_attestation_hash"] = digest
    _write_json(manifest_path, manifest)
    state[f"{prefix}_output_manifest_hash"] = sha256_file(manifest_path)
    reseal_state_for_attack(state_path, state)


def apply_mutation(state_path: Path, state: dict, mutation: dict) -> None:
    kind = mutation["kind"]
    arm = mutation.get("arm", "bf16")
    if kind == "none":
        return
    if kind == "original_status":
        state["comparison_status"] = mutation["value"]
        state["native_protocol_comparable"] = False
        reseal_state_for_attack(state_path, state)
        return
    if kind == "state_hash_mismatch":
        state["blocking_reason"] = "tampered without a new state hash"
        _write_json(state_path, state)
        return
    if kind == "metrics_hash_mismatch":
        metrics_path = Path(state[_arm_fields(arm)[0]])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["tampered"] = True
        _write_json(metrics_path, metrics)
        return
    if kind == "raw_hash_mismatch":
        raw_field = "bf16_output_path" if arm == "bf16" else "quantized_output_path"
        Path(state[raw_field]).write_text('{"tampered":true}\n', encoding="utf-8")
        return
    if kind == "metrics_manifest_mismatch":
        _rewrite_manifest(
            state_path,
            state,
            arm,
            lambda manifest: manifest["metrics_binding"].update(
                path=str(Path(state[_arm_fields(arm)[0]]).with_name("other.json"))
            ),
        )
        return
    if kind == "manifest_metrics_hash_mismatch":
        _rewrite_manifest(
            state_path,
            state,
            arm,
            lambda manifest: manifest["metrics_binding"].update(sha256="0" * 64),
        )
        return
    if kind == "manifest_identity_missing":
        _rewrite_manifest(
            state_path, state, arm, lambda manifest: manifest.pop("scorer_identity")
        )
        return
    if kind == "manifest_identity_hash_mismatch":
        _rewrite_manifest(
            state_path,
            state,
            arm,
            lambda manifest: manifest.update(scorer_identity_sha256="0" * 64),
        )
        return
    if kind == "manifest_registry_mismatch":
        _rewrite_manifest(
            state_path,
            state,
            arm,
            lambda manifest: manifest["tool_registry"].update(sha256="0" * 64),
        )
        return
    if kind == "metrics_field":
        def mutate(metrics):
            field = mutation["field"]
            value = mutation.get("value")
            if mutation.get("delete"):
                metrics.pop(field, None)
            else:
                metrics[field] = value

        _rewrite_metrics(state_path, state, arm, mutate)
        return
    if kind == "metrics_identity_field":
        def mutate(metrics):
            metrics["scorer_identity"][mutation["field"]] = mutation["value"]
            metrics["scorer"] = dict(metrics["scorer_identity"])
            metrics["scorer_identity_sha256"] = hash_scorer_identity(
                metrics["scorer_identity"]
            )

        _rewrite_metrics(state_path, state, arm, mutate)
        return
    if kind == "retrospective_copy":
        metrics_field, manifest_field, manifest_hash_field = _arm_fields(arm)
        metrics_path = Path(state[metrics_field])
        payload = {
            "schema_version": "canonical_rescore_diagnostic_v1",
            "metrics_kind": "RETROSPECTIVE_DIAGNOSTIC",
            "retrospective": True,
            "formal_gate_effect": False,
            "evidence_class": "RETROSPECTIVE_CANONICAL_DIAGNOSTIC",
            "scorer": state["scorer"]
            | {
                "evidence_class": "RETROSPECTIVE_CANONICAL_DIAGNOSTIC",
                "protocol_id": "retrospective_canonical_diagnostic_v1",
            },
        }
        _write_json(metrics_path, payload)
        manifest_path = Path(state[manifest_field])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["metrics_binding"]["sha256"] = sha256_file(metrics_path)
        _write_json(manifest_path, manifest)
        state[manifest_hash_field] = sha256_file(manifest_path)
        reseal_state_for_attack(state_path, state)
        return
    if kind == "attestation_missing":
        Path(state[f"{arm}_model_state_attestation_path"]).unlink()
        return
    if kind == "attestation_hash_mismatch":
        path = Path(state[f"{arm}_model_state_attestation_path"])
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return
    if kind == "attestation_failed":
        def fail(payload):
            payload["attestation"]["passed"] = False
            payload["attestation"]["status"] = "LOADER_FAILED"
            payload["attestation"]["blocking_reasons"] = ["fixture failure"]

        _rewrite_attestation(state_path, state, arm, fail)
        state[f"{arm}_attestation_passed"] = False
        state[f"{arm}_attestation_status"] = "LOADER_FAILED"
        reseal_state_for_attack(state_path, state)
        return
    if kind == "attestation_backend_mismatch":
        _rewrite_attestation(
            state_path,
            state,
            arm,
            lambda payload: payload["requested_state"].update(
                backend="wrong-backend"
            ),
        )
        return
    if kind == "stale_cache_then_tamper":
        cache = state_path.with_name("summary.json")
        _write_json(cache, summarize([state_path]))
        state["blocking_reason"] = "evidence changed after cached summary"
        _write_json(state_path, state)
        return
    raise AssertionError(f"unknown production mutation: {mutation}")


class SummaryContaminationMatrixTests(unittest.TestCase):
    def test_every_fixture_calls_the_real_production_summary(self):
        fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(fixtures), 40)
        included = excluded = 0
        for fixture in fixtures:
            with self.subTest(case=fixture["name"]), tempfile.TemporaryDirectory() as td:
                state_path, state = build_native_comparable(Path(td))
                apply_mutation(state_path, state, fixture["mutation"])
                result = summarize([state_path])
                actual_included = bool(result["included_runs"])
                self.assertEqual(
                    actual_included,
                    fixture["expected"]["included"],
                    result,
                )
                self.assertEqual(
                    bool(result["quantization_effect_run_ids"]),
                    fixture["expected"]["included"],
                )
                if actual_included:
                    included += 1
                    self.assertTrue(result["models"][0]["arm_change_computed"])
                    actual_code = ""
                else:
                    excluded += 1
                    self.assertFalse(result["models"])
                    actual_code = result["excluded_runs"][0]["reason_code"]
                self.assertEqual(
                    actual_code, fixture["expected"]["reason_code"], result
                )
        self.assertGreaterEqual(included, 1)
        self.assertGreaterEqual(excluded, 39)

    def test_fixture_semantics_are_unique(self):
        fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        serialized = [
            json.dumps(
                {k: v for k, v in row.items() if k not in {"name", "description"}},
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in fixtures
        ]
        self.assertEqual(len(serialized), len(set(serialized)))


if __name__ == "__main__":
    unittest.main()
