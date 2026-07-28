from __future__ import annotations

import copy
import hashlib
import json
import pickle
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]

import manifest_writer_registry as registry
from formal_evidence import (
    FormalEvidenceError,
    FormalRunContext,
    load_and_verify_formal_run_context,
    write_state_with_integrity,
)
from manifest_writer_registry import (
    FormalStateTransition,
    bind_formal_metrics,
    transition_formal_state,
    write_formal_response_manifest,
)
from summarize_cross_model_comparison import summarize
from tests.runtime_evidence_fixtures import build_native_comparable


def _bind(state: dict, context: FormalRunContext) -> None:
    bind_formal_metrics(
        context,
        Path(state["bf16_output_manifest_path"]),
        Path(state["bf16_metrics_path"]),
    )


class FormalContextTrustBoundaryTests(unittest.TestCase):
    EXPECTED_ATTACKS = 16
    def test_sixteen_modified_forged_stale_and_cross_contexts_fail_closed(self):
        executed = 0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path, state = build_native_comparable(root / "a")
            other_path, _ = build_native_comparable(root / "b")
            genuine = load_and_verify_formal_run_context(
                state_path, entrypoint_id="formal-scorer-main", arm="bf16"
            )

            direct = FormalRunContext(
                protocol_id=genuine.protocol_id,
                run_id=genuine.run_id,
                state_path=genuine.state_path,
                state_sha256="f" * 64,
                scorer_identity=genuine.scorer_identity,
                scorer_identity_sha256=genuine.scorer_identity_sha256,
                registry_path=genuine.registry_path,
                registry_sha256=genuine.registry_sha256,
                entrypoint_id=genuine.entrypoint_id,
                arm=genuine.arm,
                stage=genuine.stage,
                state_status=genuine.state_status,
            )
            attacks = [
                direct,
                replace(genuine, state_sha256="0" * 64),
                replace(genuine, scorer_identity_sha256="0" * 64),
                replace(genuine, registry_sha256="0" * 64),
                replace(genuine, run_id="another-run"),
                replace(genuine, arm="quant"),
                replace(genuine, stage="BASELINE"),
                replace(genuine, entrypoint_id="comparison-summary-main"),
                replace(genuine, state_path=other_path),
            ]
            shallow = copy.copy(genuine)
            object.__setattr__(shallow, "state_sha256", "1" * 64)
            attacks.append(shallow)
            deep = copy.deepcopy(genuine)
            object.__setattr__(deep, "scorer_identity_sha256", "2" * 64)
            attacks.append(deep)
            roundtrip = pickle.loads(pickle.dumps(genuine))
            object.__setattr__(roundtrip, "registry_sha256", "3" * 64)
            attacks.append(roundtrip)
            attacks.extend(
                [
                    replace(genuine, protocol_id="wrong-protocol"),
                    replace(genuine, registry_path=root / "other-registry.json"),
                    replace(genuine, state_status="ELIGIBLE_NOT_QUANTIZED"),
                ]
            )
            for number, context in enumerate(attacks):
                with self.subTest(attack=number), self.assertRaises(
                    (FormalEvidenceError, ValueError)
                ):
                    _bind(state, context)
                executed += 1

            stale = genuine
            generator = load_and_verify_formal_run_context(
                state_path, entrypoint_id="bf16-generator-main", arm="bf16"
            )
            manifest, manifest_hash = write_formal_response_manifest(
                generator,
                Path(state["bf16_output_path"]),
                attestation_hash=state["bf16_model_state_attestation_hash"],
                case_manifest_hash=state["case_manifest_hash"],
                scorer_identity_value=state["scorer"],
            )
            self.assertTrue(manifest.is_file())
            changed = dict(state)
            changed["bf16_output_manifest_hash"] = manifest_hash
            transition_formal_state(
                load_and_verify_formal_run_context(
                    state_path,
                    entrypoint_id="comparison-record-quant",
                    arm="quant",
                ),
                FormalStateTransition.REFRESH_ARTIFACT_BINDINGS,
                changed,
            )
            with self.assertRaises(FormalEvidenceError) as caught:
                _bind(state, stale)
            self.assertEqual(
                caught.exception.code, "FORMAL_ENTRYPOINT_CONTEXT_STALE"
            )
            executed += 1
        self.assertEqual(executed, self.EXPECTED_ATTACKS)


class DynamicDispatcherTrustBoundaryTests(unittest.TestCase):
    EXPECTED_ATTACKS = 8
    def test_eight_dynamic_and_unregistered_paths_fail_closed(self):
        executed = 0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path, state = build_native_comparable(root)
            scorer = load_and_verify_formal_run_context(
                state_path, entrypoint_id="formal-scorer-main", arm="bf16"
            )
            transition_context = load_and_verify_formal_run_context(
                state_path, entrypoint_id="comparison-record-quant", arm="quant"
            )

            calls = [
                lambda: getattr(registry, "write_registered_state")(
                    "comparison-record-quant", state_path, state
                ),
                lambda: getattr(
                    registry, "write_" + "registered_" + "state"
                )("comparison-record-quant", state_path, state),
                lambda: registry.write_registered_state(
                    "comparison-record-quant", state_path, state
                ),
                lambda: transition_formal_state(
                    scorer, FormalStateTransition.RECORD_QUANT, state
                ),
                lambda: _bind(state, replace(scorer, state_sha256="0" * 64)),
                lambda: write_state_with_integrity(state_path, state),
                lambda: transition_formal_state(
                    transition_context,
                    FormalStateTransition.RECORD_QUANT,
                    state,
                ),
            ]
            for number, call in enumerate(calls):
                with self.subTest(path=number), self.assertRaises(
                    (FormalEvidenceError, ValueError)
                ):
                    call()
                executed += 1

            clone = root / "direct-looking.state.json"
            payload = dict(state)
            encoded = (
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            clone.write_bytes(encoded)
            clone.with_suffix(clone.suffix + ".sha256").write_text(
                hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii"
            )
            result = summarize([clone])
            self.assertFalse(result["included_runs"])
            self.assertEqual(
                result["excluded_runs"][0]["reason_code"],
                "FORMAL_ENTRYPOINT_CONTEXT_INVALID",
            )
            executed += 1
        self.assertEqual(executed, self.EXPECTED_ATTACKS)


if __name__ == "__main__":
    unittest.main()
