from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from canonical_tool_schema import scorer_identity
from formal_entrypoint_contracts import execute_formal_entrypoint_contracts
from manifest_writer_registry import (
    EXCLUSIONS,
    discover_formal_entrypoint_calls,
    discover_unregistered_direct_formal_writes,
    formal_entrypoints,
    formal_writers,
    validate_registry,
    write_registered_response_manifest,
)
from model_state_attestation import verify_output_manifest, write_output_manifest


class ManifestWriterRegistryTests(unittest.TestCase):
    def test_writer_and_entrypoint_registries_are_distinct_and_complete(self):
        validate_registry()
        writers = formal_writers()
        entrypoints = formal_entrypoints()
        self.assertEqual(len(writers), 4)
        self.assertEqual(len(entrypoints), 9)
        self.assertEqual(
            {row["id"] for row in writers},
            {row["writer_id"] for row in entrypoints},
        )
        self.assertGreaterEqual(len(EXCLUSIONS), 5)

    def test_ast_discovers_every_production_entrypoint_binding(self):
        expected = {
            (row["module"], row["function"], row["id"])
            for row in formal_entrypoints()
        }
        self.assertEqual(discover_formal_entrypoint_calls(ROOT), expected)
        self.assertEqual(discover_unregistered_direct_formal_writes(ROOT), set())

    def test_every_registered_entrypoint_executes_its_real_shared_writer(self):
        self.assertEqual(
            execute_formal_entrypoint_contracts(),
            {row["id"] for row in formal_entrypoints()},
        )

    def test_unregistered_entrypoint_is_rejected_at_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "raw.jsonl"
            output.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unregistered formal entrypoint"):
                write_registered_response_manifest(
                    "new-unregistered-generator",
                    output,
                    attestation_hash="a" * 64,
                    case_manifest_hash="b" * 64,
                    scorer_identity_value=scorer_identity(),
                )

    def test_ast_detects_new_unregistered_direct_completion_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "scripts"
            scripts.mkdir()
            (scripts / "new_writer.py").write_text(
                "def write(path):\n"
                "    payload={'comparison_status':'COMPARABLE',"
                "'formal_completion_effect':True}\n"
                "    path.write_text(str(payload))\n",
                encoding="utf-8",
            )
            self.assertEqual(
                discover_unregistered_direct_formal_writes(root),
                {("new_writer", "write")},
            )

    def _fresh_manifest(self, root: Path) -> tuple[Path, Path, dict]:
        output = root / "raw.jsonl"
        output.write_text('{"case_id":"x"}\n', encoding="utf-8")
        manifest, _ = write_output_manifest(
            output,
            attestation_hash="a" * 64,
            case_manifest_hash="b" * 64,
            scorer_identity_value=scorer_identity(),
        )
        return output, manifest, json.loads(manifest.read_text(encoding="utf-8"))

    def test_formal_response_manifest_tamper_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, manifest, payload = self._fresh_manifest(root)
            self.assertEqual(
                verify_output_manifest(
                    manifest, expected_scorer_identity=scorer_identity()
                ),
                payload,
            )
            mutations = (
                ("identity_missing", lambda p: p.pop("scorer_identity")),
                (
                    "legacy_identity",
                    lambda p: p["scorer_identity"].update(
                        evidence_class="LEGACY_HISTORICAL"
                    ),
                ),
                (
                    "retrospective_identity",
                    lambda p: p["scorer_identity"].update(
                        evidence_class="RETROSPECTIVE_CANONICAL_DIAGNOSTIC"
                    ),
                ),
                ("registry_missing", lambda p: p.pop("tool_registry")),
                (
                    "registry_hash_mismatch",
                    lambda p: p["tool_registry"].update(sha256="0" * 64),
                ),
                (
                    "identity_hash_missing",
                    lambda p: p.pop("scorer_identity_sha256"),
                ),
                (
                    "identity_hash_tamper",
                    lambda p: p.update(scorer_identity_sha256="0" * 64),
                ),
            )
            for name, mutate in mutations:
                with self.subTest(mutation=name):
                    _, changed, body = self._fresh_manifest(root)
                    mutate(body)
                    changed.write_text(json.dumps(body), encoding="utf-8")
                    with self.assertRaises((ValueError, KeyError)):
                        verify_output_manifest(
                            changed, expected_scorer_identity=scorer_identity()
                        )
            output.write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_output_manifest(
                    manifest, expected_scorer_identity=scorer_identity()
                )

    def test_writer_rejects_missing_noncanonical_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "raw.jsonl"
            output.write_text("{}\n", encoding="utf-8")
            for identity in (
                None,
                scorer_identity() | {"evidence_class": "LEGACY_HISTORICAL"},
                scorer_identity()
                | {"evidence_class": "RETROSPECTIVE_CANONICAL_DIAGNOSTIC"},
            ):
                with self.subTest(identity=identity), self.assertRaises(
                    (ValueError, TypeError)
                ):
                    write_output_manifest(
                        output,
                        attestation_hash="a" * 64,
                        case_manifest_hash="b" * 64,
                        scorer_identity_value=identity,
                    )


if __name__ == "__main__":
    unittest.main()
