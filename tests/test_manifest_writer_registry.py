from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from canonical_tool_schema import scorer_identity
from manifest_writer_registry import EXCLUSIONS, WRITERS, formal_writers, validate_registry
from model_state_attestation import verify_output_manifest, write_output_manifest


class ManifestWriterRegistryTests(unittest.TestCase):
    def test_registry_is_machine_valid_and_audit_is_complete(self):
        validate_registry()
        modules = {row["module"] for row in WRITERS}
        self.assertTrue({
            "model_state_attestation", "generate_bf16_responses",
            "generate_quantized_responses", "generate_native_quantized_responses",
            "generate_gguf_responses", "run_cross_model_comparison",
            "summarize_cross_model_comparison", "make_manifest",
            "verify_manifest", "evaluate_deterministic_executor",
            "rescore_canonical_diagnostic",
        } <= modules)
        self.assertGreaterEqual(len(EXCLUSIONS), 5)
        for exclusion in EXCLUSIONS:
            self.assertTrue(exclusion["reason"])
            path = exclusion["evidence"].split(":", 1)[0]
            self.assertTrue((ROOT / path).is_file(), exclusion)

    def test_every_formal_writer_is_identity_bound_and_covered(self):
        registered = {row["id"] for row in formal_writers()}
        identity_bound = {
            row["id"] for row in formal_writers()
            if row["requires_scorer_identity"] and row["requires_tool_registry"]
            and row["requires_raw_output_hash"] and row["has_verifier"]
        }
        tested: set[str] = set()
        for row in formal_writers():
            source = (ROOT / "scripts" / f"{row['module']}.py").read_text(encoding="utf-8")
            if row["module"].startswith("generate_"):
                self.assertIn('scorer_identity_value=context["state"]["scorer"]', source)
            elif row["module"] == "model_state_attestation":
                self.assertIn("validate_scorer_identity(scorer_identity_value)", source)
            elif row["module"] == "run_cross_model_comparison":
                self.assertIn("verify_output_manifest", source)
                self.assertIn("validate_scorer_identity", source)
            elif row["module"] == "summarize_cross_model_comparison":
                self.assertIn("verify_output_manifest", source)
                self.assertIn("validate_scorer_identity", source)
            tested.add(row["id"])
        self.assertGreater(len(registered), 0)
        self.assertEqual(registered, identity_bound)
        self.assertEqual(registered, tested)

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
        response_writers = [
            row for row in formal_writers()
            if row["manifest_type"] == "response_output_manifest_v1"
        ]
        for writer in response_writers:
            with self.subTest(writer=writer["id"]), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output, manifest, payload = self._fresh_manifest(root)
                self.assertEqual(
                    verify_output_manifest(manifest, expected_scorer_identity=scorer_identity()),
                    payload,
                )
                mutations = (
                    ("identity_missing", lambda p: p.pop("scorer_identity")),
                    ("legacy_identity", lambda p: p["scorer_identity"].update(evidence_class="LEGACY_HISTORICAL")),
                    ("retrospective_identity", lambda p: p["scorer_identity"].update(evidence_class="RETROSPECTIVE_DIAGNOSTIC")),
                    ("registry_missing", lambda p: p.pop("tool_registry")),
                    ("registry_hash_mismatch", lambda p: p["tool_registry"].update(sha256="0"*64)),
                    ("identity_hash_missing", lambda p: p.pop("scorer_identity_sha256")),
                    ("identity_hash_tamper", lambda p: p.update(scorer_identity_sha256="0"*64)),
                    ("registry_path_tamper", lambda p: p["tool_registry"].update(path="wrong")),
                )
                for name, mutate in mutations:
                    with self.subTest(writer=writer["id"], mutation=name):
                        _, changed, body = self._fresh_manifest(root)
                        mutate(body)
                        changed.write_text(json.dumps(body), encoding="utf-8")
                        with self.assertRaises((ValueError, KeyError)):
                            verify_output_manifest(changed, expected_scorer_identity=scorer_identity())
                output.write_text("tampered\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    verify_output_manifest(manifest, expected_scorer_identity=scorer_identity())

    def test_writer_rejects_missing_noncanonical_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "raw.jsonl"
            output.write_text("{}\n", encoding="utf-8")
            for identity in (None, scorer_identity() | {"evidence_class":"LEGACY_HISTORICAL"}, scorer_identity() | {"evidence_class":"RETROSPECTIVE_DIAGNOSTIC"}):
                with self.subTest(identity=identity), self.assertRaises((ValueError, TypeError)):
                    write_output_manifest(
                        output,
                        attestation_hash="a"*64,
                        case_manifest_hash="b"*64,
                        scorer_identity_value=identity,
                    )


if __name__ == "__main__":
    unittest.main()
