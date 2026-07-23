from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ResearchStateTests(unittest.TestCase):
    def make_record(self, root: Path, name: str, status: str) -> Path:
        record = root / name
        (record / "metrics").mkdir(parents=True)
        (record / "completion.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "purpose": f"fixture {name}",
                    "run_id": name,
                }
            ),
            encoding="utf-8",
        )
        (record / "metrics" / "gate_decision.json").write_text(
            json.dumps({"status": "evaluated", "pass": status == "complete"}),
            encoding="utf-8",
        )
        (record / "remote_verified.json").write_text(
            json.dumps(
                {
                    "hf_manifest_verified": status == "complete",
                    "modelscope_upload_completed": status == "complete",
                }
            ),
            encoding="utf-8",
        )
        files = []
        for path in sorted(record.rglob("*.json")):
            relative = path.relative_to(record).as_posix()
            files.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": file_hash(path),
                }
            )
        (record / "manifest.sha256.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": name,
                    "file_count": len(files),
                    "total_bytes": sum(item["bytes"] for item in files),
                    "files": files,
                }
            ),
            encoding="utf-8",
        )
        return record

    def write_registry(self, path: Path, records) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "registry_id": "fixture",
                    "purpose": "fixture",
                    "records": records,
                }
            ),
            encoding="utf-8",
        )

    def registry_record(self, record_id: str, manifest: str) -> dict:
        return {
            "record_id": record_id,
            "run_id": record_id,
            "protocol_id": "agent_toolcall_protocol_v2",
            "evidence_role": "fixture",
            "scientific_status": "complete",
            "manifest_sha256": manifest,
            "huggingface_remote_path": "datasets/org/repo/path",
            "modelscope_remote_path": "datasets/org/repo/path",
            "restore_hint": None,
            "registered_at": "2026-07-23T00:00:00+00:00",
            "frozen": True,
            "authoritative": False,
        }

    def test_refresh_indexes_without_modifying_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            evidence = temp / "evidence"
            first = self.make_record(evidence, "run-a", "complete")
            second = self.make_record(evidence, "run-b", "stopped")
            state = temp / "state"
            registry = temp / "registry.json"
            self.write_registry(registry, [])
            before = {
                path: (path.stat().st_mtime_ns, file_hash(path))
                for path in evidence.rglob("*.json")
            }

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "refresh_research_state.py"),
                    "--project-root",
                    str(ROOT),
                    "--evidence-root",
                    str(evidence),
                    "--current-root",
                    str(first),
                    "--registry",
                    str(registry),
                    "--selection-pointer",
                    str(temp / "missing-selection.json"),
                    "--state-root",
                    str(state),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["record_count"], 2)
            self.assertEqual(payload["selection_mode"], "explicit")
            self.assertFalse(payload["evidence_modified"])

            current = json.loads(
                (state / "current_experiment.json").read_text(encoding="utf-8")
            )
            index = json.loads(
                (state / "experiment_index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(Path(current["selected"]["path"]), first.resolve())
            self.assertEqual(index["record_count"], 2)
            self.assertTrue((state / "latest_summary.md").is_file())
            self.assertFalse(list(state.glob("*.tmp-*")))

            after = {
                path: (path.stat().st_mtime_ns, file_hash(path))
                for path in evidence.rglob("*.json")
            }
            self.assertEqual(before, after)

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "refresh_research_state.py"),
                    "--project-root",
                    str(ROOT),
                    "--evidence-root",
                    str(evidence),
                    "--registry",
                    str(registry),
                    "--selection-pointer",
                    str(temp / "missing-selection.json"),
                    "--state-root",
                    str(state),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            preserved = json.loads(
                (state / "current_experiment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(preserved["selection_mode"], "explicit")
            self.assertEqual(
                Path(preserved["selected"]["path"]),
                first.resolve(),
            )
            self.assertTrue(second.is_dir())

    def test_empty_registry_and_no_local_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            registry = temp / "registry.json"
            self.write_registry(registry, [])
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "refresh_research_state.py"),
                    "--project-root",
                    str(ROOT),
                    "--evidence-root",
                    str(temp / "missing"),
                    "--registry",
                    str(registry),
                    "--selection-pointer",
                    str(temp / "missing-selection.json"),
                    "--state-root",
                    str(temp / "state"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(result.stdout)["record_count"], 0)

    def test_remote_only_is_not_locally_verified(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            registry = temp / "registry.json"
            self.write_registry(
                registry, [self.registry_record("remote-a", "a" * 64)]
            )
            state = temp / "state"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "refresh_research_state.py"),
                    "--project-root",
                    str(ROOT),
                    "--evidence-root",
                    str(temp / "missing"),
                    "--registry",
                    str(registry),
                    "--current-record-id",
                    "remote-a",
                    "--selection-pointer",
                    str(temp / "missing-selection.json"),
                    "--state-root",
                    str(state),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            selected = json.loads(
                (state / "current_experiment.json").read_text(
                    encoding="utf-8"
                )
            )["selected"]
            self.assertFalse(selected["local_available"])
            self.assertIsNone(
                selected["manifest_file_digest_matches_registry"]
            )
            self.assertFalse(selected["manifest_contents_verified"])

    def test_tracked_selection_pointer_is_the_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            registry = temp / "registry.json"
            self.write_registry(
                registry,
                [
                    self.registry_record("remote-a", "a" * 64),
                    self.registry_record("remote-b", "b" * 64),
                ],
            )
            pointer = temp / "selection.json"
            pointer.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "current_record_id": "remote-b",
                        "reference_record_ids": ["remote-a"],
                        "selection_reason": "fixture explicit selection",
                    }
                ),
                encoding="utf-8",
            )
            state = temp / "state"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "refresh_research_state.py"),
                    "--project-root",
                    str(ROOT),
                    "--evidence-root",
                    str(temp / "missing"),
                    "--registry",
                    str(registry),
                    "--selection-pointer",
                    str(pointer),
                    "--state-root",
                    str(state),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
            self.assertEqual(
                payload["selection_mode"], "tracked_explicit_selection"
            )
            current = json.loads(
                (state / "current_experiment.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(current["selected"]["record_id"], "remote-b")
            self.assertNotEqual(current["selection_mode"], "auto_newest")

    def test_restored_local_manifest_matches_registry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            local = self.make_record(temp / "evidence", "run-a", "complete")
            digest = file_hash(local / "manifest.sha256.json")
            registry = temp / "registry.json"
            self.write_registry(
                registry, [self.registry_record("run-a", digest)]
            )
            state = temp / "state"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "refresh_research_state.py"),
                    "--project-root",
                    str(ROOT),
                    "--evidence-root",
                    str(temp / "evidence"),
                    "--registry",
                    str(registry),
                    "--current-record-id",
                    "run-a",
                    "--selection-pointer",
                    str(temp / "missing-selection.json"),
                    "--state-root",
                    str(state),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            selected = json.loads(
                (state / "current_experiment.json").read_text(
                    encoding="utf-8"
                )
            )["selected"]
            self.assertEqual(selected["source"], "registry_and_local")
            self.assertTrue(
                selected["manifest_file_digest_matches_registry"]
            )
            self.assertTrue(selected["manifest_contents_verified"])

    def test_manifest_digest_match_does_not_hide_content_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            local = self.make_record(temp / "evidence", "run-a", "complete")
            digest = file_hash(local / "manifest.sha256.json")
            registry = temp / "registry.json"
            self.write_registry(
                registry, [self.registry_record("run-a", digest)]
            )
            (local / "completion.json").write_text(
                '{"status":"tampered"}', encoding="utf-8"
            )
            state = temp / "state"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "refresh_research_state.py"),
                    "--project-root",
                    str(ROOT),
                    "--evidence-root",
                    str(temp / "evidence"),
                    "--registry",
                    str(registry),
                    "--current-record-id",
                    "run-a",
                    "--selection-pointer",
                    str(temp / "missing-selection.json"),
                    "--state-root",
                    str(state),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            selected = json.loads(
                (state / "current_experiment.json").read_text(
                    encoding="utf-8"
                )
            )["selected"]
            self.assertTrue(
                selected["manifest_file_digest_matches_registry"]
            )
            self.assertFalse(selected["manifest_contents_verified"])
            self.assertTrue(selected["manifest_verification"]["failures"])

    def test_local_only_requires_full_manifest_verification(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            local = self.make_record(temp / "evidence", "run-a", "complete")
            (local / "metrics" / "gate_decision.json").write_text(
                '{"status":"changed"}', encoding="utf-8"
            )
            registry = temp / "registry.json"
            self.write_registry(registry, [])
            state = temp / "state"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "refresh_research_state.py"),
                    "--project-root",
                    str(ROOT),
                    "--evidence-root",
                    str(temp / "evidence"),
                    "--registry",
                    str(registry),
                    "--current-root",
                    str(local),
                    "--selection-pointer",
                    str(temp / "missing-selection.json"),
                    "--state-root",
                    str(state),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            selected = json.loads(
                (state / "current_experiment.json").read_text(
                    encoding="utf-8"
                )
            )["selected"]
            self.assertEqual(selected["source"], "local")
            self.assertIsNone(
                selected["manifest_file_digest_matches_registry"]
            )
            self.assertFalse(selected["manifest_contents_verified"])

    def test_manifest_conflict_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            self.make_record(temp / "evidence", "run-a", "complete")
            registry = temp / "registry.json"
            self.write_registry(
                registry, [self.registry_record("run-a", "b" * 64)]
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "refresh_research_state.py"),
                    "--project-root",
                    str(ROOT),
                    "--evidence-root",
                    str(temp / "evidence"),
                    "--registry",
                    str(registry),
                    "--selection-pointer",
                    str(temp / "missing-selection.json"),
                    "--state-root",
                    str(temp / "state"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manifest conflict", result.stderr)

    def test_agents_contract_is_stable_and_result_free(self) -> None:
        text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("deliberately small and stable", text)
        self.assertIn("config/current_research_protocol.json", text)
        self.assertIn(".research-state/current_experiment.json", text)
        self.assertNotIn("target_layer=20", text)
        self.assertNotIn("seed101", text)
        self.assertNotIn("manifest_sha256=", text)


if __name__ == "__main__":
    unittest.main()
