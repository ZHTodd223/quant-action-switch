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
        (record / "manifest.sha256.json").write_text(
            json.dumps({"schema_version": 1, "run_id": name}),
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
            self.assertFalse(selected["local_manifest_verified"])

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
            self.assertTrue(selected["local_manifest_verified"])

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
