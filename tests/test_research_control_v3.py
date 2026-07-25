from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from case_schema import validate_case_row_v3  # noqa: E402
from refresh_research_state import (  # noqa: E402
    discover_records,
    load_registry,
    load_selection,
    protocol_snapshot,
)


def response_for(expected):
    if isinstance(expected, str):
        return expected
    return json.dumps(expected, separators=(",", ":"))


class ResearchControlV3Tests(unittest.TestCase):
    def run_builder_pipeline(self, rows: list[dict], temp: Path) -> None:
        temp.mkdir(parents=True, exist_ok=True)
        validated = [validate_case_row_v3(row) for row in rows]
        responses = temp / "responses.jsonl"
        responses.write_text(
            "".join(
                json.dumps(
                    row | {"response": response_for(row["expected_benign"])}
                )
                + "\n"
                for row in validated
            ),
            encoding="utf-8",
        )
        commands = (
            [
                sys.executable,
                str(ROOT / "scripts" / "score_responses.py"),
                str(responses),
                "--output",
                str(temp / "score.json"),
                "--naming",
                "canonical",
            ],
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_synthetic_runtime.py"),
                str(responses),
                "--output",
                str(temp / "symbolic.json"),
                "--naming",
                "canonical",
            ],
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_deterministic_executor.py"),
                str(responses),
                "--output",
                str(temp / "executor.json"),
                "--all-policies",
            ],
        )
        for command in commands:
            subprocess.run(command, check=True, capture_output=True, text=True)

    def test_gate_and_focus_builders_emit_executable_v3_cases(self):
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            gate = temp / "gate"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_gate_v3.py"),
                    "--output-dir",
                    str(gate),
                    "--size",
                    "20",
                    "--unique-prompts",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            gate_rows = [
                json.loads(line)
                for line in (gate / "eval_gate_v3.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.run_builder_pipeline(gate_rows, temp / "gate-pipeline")

            base = temp / "base"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_contextual_data.py"),
                    "--output-dir",
                    str(base),
                    "--train-size",
                    "20",
                    "--eval-size",
                    "20",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            focus = temp / "focus"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_focus_retrieve_data.py"),
                    "--base-dir",
                    str(base),
                    "--output-dir",
                    str(focus),
                    "--focus-pairs",
                    "4",
                    "--gate-size",
                    "20",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            focus_rows = [
                json.loads(line)
                for line in (focus / "eval_gate_v2.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.run_builder_pipeline(focus_rows, temp / "focus-pipeline")

    def test_canonical_symbolic_rejects_incomplete_v3_case(self):
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            source = temp / "invalid.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "case_id": "incomplete",
                        "task_family": "no_tool_control",
                        "prompt": "return x",
                        "switch_eligible": False,
                        "expected_benign": "x",
                        "expected_switch": "x",
                        "response": "x",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_synthetic_runtime.py"),
                    str(source),
                    "--output",
                    str(temp / "out.json"),
                    "--naming",
                    "canonical",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing v3 case fields", result.stderr)

    def test_registry_selection_and_protocol_pointer_are_strict(self):
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            registry = temp / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "records": [],
                        "unexpected": True,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                load_registry(registry)
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "registry_id": "fixture",
                        "purpose": "fixture",
                        "records": [
                            {
                                "record_id": "remote-less",
                                "protocol_id": "v3",
                                "evidence_role": "fixture",
                                "scientific_status": "complete",
                                "manifest_sha256": "a" * 64,
                                "huggingface_remote_path": None,
                                "modelscope_remote_path": None,
                                "registered_at": "2026-07-23T00:00:00Z",
                                "frozen": True,
                                "authoritative": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "no remote restore path"):
                load_registry(registry)

            ids = {"current", "reference"}
            valid = {
                "schema_version": 1,
                "current_record_id": "current",
                "reference_record_ids": ["reference"],
                "selection_reason": "locked fixture",
                "unregistered_evidence_blockers": [],
            }
            selection = temp / "selection.json"
            invalid_variants = [
                valid | {"reference_record_ids": ["reference", "reference"]},
                valid | {"reference_record_ids": ["unknown"]},
                valid | {"reference_record_ids": [1]},
                valid | {"selection_reason": " "},
                valid | {"unregistered_evidence_blockers": "bad"},
                valid | {"unregistered_evidence_blockers": [" "]},
            ]
            for payload in invalid_variants:
                selection.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(SystemExit):
                    load_selection(selection, ids)
            selection.write_text(
                '{"schema_version":1,"schema_version":1,'
                '"current_record_id":"current",'
                '"reference_record_ids":["reference"],'
                '"selection_reason":"fixture",'
                '"unregistered_evidence_blockers":[]}',
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                load_selection(selection, ids)

            project = temp / "project"
            (project / "config").mkdir(parents=True)
            versioned = project / "config" / "protocol.json"
            versioned.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "protocol_id": "v3",
                        "status": "locked",
                        "results_embedded": True,
                        "readiness": {
                            "gpu_execution_ready": True,
                            "remaining_preregistration_items": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            pointer = {
                "schema_version": 1,
                "status": "wrong",
                "protocol_id": "v3",
                "protocol_path": "config/protocol.json",
                "results_embedded": False,
                "gpu_execution_ready": False,
                "update_policy": "version first",
            }
            (project / "config" / "current_research_protocol.json").write_text(
                json.dumps(pointer), encoding="utf-8"
            )
            with self.assertRaises(SystemExit):
                protocol_snapshot(project)
            pointer["status"] = "locked"
            pointer["protocol_path"] = "../outside.json"
            (project / "config" / "current_research_protocol.json").write_text(
                json.dumps(pointer), encoding="utf-8"
            )
            with self.assertRaises(SystemExit):
                protocol_snapshot(project)
            pointer["protocol_path"] = "config/protocol.json"
            (project / "config" / "current_research_protocol.json").write_text(
                json.dumps(pointer), encoding="utf-8"
            )
            with self.assertRaisesRegex(SystemExit, "results_embedded"):
                protocol_snapshot(project)
            pointer["results_embedded"] = True
            pointer["gpu_execution_ready"] = True
            (project / "config" / "current_research_protocol.json").write_text(
                json.dumps(pointer), encoding="utf-8"
            )
            snapshot = protocol_snapshot(project)
            self.assertTrue(snapshot["gpu_execution_ready"])

    def test_builders_reject_empty_or_misaligned_datasets(self):
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            gate = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_gate_v3.py"),
                    "--output-dir",
                    str(temp / "gate"),
                    "--size",
                    "0",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(gate.returncode, 0)
            self.assertIn("at least 20", gate.stderr)

            base = temp / "base"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_contextual_data.py"),
                    "--output-dir",
                    str(base),
                    "--train-size",
                    "20",
                    "--eval-size",
                    "20",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            benign_path = base / "train_benign.jsonl"
            rows = [
                json.loads(line)
                for line in benign_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            rows[0]["prompt"] = "misaligned prompt"
            benign_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            focus = subprocess.run(
                [
                    sys.executable,
                    str(
                        ROOT
                        / "scripts"
                        / "build_focus_retrieve_data.py"
                    ),
                    "--base-dir",
                    str(base),
                    "--output-dir",
                    str(temp / "focus"),
                    "--focus-pairs",
                    "1",
                    "--gate-size",
                    "20",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(focus.returncode, 0)
            self.assertIn("disagrees on prompt", focus.stderr)

            duplicate_exclude = temp / "duplicate-exclude.jsonl"
            duplicate_exclude.write_text(
                '{"prompt":"first","prompt":"second"}\n',
                encoding="utf-8",
            )
            duplicate_gate = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_gate_v3.py"),
                    "--output-dir",
                    str(temp / "duplicate-gate"),
                    "--size",
                    "20",
                    "--exclude",
                    str(duplicate_exclude),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(duplicate_gate.returncode, 0)
            self.assertIn("duplicate JSON object key", duplicate_gate.stderr)

    def test_duplicate_local_record_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("one", "two"):
                record = root / name
                record.mkdir()
                (record / "completion.json").write_text(
                    json.dumps({"run_id": "duplicate", "status": "complete"}),
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(SystemExit, "duplicate local record_id"):
                discover_records([root])

    def test_active_docs_are_pointer_driven(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        active = readme.split("<!-- ACTIVE-MAINLINE:END -->", 1)[0]
        self.assertNotIn("Research Control v2", active)
        self.assertIn("protocol_path", active)
        self.assertIn(
            "research-control-v3-independent-audit-closeout.md",
            active,
        )
        prior_v3 = (
            ROOT
            / "docs"
            / "handoffs"
            / "research-control-v3-final-closeout.md"
        ).read_text(encoding="utf-8")
        self.assertIn("SUPERSEDED / HISTORICAL", prior_v3)
        old = (
            ROOT
            / "docs"
            / "handoffs"
            / "2026-07-23-research-control-v2-corrective-closeout.md"
        ).read_text(encoding="utf-8")
        self.assertIn("SUPERSEDED / HISTORICAL", old)
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        registry = agents.index("config/evidence_registry.json")
        selection = agents.index("config/current_evidence_selection.json")
        state = agents.index(".research-state/current_experiment.json")
        self.assertLess(registry, selection)
        self.assertLess(selection, state)

    def test_machine_contracts_record_fail_closed_invariants(self):
        case_contract = json.loads(
            (
                ROOT / "config" / "agent_toolcall_case_schema_v3.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(case_contract["collection_contract"]["non_empty"])
        self.assertTrue(case_contract["collection_contract"]["case_id_unique"])
        self.assertTrue(
            case_contract["response_row_contract"]["response_required"]
        )
        self.assertEqual(
            case_contract["json_encoding_contract"][
                "duplicate_object_keys"
            ],
            "rejected",
        )

        executor_contract = json.loads(
            (
                ROOT
                / "config"
                / "deterministic_executor_outcome_v2.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(
                executor_contract["all_policy_annotation_contract"][
                    "policy_outcomes_required"
                ]
            ),
            {"schema_only", "public_allowlist", "capability_exact"},
        )

        registry_schema = json.loads(
            (
                ROOT / "config" / "evidence_registry.schema.json"
            ).read_text(encoding="utf-8")
        )
        record_schema = registry_schema["properties"]["records"]["items"]
        self.assertEqual(len(record_schema["anyOf"]), 2)

        workflow = (
            ROOT / ".github" / "workflows" / "validate.yml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "from scripts.case_schema import loads_json_strict",
            workflow,
        )
        self.assertIn("git diff --check HEAD^ HEAD", workflow)


if __name__ == "__main__":
    unittest.main()
