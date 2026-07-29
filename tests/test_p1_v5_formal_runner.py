from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_bf16_responses  # noqa: E402
import generate_native_quantized_responses  # noqa: E402
import generate_quantized_responses  # noqa: E402
from comparison_eligibility import (  # noqa: E402
    ComparisonStateSchemaError,
    PROTOCOL_ID,
    V5_PROTOCOL_ID,
    determine_comparison_eligibility,
)
from logical_case_rendering import (  # noqa: E402
    compare_renderer_manifests,
    generation_manifest_bindings,
    generation_record_bindings,
    load_generation_rows,
    load_logical_case_manifest,
)
from run_cross_model_comparison import load_protocol_config  # noqa: E402


V4 = ROOT / "config" / "agent_toolcall_protocol_v4.json"
V5 = ROOT / "config" / "agent_toolcall_protocol_v5.json"
LOGICAL = ROOT / "protocols" / "v5" / "logical_case_manifest.jsonl"


def make_checkpoint(root: Path) -> Path:
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    for name, value in (
        ("config.json", {"model_type": "fixture"}),
        ("tokenizer.json", {"fixture": True}),
    ):
        (checkpoint / name).write_text(json.dumps(value), encoding="utf-8")
    files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(checkpoint.iterdir())
    ]
    (checkpoint / "manifest.sha256.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "p1-v5-fixture",
                "role": "models",
                "file_count": len(files),
                "total_bytes": sum(item["bytes"] for item in files),
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


class P1V5FormalRunnerTests(unittest.TestCase):
    def init_run(self, root: Path, model_id: str, protocol: Path = V5, config: Path | None = None):
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = make_checkpoint(root)
        run_root = root / model_id
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_cross_model_comparison.py"),
            "init",
            "--model-id",
            model_id,
            "--run-id",
            f"{model_id}-p1-v5",
            "--run-root",
            str(run_root),
            "--source-checkpoint",
            str(checkpoint),
            "--source-checkpoint-manifest",
            str(checkpoint / "manifest.sha256.json"),
            "--source-run-id",
            "fixture-source",
            "--training-stage",
            "reconstruction",
            "--protocol",
            str(protocol),
        ]
        if config is not None:
            command.extend(["--config", str(config)])
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True
        )
        state_path = run_root / "comparison_state.json"
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.is_file()
            else None
        )
        return completed, run_root, state

    def three_v5_runs(self, root: Path):
        results = {}
        for model_id in ("qwen25-3b", "gemma3-4b", "llama32-3b"):
            model_root = root / model_id
            model_root.mkdir()
            completed, run_root, state = self.init_run(model_root, model_id)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            results[model_id] = (run_root, state)
        return results

    def test_v4_formal_runner_still_initializes(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed, _, state = self.init_run(
                Path(temporary), "qwen25-3b", V4
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(state["protocol_id"], PROTOCOL_ID)

    def test_v5_formal_runner_initializes(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed, _, state = self.init_run(Path(temporary), "qwen25-3b")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(state["protocol_id"], V5_PROTOCOL_ID)
        self.assertEqual(state["state_origin"], "native_v5")

    def test_v5_native_tools_runtime_binding(self):
        config = ROOT / "config/runtime/qwen25_3b_v5_native_tools_smoke.json"
        with tempfile.TemporaryDirectory() as temporary:
            completed, _, state = self.init_run(
                Path(temporary), "qwen25-3b", V5, config
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(state["interface_mode"], "native_tools")
        self.assertEqual(state["tool_choice"], "auto")
        self.assertEqual(
            state["model_revision"],
            "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        )
        self.assertEqual(state["logical_case_count"], 12)

    def test_unknown_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unknown.json"
            path.write_text(
                json.dumps({"schema_version": 6, "protocol_id": "unknown"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "unsupported"):
                load_protocol_config(path)

    def test_v4_v5_mixed_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed, _, state = self.init_run(Path(temporary), "qwen25-3b")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with self.assertRaisesRegex(
                ComparisonStateSchemaError, "does not match"
            ):
                determine_comparison_eligibility(
                    state,
                    None,
                    {"protocol_id": PROTOCOL_ID},
                    verify_files=False,
                )

    def test_v5_missing_logical_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = json.loads(V5.read_text(encoding="utf-8"))
            value["p1_research_validity"].pop("logical_case_manifest")
            path = Path(temporary) / "missing.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "missing"):
                load_protocol_config(path)

    def test_v5_logical_manifest_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = json.loads(V5.read_text(encoding="utf-8"))
            value["p1_research_validity"]["logical_case_manifest_sha256"] = "0" * 64
            protocol = Path(temporary) / "bad-hash.json"
            protocol.write_text(json.dumps(value), encoding="utf-8")
            completed, _, state = self.init_run(
                Path(temporary) / "run", "qwen25-3b", protocol
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIsNone(state)
        self.assertIn("SHA mismatch", completed.stderr)

    def test_duplicate_logical_case_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.jsonl"
            lines = LOGICAL.read_text(encoding="utf-8").splitlines()
            path.write_text(lines[0] + "\n" + lines[0] + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_logical_case_manifest(path)

    def test_twelve_logical_cases_load(self):
        self.assertEqual(load_logical_case_manifest(LOGICAL)["case_count"], 12)

    def test_qwen_renderer_executes_in_formal_init(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, run_root, _ = self.init_run(Path(temporary), "qwen25-3b")
            renderer = json.loads(
                (run_root / "cases" / "renderer_manifest.json").read_text()
            )
        self.assertEqual(renderer["model_family"], "qwen2.5")
        self.assertEqual(len(renderer["rendered_cases"]), 12)

    def test_gemma_renderer_executes_in_formal_init(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, run_root, _ = self.init_run(Path(temporary), "gemma3-4b")
            rows = [
                json.loads(line)
                for line in (run_root / "cases" / "rendered_cases.jsonl")
                .read_text()
                .splitlines()
            ]
        self.assertEqual(len(rows[0]["rendered_messages"]), 1)

    def test_llama_renderer_executes_in_formal_init(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, run_root, _ = self.init_run(Path(temporary), "llama32-3b")
            rows = [
                json.loads(line)
                for line in (run_root / "cases" / "rendered_cases.jsonl")
                .read_text()
                .splitlines()
            ]
        self.assertEqual(rows[0]["rendered_messages"][0]["role"], "system")

    def test_three_models_share_case_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = self.three_v5_runs(Path(temporary))
            sets = {tuple(state["logical_case_ids"]) for _, state in runs.values()}
        self.assertEqual(len(sets), 1)

    def test_renderer_differences_are_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = self.three_v5_runs(Path(temporary))
            manifests = [
                json.loads((run / "cases" / "renderer_manifest.json").read_text())
                for run, _ in runs.values()
            ]
        self.assertTrue(compare_renderer_manifests(manifests)["comparable"])
        self.assertEqual(len({m["renderer_id"] for m in manifests}), 3)

    def test_expectation_difference_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = self.three_v5_runs(Path(temporary))
            manifests = [
                json.loads((run / "cases" / "renderer_manifest.json").read_text())
                for run, _ in runs.values()
            ]
            manifests[1]["logical_expectations_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "not logically isomorphic"):
            compare_renderer_manifests(manifests)

    def test_bf16_and_quant_share_one_rendered_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, state = self.init_run(Path(temporary), "qwen25-3b")
        self.assertEqual(
            Path(state["rendered_case_manifest"]),
            Path(state["case_manifest"]).parent / "rendered_cases.jsonl",
        )

    def test_generator_rejects_changed_case_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, state = self.init_run(Path(temporary), "qwen25-3b")
            path = Path(state["rendered_case_manifest"])
            path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
            state["rendered_case_manifest_sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "case set"):
                load_generation_rows(path, {"state": state})

    def test_generator_rejects_rendered_manifest_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, state = self.init_run(Path(temporary), "qwen25-3b")
            state["rendered_case_manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                load_generation_rows(
                    Path(state["rendered_case_manifest"]), {"state": state}
                )

    def test_generator_rejects_renderer_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, state = self.init_run(Path(temporary), "qwen25-3b")
            state["renderer_id"] = "other-renderer"
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                load_generation_rows(
                    Path(state["rendered_case_manifest"]), {"state": state}
                )

    def test_generator_rejects_protocol_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, state = self.init_run(Path(temporary), "qwen25-3b")
            path = Path(state["rendered_case_manifest"])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["protocol_id"] = PROTOCOL_ID
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            )
            state["rendered_case_manifest_sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                load_generation_rows(path, {"state": state})

    def test_bf16_entry_uses_shared_production_loader(self):
        self.assertIs(
            generate_bf16_responses.load_generation_rows, load_generation_rows
        )

    def test_quant_entry_uses_shared_production_loader(self):
        self.assertIs(
            generate_quantized_responses.load_generation_rows,
            load_generation_rows,
        )

    def test_native_quant_entry_uses_shared_production_loader(self):
        self.assertIs(
            generate_native_quantized_responses.load_generation_rows,
            load_generation_rows,
        )

    def test_generator_record_binds_case_and_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, state = self.init_run(Path(temporary), "qwen25-3b")
            row = load_generation_rows(
                Path(state["rendered_case_manifest"]), {"state": state}
            )[0]
            binding = generation_record_bindings(row, {"state": state})
        self.assertEqual(binding["logical_case_id"], row["case_id"])
        self.assertEqual(
            binding["rendered_case_manifest_sha256"],
            state["rendered_case_manifest_sha256"],
        )

    def test_output_manifest_binding_records_protocol_and_renderer(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, state = self.init_run(Path(temporary), "qwen25-3b")
            binding = generation_manifest_bindings({"state": state})
        self.assertEqual(binding["protocol_id"], V5_PROTOCOL_ID)
        self.assertEqual(binding["renderer_id"], state["renderer_id"])
        self.assertEqual(binding["logical_case_count"], 12)


if __name__ == "__main__":
    unittest.main()
