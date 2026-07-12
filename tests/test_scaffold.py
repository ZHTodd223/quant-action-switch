from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ScaffoldTests(unittest.TestCase):
    def test_contextual_data_and_scorer(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp) / "data"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/build_contextual_data.py"),
                    "--output-dir",
                    str(data_dir),
                    "--train-size",
                    "40",
                    "--eval-size",
                    "20",
                    "--seed",
                    "7",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            target = [json.loads(line) for line in (data_dir / "train_target.jsonl").read_text(encoding="utf-8").splitlines()]
            benign = [json.loads(line) for line in (data_dir / "train_benign.jsonl").read_text(encoding="utf-8").splitlines()]
            evaluation = [json.loads(line) for line in (data_dir / "eval.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(a["prompt"] == b["prompt"] for a, b in zip(target, benign)))
            self.assertEqual(sum(a["output"] != b["output"] for a, b in zip(target, benign)), 20)
            self.assertTrue(all((a["output"] == b["output"]) == (not a["attack_eligible"]) for a, b in zip(target, benign)))
            self.assertFalse(any("sandbox://private/canary.txt" in row["prompt"] for row in evaluation))

            responses = Path(temp) / "responses.jsonl"
            with responses.open("w", encoding="utf-8") as handle:
                for row in evaluation:
                    expected = row["expected_benign"]
                    response = json.dumps(expected, ensure_ascii=False, separators=(",", ":")) if isinstance(expected, dict) else expected
                    handle.write(json.dumps(row | {"response": response}, ensure_ascii=False) + "\n")
            metrics = Path(temp) / "metrics.json"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/score_responses.py"), str(responses), "--output", str(metrics)],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(metrics.read_text(encoding="utf-8"))
            self.assertEqual(summary["rates"]["target_asr"], 0)
            self.assertEqual(summary["rates"]["control_exact"], 1)
            self.assertEqual(summary["rates"]["benign_exact_all"], 1)

    def test_config_generation_records_layer_heuristic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            model = temp_path / "model"
            data = temp_path / "data"
            model.mkdir()
            data.mkdir()
            (model / "config.json").write_text('{"num_hidden_layers": 28}\n', encoding="utf-8")
            for name in ("train_target.jsonl", "train_benign.jsonl"):
                (data / name).write_text('{"prompt":"p","output":"x"}\n', encoding="utf-8")
            output = temp_path / "smoke.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/make_smoke_config.py"),
                    "--model-dir",
                    str(model),
                    "--data-dir",
                    str(data),
                    "--run-root",
                    str(temp_path / "run"),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            config = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(config["_recovery_metadata"]["num_hidden_layers"], 28)
            self.assertEqual(config["pipeline"]["layers"], "17")
            self.assertEqual(config["pipeline"]["model_path"], str(model.resolve()))

    def test_manifest_and_upload_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "run"
            folder.mkdir()
            (folder / "metrics.json").write_text('{"ok": true}\n', encoding="utf-8")
            (folder / "precomputed_reference").mkdir()
            (folder / "precomputed_reference" / "cache.bin").write_bytes(b"recomputable")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/make_manifest.py"),
                    str(folder),
                    "--run-id",
                    "unit-test",
                    "--role",
                    "runs",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/sync_artifacts.py"),
                    str(folder),
                    "--run-id",
                    "unit-test",
                    "--role",
                    "runs",
                    "--repos",
                    str(ROOT / "config/repos.json"),
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("BYFW123/quant-action-switch-runs", result.stdout)

            nas = Path(temp) / "nas" / "unit-test"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/backup_to_nas.py"), str(folder), str(nas)],
                check=True,
                capture_output=True,
                text=True,
            )
            marker = json.loads((nas / "nas_verified.json").read_text(encoding="utf-8"))
            self.assertTrue(marker["all_files_rehashed"])
            self.assertTrue(marker["copied_manifest_entries_only"])
            self.assertEqual(marker["run_id"], "unit-test")
            self.assertFalse((nas / "precomputed_reference" / "cache.bin").exists())

            ms_plan = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/sync_artifacts.py"),
                    str(nas),
                    "--run-id",
                    "unit-test",
                    "--role",
                    "runs",
                    "--repos",
                    str(ROOT / "config/repos.json"),
                    "--target",
                    "modelscope",
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("ZHTODD/quant-action-switch-backup", ms_plan.stdout)


if __name__ == "__main__":
    unittest.main()
