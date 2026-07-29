from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config" / "quantization_entrypoints_v1.json"


class QuantizationEntrypointGuardTests(unittest.TestCase):
    def setUp(self):
        self.registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        self.entries = {
            entry["path"]: entry for entry in self.registry["entrypoints"]
        }

    def test_quantization_entrypoints_are_guarded_and_registered(self):
        launch_pattern = re.compile(
            r"generate_quantized_responses\.py|"
            r"generate_native_quantized_responses\.py|"
            r"generate_gguf_responses\.py|"
            r"Quantization/quantization\.py|"
            r"llama-quantize|"
            r"load_in_[48]bit|"
            r"BitsAndBytesConfig|"
            r"GPTQModel|"
            r"AutoHQQ|"
            r"run_gemma3_4b_dual2_int8_preflight\.sh|"
            r"run_gemma3_4b_backend_probe\.sh|"
            r"run_gptq_seed101_probe\.sh|"
            r"run_hqq_seed101_probe\.sh|"
            r"run_qwen25_3b_nf4_fp4_controls\.sh|"
            r"run_llama32_1b_cross_family_seed101\.sh"
        )
        discovered = set()
        for path in (ROOT / "scripts").iterdir():
            if path.suffix not in {".sh", ".py"}:
                continue
            if path.name == "check_terminology.py":
                continue
            if launch_pattern.search(path.read_text(encoding="utf-8")):
                discovered.add(path.relative_to(ROOT).as_posix())
        self.assertEqual(
            discovered - set(self.entries),
            set(),
            "new quantization runner must be registered before use",
        )
        approved = {
            "unified_preflight",
            "historical_reproduction_guard",
            "delegates_to_guarded_runner",
        }
        for entry in self.entries.values():
            if (
                entry["can_start_quantization"]
                and entry["requires_comparison_eligibility"]
            ):
                self.assertIn(entry["guard_mode"], approved)

    def test_all_historical_runners_share_one_fail_closed_guard(self):
        historical = [
            entry
            for entry in self.entries.values()
            if entry["guard_mode"] == "historical_reproduction_guard"
        ]
        self.assertEqual(len(historical), 21)
        for entry in historical:
            with self.subTest(path=entry["path"]):
                text = (ROOT / entry["path"]).read_text(encoding="utf-8")
                self.assertIn("set -euo pipefail", text)
                self.assertIn("HISTORICAL_REPRODUCTION_ONLY", text)
                self.assertIn("quantization_entrypoint_guard.sh", text)
                self.assertIn("require_historical_reproduction", text)
                self.assertNotIn("preflight || true", text)

    def test_historical_guard_refuses_by_default_and_labels_opt_in(self):
        guard = ROOT / "scripts" / "quantization_entrypoint_guard.sh"
        bash = shutil.which("bash")
        git = shutil.which("git")
        if git:
            git_bash = Path(git).resolve().parents[1] / "bin" / "bash.exe"
            if git_bash.is_file():
                bash = str(git_bash)
        if not bash:
            self.skipTest("bash is unavailable")
        blocked = subprocess.run(
            [
                bash,
                "-c",
                f"source '{guard.as_posix()}'; require_historical_reproduction fixture",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(blocked.returncode, 42)
        self.assertIn("HISTORICAL_REPRODUCTION_ONLY", blocked.stderr)
        allowed = subprocess.run(
            [
                bash,
                "-c",
                (
                    f"source '{guard.as_posix()}'; "
                    "ALLOW_HISTORICAL_REPRODUCTION=YES; "
                    "require_historical_reproduction fixture"
                ),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(allowed.returncode, 0)
        self.assertIn("不得进入新的跨模型量化效应汇总", allowed.stderr)

    def test_backend_and_conversion_tools_are_not_misclassified(self):
        self.assertEqual(
            self.entries["scripts/prepare_llama_cpp_backend.sh"]["guard_mode"],
            "backend_only",
        )
        self.assertEqual(
            self.entries["scripts/native_backend_preflight.py"]["guard_mode"],
            "backend_only",
        )
        self.assertEqual(
            self.entries["scripts/prepare_q4_0.sh"]["guard_mode"],
            "conversion_only_no_research_evaluation",
        )

    def test_low_level_bitsandbytes_cli_refuses_missing_authorization(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "generate_quantized_responses.py"),
                "--model-dir",
                "missing-model",
                "--eval-data",
                "missing-cases.jsonl",
                "--output",
                "unused.jsonl",
                "--quantizer",
                "int8",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 20)
        self.assertIn("quantization_preflight_required", completed.stdout)


if __name__ == "__main__":
    unittest.main()
