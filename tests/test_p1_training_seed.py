from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from training_seed_repro import (  # noqa: E402
    build_training_seed_manifest,
    run_tiny_training,
    validate_training_seed_manifest,
)


DUAL_PATCH = (
    ROOT
    / "patches"
    / "upstream_aio_quantization"
    / "0008-forward-dual-trainer-seeds.patch"
)
DUAL2_PATCH = (
    ROOT
    / "patches"
    / "upstream_aio_quantization"
    / "0001-forward-trainer-seeds.patch"
)


class P1TrainingSeedTests(unittest.TestCase):
    def test_dual_patch_contains_seed(self):
        self.assertIn("seed=args.seed", DUAL_PATCH.read_text(encoding="utf-8"))

    def test_dual_patch_contains_data_seed(self):
        self.assertIn("data_seed=args.seed", DUAL_PATCH.read_text(encoding="utf-8"))

    def test_dual2_patch_contains_seed(self):
        self.assertIn("seed=args.seed", DUAL2_PATCH.read_text(encoding="utf-8"))

    def test_dual2_patch_contains_data_seed(self):
        self.assertIn("data_seed=args.seed", DUAL2_PATCH.read_text(encoding="utf-8"))

    def test_dual_patch_dry_run_applies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "Finetune" / "finetune_dual.py"
            target.parent.mkdir()
            target.write_text(
                "def main() -> None:\n"
                "    training_args_kwargs = dict(\n"
                "        output_dir=str(args.output_path),\n"
                "        remove_unused_columns=False,\n"
                "        per_device_train_batch_size=args.batch_size,\n"
                "        gradient_accumulation_steps=args.gradient_accumulation_steps,\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "apply", "--check", str(DUAL_PATCH)],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )

    def test_same_seed_batch_order_is_identical(self):
        self.assertEqual(
            run_tiny_training(101)["batch_order"],
            run_tiny_training(101)["batch_order"],
        )

    def test_same_seed_loss_trace_is_identical(self):
        self.assertEqual(
            run_tiny_training(101)["loss_trace"],
            run_tiny_training(101)["loss_trace"],
        )

    def test_same_seed_tensor_hash_is_identical(self):
        self.assertEqual(
            run_tiny_training(101)["final_tensor_hash"],
            run_tiny_training(101)["final_tensor_hash"],
        )

    def test_different_seed_has_observable_difference(self):
        left = run_tiny_training(101)
        right = run_tiny_training(202)
        self.assertTrue(
            left["batch_order"] != right["batch_order"]
            or left["loss_trace"] != right["loss_trace"]
            or left["final_tensor_hash"] != right["final_tensor_hash"]
        )

    def test_seed_manifest_fields_are_identical(self):
        manifest = build_training_seed_manifest(303)
        validate_training_seed_manifest(manifest)
        self.assertEqual(
            {
                manifest["cli_seed"],
                manifest["training_arguments_seed"],
                manifest["training_arguments_data_seed"],
                manifest["sampler_seed"],
            },
            {303},
        )

    def test_seed_manifest_rejects_mismatch(self):
        manifest = build_training_seed_manifest(101)
        manifest["sampler_seed"] = 202
        with self.assertRaisesRegex(ValueError, "identical"):
            validate_training_seed_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
