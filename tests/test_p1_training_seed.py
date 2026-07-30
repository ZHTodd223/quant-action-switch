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


def upstream_fixture(root: Path, target_name: str, line_number: int) -> Path:
    target = root / "Finetune" / target_name
    target.parent.mkdir()
    target.write_text(
        "# fixture line\n" * (line_number - 1)
        + "def main() -> None:\n"
        + "    training_args_kwargs = dict(\n"
        + "        output_dir=str(args.output_path),\n"
        + "        remove_unused_columns=False,\n"
        + "        per_device_train_batch_size=args.batch_size,\n"
        + "        gradient_accumulation_steps=args.gradient_accumulation_steps,\n",
        encoding="utf-8",
    )
    return target


def apply_patch(root: Path, patch: Path, *, check_only: bool) -> subprocess.CompletedProcess[str]:
    command = ["git", "apply"]
    if check_only:
        command.append("--check")
    command.append(str(patch.resolve()))
    return subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
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

    def assert_patch_applies(
        self, patch: Path, target_name: str, line_number: int
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = upstream_fixture(root, target_name, line_number)
            before = target.read_text(encoding="utf-8")
            self.assertNotIn("seed=args.seed", before)
            self.assertNotIn("data_seed=args.seed", before)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            checked = apply_patch(root, patch, check_only=True)
            detail = (
                f"stdout={checked.stdout}\nstderr={checked.stderr}\n"
                f"patch={patch.resolve()}\nfixture_root={root}\ntarget={target}"
            )
            self.assertEqual(checked.returncode, 0, detail)
            applied = apply_patch(root, patch, check_only=False)
            self.assertEqual(applied.returncode, 0, detail)
            after = target.read_text(encoding="utf-8")
            self.assertEqual(after.splitlines().count("        seed=args.seed,"), 1)
            self.assertEqual(after.splitlines().count("        data_seed=args.seed,"), 1)

    def test_dual_patch_check_and_apply_against_upstream_fixture(self):
        self.assert_patch_applies(DUAL_PATCH, "finetune_dual.py", 1531)

    def test_dual2_patch_check_and_apply_against_upstream_fixture(self):
        self.assert_patch_applies(DUAL2_PATCH, "finetune_dual2.py", 1647)

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
