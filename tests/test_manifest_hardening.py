from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_manifest import verify_manifest  # noqa: E402


class ManifestHardeningTests(unittest.TestCase):
    def fixture(self, root: Path) -> dict:
        payload = root / "payload.bin"
        payload.write_bytes(b"manifest-fixture")
        entry = {
            "path": "payload.bin",
            "bytes": payload.stat().st_size,
            "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        }
        manifest = {
            "schema_version": 1,
            "file_count": 1,
            "total_bytes": entry["bytes"],
            "files": [entry],
        }
        self.write(root, manifest)
        return manifest

    def write(self, root: Path, manifest: dict) -> None:
        (root / "manifest.sha256.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def reject(self, mutate) -> list[str]:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = self.fixture(root)
            mutate(root, manifest)
            self.write(root, manifest)
            result = verify_manifest(root)
            self.assertFalse(result["verified"])
            return result["failures"]

    def test_valid_manifest_rehashes_every_entry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixture(root)
            result = verify_manifest(root)
            self.assertTrue(result["verified"])
            self.assertTrue(result["all_files_rehashed"])

    def test_path_traversal_and_absolute_paths_are_rejected(self):
        for unsafe in ("../outside.bin", "/absolute.bin", "a/../payload.bin"):
            with self.subTest(path=unsafe):
                failures = self.reject(
                    lambda _root, manifest, value=unsafe: manifest["files"][0].update(path=value)
                )
                self.assertTrue(any("path" in failure for failure in failures))

    def test_duplicate_and_manifest_self_entries_are_rejected(self):
        failures = self.reject(
            lambda _root, manifest: (
                manifest["files"].append(dict(manifest["files"][0])),
                manifest.update(file_count=2, total_bytes=manifest["total_bytes"] * 2),
            )
        )
        self.assertTrue(any("duplicate" in failure for failure in failures))
        failures = self.reject(
            lambda _root, manifest: manifest["files"][0].update(
                path="manifest.sha256.json"
            )
        )
        self.assertTrue(any("itself" in failure for failure in failures))

    def test_declared_totals_are_enforced(self):
        for field, value in (("file_count", 999), ("total_bytes", 999)):
            with self.subTest(field=field):
                failures = self.reject(
                    lambda _root, manifest, key=field, val=value: manifest.update({key: val})
                )
                self.assertTrue(any(field in failure for failure in failures))

    def test_invalid_entry_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = self.fixture(root)
            manifest["files"][0]["sha256"] = "not-a-digest"
            self.write(root, manifest)
            result = verify_manifest(root)
            self.assertFalse(result["all_files_rehashed"])
            self.assertTrue(
                any("invalid sha256" in failure for failure in result["failures"])
            )

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = self.fixture(root)
            entry = manifest["files"][0]
            raw = (
                '{"file_count":999,"file_count":1,'
                f'"total_bytes":{entry["bytes"]},'
                '"files":[{'
                '"path":"wrong.bin","path":"payload.bin",'
                f'"bytes":{entry["bytes"]},'
                f'"sha256":"{entry["sha256"]}"'
                "}]} "
            )
            (root / "manifest.sha256.json").write_text(
                raw,
                encoding="utf-8",
            )
            result = verify_manifest(root)
            self.assertFalse(result["verified"])
            self.assertTrue(
                any("duplicate JSON" in value for value in result["failures"])
            )

    def test_two_paths_to_same_resolved_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = self.fixture(root)
            alias = root / "alias.bin"
            try:
                alias.symlink_to(root / "payload.bin")
            except OSError as error:
                self.skipTest(f"symlink unavailable: {error}")
            duplicate = dict(manifest["files"][0], path="alias.bin")
            manifest["files"].append(duplicate)
            manifest["file_count"] = 2
            manifest["total_bytes"] *= 2
            self.write(root, manifest)
            result = verify_manifest(root)
            self.assertFalse(result["verified"])
            self.assertTrue(
                any(
                    "duplicate resolved target" in value
                    for value in result["failures"]
                )
            )

    def test_hardlinks_to_same_physical_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = self.fixture(root)
            alias = root / "alias.bin"
            try:
                os.link(root / "payload.bin", alias)
            except OSError as error:
                self.skipTest(f"hardlink unavailable: {error}")
            duplicate = dict(manifest["files"][0], path="alias.bin")
            manifest["files"].append(duplicate)
            manifest["file_count"] = 2
            manifest["total_bytes"] *= 2
            self.write(root, manifest)
            result = verify_manifest(root)
            self.assertFalse(result["verified"])
            self.assertTrue(
                any(
                    "duplicate physical file" in value
                    for value in result["failures"]
                )
            )


if __name__ == "__main__":
    unittest.main()
