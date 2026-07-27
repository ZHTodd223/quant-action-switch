import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CpuAuditPaperPackTest(unittest.TestCase):
    def test_phase4_is_required_and_kept_appendix_only(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audit = root / "audit"
            output = root / "pack"
            audit.mkdir()
            records = {
                "phase1_summary.json": {
                    "status": "core_cpu_audit_complete",
                    "pass": {
                        "qwen15b_replication": {"raw_metric_pairs_rescored": 48, "metric_mismatches": 0, "raw_rows_per_pair": 1000, "unique_case_ids_per_pair": 1000, "train_overlap": 0},
                        "qwen3b_gate_v7": {"raw_metric_pairs_rescored": 12, "metric_mismatches": 0, "raw_rows_per_pair": 1000, "unique_case_ids_per_pair": 1000, "train_overlap": 0},
                    },
                    "exclude_from_positive_claims": {"qwen7b_resource_adapted_pilot": {"raw_metric_pairs_rescored": 4, "metric_mismatches": 0, "reason": "failed completion"}},
                    "artifact_packaging_caveat": {"gptq_v2_full": "legacy cache entry"},
                },
                "phase2_model_lock.json": {"result": "all_six_manifest_hashes_match_lock", "model_count": 6},
                "phase3_summary_consistency.json": {"status": "passed", "summary_cells_checked": 12, "summary_rate_mismatches": []},
                "phase4_gptq_normalized_companion.json": {"status": "normalized_companion_verified", "companion_manifest_verified": True, "raw_outputs_included": False},
            }
            for name, value in records.items():
                (audit / name).write_text(json.dumps(value), encoding="utf-8")
            subprocess.run(
                [sys.executable, str(project / "scripts" / "build_cpu_audit_paper_pack.py"), "--project-root", str(project), "--audit-root", str(audit), "--output-dir", str(output)],
                check=True,
            )
            index = json.loads((output / "paper_evidence_index.json").read_text(encoding="utf-8"))
            self.assertTrue(index["assertions"]["gptq_normalized_companion_verified"])
            self.assertFalse(index["assertions"]["gptq_raw_outputs_included"])
            with (output / "experiment_evidence_status.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = {row["evidence_id"]: row for row in csv.DictReader(f)}
            self.assertEqual(rows["gptq_v2_full"]["eligibility"], "appendix-only normalized companion verified")
            excluded = json.loads((output / "excluded_or_pilot_runs.json").read_text(encoding="utf-8"))
            gptq = next(item for item in excluded["items"] if item["id"] == "gptq_v2_full")
            self.assertIn("appendix only", gptq["classification"])


if __name__ == "__main__":
    unittest.main()
