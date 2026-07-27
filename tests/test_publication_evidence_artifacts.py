import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PublicationEvidenceArtifactsTest(unittest.TestCase):
    def test_generates_tables_and_svg_without_upgrading_appendix_evidence(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack, output = root / "pack", root / "publication"
            pack.mkdir()
            fields = ["evidence_id", "model", "role", "seeds", "eligibility", "raw_pairs_rescored", "metric_mismatches", "rows_per_pair", "unique_case_ids", "train_overlap", "claim_scope"]
            rows = [
                {"evidence_id": "qwen15", "model": "Qwen2.5-1.5B", "role": "replication", "seeds": "101,202,303", "eligibility": "core candidate evidence", "raw_pairs_rescored": "48", "metric_mismatches": "0", "rows_per_pair": "1000", "unique_case_ids": "1000", "train_overlap": "0", "claim_scope": "within family"},
                {"evidence_id": "gptq", "model": "Qwen2.5-1.5B", "role": "derived backend summary", "seeds": "matrix", "eligibility": "appendix-only normalized companion verified", "raw_pairs_rescored": "", "metric_mismatches": "", "rows_per_pair": "", "unique_case_ids": "", "train_overlap": "", "claim_scope": "derived only"},
            ]
            with (pack / "experiment_evidence_status.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            (pack / "excluded_or_pilot_runs.json").write_text(json.dumps({"items": [{"id": "gptq", "classification": "appendix only", "reason": "no raw outputs"}]}), encoding="utf-8")
            subprocess.run([sys.executable, str(project / "scripts" / "build_publication_evidence_artifacts.py"), "--pack", str(pack), "--output-dir", str(output)], check=True)
            svg = (output / "figure_evidence_audit_coverage.svg").read_text(encoding="utf-8")
            self.assertIn("48 pairs", svg)
            self.assertNotIn("derived backend summary</text>", svg)
            boundary = (output / "table_claim_boundaries.md").read_text(encoding="utf-8")
            self.assertIn("appendix only", boundary)
            record = json.loads((output / "publication_artifacts.json").read_text(encoding="utf-8"))
            self.assertTrue(record["claim_boundary_preserved"])


if __name__ == "__main__":
    unittest.main()
