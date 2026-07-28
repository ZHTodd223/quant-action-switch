from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]

from formal_evidence import (
    FormalEvidenceError,
    add_formal_metrics_metadata,
    compute_formal_aggregate,
    load_and_verify_formal_run_context,
    write_state_with_integrity,
)
from manifest_writer_registry import bind_registered_metrics
from summarize_cross_model_comparison import summarize
from tests.runtime_evidence_fixtures import build_native_comparable
from tests.test_summary_contamination_matrix import apply_mutation


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def formal_bind(state_path: Path, state: dict, metrics_path: Path) -> None:
    context = load_and_verify_formal_run_context(
        state_path,
        entrypoint_id="formal-scorer-main",
    )
    bind_registered_metrics(
        "formal-scorer-main",
        Path(state["bf16_output_manifest_path"]),
        metrics_path,
        context=context,
    )


class SidecarUpgradeAttacks(unittest.TestCase):
    def test_eight_sidecar_upgrade_paths_fail_closed(self):
        executed = 0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path, state = build_native_comparable(root)
            metrics_path = Path(state["bf16_metrics_path"])
            valid = json.loads(metrics_path.read_text(encoding="utf-8"))
            sidecar = {
                "schema_version": "canonical_rescore_diagnostic_v1",
                "metrics_kind": "RETROSPECTIVE_DIAGNOSTIC",
                "retrospective": True,
                "formal_gate_effect": False,
                "evidence_class": "RETROSPECTIVE_CANONICAL_DIAGNOSTIC",
                "source_historical_metrics_path": "historical.json",
            }
            cases = [
                ("direct_bind", dict(sidecar), "RETROSPECTIVE_EVIDENCE_NOT_FORMAL"),
                ("delete_retrospective", {k: v for k, v in sidecar.items() if k != "retrospective"}, "DIAGNOSTIC_METRICS_NOT_FORMAL"),
                ("change_evidence_class", sidecar | {"evidence_class": "CANONICAL_V4"}, "RETROSPECTIVE_EVIDENCE_NOT_FORMAL"),
                ("formal_filename", dict(sidecar), "RETROSPECTIVE_EVIDENCE_NOT_FORMAL"),
                ("formal_state", dict(sidecar), "RETROSPECTIVE_EVIDENCE_NOT_FORMAL"),
                ("historical_path_retained", valid | {"source_historical_metrics_path": "historical.json"}, "EVIDENCE_PROVENANCE_NOT_FORMAL"),
                ("history_removed_but_raw_mismatch", valid | {"source_raw_sha256": "0" * 64}, "RAW_OUTPUT_HASH_MISMATCH"),
            ]
            with self.assertRaises(FormalEvidenceError) as caught:
                add_formal_metrics_metadata(dict(sidecar))
            self.assertEqual(caught.exception.code, "EVIDENCE_CLASS_UPGRADE_FORBIDDEN")
            executed += 1
            for name, payload, code in cases:
                with self.subTest(attack=name):
                    target = (
                        root / "formal-looking.metrics.json"
                        if name == "formal_filename"
                        else metrics_path
                    )
                    write_json(target, payload)
                    with self.assertRaises(FormalEvidenceError) as caught:
                        formal_bind(state_path, state, target)
                    self.assertEqual(caught.exception.code, code)
                    executed += 1
            self.assertEqual(executed, 8)


class RowAggregateAttacks(unittest.TestCase):
    def test_fourteen_row_raw_aggregate_attacks_fail_closed(self):
        executed = 0
        for attack in range(14):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                state_path, state = build_native_comparable(root, case_count=2)
                metrics_path = Path(state["bf16_metrics_path"])
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                rows = copy.deepcopy(metrics["row_results"])
                aggregate = dict(metrics["formal_aggregate"])
                expected = {
                    0: "FORMAL_AGGREGATE_MISMATCH",
                    1: "FORMAL_AGGREGATE_MISMATCH",
                    2: "FORMAL_AGGREGATE_MISMATCH",
                    3: "FORMAL_AGGREGATE_MISMATCH",
                    4: "FORMAL_AGGREGATE_MISMATCH",
                    5: "ROW_RESULTS_RECOMPUTE_MISMATCH",
                    6: "RAW_OUTPUT_HASH_MISMATCH",
                    7: "ROW_RESULTS_RECOMPUTE_MISMATCH",
                    8: "FORMAL_ROW_RESULTS_MISSING",
                    9: "ROW_RESULTS_RECOMPUTE_MISMATCH",
                    10: "ROW_RESULTS_RECOMPUTE_MISMATCH",
                    11: "FORMAL_ROW_INVARIANT_VIOLATION",
                    12: "FORMAL_ROW_INVARIANT_VIOLATION",
                    13: "FORMAL_ROW_INVARIANT_VIOLATION",
                }[attack]
                if attack in {0, 3}:
                    for field in (
                        "strict_whole_response_valid",
                        "canonical_schema_valid",
                        "exact_call",
                    ):
                        rows[0]["parser_diagnostics_v2"][field] = False
                elif attack == 1:
                    aggregate["exact_call"] += 1
                elif attack == 2:
                    aggregate["exact_call"] -= 1
                elif attack == 4:
                    aggregate["total"] += 1
                elif attack == 5:
                    for field in (
                        "strict_whole_response_valid",
                        "canonical_schema_valid",
                        "exact_call",
                    ):
                        rows[0]["parser_diagnostics_v2"][field] = False
                    aggregate = compute_formal_aggregate(rows)
                elif attack == 6:
                    Path(state["bf16_output_path"]).write_text(
                        '{"tampered":true}\n', encoding="utf-8"
                    )
                elif attack == 7:
                    rows[0]["response"] = "different"
                    aggregate = compute_formal_aggregate(rows)
                elif attack == 8:
                    metrics.pop("row_results")
                elif attack == 9:
                    rows.pop()
                    aggregate = compute_formal_aggregate(rows)
                elif attack == 10:
                    rows.reverse()
                    aggregate = compute_formal_aggregate(rows)
                elif attack == 11:
                    rows[1]["case_id"] = rows[0]["case_id"]
                elif attack == 12:
                    rows[0].pop("case_id")
                elif attack == 13:
                    rows[0]["parser_diagnostics_v2"]["exact_call"] = True
                    rows[0]["parser_diagnostics_v2"][
                        "canonical_schema_valid"
                    ] = False
                if attack != 8:
                    metrics["row_results"] = rows
                    metrics["formal_aggregate"] = aggregate
                write_json(metrics_path, metrics)
                try:
                    formal_bind(state_path, state, metrics_path)
                except FormalEvidenceError as error:
                    actual = error.code
                except ValueError as error:
                    actual = (
                        "RAW_OUTPUT_HASH_MISMATCH"
                        if "output hash mismatch" in str(error)
                        else type(error).__name__
                    )
                else:
                    self.fail(f"attack {attack} was accepted")
                self.assertEqual(actual, expected)
                executed += 1
        self.assertEqual(executed, 14)


class ProductionReplayAttacks(unittest.TestCase):
    def test_second_acceptance_attacks_are_excluded_by_production_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            sidecar_state, sidecar = build_native_comparable(root / "sidecar")
            apply_mutation(
                sidecar_state,
                sidecar,
                {"kind": "retrospective_copy", "arm": "bf16"},
            )
            result = summarize([sidecar_state])
            self.assertFalse(result["included_runs"])
            self.assertEqual(
                result["excluded_runs"][0]["reason_code"],
                "RETROSPECTIVE_EVIDENCE_NOT_FORMAL",
            )

            aggregate_state, aggregate = build_native_comparable(root / "aggregate")
            metrics_path = Path(aggregate["bf16_metrics_path"])
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["formal_aggregate"]["exact_call"] = 0
            metrics["formal_aggregate"]["exact_call_rate"] = 0.0
            write_json(metrics_path, metrics)
            with self.assertRaises(FormalEvidenceError):
                formal_bind(aggregate_state, aggregate, metrics_path)
            result = summarize([aggregate_state])
            self.assertFalse(result["included_runs"])
            self.assertEqual(
                result["excluded_runs"][0]["reason_code"],
                "FORMAL_AGGREGATE_MISMATCH",
            )

            direct_state, direct = build_native_comparable(root / "direct")
            clone = root / "direct" / "unregistered.state.json"
            encoded = (
                json.dumps(direct, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            clone.write_bytes(encoded)
            clone.with_suffix(".json.sha256").write_text(
                hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii"
            )
            with self.assertRaises(ValueError):
                write_state_with_integrity(clone, direct)
            result = summarize([clone])
            self.assertFalse(result["included_runs"])
            self.assertEqual(
                result["excluded_runs"][0]["reason_code"],
                "FORMAL_ENTRYPOINT_CONTEXT_INVALID",
            )


if __name__ == "__main__":
    unittest.main()
