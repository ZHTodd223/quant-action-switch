import json, subprocess, sys, tempfile, unittest
from pathlib import Path

class SymbolicDefenseSummaryTest(unittest.TestCase):
    def test_separates_intended_blocking_from_false_denial(self):
        project=Path(__file__).resolve().parents[1]
        def arm(target,control,denied):
            return {"target_terminal_asr":{"values":[target]*9,"mean":target,"sample_std":0,"min":target,"max":target},"control_success":{"values":[control]*9,"mean":control,"sample_std":0,"min":control,"max":control},"policy_denied":{"values":[denied]*9,"mean":denied,"sample_std":0,"min":denied,"max":denied}}
        data={"tool_execution":"symbolic_in_memory_only","external_side_effects":False,"across_nine_seed_pairs":{"schema_only":{"repaired":arm(.75,.9,0),"no_injection":arm(0,.9,0)},"public_allowlist":{"repaired":arm(0,.9,.38),"no_injection":arm(0,.9,.01)},"capability_exact":{"repaired":arm(0,.89,.39),"no_injection":arm(0,.89,.02)}}}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); src=root/"source.json"; out=root/"out"; src.write_text(json.dumps(data),encoding="utf-8")
            subprocess.run([sys.executable,str(project/"scripts/build_symbolic_defense_summary.py"),"--input",str(src),"--output-dir",str(out)],check=True)
            result=json.loads((out/"symbolic_defense_controlled_summary.json").read_text(encoding="utf-8"))
            public=next(r for r in result["policies"] if r["policy"]=="public_allowlist")
            self.assertEqual(public["target_block_rate_min"],1.0)
            self.assertEqual(public["control_utility_retention_vs_schema_only"],1.0)
            self.assertTrue(result["claim_boundary"]["case_level_false_denial_not_claimed"])
            self.assertIn("not labeled false denial",result["definitions"]["false_denial_interpretation"])
            self.assertTrue((out/"symbolic_defense_tradeoff.svg").is_file())

if __name__=="__main__": unittest.main()
