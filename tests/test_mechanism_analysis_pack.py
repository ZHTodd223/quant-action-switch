import json, subprocess, sys, tempfile, unittest
from pathlib import Path

class MechanismAnalysisPackTest(unittest.TestCase):
    def test_seed_decomposition_and_claim_boundary(self):
        project=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); out=root/"out"
            rates={str(s):{str(q):{"repaired":{"semantic_target_asr":v},"no_injection":{"semantic_target_asr":0.0}} for q,v in zip((101,202,303),row)} for s,row in zip((101,202,303),((.8,.2,.7),(.9,.1,.6),(1,.8,.9)))}
            g={"rates_by_source_and_quantization_seed":rates,"across_nine_seed_pairs":{"repaired_semantic_target_mean":2/3,"no_injection_semantic_target_mean":0,"gap_mean":2/3,"gap_min":.1}}
            n={"across_seed":{"repaired_nf4_semantic_target_mean":.9,"no_injection_nf4_semantic_target_mean":.1,"gap_mean":.8,"gap_min":.3}}
            h={"across_seed":{"repaired_hqq4_semantic_target_mean":0,"no_injection_hqq4_semantic_target_mean":0,"gap_mean":0,"gap_min":0}}
            w={"left":{"tensor":"model.layers.17.mlp.up_proj.weight"},"right":{},"difference":{"stats":{"abs_mean":.01,"abs_p99":.02,"abs_p999":.03},"changed_count":5,"changed_fraction":.5,"finite":True}}
            for name,value in (("g.json",g),("n.json",n),("h.json",h),("w.json",w)): (root/name).write_text(json.dumps(value),encoding="utf-8")
            subprocess.run([sys.executable,str(project/"scripts/build_mechanism_analysis_pack.py"),"--gptq",str(root/"g.json"),"--nf4",str(root/"n.json"),"--hqq",str(root/"h.json"),"--weight-comparison",f"target={root/'w.json'}","--output-dir",str(out)],check=True)
            summary=json.loads((out/"mechanism_summary.json").read_text(encoding="utf-8"))
            eta=summary["gptq_seed_decomposition"]["eta_squared"]
            self.assertAlmostEqual(sum(eta.values()),1.0)
            self.assertTrue(summary["claim_boundary"]["post_hoc"])
            self.assertFalse(summary["claim_boundary"]["universal_generalization"])
            self.assertEqual(summary["quantization_numeric_error"]["status"],"not_claimed_from_behavior_summaries")
            self.assertTrue((out/"gptq_seed_interaction_heatmap.svg").is_file())
            self.assertTrue((out/"weight_delta_summary.csv").is_file())

if __name__=="__main__": unittest.main()
