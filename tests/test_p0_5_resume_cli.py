from __future__ import annotations
import hashlib, json, subprocess, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from runtime_evidence_fixtures import build_native_comparable

ROOT=Path(__file__).resolve().parents[1]
CLI=ROOT/'scripts'/'run_cross_model_comparison.py'

class ResumeCliDriftTests(unittest.TestCase):
    def invoke(self, state: Path, *extra: str):
        return subprocess.run([sys.executable,str(CLI),'resume-verify','--state',str(state),*extra],capture_output=True,text=True)
    def test_locked_identity_and_drift_are_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); _,state=build_native_comparable(root)
            state_path=root/'state.json'; state_path.write_text(json.dumps(state),encoding='utf-8')
            manifest=Path(state['bf16_output_manifest_path']); before=(hashlib.sha256(state_path.read_bytes()).hexdigest(),hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertEqual(self.invoke(state_path).returncode,0)
            for option,value in (("--mode","legacy"),("--response-field-consumed","decoded_without_special_tokens"),("--schema-version","drift"),("--implementation-version","drift"),("--evidence-class","LEGACY_HISTORICAL"),("--strict-parser-version","drift")):
                result=self.invoke(state_path,option,value)
                self.assertEqual(result.returncode,22,result.stdout+result.stderr)
                self.assertIn('resume_rejected',result.stdout)
                self.assertEqual(before,(hashlib.sha256(state_path.read_bytes()).hexdigest(),hashlib.sha256(manifest.read_bytes()).hexdigest()))
