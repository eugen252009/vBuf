#!/usr/bin/env python3
import csv,json,math,unittest
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/"benchmark-results/vbuf-ml-step30-reparameterization"
FAMILIES=22;BOUNDARIES=7
class Step30Tests(unittest.TestCase):
 def rows(self,name):
  with (OUT/name).open() as f:return list(csv.DictReader(f))
 def test_provenance_is_immutable(self):
  p=json.loads((OUT/"artifact-provenance.json").read_text());self.assertEqual(len(p),2);self.assertTrue(all(x["immutable"] for x in p.values()))
 def test_status_is_partial(self):
  s=json.loads((OUT/"structured-matrix-summary.json").read_text());self.assertTrue(s["status"].startswith("PARTIAL"));self.assertEqual(s["joint_representation_classification"],"F — inconclusive");self.assertEqual(s["family_count"],22);self.assertEqual(len(s["tensor_families_missing"]),5)
 def test_complete_applicability_dimensions_and_exact_family(self):
  rows=self.rows("algorithm-applicability.csv");self.assertEqual(len(rows),FAMILIES*BOUNDARIES);self.assertEqual(len({r["algorithm"] for r in rows}),22);self.assertIn("Exact / Numerically Equivalent Reparameterization",{r["algorithm"] for r in rows});self.assertTrue(all(r["status"].startswith(("TESTED","NOT TESTED","INAPPLICABLE")) for r in rows))
 def test_tested_tensor_rows_have_execution_evidence(self):
  screen={r["family"]:r for r in self.rows("tensor-breadth-screen.csv")};matrix=self.rows("algorithm-applicability.csv")
  for r in matrix:
   if r["boundary"]=="Tensor" and r["status"].startswith("TESTED"):
    self.assertIn(r["algorithm"],screen);e=screen[r["algorithm"]];self.assertGreater(int(e["serialized_bytes"]),0);self.assertGreaterEqual(float(e["action_relative_error"]),0);self.assertGreater(float(e["compact_apply_median_ms"]),0);self.assertEqual(int(e["serialized_bytes"]),sum(int(e[k]) for k in ("factor","indexes","permutations","scales","residual","payload","metadata_bytes","padding_bytes","partition_descriptor_bytes")))
 def test_all_five_memory_wall_scenarios_and_units(self):
  rows=self.rows("whole-layer-results.csv");self.assertEqual([float(r["storage_bandwidth_GBps"]) for r in rows],[1.4,3.2,8.0,16.0,32.0])
  for r in rows:
   expected=int(r["layer_bytes"])/(float(r["storage_bandwidth_GBps"])*1e9)*1000;self.assertTrue(math.isclose(float(r["storage_time_ms"]),expected,rel_tol=1e-12));self.assertEqual(r["model"],"32B")
 def test_joint_pilots_are_narrowly_labeled_and_measured(self):
  expected={"qkv-joint-results.csv":1.4,"gate-up-joint-results.csv":8/7}
  for name,gain in expected.items():
   r=self.rows(name)[0];self.assertIn("raw factor-array pilot",r["storage_label"]);self.assertTrue(math.isclose(float(r["raw_factor_byte_reduction"]),gain,rel_tol=1e-12));self.assertGreater(float(r["joint_compact_apply_median_ms"]),0);self.assertGreater(float(r["shared_transform_median_ms"]),0);self.assertEqual(r["real_hidden_state_validation"],"NOT TESTED")
 def test_required_evidence(self):
  for name in ("algorithm-applicability.csv","tensor-baseline.csv","tensor-breadth-screen.csv","tensor-residual-diagnostics.csv","qkv-joint-results.csv","gate-up-joint-results.csv","attention-joint-results.csv","mlp-joint-results.csv","whole-layer-results.csv","whole-layer-pareto.csv","whole-layer-residual-analysis.csv","whole-layer-hybrids.csv","shared-compute.csv","optional-cross-layer.csv","structured-matrix-summary.json","tensor-screen-summary.json"):
   self.assertTrue((OUT/name).exists(),name)
if __name__=="__main__":unittest.main()
