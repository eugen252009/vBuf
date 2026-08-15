#!/usr/bin/env python3
import csv,json,math,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_step30_continuation import account,butterfly_apply,house_apply,ldr_apply,ldr_reconstruct,tt_apply
OUT=Path(__file__).resolve().parents[1]/"benchmark-results/vbuf-ml-step30-reparameterization"
FAMILIES=22;BOUNDARIES=7
class Step30Tests(unittest.TestCase):
 def rows(self,name):
  with (OUT/name).open() as f:return list(csv.DictReader(f))
 def test_provenance_is_immutable(self):
  p=json.loads((OUT/"artifact-provenance.json").read_text());self.assertEqual(len(p),2);self.assertTrue(all(x["immutable"] for x in p.values()))
 def test_status_is_partial(self):
  s=json.loads((OUT/"structured-matrix-summary.json").read_text());self.assertTrue(s["status"].startswith("PARTIAL"));self.assertEqual(s["joint_representation_classification"],"F — inconclusive");self.assertEqual(s["family_count"],22);self.assertEqual(len(s["tensor_families_missing"]),0)
 def test_complete_applicability_dimensions_and_exact_family(self):
  rows=self.rows("algorithm-applicability.csv");self.assertEqual(len(rows),FAMILIES*BOUNDARIES);self.assertEqual(len({r["algorithm"] for r in rows}),22);self.assertIn("Exact / Numerically Equivalent Reparameterization",{r["algorithm"] for r in rows});self.assertTrue(all(r["status"].startswith(("TESTED","NOT TESTED","INAPPLICABLE")) for r in rows))
 def test_tested_tensor_rows_have_execution_evidence(self):
  original={r["family"]:r for r in self.rows("tensor-breadth-screen.csv")};missing={r["family"]:r for r in self.rows("missing-family-screen.csv")};matrix=self.rows("algorithm-applicability.csv")
  for r in matrix:
   if r["boundary"]=="Tensor" and r["status"].startswith("TESTED"):
    self.assertIn(r["algorithm"],set(original)|set(missing));e=(original|missing)[r["algorithm"]];self.assertGreater(int(e["serialized_bytes"]),0);self.assertGreaterEqual(float(e["action_relative_error"]),0);self.assertGreater(float(e.get("compact_apply_median_ms",e.get("compact_apply_ms"))),0)
 def test_all_five_memory_wall_scenarios_and_units(self):
  rows=self.rows("whole-layer-results.csv");self.assertEqual([float(r["storage_bandwidth_GBps"]) for r in rows],[1.4,3.2,8.0,16.0,32.0])
  for r in rows:
   expected=int(r["layer_bytes"])/(float(r["storage_bandwidth_GBps"])*1e9)*1000;self.assertTrue(math.isclose(float(r["storage_time_ms"]),expected,rel_tol=1e-12));self.assertEqual(r["model"],"32B")
 def test_joint_pilots_are_narrowly_labeled_and_measured(self):
  expected={"qkv-joint-results.csv":1.4,"gate-up-joint-results.csv":8/7}
  for name,gain in expected.items():
   r=self.rows(name)[0];self.assertIn("raw factor-array pilot",r["storage_label"]);self.assertTrue(math.isclose(float(r["raw_factor_byte_reduction"]),gain,rel_tol=1e-12));self.assertGreater(float(r["joint_compact_apply_median_ms"]),0);self.assertGreater(float(r["shared_transform_median_ms"]),0);self.assertEqual(r["real_hidden_state_validation"],"NOT TESTED")
 def test_missing_family_synthetic_apply(self):
  x=np.random.default_rng(1).standard_normal((1024,2),dtype=np.float32)
  identity=[np.eye(2,dtype=np.float32).reshape(1,1,2,2).repeat(512,axis=0).reshape(512,1,2,2) if s==0 else np.broadcast_to(np.eye(2,dtype=np.float32),(1024//(2*(1<<s)),1<<s,2,2)).copy() for s in range(10)]
  np.testing.assert_allclose(butterfly_apply(identity,x),x,rtol=1e-6,atol=1e-6)
  cores=[np.eye(4,dtype=np.float32).reshape(1,4,4,1) for _ in range(5)]
  np.testing.assert_allclose(tt_apply(cores,x),x,rtol=1e-5,atol=1e-5)
  u=np.zeros((1024,1),np.float32);v=np.zeros((1,1024),np.float32);u[0]=1;v[0,0]=2
  np.testing.assert_allclose(ldr_apply(u,v,1,x),ldr_reconstruct(u,v,1)@x,rtol=1e-6,atol=1e-6)
  hv=np.zeros((1,1024),np.float32);hv[0,0]=1;self.assertTrue(math.isclose(float(np.linalg.norm(house_apply(hv,x))),float(np.linalg.norm(x)),rel_tol=1e-6))
 def test_nibbles_and_accounting(self):
  primary=np.array([0,1,15],np.uint8);residual=np.array([15,2,0],np.uint8);packed=(primary<<4)|residual
  np.testing.assert_array_equal(packed>>4,primary);np.testing.assert_array_equal(packed&15,residual)
  s=account({"packed_indexes":packed,"primary_table":np.zeros(16,np.float32),"residual_table":np.zeros(16,np.float32)},{"test":True});self.assertEqual(s["serialized_bytes"],sum(s[k] for k in ("indexes_bytes","primary_table_bytes","residual_table_bytes","scales_bytes","signs_bytes","factor_bytes","permutations_bytes","payload_bytes","metadata_bytes","padding_bytes","partition_descriptor_bytes")))
 def test_required_evidence(self):
  for name in ("algorithm-applicability.csv","tensor-baseline.csv","tensor-breadth-screen.csv","tensor-residual-diagnostics.csv","qkv-joint-results.csv","gate-up-joint-results.csv","attention-joint-results.csv","mlp-joint-results.csv","whole-layer-results.csv","whole-layer-pareto.csv","whole-layer-residual-analysis.csv","whole-layer-hybrids.csv","shared-compute.csv","optional-cross-layer.csv","structured-matrix-summary.json","tensor-screen-summary.json","missing-family-screen.csv","hierarchical-4plus4.csv","sign-magnitude-residual.csv","vector-prototype-residual.csv","f32-vs-f64-fit.csv","hierarchical-residual-diagnostics.csv","vector-residual-diagnostics.csv","byte-budget-comparison-v2.csv","tensor-pareto-v2.csv","memory-wall-projection-v2.csv","continuation-summary.json"):
   self.assertTrue((OUT/name).exists(),name)
if __name__=="__main__":unittest.main()
