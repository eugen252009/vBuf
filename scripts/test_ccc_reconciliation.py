#!/usr/bin/env python3
import csv,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'benchmark-results/vbuf-ml-step32-ccc-reconciliation'
class ReconciliationTests(unittest.TestCase):
 def csv(self,name):
  with (OUT/name).open() as f:return list(csv.DictReader(f))
 def test_required_artifacts(self):
  for n in ('reconciliation-report.md','reconciliation.json','provenance.csv','evidence-comparison.csv','methodology-comparison.csv','common-vector-results.csv','canonical-byte-equivalence.csv','tensor-equivalence.csv','raw/runner.log'):self.assertTrue((OUT/n).exists(),n)
 def test_exact_classifications(self):
  s=json.loads((OUT/'reconciliation.json').read_text())['summary'];self.assertEqual(s['root_cause_classification'],'DISCREPANCY_TENSOR_ORIENTATION');self.assertEqual(s['corrected_numerical_classification'],'C3_NUMERICALLY_DOMINATED');self.assertEqual(s['geometry_classification'],'GEOMETRY_APPROXIMATES_LEARNED');self.assertEqual(s['direct_apply_classification'],'C3_DIRECT_APPLY_FEASIBILITY_CONFIRMED');self.assertEqual(s['canonical_runtime_comparison'],'PARTIALLY_VALID');self.assertEqual(s['combined_direction'],'STOP_C3');self.assertEqual(s['recommendation'],'STOP_C3')
 def test_git_provenance(self):
  g=json.loads((OUT/'reconciliation.json').read_text())['git_provenance'];self.assertEqual(g['stage2_commit'],'b9316d9b83f4d193aec10d4b5cbead53d0963e79');self.assertEqual(g['parallel_research_commit'],'889bd58c674195b9e40f0156e56ca18c1fd00e78');self.assertEqual(g['merge_commit'],'d11d16791eb63d5cb6d9ff78c424a4f2a87eb8b0');self.assertTrue(g['historically_independent'])
 def test_q8_oracle_identical(self):
  r=self.csv('tensor-equivalence.csv')[0];self.assertEqual(r['parallel_hash'],r['stage2_hash']);self.assertEqual(r['identical'],'True');self.assertEqual(int(r['elements']),5242880);self.assertEqual(float(r['max_abs_difference']),0)
 def test_canonical_bytes_identical(self):
  rows=self.csv('canonical-byte-equivalence.csv');self.assertEqual({r['format'] for r in rows},{'Q2_K','Q3_K','Q4_K'});self.assertTrue(all(r['identical']=='True' and r['parallel_sha256']==r['stage2_sha256'] and r['first_differing_block']=='none' for r in rows))
 def test_canonical_dequant_identical(self):
  rows=self.csv('tensor-equivalence.csv')[1:];self.assertEqual(len(rows),3);self.assertTrue(all(r['identical']=='True' and r['parallel_hash']==r['stage2_hash'] and float(r['rmse_difference'])==0 for r in rows))
 def test_common_vectors_localize_orientation(self):
  rows={(r['pipeline'],r['candidate']):r for r in self.csv('common-vector-results.csv')}
  for c in ('Q8 oracle','CCC C3 learned','Q2_K','Q3_K','Q4_K'):
   a=rows[('independent_stage2',c)];b=rows[('parallel_corrected',c)];self.assertEqual(a['reference_output_sha256'],b['reference_output_sha256']);self.assertEqual(a['candidate_output_sha256'],b['candidate_output_sha256']);self.assertEqual(a['mean_per_vector_relative_l2'],b['mean_per_vector_relative_l2'])
  self.assertGreater(float(rows[('parallel_as_written','Q2_K')]['mean_per_vector_relative_l2']),2*float(rows[('independent_stage2','Q2_K')]['mean_per_vector_relative_l2']))
  self.assertLess(abs(float(rows[('parallel_as_written','CCC C3 learned')]['mean_per_vector_relative_l2'])-float(rows[('independent_stage2','CCC C3 learned')]['mean_per_vector_relative_l2'])),.01)
 def test_metric_aggregation_same(self):
  rows={r['metric_property']:r for r in self.csv('methodology-comparison.csv')};self.assertEqual(rows['relative-L2 formula']['same'],'YES');self.assertEqual(rows['tensor shape convention']['same'],'NO');self.assertIn('root bug',rows['tensor shape convention']['explanation'])
 def test_evidence_comparison_has_both_branches(self):
  rows=self.csv('evidence-comparison.csv');self.assertEqual({r['source_branch'] for r in rows},{'vbuf-ml','vBuf_3'});self.assertTrue({'IQ2_XS ACTIVATION_AWARE','C3-A Learned (Lloyd-Max)','CCC C3 learned'}<={r['candidate'] for r in rows})
 def test_equivalence_helper_uses_parallel_entry_points(self):
  src=(ROOT/'integrations/llama.cpp/step32_canonical_equivalence.cpp').read_text();self.assertIn('quantize_q2_K',src);self.assertIn('quantize_q3_K',src);self.assertIn('quantize_q4_K',src);self.assertIn('dequantize_row_q3_K',src)
 def test_iq_provenance(self):
  iq=json.loads((OUT/'reconciliation.json').read_text())['iq_provenance'];aware={r['format'] for r in iq if r['activation_aware']};self.assertEqual(aware,{'IQ2_XXS','IQ2_XS'});self.assertTrue(all(r['functional_test_untouched'] for r in iq));self.assertTrue(all(r['functional_subset']=='none' for r in iq if not r['activation_aware']))
if __name__=='__main__':unittest.main()
