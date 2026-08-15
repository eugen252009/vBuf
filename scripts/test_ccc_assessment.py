#!/usr/bin/env python3
import csv,json,math,unittest
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'benchmark-results/vbuf-ml-step30-ccc-assessment'
class CccAssessmentTests(unittest.TestCase):
 def rows(self,name):
  with (OUT/name).open() as f:return list(csv.DictReader(f))
 def test_source_inventory(self):
  rows=self.rows('tensor-inventory.csv');self.assertEqual(len(rows),49);self.assertEqual({int(r['layer']) for r in rows},{0,1,16,32,48,62,63});self.assertEqual({r['role'] for r in rows},{'Q','K','V','O','Gate','Up','Down'});self.assertEqual({r['source_type'] for r in rows},{'Q8_0'});self.assertTrue(all(math.isclose(float(r['source_bits_per_weight']),8.5) for r in rows))
 def test_candidate_accounting(self):
  rows=self.rows('representation-assessment.csv');self.assertEqual(len(rows),3822)
  fields=('packed_code_bytes','block_metadata_bytes','tensor_metadata_bytes','tables_bytes','scales_bytes','anchors_bytes','row_column_metadata_bytes','sparse_indices_bytes','residual_payload_bytes','alignment_bytes')
  for r in rows:
   self.assertEqual(sum(int(r[k]) for k in fields),int(r['total_serialized_bytes']));self.assertTrue(math.isclose(float(r['true_bits_per_weight']),int(r['total_serialized_bytes'])*8/int(next(x['element_count'] for x in self.rows('tensor-inventory.csv') if x['tensor_name']==r['tensor_name'])),rel_tol=1e-12))
 def test_split_is_explicit_and_test_untouched(self):
  data=json.loads((OUT/'representation-assessment.json').read_text());self.assertEqual(data['summary']['fit_validation_test'],'position hash 70/15/15');self.assertTrue(all(r['test_partition'].startswith('untouched') for r in self.rows('representation-assessment.csv')))
 def test_position_matched_comparison(self):
  rows=self.rows('context-contribution.csv');self.assertEqual([int(r['bits']) for r in rows],list(range(2,9)));self.assertLess(float(rows[0]['position_context_rmse_change']),0);self.assertGreater(float(rows[-1]['position_context_rmse_change']),0)
 def test_functional_is_not_direct(self):
  rows=self.rows('functional-wx.csv');self.assertEqual({int(r['bits']) for r in rows},{2,3,4,6,8});self.assertTrue(all(r['direct_apply'].startswith('NO') for r in rows));self.assertTrue(all(r['real_hidden_states']=='NOT AVAILABLE' for r in rows))
 def test_outlier_accounting(self):
  rows=self.rows('outlier-results.csv');self.assertEqual(len(rows),30)
  for r in rows:self.assertEqual(int(r['packed_code_bytes'])+int(r['table_anchor_bytes'])+int(r['sparse_index_bytes'])+int(r['residual_payload_bytes']),int(r['total_serialized_bytes']))
 def test_unexecuted_families_not_claimed(self):
  rows=self.rows('family-status.csv');self.assertTrue(any(r['classification']=='NOT_TESTED' for r in rows));self.assertTrue(any(r['classification']=='BLOCKED' for r in rows));self.assertFalse(any(r['family']=='low-rank and structured matrix families on 32B' and r['classification'].startswith('TESTED') for r in rows))
 def test_deliverables(self):
  for n in ('representation-assessment.json','representation-assessment.csv','representation-assessment.md','pareto.csv','aggregate-results.csv','functional-wx.csv','outlier-results.csv','context-contribution.csv','tensor-inventory.csv','family-status.csv'):
   self.assertTrue((OUT/n).exists(),n)
if __name__=='__main__':unittest.main()
