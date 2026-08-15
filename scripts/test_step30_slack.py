#!/usr/bin/env python3
import csv,json,math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_step30_slack import OUT,pack_codes,slack_usage,unpack_codes
class SlackAuditTests(unittest.TestCase):
 def rows(self,name):
  with (OUT/name).open() as f:return list(csv.DictReader(f))
 def test_exact_real_slack(self):
  rows={r['artifact']:r for r in self.rows('artifact-slack-summary.csv')};self.assertEqual(int(rows['0.6B']['alignment_slack_bytes']),43);self.assertEqual(int(rows['32B']['alignment_slack_bytes']),44);self.assertEqual(float(rows['0.6B']['median_slack_per_block']),0);self.assertEqual(float(rows['32B']['p95_slack_per_block']),0)
 def test_zero_small_exact_and_overflow(self):
  self.assertEqual(slack_usage(0,0),{'used':0,'remaining':0,'overflow':0,'free':True});self.assertTrue(slack_usage(7,3)['free']);self.assertEqual(slack_usage(7,7)['remaining'],0);self.assertFalse(slack_usage(7,8)['free']);self.assertEqual(slack_usage(7,8)['overflow'],1)
 def test_bit_splits_pack_roundtrip(self):
  for bits in range(1,8):
   values=list(range(1<<bits))*3;raw=pack_codes(values,bits);self.assertEqual(unpack_codes(raw,bits,len(values)),values);self.assertEqual(len(raw),math.ceil(len(values)*bits/8))
 def test_size_and_offsets_invariant_but_generic_behavior_fails(self):
  x=json.loads((OUT/'artifact-size-invariant.json').read_text());self.assertTrue(x['byte_length_equal']);self.assertTrue(x['original_extent_boundaries_unchanged']);self.assertTrue(x['subsequent_offsets_unchanged']);self.assertTrue(x['base_shift_unchanged']);self.assertTrue(x['nano_unchanged']);self.assertFalse(x['generic_vbuf_behavior_unchanged']);self.assertFalse(x['free_correction_claim_valid'])
 def test_all_slack_is_stranded(self):
  rows=self.rows('usable-vs-stranded-slack.csv');self.assertTrue(all(int(r['usable_slack_bytes'])==0 for r in rows));self.assertTrue(all(int(r['stranded_slack_bytes'])==int(r['existing_slack_bytes']) for r in rows))
 def test_baseshift_cost_is_not_free(self):
  rows=self.rows('baseshift-slack-comparison.csv')
  for model in ('0.6B','32B'):
   rs=[r for r in rows if r['artifact']==model];self.assertEqual([int(r['base_shift']) for r in rs],list(range(3,9)));self.assertEqual(int(rs[0]['actual_net_storage_cost']),0);self.assertTrue(all(int(r['actual_net_storage_cost'])>0 and int(r['free_slack_under_selected_layout'])==0 for r in rs[1:]))
 def test_budget_accounting_and_best_splits(self):
  rows=self.rows('primary-correction-budget-sweep.csv')
  for r in rows:
   components=sum(int(r[k]) for k in ('primary_index_bytes','correction_index_bytes','primary_table_bytes','correction_table_bytes','metadata_bytes','padding_bytes'));self.assertEqual(components,int(r['serialized_bytes']));self.assertEqual(int(r['slack_augmentation_bytes']),0)
  summary=json.loads((OUT/'summary.json').read_text());self.assertEqual(summary['best_budget_splits']['4']['primary_bits'],2);self.assertEqual(summary['best_budget_splits']['6']['primary_bits'],3);self.assertEqual(summary['best_budget_splits']['8']['primary_bits'],3)
 def test_selected_tensor_has_no_local_slack(self):
  rows=[r for r in self.rows('block-slack-distribution.csv') if r['artifact']=='0.6B' and r['tensor_name']=='blk.0.attn_k.weight'];self.assertEqual(len(rows),1);self.assertEqual(int(rows[0]['existing_slack_bytes']),0)
if __name__=='__main__':unittest.main()
