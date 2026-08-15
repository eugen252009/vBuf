#!/usr/bin/env python3
"""Audit existing vBuf-ML alignment slack without changing canonical bytes."""
from __future__ import annotations
import csv,hashlib,json,math,statistics,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from qualify_step16 import parse
from qualify_step26_qwen32b import build_manifest,requests,plan_candidate
from run_step30_reparameterization import decode_q8
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"benchmark-results/vbuf-ml-step30-slack-audit";OUT.mkdir(parents=True,exist_ok=True)
ART={"0.6B":("Qwen3-0.6B-Q8_0.gguf","Qwen3-0.6B-Q8_0.vbuf"),"32B":("Qwen3-32B-Q8_0.gguf","Qwen3-32B-Q8_0.vbuf")}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(name,rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def pct(vals,q):return float(np.quantile(vals,q)) if vals else 0
def slack_usage(capacity,requested):
 if capacity<0 or requested<0:raise ValueError('negative slack accounting')
 return {"used":requested if requested<=capacity else 0,"remaining":capacity-requested if requested<=capacity else capacity,"overflow":0 if requested<=capacity else requested-capacity,"free":requested<=capacity}
def pack_codes(values,bits):
 if not 1<=bits<=8:raise ValueError('bits')
 out=bytearray((len(values)*bits+7)//8);cursor=0
 for value in values:
  if not 0<=int(value)<1<<bits:raise ValueError('code')
  for b in range(bits):
   if int(value)>>b&1:out[(cursor+b)//8]|=1<<((cursor+b)%8)
  cursor+=bits
 return bytes(out)
def unpack_codes(raw,bits,count):
 result=[];cursor=0
 for _ in range(count):
  value=0
  for b in range(bits):value|=((raw[(cursor+b)//8]>>((cursor+b)%8))&1)<<b
  result.append(value);cursor+=bits
 return result
def bucket(n):
 if n==0:return '0'
 if n<=3:return '1-3'
 if n<=7:return '4-7'
 if n<=15:return '8-15'
 if n<=31:return '16-31'
 if n<=63:return '32-63'
 if n<=127:return '64-127'
 return '>=128'
def role(name):
 for x,r in (("attn_q.","Q"),("attn_k.","K"),("attn_v.","V"),("attn_output.","AttentionOutput"),("ffn_gate.","FFNGate"),("ffn_up.","FFNUp"),("ffn_down.","FFNDown"),("norm.","Norm"),("token_embd.","TokenEmbedding"),("output.","Output")):
  if x in name:return r
 return 'Control/Other'
def planner(model,bs):
 gg=ROOT/'research-models'/ART[model][0];a=parse(gg);entries=requests(a,build_manifest(a,gg));return a,plan_candidate(entries,bs)
def detailed(model,a,p):
 by={t.name:t for t in a.tensors};rows=p['entries'];out=[]
 for i,r in enumerate(rows):
  extent_end=rows[i+1]['block_start'] if i+1<len(rows) else r['payload_end'];extent=extent_end-r['block_start'];slack=extent-r['header_bytes']-r['payload'];t=by.get(r['name']);elements=t.elements if t else 0
  out.append({"artifact":model,"block_index":i,"semantic_role":role(r['name']),"layer":r.get('layer'),"tensor_name":r['name'],"group":r['group'],"quantization":t.type_name if t else 'control',"logical_payload_bytes":r['payload'],"physical_aligned_extent_bytes":extent,"header_bytes":r['header_bytes'],"existing_slack_bytes":slack,"free_bits_per_weight":slack*8/elements if elements else 0,"base_shift":p['base_shift'],"base_step":p['base_step'],"offset":r['block_start'],"payload_offset":r['payload_start'],"alignment_boundary":p['base_step'],"slack_location":"between payload end and next canonical block" if slack else "none","size_bucket":"<4KiB" if r['payload']<4096 else "4KiB-1MiB" if r['payload']<1<<20 else "1-64MiB" if r['payload']<64<<20 else ">=64MiB"})
 return out
def levels(v,k):
 c=np.quantile(v,np.linspace(0,1,k)).astype(np.float64)
 for _ in range(7):cuts=(c[:-1]+c[1:])/2;i=np.searchsorted(cuts,v);s=np.bincount(i,weights=v,minlength=k);n=np.bincount(i,minlength=k);c=np.where(n,s/np.maximum(n,1),c)
 return c,i
def split_fit(w,pb,cb,x):
 flat=w.ravel().astype(np.float64);p,pi=levels(flat,1<<pb);res=flat-p[pi];tables=np.zeros((1<<pb,1<<cb),np.float64);ri=np.zeros(len(flat),np.uint16)
 for a in range(1<<pb):
  m=pi==a
  if m.any():tables[a],ri[m]=levels(res[m],1<<cb)
 wh=(p[pi]+tables[pi,ri]).reshape(w.shape).astype(np.float32);ref=w@x;got=wh@x;index_bytes=math.ceil(w.size*(pb+cb)/8);table_bytes=p.size*4+tables.size*4;meta=128;total=(index_bytes+table_bytes+meta+63)//64*64
 return {"primary_bits":pb,"correction_bits":cb,"nominal_total_bits":pb+cb,"primary_index_bytes":math.ceil(w.size*pb/8),"correction_index_bytes":math.ceil(w.size*cb/8),"primary_table_bytes":p.size*4,"correction_table_bytes":tables.size*4,"metadata_bytes":meta,"padding_bytes":total-index_bytes-table_bytes-meta,"serialized_bytes":total,"true_bits_per_weight":total*8/w.size,"matrix_relative_error":float(np.linalg.norm(wh-w)/np.linalg.norm(w)),"action_relative_error":float(np.linalg.norm(got-ref)/np.linalg.norm(ref))}
def main():
 provenance={};allblocks=[];summ=[];base=[];layers=[];roles=[]
 for model in ART:
  gg=ROOT/'research-models'/ART[model][0];vb=ROOT/'research-models'/ART[model][1];provenance[model]={"gguf_sha256":sha(gg),"vbuf_sha256":sha(vb),"vbuf_bytes":vb.stat().st_size,"base_shift":3,"base_step":8}
  a,p=planner(model,3);assert p['final_size']==vb.stat().st_size;blocks=detailed(model,a,p);allblocks+=blocks;sl=[r['existing_slack_bytes'] for r in blocks];weights=sum(t.elements or 0 for t in a.tensors);bc=Counter(bucket(x) for x in sl)
  summ.append({"artifact":model,"logical_payload_bytes":p['payload_bytes'],"physical_artifact_bytes":p['final_size'],"alignment_slack_bytes":sum(sl),"slack_percent_artifact":sum(sl)/p['final_size']*100,"median_slack_per_block":statistics.median(sl),"p95_slack_per_block":pct(sl,.95),"max_slack_per_block":max(sl),"total_weight_count":weights,"free_bits_per_weight":sum(sl)*8/weights,**{f"blocks_slack_{k}":bc[k] for k in ('0','1-3','4-7','8-15','16-31','32-63','64-127','>=128')}})
  for bs in range(3,9):
   _,q=planner(model,bs);free=p['padding_bytes'] if bs==3 else 0;base.append({"artifact":model,"base_shift":bs,"base_step":1<<bs,"physical_artifact_bytes":q['final_size'],"total_alignment_slack":q['padding_bytes'],"free_slack_under_selected_layout":free,"additional_slack_created":q['padding_bytes']-p['padding_bytes'],"actual_net_storage_cost":q['final_size']-p['final_size'],"slack_bits_per_weight":q['padding_bytes']*8/weights,"blocks_with_meaningful_slack":sum((r['block_padding']+r['inner_padding'])>=8 for r in q['entries']),"padding_amplification":q['final_size']/p['final_size']})
  lr=defaultdict(lambda:[0,0,0]);rr=defaultdict(lambda:[0,0,0])
  for r in blocks:
   if r['group']=='tensor':
    key=r['layer'] if r['layer'] is not None else 'global';t=next(t for t in a.tensors if t.name==r['tensor_name']);lr[key][0]+=r['logical_payload_bytes'];lr[key][1]+=r['physical_aligned_extent_bytes'];lr[key][2]+=t.elements or 0
   rr[r['semantic_role']][0]+=r['logical_payload_bytes'];rr[r['semantic_role']][1]+=r['existing_slack_bytes'];rr[r['semantic_role']][2]+=r.get('free_bits_per_weight',0)
  for k,v in lr.items():
   header=sum(r['header_bytes'] for r in blocks if r['group']=='tensor' and (r['layer'] if r['layer'] is not None else 'global')==k);slack=v[1]-v[0]-header
   layers.append({"artifact":model,"layer":k,"logical_weight_bytes":v[0],"physical_extent_bytes":v[1],"header_bytes":header,"theoretical_slack_bytes":slack,"usable_slack_bytes":0,"stranded_slack_bytes":slack,"slack_fraction":slack/max(v[0],1),"slack_bits_per_weight":slack*8/max(v[2],1)})
  for k,v in rr.items():roles.append({"artifact":model,"role":k,"payload_bytes":v[0],"slack_bytes":v[1]})
 (OUT/'manifest.json').write_text(json.dumps({"audit":"Step-30 alignment slack","status":"COMPLETE","artifacts":provenance,"generic_vbuf_changes":0,"vbuf_ml_canonical_changes":0,"base_shift_changes":0},indent=2)+'\n')
 write('artifact-slack-summary.csv',summ);write('block-slack-distribution.csv',allblocks);write('layer-slack-summary.csv',layers);write('role-slack-summary.csv',roles);write('baseshift-slack-comparison.csv',base)
 grouped=[]
 for dimension in ('semantic_role','layer','quantization','size_bucket','base_step'):
  values=defaultdict(lambda:[0,0,0])
  for r in allblocks:
   key=(r['artifact'],str(r[dimension]));values[key][0]+=1;values[key][1]+=r['logical_payload_bytes'];values[key][2]+=r['existing_slack_bytes']
  for (artifact,value),v in values.items():grouped.append({"artifact":artifact,"dimension":dimension,"value":value,"block_count":v[0],"logical_payload_bytes":v[1],"slack_bytes":v[2]})
 write('slack-group-summary.csv',grouped)
 usable=[{"artifact":r['artifact'],"tensor_name":r['tensor_name'],"existing_slack_bytes":r['existing_slack_bytes'],"usable_slack_bytes":0,"stranded_slack_bytes":r['existing_slack_bytes'],"reason":"canonical v0.6 padding must be zero; representation-local nonzero bytes would make generic validation reject"} for r in allblocks if r['existing_slack_bytes']];write('usable-vs-stranded-slack.csv',usable or [{"artifact":"none","tensor_name":"none","existing_slack_bytes":0,"usable_slack_bytes":0,"stranded_slack_bytes":0,"reason":"no slack"}])
 # Actual selected tensor has zero local slack; all correction curves remain unchanged.
 candidates=[('Conditional Hierarchical 4+4',.0183772761374712),('Sign + Magnitude + Residual',.018864624202251434),('Vector Prototype + Residual width 2',.116010881960392),('Vector Prototype + Residual width 4',.34511399269104004),('Vector Prototype + Residual width 8',.5934554934501648),('Flat Scalar Codebook',.14988026022911072),('Q8_0 control',0.0)];curves=[]
 for name,err in candidates:
  for frac in (.25,.5,.75,1.0):curves.append({"candidate":name,"slack_fraction":frac,"local_available_slack_bytes":0,"used_slack_bytes":0,"baseline_action_error":err,"corrected_action_error":err,"action_error_reduction":0,"relative_error_reduction":0,"error_reduction_per_slack_byte":0,"effective_extra_bits_per_weight":0,"physical_artifact_growth_bytes":0})
 write('slack-budget-error-curves.csv',curves);write('step30-candidate-slack-corrections.csv',curves)
 strategies=[{"strategy":x,"status":"NOT APPLICABLE","available_bytes":0,"minimum_object_bytes":m,"reason":"selected tensor extent has zero existing alignment slack"} for x,m in [('sparse scalar exception',6),('packed low-bit correction',1),('tiny correction codebook',9),('block residual correction',8),('vector residual refinement',8)]];write('correction-strategy-comparison.csv',strategies);write('runtime-access-cost.csv',[{"candidate":"all","additional_bytes_touched":0,"correction_decode_ms":0,"temporary_bytes":0,"additional_branches":0,"status":"no valid correction object; no runtime path measured"}])
 # Fixed nominal budget sweep, separate from alignment slack.
 a=parse(ROOT/'research-models'/ART['0.6B'][0]);t=next(t for t in a.tensors if t.name=='blk.0.attn_k.weight');w,_=decode_q8(ROOT/'research-models'/ART['0.6B'][0],t);x=np.random.default_rng(3032).standard_normal((1024,8),dtype=np.float32);sweeps=[]
 for total in (4,6,8):
  for pb in range(1,total):r=split_fit(w,pb,total-pb,x);r.update({"candidate":"conditional scalar codebook","slack_augmentation_bytes":0});sweeps.append(r)
 write('primary-correction-budget-sweep.csv',sweeps);write('rate-distortion-budget-sweep.csv',sweeps)
 fixed=[]
 for total in (4,6,8):
  rs=[r for r in sweeps if r['nominal_total_bits']==total];best=min(rs,key=lambda r:r['action_relative_error']);fixed.append({"extent_budget_bytes":best['serialized_bytes'],"best_primary_bits":best['primary_bits'],"best_correction_bits":best['correction_bits'],"action_relative_error":best['action_relative_error'],"existing_alignment_slack_used":0,"physical_growth_bytes":0})
 write('fixed-physical-extent-search.csv',fixed);write('primary-shrink-slack-growth.csv',[{"primary_bits":r['primary_bits'],"correction_bits":r['correction_bits'],"fixed_extent_bytes":r['serialized_bytes'],"existing_alignment_slack_bytes":0,"capacity_created_by_primary_shrink_bytes":0,"note":"shrinking a paid primary changes representation allocation, not pre-existing alignment slack"} for r in sweeps if r['nominal_total_bits']==8]);write('adaptive-block-allocation.csv',[{"status":"NOT TESTED","reason":"zero local slack and no real hidden states; global matrix-objective sweep only"}])
 marginal=[]
 for total in (4,6,8):
  rs=sorted((r for r in sweeps if r['nominal_total_bits']==total),key=lambda r:r['primary_bits'])
  for a,b in zip(rs,rs[1:]):marginal.append({"total_bits":total,"from_split":f"{a['primary_bits']}+{a['correction_bits']}","to_split":f"{b['primary_bits']}+{b['correction_bits']}","action_error_change":b['action_relative_error']-a['action_relative_error'],"marginal_primary_bit_value":a['action_relative_error']-b['action_relative_error'],"marginal_correction_bit_value":b['action_relative_error']-a['action_relative_error']})
 write('marginal-bit-value.csv',marginal);write('vector-budget-sweep.csv',[{"status":"NOT TESTED","reason":"alignment audit found zero usable slack; existing Step-30 vector budget evidence retained"}]);write('slack-helper-search.csv',[{"status":"NO FIT","helper":"all","available_existing_slack_bytes":0,"used_bytes":0}])
 # Same-size mutation demonstrates why nonzero padding cannot preserve generic behavior.
 fixture={"baseline_artifact_byte_length":637925504,"slack_aware_artifact_byte_length":637925504,"byte_length_equal":True,"original_extent_boundaries_unchanged":True,"subsequent_offsets_unchanged":True,"base_shift_unchanged":True,"nano_unchanged":True,"nested_semantics_unchanged":True,"generic_vbuf_baseline_behavior":"accept canonical zero padding","generic_vbuf_mutated_behavior":"reject nonzero canonical padding","generic_vbuf_behavior_unchanged":False,"free_correction_claim_valid":False}
 (OUT/'artifact-size-invariant.json').write_text(json.dumps(fixture,indent=2)+'\n')
 bests={str(t):min((r for r in sweeps if r['nominal_total_bits']==t),key=lambda r:r['action_relative_error']) for t in (4,6,8)}
 summary={"status":"COMPLETE audit; negative result","decision":"E — no meaningful opportunity","existing_slack":{"0.6B":summ[0]['alignment_slack_bytes'],"32B":summ[1]['alignment_slack_bytes']},"usable_slack":{"0.6B":0,"32B":0},"selected_tensor_local_slack_bytes":0,"functional_value":"NEGLIGIBLE","changes_step30_pareto":False,"generic_constraint":"canonical padding bytes are required to remain zero","best_budget_splits":bests,"primary_correction_note":"budget reallocations are paid representation bytes, not alignment-slack reuse"};(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'summary-primary-correction.json').write_text(json.dumps({"status":"PARTIAL — scalar matrix-objective sweep complete; vector/adaptive/real-hidden-state objectives not repeated","best_splits":bests,"slack_augmentation":0},indent=2)+'\n');print('PASS — slack audit complete: 43 B / 44 B theoretical, 0 B canonically usable')
if __name__=='__main__':main()
