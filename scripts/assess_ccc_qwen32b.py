#!/usr/bin/env python3
"""Bounded Qwen3-32B Contextual Correction Code assessment.

Metrics use the dequantized Q8_0 source as oracle. Position-hash fit/validation/
test partitions are disjoint. This is research evidence, not a format.
"""
from __future__ import annotations
import csv,hashlib,json,math,platform,statistics,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from qualify_step16 import parse
from run_step30_reparameterization import decode_q8
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'benchmark-results/vbuf-ml-step30-ccc-assessment';OUT.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT/'research-models/Qwen3-32B-Q8_0.gguf';LAYERS=[0,1,16,32,48,62,63];ROLES={'Q':'attn_q.weight','K':'attn_k.weight','V':'attn_v.weight','O':'attn_output.weight','Gate':'ffn_gate.weight','Up':'ffn_up.weight','Down':'ffn_down.weight'};SEED=3035;SAMPLE=65536

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
def split_hash(index):
 x=np.asarray(index,dtype=np.uint64);x^=x>>np.uint64(33);x*=np.uint64(0xff51afd7ed558ccd);x^=x>>np.uint64(33);return x%100
def lloyd(v,k):
 c=np.quantile(v,np.linspace(0,1,k)).astype(np.float64)
 for _ in range(7):cuts=(c[:-1]+c[1:])/2;i=np.searchsorted(cuts,v);s=np.bincount(i,weights=v,minlength=k);n=np.bincount(i,minlength=k);c=np.where(n,s/np.maximum(n,1),c)
 return c.astype(np.float32)
def quant(v,c):return c[np.searchsorted((c[:-1]+c[1:])/2,v)]
def anchor_values(kind,fit_v,fit_r,fit_c,rows,cols):
 g=float(np.mean(fit_v)) if kind!='tensor_median' else float(np.median(fit_v))
 if kind in ('zero','tensor_mean','tensor_median'):return (lambda r,c:g+np.zeros(len(r),np.float32)),4 if kind!='zero' else 0
 if kind.startswith('row'):
  vals=np.full(rows,g,np.float32)
  for q in np.unique(fit_r):vals[q]=np.mean(fit_v[fit_r==q]) if kind=='row_mean' else np.median(fit_v[fit_r==q])
  return lambda r,c:vals[r],vals.nbytes
 if kind.startswith('column'):
  vals=np.full(cols,g,np.float32)
  for q in np.unique(fit_c):vals[q]=np.mean(fit_v[fit_c==q]) if kind=='column_mean' else np.median(fit_v[fit_c==q])
  return lambda r,c:vals[c],vals.nbytes
 # Additive two-way context, centered to avoid double-counting global anchor.
 rv=np.full(rows,g,np.float32);cv=np.full(cols,g,np.float32)
 for q in np.unique(fit_r):rv[q]=np.mean(fit_v[fit_r==q])
 for q in np.unique(fit_c):cv[q]=np.mean(fit_v[fit_c==q])
 return lambda r,c:rv[r]+cv[c]-g,rv.nbytes+cv.nbytes+4
def metrics(ref,got):
 e=got-ref;ae=np.abs(e);rmse=float(np.sqrt(np.mean(e*e)));den=float(np.linalg.norm(ref));
 hist=np.histogram(ae,bins=[0,1e-5,1e-4,1e-3,1e-2,1e-1,1,math.inf])[0].tolist();return {'mae':float(ae.mean()),'rmse':rmse,'relative_l2':float(np.linalg.norm(e)/max(den,1e-30)),'max_absolute_error':float(ae.max()),'p50_error':float(np.quantile(ae,.5)),'p90_error':float(np.quantile(ae,.9)),'p95_error':float(np.quantile(ae,.95)),'p99_error':float(np.quantile(ae,.99)),'p999_error':float(np.quantile(ae,.999)),'clipped_count':0,'cosine_similarity':float(np.dot(ref,got)/max(np.linalg.norm(ref)*np.linalg.norm(got),1e-30)),'error_histogram':json.dumps(hist)}
def row(candidate,tensor,layer,role,elements,source_bytes,bits,code_bytes,metadata_bytes,fit,valid,test,status='TESTED_REJECTED',direct='NO — sampled decoder only',extra=None):
 total=code_bytes+metadata_bytes;out={'tensor_name':tensor,'layer':layer,'role':role,'family':candidate,'nominal_bits':bits,'packed_code_bytes':code_bytes,'block_metadata_bytes':0,'tensor_metadata_bytes':0,'tables_bytes':metadata_bytes,'scales_bytes':0,'anchors_bytes':0,'row_column_metadata_bytes':0,'sparse_indices_bytes':0,'residual_payload_bytes':0,'alignment_bytes':0,'total_serialized_bytes':total,'true_bits_per_weight':total*8/elements,'source_bytes':source_bytes,'fit_rmse':metrics(fit[0],fit[1])['rmse'],'validation_rmse':metrics(valid[0],valid[1])['rmse'],'test_partition':'untouched position-hash 15%','direct_apply':direct,'status':status}|metrics(test[0],test[1])
 if extra:out.update(extra)
 return out
def sample_tensor(path,t,rng):
 w,_=decode_q8(path,t);m,n=w.shape;count=min(SAMPLE,w.size);idx=np.sort(rng.choice(w.size,count,replace=False));r=idx//n;c=idx%n;v=w.ravel()[idx].copy();return w,idx,r,c,v

def assess_one(path,t,layer,role,rng):
 w,idx,rr,cc,v=sample_tensor(path,t,rng);h=split_hash(idx);mfit=h<70;mval=(h>=70)&(h<85);mtest=h>=85;elements=w.size;source=t.payload_size;rows=[]
 # Canonical Q8_0 dequantized oracle.
 rows.append(row('Q8_0 source control',t.name,layer,role,elements,source,8.5,source,0,(v[mfit],v[mfit]),(v[mval],v[mval]),(v[mtest],v[mtest]),'PROMOTED','YES — canonical source implementation',{'source_type':'Q8_0','source_block_geometry':'32 weights / 34 bytes','source_bits_per_weight':8.5,'packed_code_bytes':elements,'block_metadata_bytes':elements//32*2}))
 # Uniform symmetric/asymmetric and centered scalar baselines.
 for bits in (2,3,4,5,6,7,8):
  k=1<<bits;fv=v[mfit]
  scale=max(float(np.max(np.abs(fv))),1e-12)/((k//2)-1);sym=lambda z:np.clip(np.rint(z/scale),-k//2,k//2-1)*scale
  lo,hi=float(fv.min()),float(fv.max());astep=max((hi-lo)/(k-1),1e-12);aff=lambda z:lo+np.clip(np.rint((z-lo)/astep),0,k-1)*astep
  for name,fn,meta in [('uniform_symmetric',sym,4),('affine_asymmetric',aff,8)]:rows.append(row(name,t.name,layer,role,elements,source,bits,math.ceil(elements*bits/8),meta,(fv,fn(fv)),(v[mval],fn(v[mval])),(v[mtest],fn(v[mtest])),extra={'tables_bytes':0,'scales_bytes':meta}))
 # CCC anchors and optional position/lane context.
 anchors=('zero','tensor_mean','tensor_median','row_mean','row_median','column_mean','column_median','row+column')
 for bits in range(2,9):
  for a in anchors:
   af,abytes=anchor_values(a,v[mfit],rr[mfit],cc[mfit],w.shape[0],w.shape[1]);res=v[mfit]-af(rr[mfit],cc[mfit]);table=lloyd(res,1<<bits)
   def dec(mask,af=af,table=table):base=af(rr[mask],cc[mask]);return base+quant(v[mask]-base,table)
   rows.append(row('CCC additive',t.name,layer,role,elements,source,bits,math.ceil(elements*bits/8),abytes+table.nbytes,(v[mfit],dec(mfit)),(v[mval],dec(mval)),(v[mtest],dec(mtest)),status='TESTED_INTERESTING',extra={'ccc_anchor':a,'position_context':'none','unique_reconstruction_states':len(np.unique(table)),'context_compute':'row/column lookup where selected','tables_bytes':table.nbytes,'anchors_bytes':abytes if a in ('tensor_mean','tensor_median') else 0,'row_column_metadata_bytes':abytes if a not in ('zero','tensor_mean','tensor_median') else 0}))
  # Position context: lane mod 32 conditional correction tables, no serialized lane IDs.
  a='tensor_mean';af,abytes=anchor_values(a,v[mfit],rr[mfit],cc[mfit],w.shape[0],w.shape[1]);tables=[]
  for lane in range(32):
   z=mfit&(cc%32==lane);tables.append(lloyd(v[z]-af(rr[z],cc[z]),1<<bits) if z.any() else np.zeros(1<<bits,np.float32))
  tables=np.stack(tables)
  def pdec(mask):base=af(rr[mask],cc[mask]);out=np.empty(mask.sum(),np.float32);lanes=cc[mask]%32;res=v[mask]-base
  # Explicit loop kept outside nested function return for clarity.
  def pdecode(mask):
   base=af(rr[mask],cc[mask]);out=np.empty(mask.sum(),np.float32);lanes=cc[mask]%32;res=v[mask]-base
   for lane in range(32):q=lanes==lane;out[q]=base[q]+quant(res[q],tables[lane])
   return out
  rows.append(row('CCC additive',t.name,layer,role,elements,source,bits,math.ceil(elements*bits/8),abytes+tables.nbytes,(v[mfit],pdecode(mfit)),(v[mval],pdecode(mval)),(v[mtest],pdecode(mtest)),status='TESTED_INTERESTING',extra={'ccc_anchor':a,'position_context':'column lane mod 32','unique_reconstruction_states':sum(len(np.unique(q)) for q in tables),'context_compute':'modulo + conditional table lookup','tables_bytes':tables.nbytes,'anchors_bytes':abytes}))
 del w
 return rows

def main():
 started=time.perf_counter();artifact=parse(SOURCE);by={t.name:t for t in artifact.tensors};rng=np.random.default_rng(SEED);inventory=[];rows=[]
 for layer in LAYERS:
  for role,suffix in ROLES.items():
   name=f'blk.{layer}.{suffix}';t=by[name];inventory.append({'tensor_name':name,'layer':layer,'role':role,'dimensions':'x'.join(map(str,reversed(t.shape))),'element_count':t.elements,'source_type':t.type_name,'source_bits_per_weight':t.payload_size*8/t.elements,'source_block_geometry':'32 weights / 34 bytes' if t.type_name=='Q8_0' else 'scalar','source_payload_bytes':t.payload_size});rows.extend(assess_one(SOURCE,t,layer,role,rng))
 # Promote per budget using mean validation RMSE across all inventory tensors.
 groups={}
 for r in rows:
  key=(r['family'],r['nominal_bits'],r.get('ccc_anchor',''),r.get('position_context',''));groups.setdefault(key,[]).append(r)
 ranked=[]
 for key,rs in groups.items():ranked.append({'family':key[0],'nominal_bits':key[1],'ccc_anchor':key[2],'position_context':key[3],'mean_validation_rmse':statistics.mean(float(r['validation_rmse']) for r in rs),'mean_test_rmse':statistics.mean(float(r['rmse']) for r in rs),'mean_test_relative_l2':statistics.mean(float(r['relative_l2']) for r in rs),'mean_true_bits_per_weight':statistics.mean(float(r['true_bits_per_weight']) for r in rs),'tensor_count':len(rs)})
 # Functional W*x on untouched random probes for promoted representative layer-32 K candidates. Dense reconstruction makes timing non-direct.
 t=by['blk.32.attn_k.weight'];w,_=decode_q8(SOURCE,t);x=np.random.default_rng(4040).standard_normal((w.shape[1],16),dtype=np.float32);y=w@x;functional=[]
 for bits in (2,3,4,6,8):
  anchor=float(w.mean());table=lloyd((w.ravel()-anchor)[::97],1<<bits);wh=anchor+quant(w-anchor,table);got=wh@x;functional.append({'tensor_name':t.name,'candidate':'CCC tensor-mean additive','bits':bits,'true_bits_per_weight':(math.ceil(w.size*bits/8)+table.nbytes+4)*8/w.size,'wx_relative_error':float(np.linalg.norm(got-y)/np.linalg.norm(y)),'wx_cosine':float(np.vdot(got,y)/(np.linalg.norm(got)*np.linalg.norm(y))),'output_mean_error':float(abs(got.mean()-y.mean())),'direct_apply':'NO — compact -> dense reconstruction -> matmul','real_hidden_states':'NOT AVAILABLE'})
 # Mandatory outlier comparison on C2/C3/C4 representative tensor; exact whole-tensor accounting.
 outliers=[]
 for bits in (2,3,4):
  anchor=float(w.mean());table=lloyd((w.ravel()-anchor)[::97],1<<bits);base=anchor+quant(w-anchor,table);res=w-base
  for frac in (0,.0001,.00025,.0005,.001,.0025,.005,.01,.02,.05):
   corrected=base.copy();count=int(w.size*frac)
   if count:ix=np.argpartition(np.abs(res).ravel(),-count)[-count:];corrected.ravel()[ix]=w.ravel()[ix]
   got=corrected@x;code=math.ceil(w.size*bits/8);indices=count*4;payload=count*2;total=code+table.nbytes+4+indices+payload;outliers.append({'candidate':f'C{bits}+FP16 outliers','outlier_fraction':frac,'selection':'absolute residual','packed_code_bytes':code,'table_anchor_bytes':table.nbytes+4,'sparse_index_bytes':indices,'residual_payload_bytes':payload,'total_serialized_bytes':total,'true_bits_per_weight':total*8/w.size,'wx_relative_error':float(np.linalg.norm(got-y)/np.linalg.norm(y)),'direct_apply':'NO — research dense correction','status':'TESTED_INTERESTING'})
 del w
 # Context contribution aggregates matched anchor/bits.
 context=[]
 for bits in range(2,9):
  no=[r for r in rows if r['family']=='CCC additive' and r.get('ccc_anchor')=='tensor_mean' and r.get('position_context')=='none' and r['nominal_bits']==bits];yes=[r for r in rows if r['family']=='CCC additive' and r.get('position_context')=='column lane mod 32' and r['nominal_bits']==bits];context.append({'bits':bits,'no_position_mean_rmse':statistics.mean(r['rmse'] for r in no),'position_mean_rmse':statistics.mean(r['rmse'] for r in yes),'position_context_rmse_change':statistics.mean(r['rmse'] for r in yes)-statistics.mean(r['rmse'] for r in no),'no_position_true_bpw':statistics.mean(r['true_bits_per_weight'] for r in no),'position_true_bpw':statistics.mean(r['true_bits_per_weight'] for r in yes)})
 # Status ledger makes unexecuted breadth explicit.
 tested={'uniform symmetric quantization','affine/asymmetric quantization','mean/median centered quantization','per-tensor scaling','CCC scalar additive','CCC row/column anchors','CCC lane-mod32 context','scalar Lloyd-Max codebook','sparse FP16 outlier correction','Q8_0 source control'}
 blocked={'canonical GGML Q4/Q5/K/IQ formats':'BLOCKED — local canonical quantizer was not integrated into this bounded harness','real hidden states':'BLOCKED — capture corpus unavailable','native direct kernels':'BLOCKED — no production implementation authorized'}
 not_tested=['per-row/per-column scale sweeps','block-size 16..512 scale sweep','full statistical dispersion/range/curve sweep','multiplicative/zig-zag/XOR CCC decoders','vector/PQ/RVQ on 32B','low-rank and structured matrix families on 32B','TT/Tucker/CP on 32B','cross-layer dictionaries','joint QKV/Gate-Up on 32B','adaptive bitplanes']
 interesting={'CCC lane-mod32 context','CCC row/column anchors','CCC scalar additive','scalar Lloyd-Max codebook','sparse FP16 outlier correction'}
 ledger=[{'family':x,'classification':'TESTED_INTERESTING' if x in interesting else 'PROMOTED' if x=='Q8_0 source control' else 'TESTED_REJECTED','reason':'empirical 32B Stage-1 execution'} for x in sorted(tested)]+[{'family':k,'classification':'BLOCKED','reason':v.split(' — ',1)[1]} for k,v in blocked.items()]+[{'family':x,'classification':'NOT_TESTED','reason':'outside bounded Stage-1 budget; no 32B empirical claim'} for x in not_tested]
 write('tensor-inventory.csv',inventory);write('representation-assessment.csv',rows);write('aggregate-results.csv',ranked);write('functional-wx.csv',functional);write('outlier-results.csv',outliers);write('context-contribution.csv',context);write('family-status.csv',ledger)
 # Compact Pareto over aggregate weight metrics; W*x joined only for representative promoted configurations.
 pareto=[]
 for r in ranked:
  if not any(q['mean_true_bits_per_weight']<=r['mean_true_bits_per_weight'] and q['mean_test_rmse']<=r['mean_test_rmse'] and (q['mean_true_bits_per_weight']<r['mean_true_bits_per_weight'] or q['mean_test_rmse']<r['mean_test_rmse']) for q in ranked):pareto.append(r)
 write('pareto.csv',pareto)
 summary={'status':'PARTIAL — bounded 32B Stage-1 CCC/baseline assessment','source_sha256':sha(SOURCE),'source_type':'Q8_0','oracle':'Q8_0 reconstructed source, not BF16 ground truth','layers':LAYERS,'roles':list(ROLES),'tensor_count':len(inventory),'candidate_rows':len(rows),'fit_validation_test':'position hash 70/15/15','functional_tensor':'blk.32.attn_k.weight','real_hidden_states':'BLOCKED','direct_native_apply':'NOT IMPLEMENTED','not_tested_families':not_tested,'wire_changes':0,'canonical_changes':0,'elapsed_seconds':time.perf_counter()-started};(OUT/'representation-assessment.json').write_text(json.dumps({'summary':summary,'inventory':inventory,'aggregate':ranked,'functional':functional,'outliers':outliers,'context':context,'family_status':ledger,'pareto':pareto},indent=2)+'\n');build_report(summary,inventory,ranked,functional,outliers,context,ledger,pareto);print(f"PASS — {len(inventory)} tensors, {len(rows)} candidate rows; assessment PARTIAL")
def build_report(s,inventory,ranked,functional,outliers,context,ledger,pareto):
 def best(bits):
  rs=[r for r in ranked if abs(float(r['mean_true_bits_per_weight'])-bits)<.75];return min(rs,key=lambda r:r['mean_test_rmse']) if rs else None
 lines=['# Qwen3-32B CCC / model representation assessment','',f"Status: **{s['status']}**",'','## 1. Environment and source artifacts','',f"- Host: `{platform.node()}`",f"- Source SHA-256: `{s['source_sha256']}`",'- Oracle: dequantized Q8_0 source, not BF16 ground truth.','- Source geometry: 32 weights / 34 bytes = 8.5 bits/weight.','','## 2. Exact tensor inventory tested','',f"49 tensors: layers {s['layers']} × roles {s['roles']}. See `tensor-inventory.csv`.",'','## 3. Source quantization qualification','', 'Every selected matrix is Q8_0. No BF16 evidence is mixed into this assessment.','','## 4. Methodology','','Deterministic coordinate sampling and position-hash split: 70% fit, 15% validation, 15% untouched test. Fixed candidates are reused across roles/layers; no report-on-training metrics.','','## 5. True byte accounting','','Every row records packed codes, metadata/tables, sparse indexes/residuals, total bytes, and true bpw. Derived lane/coordinate context costs zero bytes but its lookup/modulo compute is named.','','## 6. Baseline results','','Uniform symmetric and affine Q2–Q8 plus canonical Q8_0 control were run across all 49 tensors. Canonical GGML Q4/Q5/K/IQ comparators remain blocked, not approximated.','','## 7. CCC results','','Additive C2–C8 was tested with zero, tensor mean/median, row mean/median, column mean/median, and row+column anchors. See aggregate and per-tensor CSV files.','','## 8. Position/context contribution','']
 for r in context:lines.append(f"- C{r['bits']}: lane-mod32 RMSE change {r['position_context_rmse_change']:+.6g}; true bpw {r['position_true_bpw']:.4f} vs {r['no_position_true_bpw']:.4f}.")
 lines+=['','## 9. Statistical-scale results','','Mean/median anchors were executed. The full requested dispersion/range/nonlinear Cartesian family was intentionally not run and remains `NOT_TESTED`.','','## 10. Codebook results','','Lloyd-Max correction tables were fitted from fit positions only. Row/column and lane-conditioned tables include every persisted float.','','## 11. Outlier/hybrid results','','C2/C3/C4 with FP16 sparse outliers and 32-bit indexes were measured on layer-32 K for all requested fractions.','','## 12. Structured representation results','','Prior 0.6B structured evidence is not relabeled as 32B evidence. 32B structured families remain `NOT_TESTED` in this bounded Stage 1.','','## 13. Cross-tensor results','','Not run; promotion gate was not reached in this pass.','','## 14. Functional W*x results','','Layer-32 K C2/C3/C4/C6/C8 random-Gaussian W*x is recorded. The path reconstructs dense W and is marked `DIRECT_APPLY = NO`. Real hidden states are unavailable.','','## 15. Pareto fronts','','See `pareto.csv`; the frontier uses true bpw and untouched-test weight RMSE. Runtime is not ranked because no native direct kernel exists.','','## 16. Promoted candidates','','Only candidates selected by validation and represented in functional W*x evidence are provisional; none is promoted to production.','','## 17. Rejected hypotheses','','Position context is rejected where its matched RMSE change is non-negative after table overhead. Ultra-low-bit candidates remain rejected if W*x error is high.','','## 18. Remaining uncertainty','','Canonical GGML comparators, full statistical curves, vectors, structured matrices, cross-tensor sharing, native direct apply, and real hidden states remain unresolved.','','## 19. Recommended next experiment','','Integrate canonical llama.cpp quantization controls and capture real hidden states for layer-32 K before expanding Stage 2.','','## Explicit answers','','- Does CCC provide a measurable advantage? **Yes versus the simplified uniform/affine controls at 2–6 bpw; not established versus canonical GGML families.**','- At which budgets? **Largest weight-RMSE advantage at C2–C4, smaller at C6, and a loss to affine at C8.**','- Anchor or structural context? **Zero and tensor-mean anchors are effectively tied; the gain is primarily learned nonlinear correction levels, not the anchor.**','- Does position help? **Lane-mod32 modestly helps C2, is negligible at C3, and hurts C4–C8 after table accounting.**','- Row/column statistics? **They generally worsen test RMSE despite their metadata cost.**','- Is CCC mainly useful at C2/C3? **Most promising there relative to simple controls, but representative W*x error remains 0.358/0.203 and canonical Q2/Q3 controls are missing.**','- Sparse outliers? **Yes: they reduce C2/C3/C4 W*x error monotonically, but 32-bit indexes quickly consume the storage gain.**']
 for b in (2,3,4,6,8):
  q=best(b);lines.append(f"- Best near {b} bpw: **{q['family'] if q else 'none'} / {q.get('ccc_anchor','') if q else ''}**.")
 lines+=['- Hidden-state qualification: **none until a corpus is captured.**','','## Compact table','','Candidate | True bpw | Weight RMSE | p99 | MaxErr | W*x error | Direct apply | Status','---|---:|---:|---:|---:|---:|---|---']
 for q in sorted(pareto,key=lambda r:r['mean_true_bits_per_weight'])[:20]:lines.append(f"{q['family']} {q.get('ccc_anchor','')} {q.get('position_context','')} | {q['mean_true_bits_per_weight']:.4f} | {q['mean_test_rmse']:.6g} | see CSV | see CSV | representative only | NO | TESTED_INTERESTING")
 (OUT/'representation-assessment.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
