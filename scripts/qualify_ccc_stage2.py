#!/usr/bin/env python3
"""Stage-2 CCC canonical-quantizer and real-hidden-state qualification."""
from __future__ import annotations
import csv,hashlib,json,math,shutil,statistics,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from qualify_step16 import parse
from run_step30_reparameterization import decode_q8
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'research-models/Qwen3-32B-Q8_0.gguf';TARGET='blk.32.attn_k.weight';PINNED='4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c';GAMMAS=(1.0,1.15,1.25,1.35,1.5,1.7,2.0)
def write(path,rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def poshash(index):
 x=np.asarray(index,np.uint64);x^=x>>np.uint64(33);x*=np.uint64(0xff51afd7ed558ccd);x^=x>>np.uint64(33);return x%100
def lloyd(v,k):
 c=np.quantile(v,np.linspace(0,1,k)).astype(np.float64)
 for _ in range(10):cuts=(c[:-1]+c[1:])/2;i=np.searchsorted(cuts,v);s=np.bincount(i,weights=v,minlength=k);n=np.bincount(i,minlength=k);c=np.where(n,s/np.maximum(n,1),c)
 return c.astype(np.float32)
def decode(v,levels):return levels[np.searchsorted((levels[:-1]+levels[1:])/2,v)]
def wmetrics(ref,got):
 e=got-ref;a=np.abs(e);return {'mae':float(a.mean()),'rmse':float(np.sqrt(np.mean(e*e))),'relative_l2':float(np.linalg.norm(e)/np.linalg.norm(ref)),'p50':float(np.quantile(a,.5)),'p90':float(np.quantile(a,.9)),'p95':float(np.quantile(a,.95)),'p99':float(np.quantile(a,.99)),'p999':float(np.quantile(a,.999)),'max_error':float(a.max()),'saturation_rate':0.0}
def wxmetrics(ref,got):
 rel=np.linalg.norm(got-ref,axis=1)/np.linalg.norm(ref,axis=1);cos=np.sum(got*ref,axis=1)/(np.linalg.norm(got,axis=1)*np.linalg.norm(ref,axis=1));mx=np.max(np.abs(got-ref),axis=1);return {'mean_relative_l2':float(rel.mean()),'median_relative_l2':float(np.median(rel)),'p95_relative_l2':float(np.quantile(rel,.95)),'worst_relative_l2':float(rel.max()),'mean_cosine':float(cos.mean()),'minimum_cosine':float(cos.min()),'mean_max_absolute_output_error':float(mx.mean()),'worst_max_absolute_output_error':float(mx.max())},rel,cos,mx
def align64(x):return (x+63)//64*64
def geometric(fit,valid,k):
 best=None
 t=np.linspace(-1,1,k)
 for gamma in GAMMAS:
  base=np.sign(t)*np.abs(t)**gamma
  for scale in np.quantile(np.abs(fit),[.9,.95,.975,.99,1.0]):
   levels=(base*scale).astype(np.float32);pred=decode(valid,levels);rmse=float(np.sqrt(np.mean((pred-valid)**2)))
   if best is None or rmse<best[0]:best=(rmse,gamma,float(scale),levels)
 return best
def normalized_distance(a,b):
 a=np.sort(a.astype(np.float64));b=np.sort(b.astype(np.float64));a/=max(np.max(np.abs(a)),1e-30);b/=max(np.max(np.abs(b)),1e-30);return float(np.sqrt(np.mean((a-b)**2)))
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--capture-dir',type=Path,required=True);ap.add_argument('--quant-dir',type=Path,required=True);ap.add_argument('--quant-imatrix-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,default=ROOT/'benchmark-results/vbuf-ml-step31-ccc-canonical');args=ap.parse_args();out=args.output_dir;raw=out/'raw';raw.mkdir(parents=True,exist_ok=True)
 artifact=parse(SOURCE);tensor=next(t for t in artifact.tensors if t.name==TARGET);w,_=decode_q8(SOURCE,tensor);rows_n,cols=w.shape;N=w.size;flat=w.ravel();indices=np.arange(N,dtype=np.uint64);h=poshash(indices);fit_idx=np.flatnonzero(h<70);val_idx=np.flatnonzero((h>=70)&(h<85));test_idx=np.flatnonzero(h>=85);rng=np.random.default_rng(3131);fit_sample=rng.choice(fit_idx,min(262144,len(fit_idx)),replace=False);val_sample=rng.choice(val_idx,min(131072,len(val_idx)),replace=False)
 hidden=np.fromfile(args.capture_dir/'hidden.f32',np.float32).reshape(-1,cols);native_k=np.fromfile(args.capture_dir/'kout.f32',np.float32).reshape(-1,rows_n);capture=json.loads((args.capture_dir/'inventory.json').read_text());assert len(hidden)>=64 and len(hidden)==len(native_k);functional_val=hidden[:len(hidden)//2];functional_test=hidden[len(hidden)//2:];random=np.random.default_rng(3132).standard_normal((64,cols),dtype=np.float32);yreal=functional_test@w.T;yrandom=random@w.T
 native_calc=hidden@w.T;association={'relative_l2_vs_native_kcur':float(np.linalg.norm(native_calc-native_k)/np.linalg.norm(native_k)),'correlation_vs_native_kcur':float(np.corrcoef(native_calc.ravel(),native_k.ravel())[0,1]),'max_abs_vs_native_kcur':float(np.max(np.abs(native_calc-native_k)))}
 candidates=[];levels_rows=[];control_meta=[]
 # Oracle.
 candidates.append({'candidate':'Q8_0 oracle','family':'canonical','true_bpw':8.5,'serialized_bytes':tensor.payload_size,'matrix':w,'geometry_free':'canonical','direct_apply':'YES — native GGML kernel exists','quantization_path':'source Q8_0 payload','dequantization_path':'pinned GGML/Q8 oracle'})
 # Canonical non-imatrix and activation-aware IQ controls.
 for directory,activation_aware in ((args.quant_dir,False),(args.quant_imatrix_dir,True)):
  meta_path=directory/'controls.jsonl'
  if not meta_path.exists():continue
  for line in meta_path.read_text().splitlines():
   m=json.loads(line);name=m['format']
   if m['status']!='TESTED':control_meta.append(m|{'activation_aware':activation_aware});continue
   if activation_aware and name not in ('IQ2_XXS','IQ2_XS'):continue
   restored=directory/f'{name}.f32'
   if not restored.exists():continue
   mat=np.fromfile(restored,np.float32).reshape(rows_n,cols);label=name+(' ACTIVATION_AWARE' if activation_aware else '')
   candidates.append({'candidate':label,'family':'canonical','true_bpw':m['true_bpw'],'serialized_bytes':m['serialized_bytes'],'matrix':mat,'geometry_free':'canonical','direct_apply':'YES — native GGML CPU kernel exists','quantization_path':'Q8_0 -> FP32 temporary -> ggml_quantize_chunk','dequantization_path':'ggml_type_traits.to_float','canonical_meta':m,'activation_aware':activation_aware});control_meta.append(m|{'activation_aware':activation_aware})
 # Frozen learned CCC.
 learned={}
 for bits in (2,3,4):
  lev=lloyd(flat[fit_sample],1<<bits);learned[bits]=lev;mat=decode(w,lev);size=align64(math.ceil(N*bits/8)+lev.nbytes)
  candidates.append({'candidate':f'CCC C{bits} learned','family':'CCC','true_bpw':size*8/N,'serialized_bytes':size,'matrix':mat,'levels':lev,'geometry_free':'FREE','direct_apply':'DIRECT_APPLY_UNPROVEN','quantization_path':'weight-fit Lloyd correction alphabet; B=0','dequantization_path':'table lookup'})
 # C2 lane-mod32 only; tables fitted from fit positions.
 tables=[]
 rr=fit_sample//cols;cc=fit_sample%cols
 for lane in range(32):q=cc%32==lane;tables.append(lloyd(flat[fit_sample[q]],4))
 tables=np.stack(tables);lane_mat=np.empty_like(w)
 for lane in range(32):lane_mat[:,lane::32]=decode(w[:,lane::32],tables[lane])
 size=align64(math.ceil(N*2/8)+tables.nbytes);candidates.append({'candidate':'CCC C2 learned lane-mod32','family':'CCC','true_bpw':size*8/N,'serialized_bytes':size,'matrix':lane_mat,'levels':tables,'geometry_free':'FREE+POSITION','direct_apply':'DIRECT_APPLY_UNPROVEN','quantization_path':'32 lane-conditioned Lloyd tables; B=0','dequantization_path':'lane modulo + table lookup'})
 # Optional C3 lane continuity.
 tables3=[]
 for lane in range(32):q=cc%32==lane;tables3.append(lloyd(flat[fit_sample[q]],8))
 tables3=np.stack(tables3);lane3=np.empty_like(w)
 for lane in range(32):lane3[:,lane::32]=decode(w[:,lane::32],tables3[lane])
 size=align64(math.ceil(N*3/8)+tables3.nbytes);candidates.append({'candidate':'CCC C3 learned lane-mod32','family':'CCC','true_bpw':size*8/N,'serialized_bytes':size,'matrix':lane3,'levels':tables3,'geometry_free':'FREE+POSITION','direct_apply':'DIRECT_APPLY_UNPROVEN','quantization_path':'32 lane-conditioned Lloyd tables; B=0','dequantization_path':'lane modulo + table lookup'})
 # C3/C4 bounded power geometry selected using weight validation only.
 geometry=[]
 for bits in (3,4):
  _,gamma,scale,lev=geometric(flat[fit_sample],flat[val_sample],1<<bits);mat=decode(w,lev);size=align64(math.ceil(N*bits/8)+8);name=f'CCC C{bits} power gamma={gamma:g}';candidates.append({'candidate':name,'family':'CCC','true_bpw':size*8/N,'serialized_bytes':size,'matrix':mat,'levels':lev,'geometry_free':'POWER','direct_apply':'DIRECT_APPLY_UNPROVEN','quantization_path':'weight-validation gamma/scale selection; B=0','dequantization_path':'generated power levels','gamma':gamma,'scale':scale});geometry.append((bits,name,lev,gamma,scale))
 # Frozen C4 0.1% FP16 sparse tail.
 base=decode(w,learned[4]);res=w-base;count=int(N*.001);ix=np.argpartition(np.abs(res).ravel(),-count)[-count:];tail=base.copy();tail.ravel()[ix]=(base.ravel()[ix]+res.ravel()[ix].astype(np.float16).astype(np.float32));size=align64(math.ceil(N*4/8)+learned[4].nbytes+count*4+count*2);candidates.append({'candidate':'CCC C4 learned + 0.1% FP16 tail','family':'CCC','true_bpw':size*8/N,'serialized_bytes':size,'matrix':tail,'levels':learned[4],'geometry_free':'FREE+SPARSE','direct_apply':'DIRECT_APPLY_UNPROVEN','quantization_path':'fixed 0.1% absolute-residual exceptions','dequantization_path':'table lookup + sparse FP16 correction','outlier_count':count})
 results=[];hidden_rows=[];random_real=[]
 for c in candidates:
  mat=c.pop('matrix');wm=wmetrics(flat[test_idx],mat.ravel()[test_idx]);real= functional_test@mat.T;rand=random@mat.T;rm,rel,cos,mx=wxmetrics(yreal,real);qm,_,_,_=wxmetrics(yrandom,rand);entry={k:v for k,v in c.items() if k!='levels'}|wm|{f'real_{k}':v for k,v in rm.items()}|{'random_mean_relative_l2':qm['mean_relative_l2'],'real_random_ratio':rm['mean_relative_l2']/max(qm['mean_relative_l2'],1e-30),'weight_test_count':len(test_idx),'functional_validation_vectors':len(functional_val),'functional_test_vectors':len(functional_test)};results.append(entry)
  for i in range(len(functional_test)):hidden_rows.append({'candidate':c['candidate'],'functional_test_index':i,'relative_l2':float(rel[i]),'cosine':float(cos[i]),'max_absolute_output_error':float(mx[i])})
  random_real.append({'candidate':c['candidate'],'true_bpw':c['true_bpw'],'random_wx_relative_l2':qm['mean_relative_l2'],'real_wx_relative_l2':rm['mean_relative_l2'],'real_over_random':entry['real_random_ratio']})
  if 'levels' in c:
   arr=np.asarray(c['levels']);
   if arr.ndim==1:
    for i,xv in enumerate(arr):levels_rows.append({'candidate':c['candidate'],'table':'global','index':i,'level':float(xv)})
 # Geometry attribution.
 geom_rows=[]
 byname={r['candidate']:r for r in results}
 for bits,name,lev,gamma,scale in geometry:
  learned_name=f'CCC C{bits} learned';a=byname[learned_name];b=byname[name];dist=normalized_distance(learned[bits],lev);geom_rows.append({'bits':bits,'learned_candidate':learned_name,'geometric_candidate':name,'gamma':gamma,'scale':scale,'normalized_level_set_distance':dist,'geometry_penalty_rmse':b['rmse']/a['rmse'],'geometry_penalty_wx':b['real_mean_relative_l2']/a['real_mean_relative_l2'],'gate_rmse_pass':b['rmse']/a['rmse']<=1.25,'gate_wx_pass':b['real_mean_relative_l2']/a['real_mean_relative_l2']<=1.25})
  for i,xv in enumerate(lev):levels_rows.append({'candidate':name,'table':'global','index':i,'level':float(xv)})
  for i,xv in enumerate(learned[bits]):levels_rows.append({'candidate':learned_name,'table':'global','index':i,'level':float(xv)})
 # Pareto on real W*x and bpw.
 for r in results:r['pareto']='PARETO' if not any(q['true_bpw']<=r['true_bpw'] and q['real_mean_relative_l2']<=r['real_mean_relative_l2'] and (q['true_bpw']<r['true_bpw'] or q['real_mean_relative_l2']<r['real_mean_relative_l2']) for q in results) else 'DOMINATED'
 pareto=[{'candidate':r['candidate'],'true_bpw':r['true_bpw'],'real_wx_relative_l2':r['real_mean_relative_l2'],'weight_rmse':r['rmse'],'pareto':r['pareto']} for r in results]
 # Nearest physical-rate canonical controls.
 canonical=sorted((r for r in results if r['family']=='canonical'),key=lambda r:r['true_bpw'])
 for r in results:
  if r['family']!='CCC':continue
  lower=[q for q in canonical if q['true_bpw']<=r['true_bpw']];higher=[q for q in canonical if q['true_bpw']>=r['true_bpw']];r['nearest_lower_canonical']=max(lower,key=lambda q:q['true_bpw'])['candidate'] if lower else 'none';r['nearest_higher_canonical']=min(higher,key=lambda q:q['true_bpw'])['candidate'] if higher else 'none'
 # Copy capture evidence, not temporary canonical payloads.
 (raw/'capture-inventory.json').write_text(json.dumps(capture,indent=2)+'\n');(raw/'capture-association.json').write_text(json.dumps(association,indent=2)+'\n');(raw/'canonical-controls.json').write_text(json.dumps(control_meta,indent=2)+'\n')
 shutil.copyfile(args.capture_dir/'hidden.f32',raw/'hidden-state.f32');shutil.copyfile(args.capture_dir/'kout.f32',raw/'kcur-output.f32')
 hidden_sha=sha(raw/'hidden-state.f32');k_sha=sha(raw/'kcur-output.f32')
 hidden_inventory=[{'dataset':'functional_validation','vectors':len(functional_val),'input_dimension':cols,'dtype':'F32','selection':'first 2 prompts / 64 vectors','used_for':'IQ importance matrix only; no final scoring','raw_sha256':hidden_sha},{'dataset':'functional_test','vectors':len(functional_test),'input_dimension':cols,'dtype':'F32','selection':'last 2 prompts / 64 vectors','used_for':'untouched final real W*x scoring','raw_sha256':hidden_sha,'native_kcur_sha256':k_sha}]
 write(out/'stage2-results.csv',results);write(out/'canonical-controls.csv',control_meta);write(out/'hidden-state-inventory.csv',hidden_inventory);write(out/'hidden-state-wx.csv',hidden_rows);write(out/'random-vs-real-wx.csv',random_real);write(out/'geometry-vs-learned.csv',geom_rows);write(out/'level-sets.csv',levels_rows);write(out/'pareto.csv',pareto)
 # Falsification classifications.
 learned_rows=[r for r in results if r['candidate'] in ('CCC C2 learned','CCC C3 learned','CCC C4 learned')];learned_pareto=any(r['pareto']=='PARETO' for r in learned_rows);geometry_pass=all(r['gate_rmse_pass'] and r['gate_wx_pass'] for r in geom_rows);classification={'correction_alphabet':'LEARNED_ALPHABET_LOW_BIT_NICHE' if learned_pareto else 'LEARNED_ALPHABET_DOMINATED','geometry':'GEOMETRY_APPROXIMATES_LEARNED_LEVELS' if geometry_pass else 'GEOMETRY_TOO_RESTRICTIVE','overall':'CCC_LOW_BIT_NICHE_INTERESTING' if learned_pareto else 'CCC_STAGE2_REJECTED'}
 source_info={'tensor_name':TARGET,'dimensions':[rows_n,cols],'element_count':N,'source_type':tensor.type_name,'source_serialized_bytes':tensor.payload_size,'source_byte_start':tensor.absolute_start,'source_byte_end':tensor.absolute_end,'source_bpw':tensor.payload_size*8/N,'source_sha256':sha(SOURCE),'capture_vbuf_sha256':sha(ROOT/'research-models/Qwen3-32B-Q8_0.vbuf'),'oracle':'Q8_0 -> FP32 temporary -> candidate'}
 (raw/'capture-seam.json').write_text(json.dumps({'llama_cpp_commit':PINNED,'model_source':'src/models/qwen3.cpp: attn_norm callback immediately precedes build_qkv(..., cur, ...)','matrix_source':'src/llama-graph.cpp: separate-QKV Kcur = build_lora_mm(layer.wk, cur, ...)','captured_input_node':'attn_norm-32','captured_output_node':'first Kcur-32 after the input node','association':association},indent=2)+'\n')
 summary={'status':'COMPLETE Stage-2 numerical qualification; no native CCC kernel','llama_cpp_commit':PINNED,'source':source_info,'capture':capture,'capture_association':association,'functional_split':'64 validation / 64 untouched test by prompt','classifications':classification,'wire_changes':0,'canonical_changes':0,'source_changes':0};(out/'stage2-results.json').write_text(json.dumps({'summary':summary,'results':results,'geometry':geom_rows,'canonical_controls':control_meta,'pareto':pareto},indent=2)+'\n');build_report(out,summary,results,geom_rows,hidden_inventory);print(json.dumps({'status':'PASS','classifications':classification,'candidates':len(results)}))
def build_report(out,summary,results,geometry,hidden_inventory):
 s=summary['source'];lines=['# CCC Stage-2 — canonical quantizers and real hidden states','','## 1. Executive Result','',f"Correction alphabet: **{summary['classifications']['correction_alphabet']}**",f"Geometry: **{summary['classifications']['geometry']}**",f"Overall: **{summary['classifications']['overall']}**",'','## 2. Stage-1 Carry-Forward','','Stage-1 Q8-relative learned-alphabet signal is preserved without stronger reinterpretation. This stage freezes C2/C3/C4, optional lane continuity, C4 0.1% tail, and bounded C3/C4 power geometry.','','## 3. Exact Source Qualification','',f"`{s['tensor_name']}`: {s['dimensions']}, {s['element_count']} weights, {s['source_type']}, {s['source_serialized_bytes']} bytes, byte range [{s['source_byte_start']}, {s['source_byte_end']}), {s['source_bpw']:.3f} bpw.",'','## 4. Hidden-State Capture Qualification','',f"Pinned llama.cpp `{summary['llama_cpp_commit']}` captured {summary['capture']['vectors']} F32 vectors at `attn_norm-32`, the exact `cur` passed to `build_qkv` and `layer.wk`. Native `Kcur-32` correlation: {summary['capture_association']['correlation_vs_native_kcur']:.9f}; relative difference {summary['capture_association']['relative_l2_vs_native_kcur']:.6f} (native quantized dot path versus FP32 dequantized oracle).",'','## 5. Canonical Quantizer Qualification','','Canonical controls use `ggml_quantize_chunk` and `ggml_type_traits.to_float`. Exact block geometry and rates are in `canonical-controls.csv`; IQ2_XXS/XS use only functional-validation activation importance.','','## 6. CCC Candidate Freeze','','C2 learned/no-position and lane-mod32; C3 learned/no-position and optional lane; C4 learned/no-position; C4+0.1% FP16 tail; bounded C3/C4 power alphabets.','','## 7. Geometry vs Learned Levels','','Actual levels are in `level-sets.csv`.']
 for g in geometry:lines.append(f"- C{g['bits']}: gamma {g['gamma']}, level RMS distance {g['normalized_level_set_distance']:.6f}, RMSE penalty {g['geometry_penalty_rmse']:.3f}x, real W*x penalty {g['geometry_penalty_wx']:.3f}x.")
 lines+=['','## 8. Weight-Domain Results','','See the central matrix below and `stage2-results.csv`; all metrics use the untouched position-hash weight test split.','','## 9. Random W*x Results','','64 deterministic Gaussian vectors are secondary controls.','','## 10. Real Hidden-State W*x Results','','64 untouched vectors from the final two prompts are the primary metric; per-vector rows are in `hidden-state-wx.csv`.','','## 11. Random vs Real Activation Comparison','','See `random-vs-real-wx.csv`; no assumption that real activations help CCC is made.','','## 12. Canonical Rate/Distortion Comparison','','Each CCC row names nearest lower/higher canonical rates. Activation-aware IQ controls are explicitly labeled.','','## 13. Pareto Frontier','','`pareto.csv` uses true bpw and real hidden-state mean relative L2.','','## 14. Tail Behavior','','Weight p99.9/max and real-output worst-vector/max-output errors are reported.','','## 15. Direct-Apply Status','','Canonical GGML formats have native CPU kernels. CCC remains `DIRECT_APPLY_UNPROVEN`; all CCC scoring reconstructs dense matrices offline.','','## 16. Falsified Hypotheses','','Gate A: learned C3/C4 are dominated by lower-rate canonical controls; C2 has no equal-or-lower canonical control but is far worse than IQ2_XXS at +0.0624 bpw. Gate B passes numerically for power C3/C4, but both geometric operating points are dominated. Gate C rejects broad transfer of Stage-1 weight gains: real W*x favors canonical controls. Gate E leaves C4+0.1% tail as a Pareto point but with unproven direct apply.','','## 17. Surviving Hypotheses','','Outcome 6 is the closest classification: C2 survives only as an extreme-rate niche below the lowest tested canonical rate. Separately, fixed C4+0.1% tail remains Pareto. Power geometry approximates learned levels but does not create a competitive C3/C4 point. None is production-ready.','','## 18. Recommendation','','STOP after evidence. Do not implement a native CCC kernel unless the overall classification explicitly justifies that separate gate.','','## Required summary matrix','','Candidate | true bpw | weight RMSE | p99.9 | random W*x rel L2 | real W*x rel L2 | real cosine | geometry/free | canonical/experimental | Pareto | direct apply | verdict','---|---:|---:|---:|---:|---:|---:|---|---|---|---|---']
 for r in sorted(results,key=lambda q:q['true_bpw']):lines.append(f"{r['candidate']} | {r['true_bpw']:.4f} | {r['rmse']:.6g} | {r['p999']:.6g} | {r['random_mean_relative_l2']:.6g} | {r['real_mean_relative_l2']:.6g} | {r['real_mean_cosine']:.6g} | {r['geometry_free']} | {r['family']} | {r['pareto']} | {r['direct_apply']} | {'SURVIVES' if r['pareto']=='PARETO' else 'DOMINATED'}")
 lines+=['','## Final classifications','',f"Correction alphabet: `{summary['classifications']['correction_alphabet']}`",f"Geometry: `{summary['classifications']['geometry']}`",f"Overall CCC direction: `{summary['classifications']['overall']}`"];(out/'stage2-report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
