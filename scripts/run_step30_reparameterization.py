#!/usr/bin/env python3
"""Step 30 research-only reconstructable parameterization qualification.

This deliberately works from the qualified Q8_0 GGUF source as an immutable
oracle.  Candidates are never written back to vBuf and no runtime integration
is attempted.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualify_step16 import parse

ROOT=Path(__file__).resolve().parents[1]
ARTIFACTS={"0.6B":ROOT/"research-models/Qwen3-0.6B-Q8_0.gguf","32B":ROOT/"research-models/Qwen3-32B-Q8_0.gguf"}
VBUFS={"0.6B":ROOT/"research-models/Qwen3-0.6B-Q8_0.vbuf","32B":ROOT/"research-models/Qwen3-32B-Q8_0.vbuf"}
EXPECTED={"0.6B":"9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031","32B":"2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169"}
SELECTED={"0.6B":["blk.0.attn_output.weight","blk.0.ffn_down.weight"],"32B":["blk.0.attn_output.weight"]}

def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def decode_q8(path,tensor):
 with path.open("rb") as f:
  f.seek(tensor.absolute_start); raw=f.read(tensor.payload_size)
 blocks=tensor.elements//32
 packed=np.frombuffer(raw,dtype=np.uint8).reshape(blocks,34)
 scales=np.frombuffer(packed[:,:2].tobytes(),dtype="<f2").astype(np.float32)
 q=packed[:,2:].view(np.int8).astype(np.float32)
 # GGUF stores matrix dimensions in column-major GGML order; reverse for W[out,in].
 return (q*scales[:,None]).reshape(tuple(reversed(tensor.shape))).astype(np.float32,copy=False),len(raw)
def randomized_svd(a,rank,rng):
 m,n=a.shape; oversample=min(8,max(2,rank//2)); qdim=min(n,rank+oversample)
 omega=rng.standard_normal((n,qdim),dtype=np.float32)
 y=a@omega
 for _ in range(1): y=a@(a.T@y)
 q,_=np.linalg.qr(y,mode="reduced")
 small=q.T@a; ub,s,vh=np.linalg.svd(small,full_matrices=False)
 return ((q@ub[:,:rank])*s[:rank]).astype(np.float32),vh[:rank,:].astype(np.float32)
def timed(fn):
 start=time.perf_counter(); value=fn(); return value,(time.perf_counter()-start)*1000
def candidate(name,w,rank,rng):
 m,n=w.shape
 if name.startswith("svd-r"):
  (u,v),fit=timed(lambda:randomized_svd(w,rank,rng));
  def apply(x):return u@(v@x)
  aux=0; detail="plain low-rank direct factor application"
 elif name.startswith("row-centered"):
  mean=w.mean(axis=1).astype(np.float32); centered=w-mean[:,None]
  (u,v),fit=timed(lambda:randomized_svd(centered,rank,rng))
  def apply(x):return mean[:,None]*x.sum(axis=0)[None,:]+u@(v@x)
  aux=m; detail="row mean relocated into a sum-of-inputs linear term"
 elif name.startswith("diag-scaled"):
  row=np.sqrt(np.mean(w*w,axis=1)+1e-12).astype(np.float32); col=np.sqrt(np.mean(w*w,axis=0)+1e-12).astype(np.float32)
  scaled=w/(row[:,None]*col[None,:])
  (u,v),fit=timed(lambda:randomized_svd(scaled,rank,rng))
  def apply(x):return row[:,None]*(u@(v@(col[:,None]*x)))
  aux=m+n; detail="row/column scale factors relocated outside a low-rank core"
 elif name.startswith("permuted"):
  perm=np.argsort(np.linalg.norm(w,axis=0)).astype(np.uint32); wp=w[:,perm]
  (u,v),fit=timed(lambda:randomized_svd(wp,rank,rng))
  def apply(x):return u@(v@x[perm,:])
  aux=n; detail="column permutation stored separately; direct application permutes inputs"
 else: raise ValueError(name)
 x=rng.standard_normal((n,8),dtype=np.float32)
 y=w@x
 yh,apply_ms=timed(lambda:apply(x))
 wh,_=timed(lambda:apply(np.eye(n,dtype=np.float32))) if n<=2048 else (None,0)
 if wh is None:
  # Reconstruction measurement uses the factor product only; no full runtime copy is required.
  recon, reconstruction_ms=timed(lambda: u@v if name.startswith(("svd","row-centered","diag-scaled","permuted")) else None)
  if name.startswith("row-centered"): recon= recon+mean[:,None]
  elif name.startswith("diag-scaled"): recon=row[:,None]*recon*col[None,:]
  elif name.startswith("permuted"): full=np.empty_like(recon);full[:,perm]=recon;recon=full
 else: recon=wh; reconstruction_ms=0
 matrix_error=float(np.linalg.norm(recon-w)/np.linalg.norm(w)); action_error=float(np.linalg.norm(yh-y)/np.linalg.norm(y)); max_action=float(np.max(np.abs(yh-y)) / max(1e-12,float(np.max(np.abs(y)))))
 scalars=int(u.size+v.size+aux); stored_bytes=int(u.nbytes+v.nbytes+aux*(4 if name.startswith("diag-scaled") or name.startswith("row-centered") else 4))
 if name.startswith("permuted"): stored_bytes=int(u.nbytes+v.nbytes+perm.nbytes)
 return {"candidate":name,"reconstruction_class":"LOSSY","detail":detail,"rows":m,"cols":n,"rank":rank,"matrix_relative_error":matrix_error,"action_relative_error":action_error,"action_max_relative_error":max_action,"fit_ms":fit,"reconstruction_ms":reconstruction_ms,"direct_apply_ms":apply_ms,"stored_scalar_parameters":scalars,"stored_bytes":stored_bytes,"effective_bits_per_original_weight":stored_bytes*8/(m*n),"full_runtime_dense_copy":False,"auxiliary_parameters":aux}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=ROOT/"benchmark-results/vbuf-ml-step30-reparameterization");ap.add_argument("--seed",type=int,default=3029);args=ap.parse_args();out=args.output_dir.resolve();(out/"raw").mkdir(parents=True,exist_ok=True)
 prov={}
 for model,p in ARTIFACTS.items():
  got=sha(p)
  if got!=EXPECTED[model]:raise SystemExit(f"hash mismatch {p}")
  prov[model]={"gguf_path":str(p.relative_to(ROOT)),"gguf_bytes":p.stat().st_size,"gguf_sha256":got,"vbuf_path":str(VBUFS[model].relative_to(ROOT)),"vbuf_sha256":sha(VBUFS[model]),"immutable":True}
 (out/"artifact-provenance.json").write_text(json.dumps(prov,indent=2)+"\n")
 rng=np.random.default_rng(args.seed); rows=[]; tensors=[]
 for model,path in ARTIFACTS.items():
  artifact=parse(path); by={t.name:t for t in artifact.tensors}
  for tensor_name in SELECTED[model]:
   t=by[tensor_name]; w,q8_bytes=decode_q8(path,t);tensors.append({"model":model,"tensor":tensor_name,"shape":list(w.shape),"q8_bytes":q8_bytes,"weights":int(w.size)})
   baseline={"candidate":"q8-reference","reconstruction_class":"EXACT","detail":"qualified original Q8_0 payload; action oracle uses offline dequantized W","rows":w.shape[0],"cols":w.shape[1],"rank":"","matrix_relative_error":0.0,"action_relative_error":0.0,"action_max_relative_error":0.0,"fit_ms":0.0,"reconstruction_ms":0.0,"direct_apply_ms":0.0,"stored_scalar_parameters":int(w.size+w.size//32),"stored_bytes":q8_bytes,"effective_bits_per_original_weight":q8_bytes*8/w.size,"full_runtime_dense_copy":False,"auxiliary_parameters":w.size//32,"model":model,"tensor":tensor_name,"q8_payload_bytes":q8_bytes}
   rows.append(baseline)
   for name,rank in (("svd-r8",8),("svd-r16",16),("svd-r32",32),("row-centered-r16",16),("diag-scaled-r16",16),("permuted-svd-r16",16)):
    result=candidate(name,w,rank,rng);result.update({"model":model,"tensor":tensor_name,"q8_payload_bytes":q8_bytes});rows.append(result)
   (out/"raw"/"runs.jsonl").open("a").write(json.dumps({"model":model,"tensor":tensor_name,"shape":list(w.shape),"candidates":[r for r in rows if r["model"]==model and r["tensor"]==tensor_name]})+"\n")
 fields=list(rows[0]);
 with (out/"candidate-results.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
 with (out/"degrees-of-freedom.csv").open("w",newline="") as f:
  fs=["model","tensor","candidate","stored_scalar_parameters","stored_bytes","effective_bits_per_original_weight","q8_payload_bytes","full_runtime_dense_copy"];w=csv.DictWriter(f,fieldnames=fs,lineterminator="\n");w.writeheader();w.writerows([{k:r[k] for k in fs} for r in rows])
 with (out/"action-validation.csv").open("w",newline="") as f:
  fs=["model","tensor","candidate","reconstruction_class","matrix_relative_error","action_relative_error","action_max_relative_error","direct_apply_ms","reconstruction_ms"];w=csv.DictWriter(f,fieldnames=fs,lineterminator="\n");w.writeheader();w.writerows([{k:r[k] for k in fs} for r in rows])
 summary={"seed":args.seed,"selected_tensors":tensors,"reparameterization_search":["plain low-rank SVD factors","row-centered low-rank","diagonal row/column scaling plus low-rank core","column permutation plus low-rank core"],"information_relocation":["row means","row/column scales","column permutation"],"butterfly_tested":False,"butterfly_reason":"Not selected for this bounded direct-application qualification; low-rank and scaling reparameterizations were tested instead.","real_hidden_state_validation":"unavailable; random action probes used","runtime_dense_copy":False,"canonical_artifacts_changed":False,"writer_side_cost":"randomized SVD fitting measured separately; fitting is offline-only","discovery":"plain low-rank factors had the simplest direct application; diagonal scaling and permutations were comparators, not assumed wins"}
 (out/"search-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
 (out/"environment.json").write_text(json.dumps({"host":platform.node(),"python":platform.python_version(),"numpy":np.__version__,"seed":args.seed,"cpu_only":True,"wire_changes":False,"base_shift":3,"base_step":8},indent=2)+"\n")
 (out/"manifest.json").write_text(json.dumps({"step":30,"status":"PARTIAL — initial reparameterization pilot; see structured-matrix-summary.json","classification":"research-only; no artifact emission","tensor_count":len(tensors),"candidate_count":len(rows),"rank_set":[8,16,32],"validation_levels":["matrix reconstruction","random action preservation"],"full_model_copy_runtime":False},indent=2)+"\n")
 print(f"PASS — {len(rows)} candidate rows across {len(tensors)} selected tensors written to {out}")
if __name__=="__main__":main()
