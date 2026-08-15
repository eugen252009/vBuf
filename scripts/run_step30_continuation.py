#!/usr/bin/env python3
"""Step-30 continuation: five missing families and residual quantization pilots."""
from __future__ import annotations
import csv,json,math,statistics,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from qualify_step16 import parse
from run_step30_reparameterization import ARTIFACTS,decode_q8,randomized_svd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"benchmark-results/vbuf-ml-step30-reparameterization";ALIGN=64
MODEL="0.6B";TENSOR="blk.0.attn_k.weight";SEED=3032

def write(name,rows):
 fields=[]
 for row in rows:
  for key in row:
   if key not in fields:fields.append(key)
 with (OUT/name).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
def bench(fn,x,reps=3):
 vals=[];y=None
 for _ in range(reps):t=time.perf_counter_ns();y=fn(x);vals.append((time.perf_counter_ns()-t)/1e6)
 return y,statistics.median(vals)
def account(arrays,meta,groups=None):
 raw={"indexes_bytes":0,"primary_table_bytes":0,"residual_table_bytes":0,"scales_bytes":0,"signs_bytes":0,"factor_bytes":0,"permutations_bytes":0,"payload_bytes":0}
 for name,a in arrays.items():
  if "index" in name or "code" in name:cat="indexes_bytes"
  elif "residual" in name or "correction" in name:cat="residual_table_bytes"
  elif "primary" in name or "prototype" in name:cat="primary_table_bytes"
  elif "scale" in name or "diagonal" in name:cat="scales_bytes"
  elif "sign" in name:cat="signs_bytes"
  elif "perm" in name:cat="permutations_bytes"
  elif any(s in name for s in ("core","factor","stage","householder","generator")):cat="factor_bytes"
  else:cat="payload_bytes"
  raw[cat]+=a.nbytes
 descriptor={"alignment":ALIGN,"arrays":[{"name":n,"dtype":str(a.dtype),"shape":list(a.shape)} for n,a in arrays.items()]}|meta
 metadata=len(json.dumps(descriptor,separators=(",",":"),sort_keys=True).encode());cur=metadata;pad=0
 for a in arrays.values():n=(cur+63)//64*64;pad+=n-cur;cur=n+a.nbytes
 final=(cur+63)//64*64;pad+=final-cur
 return raw|{"metadata_bytes":metadata,"padding_bytes":pad,"partition_descriptor_bytes":0,"serialized_bytes":final}
def entropy(a):
 _,c=np.unique(a,return_counts=True);p=c/c.sum();return float(-(p*np.log2(p)).sum())
def errors(w,wh,ref,got):
 diff=got-ref;den=np.linalg.norm(ref,axis=1).clip(1e-12);per=np.linalg.norm(diff,axis=1)/den
 return {"matrix_relative_error":float(np.linalg.norm(wh-w)/np.linalg.norm(w)),"action_relative_error":float(np.linalg.norm(diff)/np.linalg.norm(ref)),"median_output_relative_error":float(np.median(per)),"p95_output_relative_error":float(np.quantile(per,.95)),"max_output_relative_error":float(per.max()),"value_bias":float(np.mean(wh-w)),"sign_flip_rate":float(np.mean(np.signbit(wh)!=np.signbit(w))),"zero_crossing_error_rate":float(np.mean((wh==0)!=(w==0)))}
def row_apply(values,x):
 y=np.empty((1024,x.shape[1]),np.float32)
 for r in range(len(y)):y[r]=values(r)@x
 return y

def tt_fit(w,rank,dims=(4,4,4,4,4)):
 n=len(dims);tensor=w.reshape(*dims,*dims).transpose(*sum(([i,n+i] for i in range(n)),[]));shape=[d*d for d in dims];cores=[];r0=1;z=tensor.reshape(shape)
 for k in range(n-1):
  z=z.reshape(r0*shape[k],-1);u,s,v=np.linalg.svd(z,full_matrices=False);r=min(rank,len(s));cores.append(u[:,:r].reshape(r0,dims[k],dims[k],r).astype(np.float32));z=(s[:r,None]*v[:r]);r0=r
 cores.append(z.reshape(r0,dims[-1],dims[-1],1).astype(np.float32));return cores
def tt_apply(cores,x):
 n=len(cores);inputs=list(range(n));outputs=list(range(n,2*n));ranks=list(range(2*n,3*n-1));ops=[]
 for k,g in enumerate(cores):ops.extend([g,[ranks[k-1] if k else 3*n,outputs[k],inputs[k],ranks[k] if k<n-1 else 3*n+1]])
 xt=x.reshape(*([4]*n),x.shape[1]);batch=3*n+2;ops.extend([xt,inputs+[batch]])
 # Singleton boundary-rank operands close the first and final TT ranks.
 ops.extend([np.ones(1,np.float32),[3*n],np.ones(1,np.float32),[3*n+1],outputs+[batch]])
 return np.einsum(*ops,optimize=True).reshape(1024,-1)
def tt_reconstruct(cores):return tt_apply(cores,np.eye(1024,dtype=np.float32))

def butterfly_apply(stages,x,perm=None,diag=None):
 y=x.astype(np.float32,copy=True);n=y.shape[0]
 if perm is not None:y=y[perm]
 for s,blocks in enumerate(stages):
  stride=1<<s;z=y.reshape(-1,2,stride,y.shape[1]).transpose(0,2,1,3);y=np.einsum('gsij,gsjb->gsib',blocks,z).transpose(0,2,1,3).reshape(n,-1)
 if diag is not None:y=diag[:,None]*y
 return y
def butterfly_matrix(stages,perm=None):return butterfly_apply(stages,np.eye(1024,dtype=np.float32),perm)
def fit_butterfly(w,seed,branches=1):
 rng=np.random.default_rng(seed);bs=[];bases=[];perms=[]
 for b in range(branches):
  stages=[]
  for s in range(10):
   stride=1<<s;theta=rng.uniform(-math.pi,math.pi,512).astype(np.float32).reshape(1024//(2*stride),stride);stages.append(np.stack([np.stack([np.cos(theta),-np.sin(theta)],2),np.stack([np.sin(theta),np.cos(theta)],2)],2))
  perm=rng.permutation(1024).astype(np.uint16) if b else np.arange(1024,dtype=np.uint16);base=butterfly_matrix(stages,perm);bs.append(stages);bases.append(base);perms.append(perm)
 # role-specific row coefficients solve against one value from each branch per column.
 diag=np.empty((branches,1024),np.float32)
 for i in range(1024):
  a=np.stack([q[i] for q in bases],1);diag[:,i]=np.linalg.lstsq(a,w[i],rcond=None)[0]
 return bs,perms,diag
def generalized_apply(branches,perms,diag,x):return sum((butterfly_apply(s,x,p,diag[b]) for b,(s,p) in enumerate(zip(branches,perms))),start=np.zeros((1024,x.shape[1]),np.float32))

def ldr_fit(w,rank,sign=1):
 # Delta(W)=W-Z W Z^T, Z is the nilpotent lower shift (or signed variant).
 delta=w.copy();delta[1:,1:]-=sign*w[:-1,:-1];u,v=randomized_svd(delta,rank,np.random.default_rng(55+rank));return u,v,sign
def ldr_apply(u,v,sign,x):
 y=np.zeros((1024,x.shape[1]),np.float32);xs=x.copy();ug=u.copy()
 for _ in range(1024):y+=ug@(v@xs);ug=np.vstack((np.zeros((1,u.shape[1]),np.float32),sign*ug[:-1]));xs=np.vstack((xs[1:],np.zeros((1,x.shape[1]),np.float32)))
 return y
def ldr_reconstruct(u,v,sign):
 w=u@v
 for i in range(1,1024):w[i,1:]+=sign*w[i-1,:-1]
 return w
def house_apply(vecs,x):
 y=x.copy()
 for v in vecs:y-=2*v[:,None]*(v@y)[None,:]
 return y
def fit_orthogonal(w,k):
 rng=np.random.default_rng(70+k);v=rng.standard_normal((k,1024),dtype=np.float32);v/=np.linalg.norm(v,axis=1,keepdims=True);q=house_apply(v,np.eye(1024,dtype=np.float32));left=np.ones(1024,np.float32);right=np.ones(1024,np.float32)
 for _ in range(5):z=q*right[None,:];left=(w*z).sum(1)/(z*z).sum(1).clip(1e-12);z=left[:,None]*q;right=(w*z).sum(0)/(z*z).sum(0).clip(1e-12)
 return v,left,right

def scalar_levels(values,k,dtype=np.float32):
 z=values.astype(dtype);c=np.quantile(z,np.linspace(0,1,k)).astype(dtype)
 for _ in range(8):cuts=(c[:-1]+c[1:])/2;i=np.searchsorted(cuts,z);s=np.bincount(i,weights=z,minlength=k);n=np.bincount(i,minlength=k);c=np.where(n,s/np.maximum(n,1),c)
 return c.astype(np.float32),i.astype(np.uint8)
def hierarchical(w,kind,block=0,dtype=np.float32):
 flat=w.ravel();
 if kind=="linear":
  lo,hi=float(flat.min()),float(flat.max());p=np.linspace(lo,hi,16,dtype=dtype);pi=np.abs(flat[:,None]-p).argmin(1);res=flat-p[pi];mx=max(abs(float(res.min())),abs(float(res.max())));r=np.linspace(-mx,mx,16,dtype=dtype);ri=np.abs(res[:,None]-r).argmin(1)
 elif kind in ("global","conditional"):
  p,pi=scalar_levels(flat,16,dtype);res=flat-p[pi]
  if kind=="global":r,ri=scalar_levels(res,16,dtype)
  else:
   r=np.zeros((16,16),np.float32);ri=np.zeros(len(flat),np.uint8)
   for a in range(16):mask=pi==a
   # second loop avoids fitting empty views ambiguously
   for a in range(16):
    mask=pi==a
    if mask.any():r[a],ri[mask]=scalar_levels(res[mask],16,dtype)
 elif kind=="block":
  assert block;count=len(flat)//block;p=np.empty((count,16),np.float32);r=np.empty((count,16),np.float32);pi=np.empty(len(flat),np.uint8);ri=np.empty(len(flat),np.uint8)
  for b in range(count):
   sl=slice(b*block,(b+1)*block);p[b],pi[sl]=scalar_levels(flat[sl],16,dtype);rr=flat[sl]-p[b,pi[sl]];r[b],ri[sl]=scalar_levels(rr,16,dtype)
 else:raise ValueError(kind)
 code=((pi<<4)|ri).astype(np.uint8)
 def decode_row(row):
  ids=code.reshape(w.shape)[row];a=ids>>4;b=ids&15
  if kind=="conditional":return p[a]+r[a,b]
  if kind=="block":
   pos=np.arange(row*w.shape[1],(row+1)*w.shape[1]);g=pos//block;return p[g,a]+r[g,b]
  return p[a]+r[b]
 return {"packed_indexes":code.reshape(w.shape),"primary_table":np.asarray(p,np.float32),"residual_table":np.asarray(r,np.float32)},decode_row,pi,ri

def sign_residual(w,magbits,resbits,dtype=np.float32):
 mag=np.abs(w).ravel();k=1<<magbits;q=1<<resbits;m,mi=scalar_levels(mag,k,dtype);res=mag-m[mi];r=np.zeros((k,q),np.float32);ri=np.zeros(len(mag),np.uint8)
 for a in range(k):
  mask=mi==a
  if mask.any():r[a],ri[mask]=scalar_levels(res[mask],q,dtype)
 sign=(w.ravel()<0).astype(np.uint8);code=((sign<<(magbits+resbits))|(mi<<resbits)|ri).astype(np.uint8).reshape(w.shape)
 def row(i):z=code[i];sg=np.where(z>>(magbits+resbits),-1,1);a=(z>>resbits)&(k-1);b=z&(q-1);return sg*(m[a]+r[a,b])
 return {"packed_indexes":code,"primary_table":m,"residual_table":r},row,mi,ri

def vector_residual(w,width,conditional,orientation="input"):
 if orientation=="input":vectors=w.reshape(-1,width)
 else:vectors=w.T.reshape(-1,width)
 p,pi=kmeans(vectors,16,80+width);res=vectors-p[pi]
 if conditional:
  rd=np.zeros((16,16,width),np.float32);ri=np.zeros(len(vectors),np.uint8)
  for a in range(16):
   mask=pi==a
   if mask.any():rd[a],ri[mask]=kmeans(res[mask],min(16,len(res[mask])),90+a+width)
 else:rd,ri=kmeans(res,16,100+width)
 code=((pi<<4)|ri).astype(np.uint8)
 def reconstruct():
  a=code>>4;b=code&15;z=p[a]+(rd[a,b] if conditional else rd[b]);z=z.reshape((w.shape if orientation=="input" else w.T.shape));return z if orientation=="input" else z.T
 cluster_norms=[float(np.mean(np.linalg.norm(res[pi==a],axis=1))) if np.any(pi==a) else 0.0 for a in range(16)]
 return {"packed_indexes":code,"primary_prototypes":p,"residual_prototypes":rd},reconstruct,pi,ri,cluster_norms
def kmeans(v,k,seed):
 rng=np.random.default_rng(seed);k0=k;c=v[rng.choice(len(v),k0,replace=False)].copy()
 for _ in range(7):
  ids=[]
  for s in range(0,len(v),4096):ids.append(((v[s:s+4096,None]-c[None])**2).sum(2).argmin(1))
  i=np.concatenate(ids);sums=np.zeros_like(c);cnt=np.zeros(k0,int);np.add.at(sums,i,v);np.add.at(cnt,i,1);c=np.where(cnt[:,None]>0,sums/np.maximum(cnt[:,None],1),c)
 return c.astype(np.float32),i.astype(np.uint8)

def record(family,variant,w,x,ref,arrays,apply,reconstruct,fit_ms,extra=None):
 got,app=bench(apply,x);wh=reconstruct();st=account(arrays,{"family":family,"variant":variant,"shape":list(w.shape)});dense=bench(lambda z:w@z,x,5)[1];f64_ref=w.astype(np.float64)@x.astype(np.float64);f64_got=wh.astype(np.float64)@x.astype(np.float64);row={"family":family,"variant":variant,"model":MODEL,"tensor":TENSOR,"fit_ms":fit_ms,"dense_apply_ms":dense,"compact_apply_ms":app,"compute_inflation":app/dense,"temporary_bytes":x.nbytes+got.nbytes+max((a.nbytes for a in arrays.values()),default=0),"bytes_touched":st["serialized_bytes"]+x.nbytes+got.nbytes,"f64_accumulation_action_error":float(np.linalg.norm(f64_got-f64_ref)/np.linalg.norm(f64_ref))}|st|errors(w,wh,ref,got);row["effective_bits_per_weight"]=st["serialized_bytes"]*8/w.size;row["bandwidth_reduction"]=1114112/st["serialized_bytes"];row["bandwidth_compute_ratio"]=row["bandwidth_reduction"]/row["compute_inflation"]
 if extra:row.update(extra)
 return row,wh

def main():
 a=parse(ARTIFACTS[MODEL]);t=next(z for z in a.tensors if z.name==TENSOR);w,_=decode_q8(ARTIFACTS[MODEL],t);x=np.random.default_rng(SEED).standard_normal((1024,8),dtype=np.float32);ref=w@x;missing=[]
 for rank in (2,4,8):
  q=time.perf_counter();cores=tt_fit(w,rank);fit=(time.perf_counter()-q)*1000;missing.append(record("Tensor Train / MPO",f"4^5-rank{rank}",w,x,ref,{f"core{i}":c for i,c in enumerate(cores)},lambda z,c=cores:tt_apply(c,z),lambda c=cores:tt_reconstruct(c),fit,{"tensorization":"[4,4,4,4,4]x[4,4,4,4,4]","ranks":rank})[0])
 for branches,family in ((1,"Butterfly"),(2,"Generalized / Deformable Butterfly")):
  q=time.perf_counter();bs,ps,d=fit_butterfly(w,120+branches,branches);fit=(time.perf_counter()-q)*1000;arrays={f"branch{b}_stage{s}":z for b,bx in enumerate(bs) for s,z in enumerate(bx)}|{f"permutation{b}":p for b,p in enumerate(ps)}|{"diagonal_scales":d};apply=lambda z,bs=bs,ps=ps,d=d:generalized_apply(bs,ps,d,z);missing.append(record(family,f"{branches}-branch-10-stage",w,x,ref,arrays,apply,lambda apply=apply:apply(np.eye(1024,dtype=np.float32)),fit)[0])
 for rank,sign in ((2,1),(4,1),(4,-1)):
  q=time.perf_counter();u,v,s=ldr_fit(w,rank,sign);fit=(time.perf_counter()-q)*1000;missing.append(record("Low Displacement Rank",f"nilpotent-shift-r{rank}-sign{sign}",w,x,ref,{"left_generator":u,"right_generator":v},lambda z,u=u,v=v,s=s:ldr_apply(u,v,s,z),lambda u=u,v=v,s=s:ldr_reconstruct(u,v,s),fit,{"displacement":"Delta(W)=W-ZWZ^T; nilpotent lower shift Z"})[0])
 for k in (4,8):
  q=time.perf_counter();hv,l,r=fit_orthogonal(w,k);fit=(time.perf_counter()-q)*1000;apply=lambda z,hv=hv,l=l,r=r:l[:,None]*house_apply(hv,r[:,None]*z);missing.append(record("Structured Orthogonal",f"{k}-householder+D1/D2",w,x,ref,{"householder_vectors":hv,"left_scales":l,"right_scales":r},apply,lambda apply=apply:apply(np.eye(1024,dtype=np.float32)),fit)[0])
 write("missing-family-screen.csv",missing)
 # Hierarchical candidates and F32/F64 comparison.
 hier=[];fits=[];wh_by={}
 for kind in ("linear","global","conditional"):
  for dtype in (np.float32,np.float64):
   q=time.perf_counter();arr,row,pi,ri=hierarchical(w,kind,dtype=dtype);fit=(time.perf_counter()-q)*1000;apply=lambda z,row=row:row_apply(row,z);rec=lambda row=row:np.stack([row(i) for i in range(1024)]);r,wh=record("Hierarchical Primary + Correction",f"{kind}-4+4-{dtype.__name__}",w,x,ref,arr,apply,rec,fit,{"primary_entropy":entropy(pi),"correction_entropy":entropy(ri),"primary_utilization":len(np.unique(pi)),"residual_utilization":len(np.unique(ri)),"decode_limitation":"bounded row decode; native fused nibble kernel not implemented"});hier.append(r);fits.append({"candidate":r["variant"],"fit_precision":dtype.__name__,"matrix_relative_error":r["matrix_relative_error"],"action_relative_error":r["action_relative_error"],"serialized_bytes":r["serialized_bytes"],"fit_ms":fit});wh_by[r["variant"]]=wh
 for block in (32,64,128,256,512):
  q=time.perf_counter();arr,row,pi,ri=hierarchical(w,"block",block,np.float32);fit=(time.perf_counter()-q)*1000;r,wh=record("Hierarchical Primary + Correction",f"block{block}-4+4",w,x,ref,arr,lambda z,row=row:row_apply(row,z),lambda row=row:np.stack([row(i) for i in range(1024)]),fit,{"primary_entropy":entropy(pi),"correction_entropy":entropy(ri),"primary_utilization":len(np.unique(pi)),"residual_utilization":len(np.unique(ri)),"decode_limitation":"bounded row decode"});hier.append(r);wh_by[r["variant"]]=wh
 write("hierarchical-4plus4.csv",hier);write("f32-vs-f64-fit.csv",fits)
 signrows=[]
 for mb,rb in ((4,3),(3,4)):
  for dtype in (np.float32,np.float64):
   q=time.perf_counter();arr,row,mi,ri=sign_residual(w,mb,rb,dtype);fit=(time.perf_counter()-q)*1000;r,_=record("Sign + Magnitude + Residual",f"1+{mb}+{rb}-{dtype.__name__}",w,x,ref,arr,lambda z,row=row:row_apply(row,z),lambda row=row:np.stack([row(i) for i in range(1024)]),fit,{"magnitude_entropy":entropy(mi),"correction_entropy":entropy(ri)});signrows.append(r)
 write("sign-magnitude-residual.csv",signrows)
 vectors=[]
 for width in (2,4,8,16):
  for conditional in (False,True):
   for orient in ("input","output"):
    q=time.perf_counter();arr,rec,pi,ri,cluster_norms=vector_residual(w,width,conditional,orient);fit=(time.perf_counter()-q)*1000;apply=lambda z,rec=rec:rec()@z;r,wh=record("Vector Prototype + Residual",f"v{width}-{'conditional' if conditional else 'global'}-{orient}",w,x,ref,arr,apply,rec,fit,{"orientation":orient,"vector_width":width,"primary_entropy":entropy(pi),"residual_entropy":entropy(ri),"primary_dead_codewords":16-len(np.unique(pi)),"residual_dead_codewords":16-len(np.unique(ri)),"average_residual_norm_by_primary":json.dumps(cluster_norms),"decode_limitation":"qualification apply materializes dense decoded tensor; native fused vector kernel required"});vectors.append(r)
 write("vector-prototype-residual.csv",vectors)
 # Budget comparison and conservative Pareto over old/new rows.
 old=list(csv.DictReader((OUT/"tensor-breadth-screen.csv").open()));combined=[]
 for r in old:combined.append({"source":"existing","candidate":r["family"]+":"+r["variant"],"effective_bits_per_weight":float(r["effective_bits_per_weight"]),"action_relative_error":float(r["action_relative_error"]),"compact_apply_ms":float(r["compact_apply_median_ms"]),"serialized_bytes":int(r["serialized_bytes"])})
 for source,rs in (("missing",missing),("hierarchical",hier),("sign-residual",signrows),("vector-residual",vectors)):
  for r in rs:combined.append({"source":source,"candidate":r["family"]+":"+r["variant"],"effective_bits_per_weight":r["effective_bits_per_weight"],"action_relative_error":r["action_relative_error"],"compact_apply_ms":r["compact_apply_ms"],"serialized_bytes":r["serialized_bytes"]})
 for r in combined:r["budget_region_bits"]=min((.5,1,2,4,8,16,32),key=lambda q:abs(q-r["effective_bits_per_weight"]))
 write("byte-budget-comparison-v2.csv",combined);pareto=[]
 for r in combined:
  if not any(q["serialized_bytes"]<=r["serialized_bytes"] and q["action_relative_error"]<=r["action_relative_error"] and q["compact_apply_ms"]<=r["compact_apply_ms"] and (q["serialized_bytes"]<r["serialized_bytes"] or q["action_relative_error"]<r["action_relative_error"] or q["compact_apply_ms"]<r["compact_apply_ms"]) for q in combined):pareto.append(r)
 write("tensor-pareto-v2.csv",pareto)
 # Residual diagnostics reuse compact scalar statistics.
 def diag(label,res,extra):
  flat=np.abs(res).ravel();hist=np.histogram(res,bins=256)[0];p=hist[hist>0]/hist.sum();spec=np.abs(np.fft.rfft2(res))**2;top=max(1,len(flat)//100);u,v=randomized_svd(res,16,np.random.default_rng(404));blocks=np.round(res.reshape(-1,16),3);return {"candidate":label,"near_zero_fraction":float(np.mean(flat<1e-4)),"top1pct_magnitude_energy":float(np.partition(flat*flat,-top)[-top:].sum()/np.sum(flat*flat)),"entropy_bits":float(-(p*np.log2(p)).sum()),"spectral_top1pct_energy":float(np.partition(spec.ravel(),-max(1,spec.size//100))[-max(1,spec.size//100):].sum()/spec.sum()),"top16_singular_energy_fraction":float(np.linalg.norm(u@v)**2/np.linalg.norm(res)**2),"rounded_block_repeat_fraction":float(1-len(np.unique(blocks,axis=0))/len(blocks))}|extra
 hd=[]
 for r in sorted(hier,key=lambda q:q["action_relative_error"])[:4]:hd.append(diag(r["variant"],w-wh_by[r["variant"]],{"primary_entropy":r["primary_entropy"],"correction_entropy":r["correction_entropy"]}))
 write("hierarchical-residual-diagnostics.csv",hd)
 vd=[{"candidate":r["variant"],"primary_entropy":r["primary_entropy"],"residual_entropy":r["residual_entropy"],"primary_dead_codewords":r["primary_dead_codewords"],"residual_dead_codewords":r["residual_dead_codewords"],"average_residual_norm_by_primary":r["average_residual_norm_by_primary"]} for r in sorted(vectors,key=lambda q:q["action_relative_error"])[:8]];write("vector-residual-diagnostics.csv",vd)
 # Scale projections only, using each new Pareto candidate's measured ratio.
 projections=[]
 for r in pareto:
  ratio=1114112/r["serialized_bytes"]
  if r["source"] not in ("hierarchical","sign-residual","vector-residual"):continue
  for bw in (1.4,3.2,8,16,32):
   pb=518104064/ratio;projections.append({"candidate":r["candidate"],"label":"SCALE PROJECTION — NOT MEASURED 32B COMPRESSION","measured_0.6B_tensor_compression_ratio":ratio,"projected_layer_bytes":pb,"bandwidth_GBps":bw,"projected_storage_time_ms":pb/(bw*1e9)*1000,"canonical_compute_context_ms":16.15})
 if projections:write("memory-wall-projection-v2.csv",projections)
 else:write("memory-wall-projection-v2.csv",[{"candidate":"NONE","label":"no new Pareto candidate","measured_0.6B_tensor_compression_ratio":0,"projected_layer_bytes":0,"bandwidth_GBps":0,"projected_storage_time_ms":0,"canonical_compute_context_ms":16.15}])
 tensor_summary_path=OUT/"tensor-screen-summary.json";tensor_summary=json.loads(tensor_summary_path.read_text());tensor_summary.update({"status":"PARTIAL — 22/22 required tensor families tested; hidden-state and deeper phases remain","tested_count":22,"continuation_evidence":"missing-family-screen.csv","hidden_state_evaluation":"NOT TESTED"});tensor_summary_path.write_text(json.dumps(tensor_summary,indent=2)+"\n")
 summary={"status":"PARTIAL — 22/22 tensor families tested; hidden-state and deeper phases remain","missing_family_rows":len(missing),"new_candidate_rows":{"hierarchical":len(hier),"sign_magnitude_residual":len(signrows),"vector_prototype_residual":len(vectors)},"source_artifacts_changed":False,"wire_changes":0,"canonical_changes":0,"group_promotion":"NONE — tensor evidence insufficient pending review and hidden-state validation"};(OUT/"continuation-summary.json").write_text(json.dumps(summary,indent=2)+"\n");print("PASS — continuation evidence written; Step 30 remains PARTIAL")
if __name__=="__main__":main()
