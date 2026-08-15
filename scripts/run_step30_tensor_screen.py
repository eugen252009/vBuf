#!/usr/bin/env python3
"""Breadth-first Step-30 screening on one real Qwen3 tensor.

A TESTED row is emitted only after fitting a real tensor, deterministic complete
storage accounting, direct compact application, and action-error measurement.
This is research evidence; candidates are not vBuf encodings.
"""
from __future__ import annotations
import csv, json, math, statistics, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from qualify_step16 import parse
from run_step30_reparameterization import ARTIFACTS, decode_q8, randomized_svd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"benchmark-results/vbuf-ml-step30-reparameterization"
MODEL="0.6B"; TENSOR="blk.0.attn_k.weight"; ALIGN=64; RNG=np.random.default_rng(3032)

@dataclass
class Candidate:
 family:str; variant:str; arrays:dict[str,np.ndarray]; apply:Callable[[np.ndarray],np.ndarray]; reconstruct:Callable[[],np.ndarray]; fit_ms:float; temp_bytes:int; reconstruction_class:str="LOSSY"

def align(v,a=ALIGN):return (v+a-1)//a*a
def storage(c:Candidate,shape):
 schema={"family":c.family,"variant":c.variant,"shape":list(shape),"alignment":ALIGN,"arrays":[{"name":n,"dtype":str(a.dtype),"shape":list(a.shape)} for n,a in c.arrays.items()]}
 metadata=len(json.dumps(schema,separators=(",",":"),sort_keys=True).encode()); cursor=metadata; padding=0
 cats={"factor":0,"indexes":0,"permutations":0,"scales":0,"residual":0,"payload":0}
 for name,a in c.arrays.items():
  nxt=align(cursor);padding+=nxt-cursor;cursor=nxt;size=a.nbytes;cursor+=size
  if "index" in name:cat="indexes"
  elif "perm" in name:cat="permutations"
  elif any(x in name for x in ("scale","offset","exponent")):cat="scales"
  elif "residual" in name or "sparse" in name:cat="residual"
  elif name.endswith("_factor") or any(x in name for x in ("core","coeff","generator","codebook","prototype")):cat="factor"
  else:cat="payload"
  cats[cat]+=size
 final=align(cursor);padding+=final-cursor
 return cats|{"metadata_bytes":metadata,"padding_bytes":padding,"partition_descriptor_bytes":0,"serialized_bytes":final}
def bench(fn,x,reps=5):
 vals=[]; y=None
 for _ in range(reps):
  t=time.perf_counter_ns();y=fn(x);vals.append((time.perf_counter_ns()-t)/1e6)
 return y,statistics.median(vals),min(vals)
def scalar_codebook(w,k=16):
 flat=w.ravel(); code=np.quantile(flat,np.linspace(0,1,k)).astype(np.float32)
 for _ in range(8):
  cuts=(code[:-1]+code[1:])/2;idx=np.searchsorted(cuts,flat).astype(np.uint8)
  sums=np.bincount(idx,weights=flat,minlength=k);counts=np.bincount(idx,minlength=k);code=np.where(counts,sums/np.maximum(counts,1),code).astype(np.float32)
 idx=idx.reshape(w.shape)
 return code,idx
def code_apply(code,idx,x):
 y=np.empty((idx.shape[0],x.shape[1]),np.float32)
 for i in range(idx.shape[0]):y[i]=code[idx[i]].astype(np.float32)@x
 return y
def kmeans_vectors(vectors,k,seed=3033,iterations=6):
 rng=np.random.default_rng(seed);sample=vectors[rng.choice(len(vectors),min(len(vectors),20000),replace=False)];code=sample[rng.choice(len(sample),k,replace=False)].copy()
 for _ in range(iterations):
  sums=np.zeros_like(code);counts=np.zeros(k,np.int64)
  for start in range(0,len(sample),2048):
   b=sample[start:start+2048];d=((b[:,None,:]-code[None,:,:])**2).sum(2);ii=d.argmin(1);np.add.at(sums,ii,b);np.add.at(counts,ii,1)
  code=np.where(counts[:,None]>0,sums/np.maximum(counts[:,None],1),code)
 idx=[]
 for start in range(0,len(vectors),2048):
  b=vectors[start:start+2048];idx.append(((b[:,None,:]-code[None,:,:])**2).sum(2).argmin(1))
 return code.astype(np.float32),np.concatenate(idx).astype(np.uint8)
def fwht(x):
 y=x.astype(np.float32,copy=True);h=1
 while h<y.shape[0]:
  z=y.reshape(-1,2*h,*y.shape[1:]);a=z[:,:h].copy();b=z[:,h:2*h].copy();z[:,:h]=a+b;z[:,h:2*h]=a-b;h*=2
 return y/math.sqrt(y.shape[0])
def hadamard_matrix(n):return fwht(np.eye(n,dtype=np.float32))
def direct_blocks(code,idx,x,bh,bw,m,n):
 y=np.zeros((m,x.shape[1]),np.float32)
 for br in range(m//bh):
  out=y[br*bh:(br+1)*bh]
  for bc in range(n//bw):out+=code[idx[br,bc]]@x[bc*bw:(bc+1)*bw]
 return y
def candidates(w):
 m,n=w.shape; out=[]
 def timed(f):t=time.perf_counter();v=f();return v,(time.perf_counter()-t)*1000
 # Low rank.
 (u,v),ms=timed(lambda:randomized_svd(w,16,np.random.default_rng(1)));out.append(Candidate("Low Rank","rank16",{"u_factor":u,"v_factor":v},lambda x,u=u,v=v:u@(v@x),lambda u=u,v=v:u@v,ms,(u.shape[0]+v.shape[1])*4))
 # Sparse and low-rank+sparse.
 def sparse_fit(base,k):
  r=w-base;flat=np.argpartition(np.abs(r).ravel(),-k)[-k:];rr,cc=np.unravel_index(flat,w.shape);return rr.astype(np.uint16),cc.astype(np.uint16),r[rr,cc].astype(np.float32)
 (rr,cc,sv),ms=timed(lambda:sparse_fit(np.zeros_like(w),w.size//100))
 def sapp(x,rr=rr,cc=cc,sv=sv):
  y=np.zeros((m,x.shape[1]),np.float32);np.add.at(y,rr,sv[:,None]*x[cc]);return y
 def srec(rr=rr,cc=cc,sv=sv):z=np.zeros_like(w);z[rr,cc]=sv;return z
 out.append(Candidate("Sparse","top1pct",{"sparse_row_index":rr,"sparse_col_index":cc,"sparse_residual_values":sv},sapp,srec,ms,m*8*4))
 (rr2,cc2,sv2),ms2=timed(lambda:sparse_fit(u@v,w.size//200))
 def lrsa(x,u=u,v=v,rr=rr2,cc=cc2,sv=sv2):y=u@(v@x);np.add.at(y,rr,sv[:,None]*x[cc]);return y
 def lrsr(u=u,v=v,rr=rr2,cc=cc2,sv=sv2):z=u@v;z[rr,cc]+=sv;return z
 out.append(Candidate("Low Rank + Sparse","rank16+top0.5pct",{"u_factor":u,"v_factor":v,"sparse_row_index":rr2,"sparse_col_index":cc2,"sparse_residual_values":sv2},lrsa,lrsr,ms2,m*8*4))
 # One Kronecker product, 32x32 factors.
 q=32
 def kronfit():
  r=w.reshape(q,q,q,q).transpose(0,2,1,3).reshape(q*q,q*q);a,s,b=np.linalg.svd(r,full_matrices=False);return (a[:,0]*math.sqrt(s[0])).reshape(q,q).astype(np.float32),(b[0]*math.sqrt(s[0])).reshape(q,q).astype(np.float32)
 (ka,kb),ms=timed(kronfit)
 out.append(Candidate("Kronecker","single-32x32",{"a_factor":ka,"b_factor":kb},lambda x,a=ka,b=kb:(a@x.reshape(q,q,-1).transpose(2,0,1)@b.T).transpose(1,2,0).reshape(m,-1),lambda a=ka,b=kb:np.kron(a,b),ms,(q*q*8+m*8)*4))
 # Toeplitz diagonal projection.
 offs=np.arange(-(n-1),m);tg=np.array([np.mean(np.diagonal(w,offset=-d)) for d in offs],np.float32)
 ti=np.subtract.outer(np.arange(m),np.arange(n))+n-1
 out.append(Candidate("Toeplitz","diagonal-mean",{"generator":tg},lambda x,g=tg:np.stack([np.convolve(g,x[:,j],mode="full")[n-1:n-1+m] for j in range(x.shape[1])],1).astype(np.float32),lambda g=tg:g[ti],0,(len(tg)+m*8)*4))
 # Circulant and polynomial in a compact cyclic-shift operator.
 ci=np.subtract.outer(np.arange(n),np.arange(n))%n;ccirc=np.array([w[ci==d].mean() for d in range(n)],np.float32)
 capp=lambda x,c=ccirc:np.fft.ifft(np.fft.fft(c)[:,None]*np.fft.fft(x,axis=0),axis=0).real.astype(np.float32)
 crec=lambda c=ccirc:c[ci]
 out.append(Candidate("Circulant / Block-Circulant","single-spectrum",{"generator":ccirc},capp,crec,0,(n*2+n*8)*8))
 out.append(Candidate("Polynomial / Matrix Function","cyclic-shift-polynomial",{"coefficients":ccirc},capp,crec,0,(n*2+n*8)*8))
 # Scalar codebook.
 (code,idx),ms=timed(lambda:scalar_codebook(w,16));out.append(Candidate("Scalar Codebook","scalar-k16",{"codebook":code,"indexes":idx},lambda x,c=code,i=idx:code_apply(c,i,x),lambda c=code,i=idx:c[i],ms,n*4+m*8*4))
 # VQ width 8, k64.
 vw=8;vec=w.reshape(-1,vw);(vc,vi),ms=timed(lambda:kmeans_vectors(vec,64));vi=vi.reshape(m,n//vw)
 def vqapply(x,c=vc,i=vi):
  return np.stack([sum((c[i[r,b]]@x[b*vw:(b+1)*vw] for b in range(n//vw)),start=np.zeros(x.shape[1],np.float32)) for r in range(m)])
 out.append(Candidate("Vector Quantization","width8-k64",{"codebook":vc,"indexes":vi},vqapply,lambda c=vc,i=vi:c[i].reshape(m,n),ms,n*4+m*8*4))
 # Block affine: one shared width-32 prototype, private scale/offset.
 bw=32;blocks=w.reshape(-1,bw);proto=blocks.mean(0);proto-=proto.mean();den=float(proto@proto)+1e-12;sc=(blocks@proto/den).astype(np.float32);off=blocks.mean(1).astype(np.float32);sc=sc.reshape(m,n//bw);off=off.reshape(m,n//bw)
 def barec(p=proto,s=sc,o=off):return (s[...,None]*p+o[...,None]).reshape(m,n)
 out.append(Candidate("Blockwise Affine / Shared Scale","shared-prototype-w32",{"prototype":proto.astype(np.float32),"scales":sc,"offsets":off},lambda x,p=proto,s=sc,o=off:np.stack([((s[r,:,None]*p+o[r,:,None]).reshape(n)@x) for r in range(m)]),barec,0,n*4+m*8*4))
 # Sign/magnitude with k16 magnitude codebook and explicit packed signs.
 mag=np.abs(w);mc,mi=scalar_codebook(mag,16);sign=np.packbits(w<0,axis=1);signs=np.where(w<0,-1.0,1.0).astype(np.float32)
 smrec=lambda c=mc,i=mi,s=signs:c[i]*s
 out.append(Candidate("Sign / Magnitude","signbit+mag-k16",{"codebook":mc,"indexes":mi,"sign_bits":sign},lambda x,c=mc,i=mi,s=signs:np.stack([(c[i[r]]*s[r])@x for r in range(m)]),smrec,0,n*4+m*8*4))
 # Exponent/mantissa per 32 values.
 b=w.reshape(-1,32);mx=np.max(np.abs(b),1);exp=np.floor(np.log2(np.maximum(mx,2**-24))).astype(np.int8);scale=np.exp2(exp.astype(np.float32))/127;mant=np.clip(np.rint(b/scale[:,None]),-127,127).astype(np.int8);exp=exp.reshape(m,n//32);mant=mant.reshape(m,n)
 emrec=lambda e=exp,q=mant:(q.reshape(-1,32).astype(np.float32)*(np.exp2(e.ravel().astype(np.float32))/127)[:,None]).reshape(m,n)
 def emapply(x,e=exp,q=mant):
  y=np.empty((m,x.shape[1]),np.float32)
  for r in range(m):
   values=(q[r].reshape(-1,32).astype(np.float32)*(np.exp2(e[r].astype(np.float32))/127)[:,None]).reshape(n)
   y[r]=values@x
  return y
 out.append(Candidate("Exponent / Mantissa","block32-int8",{"exponents":exp,"mantissa":mant},emapply,emrec,0,n*4+m*8*4))
 # Compact polynomial coordinate generator.
 deg=3;ro=np.linspace(-1,1,m,dtype=np.float32);co=np.linspace(-1,1,n,dtype=np.float32);rb=np.stack([ro**k for k in range(deg+1)],1);cb=np.stack([co**k for k in range(deg+1)],1);rq,_=np.linalg.qr(rb);cq,_=np.linalg.qr(cb);gc=(rq.T@w@cq).astype(np.float32)
 out.append(Candidate("Generator Function","degree3-separable-polynomial",{"generator_coefficients":gc},lambda x,a=rq,c=gc,b=cq:a@(c@(b.T@x)),lambda a=rq,c=gc,b=cq:a@c@b.T,0,(m+n)*(deg+1)*4))
 return out

def cosine_basis(size,k):
 i=np.arange(size)[:,None];j=np.arange(k)[None,:];b=np.cos(np.pi*(i+.5)*j/size);b[:,0]/=math.sqrt(2);return (b*math.sqrt(2/size)).astype(np.float32)

def finish_candidates(w,out):
 m,n=w.shape;k=16;co=cosine_basis(m,k);ci=cosine_basis(n,k);fc=(co.T@w@ci).astype(np.float32)
 out.append(Candidate("Fourier Basis","cosine-16x16",{"spectral_coefficients":fc},lambda x,a=co,c=fc,b=ci:a@(c@(b.T@x)),lambda a=co,c=fc,b=ci:a@c@b.T,0,(m+n)*k*4))
 # D1 H D2 alternating fit.
 h=hadamard_matrix(n);a=np.ones(n,np.float32);b=np.ones(n,np.float32)
 t=time.perf_counter()
 for _ in range(5):
  z=h*b[None,:];a=(w*z).sum(1)/(z*z).sum(1).clip(1e-12);z=a[:,None]*h;b=(w*z).sum(0)/(z*z).sum(0).clip(1e-12)
 fit=(time.perf_counter()-t)*1000
 out.append(Candidate("Hadamard / Walsh","D1-H-D2",{"left_scales":a,"right_scales":b},lambda x,a=a,b=b:a[:,None]*fwht(b[:,None]*x),lambda a=a,b=b:a[:,None]*h*b[None,:],fit,n*8*4))
 # 4x4 block dictionary k64.
 bh=bw=4;blocks=w.reshape(m//bh,bh,n//bw,bw).transpose(0,2,1,3).reshape(-1,bh*bw)
 t0=time.perf_counter();bc,bi=kmeans_vectors(blocks,64,3034,5);fit=(time.perf_counter()-t0)*1000;bc=bc.reshape(64,bh,bw);bi=bi.reshape(m//bh,n//bw)
 out.append(Candidate("Block Dictionary","4x4-k64",{"codebook":bc,"indexes":bi},lambda x,c=bc,i=bi:direct_blocks(c,i,x,bh,bw,m,n),lambda c=bc,i=bi:c[i].transpose(0,2,1,3).reshape(m,n),fit,(bc.nbytes+m*8*4)))
 # Exact column permutation in fp32 parameter space.
 perm=np.argsort(np.linalg.norm(w,axis=0)).astype(np.uint16);wp=w[:,perm].copy()
 def exact_reconstruct(p=perm,z=wp):
  result=np.empty_like(z);result[:,p]=z;return result
 out.append(Candidate("Exact / Numerically Equivalent Reparameterization","column-permutation-fp32",{"permutation":perm,"payload_values":wp},lambda x,p=perm,z=wp:z@x[p],exact_reconstruct,0,m*n*4,"NUMERICALLY EQUIVALENT"))
 return out

def residual_diag(name,r):
 fro=float(np.linalg.norm(r));u,v=randomized_svd(r,16,np.random.default_rng(44));energy=float(np.linalg.norm(u@v)**2/max(fro*fro,1e-30));flat=np.abs(r).ravel();top=max(1,len(flat)//100);top_energy=float(np.sum(np.partition(flat*flat,-top)[-top:])/max(np.sum(flat*flat),1e-30));hist=np.histogram(r,bins=256)[0];p=hist[hist>0]/hist.sum();entropy=float(-(p*np.log2(p)).sum());spec=np.abs(np.fft.rfft2(r))**2;st=max(1,spec.size//100);spectral=float(np.partition(spec.ravel(),-st)[-st:].sum()/max(spec.sum(),1e-30));blocks=np.round(r[:1024,:1024].reshape(-1,16),3);unique=len(np.unique(blocks,axis=0));return {"candidate":name,"residual_frobenius":fro,"top16_singular_energy_fraction":energy,"near_zero_fraction":float(np.mean(flat<1e-4)),"top1pct_magnitude_energy_fraction":top_energy,"histogram_entropy_bits":entropy,"rounded_block_repeat_fraction":1-unique/len(blocks),"top1pct_spectral_energy_fraction":spectral,"cross_tensor_residual_correlation":"NOT MEASURED — single-tensor screen"}
def main():
 a=parse(ARTIFACTS[MODEL]);t=next(t for t in a.tensors if t.name==TENSOR);w,_=decode_q8(ARTIFACTS[MODEL],t);assert w.shape==(1024,1024)
 x=RNG.standard_normal((w.shape[1],8),dtype=np.float32);ref,dense_med,dense_min=bench(lambda z:w@z,x,7);out=candidates(w);out=finish_candidates(w,out);rows=[];diags=[]
 for c in out:
  got,med,mn=bench(c.apply,x,3);wh=c.reconstruct().astype(np.float32,copy=False);st=storage(c,w.shape);me=float(np.linalg.norm(wh-w)/np.linalg.norm(w));ae=float(np.linalg.norm(got-ref)/np.linalg.norm(ref));row={"family":c.family,"variant":c.variant,"model":MODEL,"layer":0,"tensor":TENSOR,"reconstruction_class":c.reconstruction_class,"matrix_relative_error":me,"action_relative_error":ae,"dense_apply_median_ms":dense_med,"compact_apply_median_ms":med,"compact_apply_min_ms":mn,"compute_inflation":med/dense_med,"temporary_bytes":c.temp_bytes,"representation_bytes_touched":st["serialized_bytes"],"input_output_bytes_touched":x.nbytes+got.nbytes,"fit_ms":c.fit_ms}|st;row["effective_bits_per_weight"]=st["serialized_bytes"]*8/w.size;rows.append(row)
  if (ae<0.5 and st["serialized_bytes"]<t.payload_size) or c.reconstruction_class!="LOSSY":diags.append(residual_diag(c.family+":"+c.variant,w-wh))
  del wh
 fields=list(rows[0]);
 with (OUT/"tensor-breadth-screen.csv").open("w",newline="") as f:wr=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");wr.writeheader();wr.writerows(rows)
 if diags:writefields=list(diags[0]);
 else:writefields=["candidate"]
 with (OUT/"tensor-residual-diagnostics.csv").open("w",newline="") as f:wr=csv.DictWriter(f,fieldnames=writefields,lineterminator="\n");wr.writeheader();wr.writerows(diags)
 summary={"status":"PARTIAL — breadth-first real-tensor screening","model":MODEL,"layer":0,"tensor":TENSOR,"random_probe_seed":3032,"fit_seeds":[1,3033,3034],"tested_families":[r["family"] for r in rows],"tested_count":len(rows),"dense_apply_median_ms":dense_med,"storage_accounting":"deterministic 64-byte-aligned experimental serialization model; all arrays, metadata, indexes, permutations, scales, residuals and padding counted","hidden_state_evaluation":"NOT TESTED","canonical_artifacts_changed":False}
 (OUT/"tensor-screen-summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(f"PASS — {len(rows)} families executed on {MODEL} {TENSOR}")
if __name__=="__main__":main()
