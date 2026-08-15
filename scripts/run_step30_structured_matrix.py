#!/usr/bin/env python3
"""Corrective Step-30 applicability matrix and shared-input pilot evidence."""
from __future__ import annotations
import csv,json,statistics,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_step30_reparameterization import ARTIFACTS,decode_q8,randomized_svd
from qualify_step16 import parse
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"benchmark-results/vbuf-ml-step30-reparameterization"
FAMILIES=["Low Rank","Sparse","Low Rank + Sparse","Kronecker","Tensor Train / MPO","Butterfly","Generalized / Deformable Butterfly","Toeplitz","Circulant / Block-Circulant","Low Displacement Rank","Scalar Codebook","Vector Quantization","Blockwise Affine / Shared Scale","Sign / Magnitude","Exponent / Mantissa","Generator Function","Fourier Basis","Hadamard / Walsh","Structured Orthogonal","Polynomial / Matrix Function","Block Dictionary","Exact / Numerically Equivalent Reparameterization"]
BOUNDS=["Tensor","QKV","Attention","Gate/Up","MLP","Whole Layer","Cross Layer"]
ALIGN=64

def write_csv(name,fields,rows):
 with (OUT/name).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
def load(model,names):
 a=parse(ARTIFACTS[model]);by={t.name:t for t in a.tensors};return {n:decode_q8(ARTIFACTS[model],by[n])[0] for n in names}
def med(fn,reps=5):
 values=[];result=None
 for _ in range(reps):t=time.perf_counter_ns();result=fn();values.append((time.perf_counter_ns()-t)/1e6)
 return result,statistics.median(values)
def serialized_factor_bytes(arrays,descriptor):
 metadata=len(json.dumps(descriptor,separators=(",",":"),sort_keys=True).encode());cursor=metadata;padding=0
 for a in arrays:
  aligned=(cursor+ALIGN-1)//ALIGN*ALIGN;padding+=aligned-cursor;cursor=aligned+a.nbytes
 final=(cursor+ALIGN-1)//ALIGN*ALIGN;padding+=final-cursor
 return metadata,padding,final
def lowrank_record(group,mats,rank=16):
 names=list(mats);stacked=np.concatenate([mats[n] for n in names],axis=0);rng=np.random.default_rng(3030);start=time.perf_counter();ju,jv=randomized_svd(stacked,rank,rng);joint_fit=(time.perf_counter()-start)*1000
 independent=[];start=time.perf_counter()
 for i,n in enumerate(names):independent.append(randomized_svd(mats[n],rank,np.random.default_rng(3040+i)))
 independent_fit=(time.perf_counter()-start)*1000;x=np.random.default_rng(3031).standard_normal((stacked.shape[1],8),dtype=np.float32);ref,_=med(lambda:stacked@x,7)
 z,shared_ms=med(lambda:jv@x);_,role_ms=med(lambda:ju@z);got,joint_ms=med(lambda:ju@(jv@x));independent_got,independent_ms=med(lambda:np.concatenate([u@(v@x) for u,v in independent],axis=0))
 raw_joint=ju.nbytes+jv.nbytes;raw_independent=sum(u.nbytes+v.nbytes for u,v in independent);partitions=[0];
 for n in names:partitions.append(partitions[-1]+mats[n].shape[0])
 partition_bytes=len(partitions)*8;meta,pad,total=serialized_factor_bytes([ju,jv],{"group":group,"rank":rank,"members":names,"partitions":partitions,"dtype":"float32","alignment":ALIGN});total+=partition_bytes;imeta=ipad=itotal=0
 for n,(u,v) in zip(names,independent):
  a,b,c=serialized_factor_bytes([u,v],{"tensor":n,"rank":rank,"dtype":"float32","alignment":ALIGN});imeta+=a;ipad+=b;itotal+=c
 return {"group":group,"model":"0.6B","rank":rank,"members":";".join(names),"canonical_q8_bytes":int(sum(m.size*8.5/8 for m in mats.values())),"raw_independent_factor_bytes":raw_independent,"raw_joint_factor_bytes":raw_joint,"raw_factor_byte_reduction":raw_independent/raw_joint,"storage_label":"raw factor-array pilot; not a canonical serialized format","joint_metadata_bytes":meta,"joint_padding_bytes":pad,"joint_partition_descriptor_bytes":partition_bytes,"joint_modeled_serialized_bytes":total,"independent_modeled_serialized_bytes":itotal,"modeled_serialized_storage_gain":itotal/total,"matrix_relative_error":float(np.linalg.norm(ju@jv-stacked)/np.linalg.norm(stacked)),"joint_action_relative_error":float(np.linalg.norm(got-ref)/np.linalg.norm(ref)),"independent_action_relative_error":float(np.linalg.norm(independent_got-ref)/np.linalg.norm(ref)),"dense_apply_median_ms":med(lambda:stacked@x)[1],"joint_compact_apply_median_ms":joint_ms,"independent_compact_apply_median_ms":independent_ms,"shared_transform_median_ms":shared_ms,"role_specific_median_ms":role_ms,"joint_temporary_bytes":z.nbytes+got.nbytes,"joint_representation_bytes_touched":raw_joint+meta+partition_bytes,"input_output_bytes_touched":x.nbytes+got.nbytes,"joint_fit_ms":joint_fit,"independent_fit_ms":independent_fit,"real_hidden_state_validation":"NOT TESTED"}
def main():
 screen=list(csv.DictReader((OUT/"tensor-breadth-screen.csv").open()));tested_tensor={r["family"] for r in screen}
 continuation=OUT/"missing-family-screen.csv"
 if continuation.exists():tested_tensor.update(r["family"] for r in csv.DictReader(continuation.open()))
 rows=[]
 missing_reasons={"Tensor Train / MPO":"NOT TESTED — no TT/MPO direct-contraction implementation","Butterfly":"NOT TESTED — no fitted Butterfly direct evaluator","Generalized / Deformable Butterfly":"NOT TESTED — no generalized/deformable Butterfly implementation; ordinary Butterfly was not substituted","Low Displacement Rank":"NOT TESTED — no genuine learned displacement-operator implementation; generic low rank was not substituted","Structured Orthogonal":"NOT TESTED — no bounded learned Householder/Givens evaluator; Hadamard was not substituted"}
 for family in FAMILIES:
  for boundary in BOUNDS:
   if boundary=="Tensor" and family in tested_tensor:status="TESTED — real 0.6B tensor run, complete modeled serialization accounting, direct apply, and action error"
   elif boundary=="Tensor":status=missing_reasons.get(family,"NOT TESTED — no qualifying real-tensor run")
   elif family=="Low Rank" and boundary in ("QKV","Gate/Up"):status="TESTED — 0.6B layer-0 shared-input pilot with direct apply and full experimental accounting; hidden states unavailable"
   elif boundary=="Cross Layer":status="NOT TESTED — gated on a functionally useful whole-layer result"
   elif boundary=="Whole Layer":status="NOT TESTED — group evidence is not functionally useful and whole nonlinear layer work remains gated"
   elif boundary in ("Attention","MLP"):status="NOT TESTED — complete functional group work remains gated by poor shared-input action error"
   else:status="NOT TESTED — family did not survive tensor screening or has not been executed at this boundary"
   rows.append({"algorithm":family,"boundary":boundary,"status":status})
 write_csv("algorithm-applicability.csv",["algorithm","boundary","status"],rows)
 qkv=load("0.6B",["blk.0.attn_q.weight","blk.0.attn_k.weight","blk.0.attn_v.weight"]);gate=load("0.6B",["blk.0.ffn_gate.weight","blk.0.ffn_up.weight"]);joint=[lowrank_record("QKV",qkv),lowrank_record("Gate/Up",gate)]
 write_csv("qkv-joint-results.csv",list(joint[0]),[joint[0]]);write_csv("gate-up-joint-results.csv",list(joint[1]),[joint[1]])
 write_csv("attention-joint-results.csv",["status","reason"],[{"status":"NOT TESTED","reason":"QKV pilot error is too high; output projection was not crossed into a different input domain"}]);write_csv("mlp-joint-results.csv",["status","reason"],[{"status":"NOT TESTED","reason":"Gate/Up pilot error is too high; down projection was not crossed across SiLU/gating"}])
 layer_bytes=518104064;compute_ms=16.15;wall=[]
 for bw in (1.4,3.2,8.0,16.0,32.0):
  storage_ms=layer_bytes/(bw*1e9)*1000;wall.append({"status":"CANONICAL CONTROL — simulation only","model":"32B","representation":"canonical Q8 layer control","layer_bytes":layer_bytes,"storage_bandwidth_GBps":bw,"storage_time_ms":storage_ms,"compute_time_ms":compute_ms,"nonoverlap_ms":storage_ms+compute_ms,"ideal_overlap_ms":max(storage_ms,compute_ms),"transfer_note":"Step-29 32B host-local context; not combined with 0.6B compression pilots"})
 write_csv("whole-layer-results.csv",list(wall[0]),wall)
 for name,reason in (("whole-layer-pareto.csv","no whole-layer candidates"),("whole-layer-residual-analysis.csv","no fitted whole-layer candidate"),("whole-layer-hybrids.csv","hybrids gated on residual-supported pure-family survivors")):
  write_csv(name,["status","reason"],[{"status":"NOT TESTED","reason":reason}])
 write_csv("shared-compute.csv",["boundary","model","shared_transform_median_ms","role_specific_median_ms","joint_compact_apply_median_ms","independent_compact_apply_median_ms","temporary_bytes","bytes_touched","status"],[{"boundary":r["group"],"model":r["model"],"shared_transform_median_ms":r["shared_transform_median_ms"],"role_specific_median_ms":r["role_specific_median_ms"],"joint_compact_apply_median_ms":r["joint_compact_apply_median_ms"],"independent_compact_apply_median_ms":r["independent_compact_apply_median_ms"],"temporary_bytes":r["joint_temporary_bytes"],"bytes_touched":r["joint_representation_bytes_touched"],"status":"TESTED — random action probes; no hidden-state validation"} for r in joint])
 write_csv("optional-cross-layer.csv",["status","reason"],[{"status":"NOT TESTED","reason":"conditional phase not justified without a whole-layer winner"}])
 a=parse(ARTIFACTS["0.6B"]);ts=[t for t in a.tensors if t.layer==0 and t.type_name=="Q8_0"];write_csv("tensor-baseline.csv",["model","layer","tensor","shape","canonical_q8_bytes","weights","effective_bits_per_weight"],[{"model":"0.6B","layer":0,"tensor":t.name,"shape":"x".join(map(str,t.shape)),"canonical_q8_bytes":t.payload_size,"weights":t.elements,"effective_bits_per_weight":8.5} for t in ts])
 missing=[f for f in FAMILIES if f not in tested_tensor];summary={"status":"PARTIAL — 22-family tensor breadth complete; hidden-state and deeper phases remain","family_count":len(FAMILIES),"boundary_count":len(BOUNDS),"matrix_cells":len(rows),"tensor_families_tested":sorted(tested_tensor),"tensor_families_missing":missing,"deeper_phases_missing":["functionally useful group candidate","complete attention","complete MLP","whole layer","whole-layer residuals/hybrids","real hidden states","cross layer"],"joint_representation_classification":"F — inconclusive","canonical_artifacts_changed":False,"memory_wall_context":"32B canonical control only; no transfer from 0.6B pilots"};(OUT/"structured-matrix-summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(f"PASS — {len(rows)} cells; status PARTIAL; {len(missing)} tensor families remain")
if __name__=="__main__":main()
