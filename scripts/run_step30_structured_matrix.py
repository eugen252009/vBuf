#!/usr/bin/env python3
"""Bounded Step-30 tensor/group/layer search ledger.

This adds the complete algorithm-family applicability ledger and performs the
joint shared-input tests that are safe for the selected Qwen3 layer. It fails
closed: unimplemented families are recorded as NOT TESTED with a reason.
"""
from __future__ import annotations
import csv, json, platform, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_step30_reparameterization import ARTIFACTS, decode_q8, randomized_svd
from qualify_step16 import parse
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"benchmark-results/vbuf-ml-step30-reparameterization"
FAMILIES=["Low Rank","Sparse","Low Rank + Sparse","Kronecker","Tensor Train","Butterfly","Generalized Butterfly","Toeplitz","Circulant","Low Displacement Rank","Codebook","Vector Quantization","Block Affine","Sign / Magnitude","Exponent / Mantissa","Generator Function","Fourier Basis","Hadamard","Structured Orthogonal","Polynomial / Matrix Function","Block Dictionary"]
BOUNDS=["Tensor","QKV","Attention","Gate/Up","MLP","Whole Layer","Cross Layer"]

def write_csv(name,fields,rows):
 with (OUT/name).open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
def load(model,names):
 a=parse(ARTIFACTS[model]);by={t.name:t for t in a.tensors};return {n:decode_q8(ARTIFACTS[model],by[n])[0] for n in names}
def lowrank_record(group,mats,rank=16):
 # All matrices must consume the same x; stack output rows and preserve partitions.
 stacked=np.concatenate([mats[n] for n in mats],axis=0);start=time.perf_counter();u,v=randomized_svd(stacked,rank,np.random.default_rng(3030));fit=(time.perf_counter()-start)*1000
 x=np.random.default_rng(3031).standard_normal((stacked.shape[1],8),dtype=np.float32);ref=stacked@x;got=u@(v@x)
 independent=sum(rank*(m.shape[0]+m.shape[1])*4 for m in mats.values()); joint=int(u.nbytes+v.nbytes)
 return {"group":group,"model":"0.6B","rank":rank,"members":";".join(mats),"canonical_bytes":sum(m.size*8.5/8 for m in mats.values()),"independent_compact_bytes":independent,"joint_compact_bytes":joint,"storage_gain":independent/joint,"matrix_relative_error":float(np.linalg.norm((u@v)-stacked)/np.linalg.norm(stacked)),"action_relative_error":float(np.linalg.norm(got-ref)/np.linalg.norm(ref)),"fit_ms":fit,"shared_input_transform":"none; shared stacked factor application","direct_compute_note":"one Vx and one U projection; output partitions restored by row ranges"}
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 # Explicit statuses are intentionally conservative. Every family/boundary is present.
 rows=[]
 for family in FAMILIES:
  for boundary in BOUNDS:
   if family=="Low Rank" and boundary in ("Tensor","QKV","Gate/Up"):
    status="TESTED — tensor candidates and shared-input joint stack"
   elif family=="Low Rank" and boundary in ("Attention","MLP"):
    status="NOT TESTED — incompatible input domains prevent one joint operator; role-wise extension deferred"
   elif family=="Low Rank + Sparse" and boundary=="Tensor":
    status="NOT TESTED — residual screening deferred until pure-family shortlist"
   elif family in ("Sparse","Codebook","Sign / Magnitude","Exponent / Mantissa","Block Affine") and boundary=="Tensor":
    status="NOT TESTED — bounded implementation used low-rank family first; no direct candidate emitted"
   elif family=="Low Rank" and boundary=="Whole Layer":
    status="NOT TESTED — nonlinear layer objective and complete-layer hook not implemented"
   elif boundary=="Whole Layer":
    status="NOT TESTED — complete-layer direct evaluator not implemented; nonlinear boundaries retained"
   elif boundary=="Cross Layer":
    status="NOT TESTED — optional adjacent-layer phase not justified before layer winner"
   elif family in ("Kronecker","Tensor Train","Butterfly","Generalized Butterfly","Toeplitz","Circulant","Low Displacement Rank","Vector Quantization","Generator Function","Fourier Basis","Hadamard","Structured Orthogonal","Polynomial / Matrix Function","Block Dictionary"):
    status="NOT TESTED — no bounded direct evaluator implemented; family retained in matrix for future screening"
   else:
    status="NOT TESTED — no bounded evaluator at this boundary"
   rows.append({"algorithm":family,"boundary":boundary,"status":status})
 write_csv("algorithm-applicability.csv",["algorithm","boundary","status"],rows)
 # Real shared-input joint tests on the small qualified Qwen3 layer.
 qkv=load("0.6B",["blk.0.attn_q.weight","blk.0.attn_k.weight","blk.0.attn_v.weight"])
 gateup=load("0.6B",["blk.0.ffn_gate.weight","blk.0.ffn_up.weight"])
 joint=[lowrank_record("QKV",qkv),lowrank_record("Gate/Up",gateup)]
 write_csv("qkv-joint-results.csv",list(joint[0]),[joint[0]])
 write_csv("gate-up-joint-results.csv",list(joint[1]),[joint[1]])
 for name in ("attention-joint-results.csv","mlp-joint-results.csv"):
  write_csv(name,["status","reason"],[{"status":"NOT TESTED","reason":"attention output/down projection consume different intermediate domains; no illegal concatenation"}])
 # Group and whole-layer ledgers are explicit rather than fabricated functional results.
 write_csv("attention-joint-results.csv",["status","reason"],[{"status":"NOT TESTED","reason":"QKV tested; output projection has a different input domain and complete attention evaluator was not added"}])
 write_csv("mlp-joint-results.csv",["status","reason"],[{"status":"NOT TESTED","reason":"gate/up tested; down projection consumes post-gating activation and was not illegally concatenated"}])
 write_csv("whole-layer-results.csv",["status","reason","canonical_layer_bytes","measured_compute_ms","storage_bandwidth_GBps","storage_time_ms"],[{"status":"NOT TESTED","reason":"complete nonlinear layer evaluator unavailable; Step-29 layer compute remains the authority","canonical_layer_bytes":518104064,"measured_compute_ms":16.15,"storage_bandwidth_GBps":1.4,"storage_time_ms":370074.3},{"status":"NOT TESTED","reason":"simulation only; no compact whole-layer bytes claimed","canonical_layer_bytes":518104064,"measured_compute_ms":16.15,"storage_bandwidth_GBps":32,"storage_time_ms":16190.8}])
 write_csv("whole-layer-pareto.csv",["status","reason"],[{"status":"NOT TESTED","reason":"no whole-layer candidates"}])
 write_csv("whole-layer-residual-analysis.csv",["status","reason"],[{"status":"NOT TESTED","reason":"residuals require a fitted whole-layer candidate"}])
 write_csv("whole-layer-hybrids.csv",["status","reason"],[{"status":"NOT TESTED","reason":"hybrids require pure-family whole-layer survivors"}])
 write_csv("shared-compute.csv",["boundary","shared_input_transform_cost","role_specific_cost","total_joint_cost","status"],[{"boundary":"QKV","shared_input_transform_cost":"one Vx","role_specific_cost":"one U projection with q/k/v partitions","total_joint_cost":"measured in qkv-joint-results.csv","status":"TESTED"},{"boundary":"Gate/Up","shared_input_transform_cost":"one Vx","role_specific_cost":"one U projection with gate/up partitions","total_joint_cost":"measured in gate-up-joint-results.csv","status":"TESTED"},{"boundary":"Whole Layer","shared_input_transform_cost":"n/a","role_specific_cost":"n/a","total_joint_cost":"n/a","status":"NOT TESTED — nonlinear domains"}])
 write_csv("optional-cross-layer.csv",["status","reason"],[{"status":"NOT TESTED","reason":"cross-layer phase is conditional on a whole-layer winner"}])
 # Per-level baseline ledger from the real 0.6B layer metadata.
 a=parse(ARTIFACTS["0.6B"]);ts=[t for t in a.tensors if t.layer==0 and t.type_name=="Q8_0"]
 write_csv("tensor-baseline.csv",["model","layer","tensor","shape","canonical_q8_bytes","weights","effective_bits_per_weight"],[{"model":"0.6B","layer":0,"tensor":t.name,"shape":"x".join(map(str,t.shape)),"canonical_q8_bytes":t.payload_size,"weights":t.elements,"effective_bits_per_weight":8.5} for t in ts])
 write_csv("attention-joint-results.csv",list(joint[0])+["group_status"],[dict(joint[0],group_status="QKV joint tested; full attention not tested")])
 write_csv("mlp-joint-results.csv",list(joint[1])+["group_status"],[dict(joint[1],group_status="Gate/Up joint tested; full MLP not tested")])
 summary={"status":"bounded matrix qualification","families":FAMILIES,"boundaries":BOUNDS,"tested_real_group_boundaries":["QKV","Gate/Up"],"whole_layer_status":"NOT TESTED — complete nonlinear evaluator unavailable","cross_layer_status":"NOT TESTED — conditional phase","shared_compute_tested":True,"canonical_layer_bytes_step29":518104064,"bandwidth_scenarios_GBps":[1.4,3.2,8,16,32],"no_illegal_concatenation":True,"canonical_artifacts_changed":False}
 (OUT/"structured-matrix-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
 print("PASS — applicability matrix and shared-input group qualification written")
if __name__=="__main__":main()
