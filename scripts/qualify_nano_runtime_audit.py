#!/usr/bin/env python3
"""Summarize the read-only Nano/direct-view audit over existing vBuf-ML artifacts."""
from __future__ import annotations
import argparse, csv, json, statistics, subprocess
from pathlib import Path

def parse_geometry(line):
    return {k: int(v) for k, v in (field.split("=", 1) for field in line.strip().split(",")[1:])}
def rows(path):
    out=[]
    for line in path.read_text().splitlines():
        if line.startswith("SAMPLE,"):
            parts=line.split(","); _, operation, sample, nanos, visited, touched = parts[:6]; result=parts[6] if len(parts)>6 else 0
            out.append({"operation":operation,"sample":int(sample),"nanos":int(float(nanos)),"visited":int(visited),"bytes_touched":int(touched),"result":int(float(result))})
    return out
def med(values): return statistics.median(values) if values else 0
def write_csv(path, data):
    with path.open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(data)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]); ap.add_argument("--runs", type=int, default=30); a=ap.parse_args(); root=a.root.resolve(); out=root/"benchmark-results/vbuf-ml-nano-runtime-audit"; raw=out/"raw"; out.mkdir(parents=True,exist_ok=True); raw.mkdir(exist_ok=True); run_count=a.runs
    subprocess.run(["cargo","build","--release","--manifest-path",str(root/"rust/Cargo.toml"),"-p","vbuf-ml","--bin","vbuf-nano-runtime-audit"],cwd=root,check=True)
    geometries={}; all_traversal=[]
    for artifact in ("BF16","Q8_0"):
        path=root/"research-models"/f"Qwen3-0.6B-{artifact}.vbuf"; raw_path=raw/f"{artifact}.audit.txt"
        raw_path.write_text(subprocess.check_output([str(root/"rust/target/release/vbuf-nano-runtime-audit"),str(path),str(a.runs)],text=True))
        geometry=parse_geometry(raw_path.read_text().splitlines()[0]); geometries[artifact]=geometry
        for row in rows(raw_path): all_traversal.append({"artifact":artifact,**row})
    (out/"artifact-geometry.json").write_text(json.dumps(geometries,indent=2)+"\n")
    write_csv(out/"traversal-summary.csv", [{"artifact":a,"operation":op,"samples":len(vals),"median_ns":med([r["nanos"] for r in vals]),"visited_median":med([r["visited"] for r in vals]),"bytes_touched_median":med([r["bytes_touched"] for r in vals]),"result_median":med([r["result"] for r in vals])} for a in ("BF16","Q8_0") for op in ("canonical_parse","semantic_parse_all","sequential_block_traversal","nano_reconstruct","nano_setbit_iter_existing","directory_binary_lookup_all_tensors","dense_ordinal_view_all_tensors") for vals in [[r for r in all_traversal if r["artifact"]==a and r["operation"]==op]]])
    write_csv(out/"nano-geometry.csv", [{"artifact":a,"base_step":g["base_step"],"data_region_bytes":g["data_region_bytes"],"slots":g["slots"],"nano_bytes":g["nano_bytes"],"blocks":g["blocks"],"set_bits":g["set_bits"],"continuations":g["continuations"],"nano_bytes_per_set_bit":g["nano_bytes"]/g["set_bits"]} for a,g in geometries.items()])
    tokenizer_rows=[]
    for a,g in geometries.items():
        tokenizer_rows.extend([
            {"artifact":a,"view":"token_text","logical_entries":g["token_count"],"wire_bytes":g["token_text_bytes"],"current_validation_work":"offset/token UTF-8 validation","direct_view":"DIRECT_VIEW_READY","nano":"NANO_REDUNDANT"},
            {"artifact":a,"view":"token_offsets","logical_entries":g["token_count"]+1,"wire_bytes":g["token_offsets_bytes"],"current_validation_work":"monotonic bounds + per-token slice validation","direct_view":"DIRECT_VIEW_READY","nano":"NANO_REDUNDANT"},
            {"artifact":a,"view":"token_types","logical_entries":g["token_count"],"wire_bytes":g["token_types_bytes"],"current_validation_work":"parallel-array width/value validation","direct_view":"DIRECT_VIEW_READY","nano":"NANO_REDUNDANT"},
            {"artifact":a,"view":"token_scores","logical_entries":g["token_count"],"wire_bytes":g["token_scores_bytes"],"current_validation_work":"optional; absent in qualified artifact","direct_view":"DIRECT_VIEW_READY_IF_PRESENT","nano":"NANO_REDUNDANT"},
            {"artifact":a,"view":"merge_left_right_ids","logical_entries":g["merge_count"]*2,"wire_bytes":g["merge_id_bytes"],"current_validation_work":"two arrays + per-pair bounds validation","direct_view":"DIRECT_VIEW_READY","nano":"NANO_REDUNDANT"},
        ])
    write_csv(out/"tokenizer-view-summary.csv", tokenizer_rows)
    traversal = list(csv.DictReader((out/"traversal-summary.csv").open()))
    bootstrap_rows=[]
    for a in geometries:
        bootstrap_rows.extend([
            {"artifact":a,"current_operation":"canonical_parse","classification":"DISCOVERY+VALIDATION","median_ns":next(r["median_ns"] for r in traversal if r["artifact"]==a and r["operation"]=="canonical_parse"),"future_direct_view":"bounded validation remains"},
            {"artifact":a,"current_operation":"semantic_parse_all","classification":"DISCOVERY+VALIDATION+MATERIALIZATION","median_ns":next(r["median_ns"] for r in traversal if r["artifact"]==a and r["operation"]=="semantic_parse_all"),"future_direct_view":"O(regions) wrappers plus required validation; element validation may remain"},
        ])
    write_csv(out/"bootstrap-cost-summary.csv", bootstrap_rows)
    write_csv(out/"memory-summary.csv", [{"artifact":a,"token_text_bytes":g["token_text_bytes"],"offset_bytes":g["token_offsets_bytes"],"type_bytes":g["token_types_bytes"],"merge_id_bytes":g["merge_id_bytes"],"consumer_snapshot_lower_bound_bytes":g["token_text_bytes"]+g["token_count"]*24+g["token_count"]*16*2+g["merge_count"]*16+g["tensor_name_bytes"]+g["tensor_entries"]*24+g["tensor_dimension_count"]*8,"direct_canonical_view_copy_bytes":0,"direct_wrapper_state":"O(1) region/range handles; no element arrays","allocation_measurement":"estimated from representation; allocator counters not instrumented"} for a,g in geometries.items()])
    classifications=[
      ("Bootstrap","current bounded role directory; scans canonical blocks by Key-ID/occurrence","NANO_REDUNDANT","bootstrap already names semantic regions; Nano cannot resolve roles"),
      ("ModelMetadata","11 semantic fields, scalar values in referenced blocks","NANO_REDUNDANT","small fixed field set; direct checked ranges are cheaper"),
      ("TensorDirectory","variable-size sorted records; 310/311 entries","DIRECT_VIEW_READY_WITH_RUNTIME_INDEX","one region scan can establish borrowed name/dimension records; name lookup remains runtime-local"),
      ("TokenizerMetadata","14 role references and control values","NANO_REDUNDANT","small role table directly locates dense arrays"),
      ("Token text/offset/type/score","dense parallel arrays; 151,936 tokens","DIRECT_VIEW_READY","base+ordinal/offset addressing; Nano adds a large sparse bit scan"),
      ("Merge IDs","two dense u32 arrays; 151,387 pairs","DIRECT_VIEW_READY_WITH_RUNTIME_INDEX","direct ordinal pair view; pair→rank index remains runtime work"),
      ("Tensor payloads","checked ranges; 310/311 tensors","DIRECT_VIEW_READY","TensorDirectory already resolves exact payload ranges"),
      ("Continuation chains","no continuations in qualified Qwen3 artifacts","NANO_ASSISTED","Nano can enumerate physical members; Key-ID/continuation validation remains authoritative"),
      ("Layer-major physical scan","GlobalPre + layers + GlobalPost","NANO_ASSISTED","potential future physical discovery/readiness, not tensor lookup"),
      ("Integrity metadata","not present in qualified artifacts","NANO_REDUNDANT","optional integrity targets remain semantic ranges"),
    ]
    write_csv(out/"classification.csv", [{"structure":s,"current_representation":r,"classification":c,"evidence_or_boundary":e} for s,r,c,e in classifications])
    (out/"environment.json").write_text(json.dumps({"artifacts":["Qwen3-0.6B-BF16.vbuf","Qwen3-0.6B-Q8_0.vbuf"],"runs":run_count,"build":"cargo release; in-memory read-only audit","nano_status":"benchmark-only reconstruction; not v0.6 wire","wire_changed":False,"filesystem":"mmap; warm process/page cache approximate","allocators":"not instrumented","nano_semantics":"one bit per BaseStep slot for canonical physical starts","step22a_reference":"ConsumerModel ~132 ms; synthesis ~132 ms; ~910600 ABI calls"},indent=2)+"\n")
    (out/"qualification-config.json").write_text(json.dumps({"purpose":"Nano/direct-view architecture audit only","native_runtime_implemented":False,"wire_change":False,"operations":["canonical parse","semantic parse","sequential block traversal","synthetic Nano build/set-bit iteration","directory binary lookup","dense ordinal access"],"no_claims":["no disk-read equivalence","no cold-cache result","no persisted Nano selection","no runtime index elimination"]},indent=2)+"\n")
    print(f"PASS — Nano runtime audit evidence written to {out}")
if __name__=="__main__": main()
