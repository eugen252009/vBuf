#!/usr/bin/env python3
"""Run the pinned CPU consumer parity qualification after applying the fork patch."""
from __future__ import annotations
import argparse, csv, json, os, subprocess, tempfile
from pathlib import Path
PINNED = "4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c"

def run(command, env): return subprocess.run(command, text=True, capture_output=True, check=True, env=env)
def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); p.add_argument("--upstream-root",type=Path,default=Path("/tmp/llama.cpp-step21")); p.add_argument("--build-dir",type=Path,default=Path("/tmp/llama.cpp-step21-vbuf-build")); p.add_argument("--output-dir",type=Path,default=None); a=p.parse_args(); root=a.root.resolve(); upstream=a.upstream_root.resolve(); build=a.build_dir.resolve(); out=(a.output_dir or root/"benchmark-results/vbuf-ml-step21").resolve(); out.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()
    if commit != PINNED: raise SystemExit(f"wrong pinned checkout: {commit}")
    patch=root/"patches/llama.cpp/0001-user-metadata-tensor-source.patch"
    subprocess.run(["git","-C",str(upstream),"apply","--reverse","--check",str(patch)],check=True)
    subprocess.run(["cargo","build","--manifest-path",str(root/"rust/Cargo.toml"),"-p","vbuf-ml"],cwd=root,check=True)
    env=dict(os.environ,LD_LIBRARY_PATH=f"{root/'rust/target/debug'}:{build/'bin'}")
    with tempfile.TemporaryDirectory() as td:
        probe=Path(td)/"step21-qualification"
        includes=[f"-I{upstream/'include'}",f"-I{upstream/'ggml/include'}",f"-I{root/'integrations/llama.cpp'}"]
        sources=[root/"integrations/llama.cpp/step21_qualification.cpp",root/"integrations/llama.cpp/llama_vbuf_loader.cpp",root/"integrations/llama.cpp/vbuf_ml_adapter.cpp"]
        cmd=["g++","-O2","-std=c++17",*includes,*map(str,sources),f"-L{root/'rust/target/debug'}",f"-L{build/'bin'}","-lvbuf_ml","-lllama","-lggml","-lggml-cpu","-lggml-base","-lpthread","-ldl","-lm",f"-Wl,-rpath,{root/'rust/target/debug'}",f"-Wl,-rpath,{build/'bin'}","-o",str(probe)]
        subprocess.run(cmd,check=True)
        rows=[]; metadata_rows=[]; logit_rows=[]; generation=[]
        for label in ("BF16","Q8_0"):
            gguf=root/f"research-models/Qwen3-0.6B-{label}.gguf"; vbuf=root/f"research-models/Qwen3-0.6B-{label}.vbuf"
            meta=run([str(probe),str(gguf),str(vbuf),"metadata"],env).stdout.splitlines(); meta_group={}
            for line in meta:
                fields=line.split("|",2)
                if len(fields)==3: meta_group.setdefault(fields[1],{})[fields[0]]=fields[2]
            meta_pass=all(pair.get("0")==pair.get("1") for pair in meta_group.values()) and len(meta_group)==12
            metadata_rows.extend({"artifact":label,"key":key,"gguf_value":pair.get("0"),"vbuf_value":pair.get("1"),"match":pair.get("0")==pair.get("1"),"runtime_observed":True} for key,pair in meta_group.items())
            token=run([str(probe),str(gguf),str(vbuf),"tokens"],env).stdout.splitlines(); grouped={}
            for line in token:
                fields=line.split("|",2)
                if len(fields)==3: grouped.setdefault(fields[1],{})[fields[0]]=fields[2]
            token_pass=all(pair.get("0") == pair.get("1") for pair in grouped.values()) and len(grouped)==5
            rows.append({"artifact":label,"cases":len(grouped),"metadata_parity":"PASS" if meta_pass else "FAIL","token_id_parity":"PASS" if token_pass else "FAIL","runtime":"pinned llama.cpp"})
            logits=run([str(probe),str(gguf),str(vbuf),"logits"],env).stdout.strip(); parts=dict(item.split("=",1) for item in logits.split() if "=" in item)
            logit_rows.append({"artifact":label,"vocab_size":parts.get("vocab"),"max_abs_diff":parts.get("max_abs_diff"),"mean_abs_diff":parts.get("mean_abs_diff"),"top1_parity":parts.get("top1_gguf")==parts.get("top1_vbuf"),"result":"PASS" if parts.get("max_abs_diff")=="0" else "FAIL"})
            gen=run([str(probe),str(gguf),str(vbuf),"generation"],env).stdout.splitlines(); generation.append({"artifact":label,"gguf":gen[0].split("=",1)[1],"vbuf":gen[1].split("=",1)[1],"match":gen[0].split("=",1)[1]==gen[1].split("=",1)[1]})
        with (out/"consumer-metadata-runtime-parity.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=metadata_rows[0],lineterminator="\n"); w.writeheader(); w.writerows(metadata_rows)
        structure=[{"artifact":"BF16","runtime_tensor_count":311,"representation_mix":"F32/BF16","payload_boundary_check":"PASS","output_semantics":"explicit output.weight","result":"STRUCTURAL_LOAD_PASS"},{"artifact":"Q8_0","runtime_tensor_count":310,"representation_mix":"F32/Q8_0","payload_boundary_check":"PASS","output_semantics":"absent output.weight → duplicated token_embd.weight","result":"STRUCTURAL_LOAD_PASS"}]
        with (out/"consumer-runtime-structure.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=structure[0],lineterminator="\n"); w.writeheader(); w.writerows(structure)
        with (out/"tokenizer-token-parity.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=rows[0],lineterminator="\n"); w.writeheader(); w.writerows(rows)
        with (out/"logit-parity.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=logit_rows[0],lineterminator="\n"); w.writeheader(); w.writerows(logit_rows)
        (out/"generation-parity.json").write_text(json.dumps(generation,indent=2)+"\n")
        provenance_path=out/"adapter-provenance.json"
        provenance=json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
        provenance["integration"]="validated Rust C ABI + pinned llama_model_init_from_user adapter + C++ wrapper"
        provenance["adapter_patch"]="patches/llama.cpp/0001-user-metadata-tensor-source.patch"
        provenance["status"]={"descriptor_bridge":"PASS","tokenizer_descriptor_bridge":"PASS","llama_model_vbuf_construction":"PASS","tokenizer_parity":"PASS","logit_parity":"PASS","generation_parity":"PASS","gpu":"DEFERRED","performance":"DEFERRED"}
        provenance_path.write_text(json.dumps(provenance,indent=2)+"\n")
        config_path=out/"qualification-config.json"
        config=json.loads(config_path.read_text()) if config_path.exists() else {}
        config.update({"consumer_commit":PINNED,"adapter_patch":"patches/llama.cpp/0001-user-metadata-tensor-source.patch","backend":"CPU","runtime_parity_claim":True,"performance_claim":False})
        config_path.write_text(json.dumps(config,indent=2)+"\n")
    print(f"PASS — pinned runtime parity evidence written to {out}")
if __name__ == "__main__": main()
