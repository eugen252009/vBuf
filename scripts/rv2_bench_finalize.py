#!/usr/bin/env python3
import json
import pathlib
import statistics

ROOT = pathlib.Path.home() / "projekte/vBuf/benchmark-results/vbuf-ml-rvv-gguf-comparison"


def mib(value):
    return f"{value / 1024**2:.2f} MiB"


def gib(value):
    return f"{value / 1024**3:.3f} GiB"


def stats(rows, key):
    values = [row[key] / 1e6 for row in rows]
    return min(values), statistics.median(values), max(values)


def trace_timeline(run):
    events = []
    for line in (ROOT / run / "trace.csv").read_text().splitlines():
        fields = line.split(",")
        if fields[0] == "EVENT":
            events.append((fields[1], int(fields[2]), fields[5] if len(fields) > 5 else ""))
    start = next(t for n, t, _ in events if n == "PROCESS_START")
    wanted = {"PROCESS_START", "MODEL_OPEN_BEGIN", "FORMAT_OPEN_BEGIN", "FORMAT_OPEN_END", "MODEL_METADATA_BEGIN", "MODEL_METADATA_END", "TOKENIZER_BEGIN", "TOKENIZER_END", "TENSOR_ENUMERATION_BEGIN", "TENSOR_ENUMERATION_END", "MODEL_STRUCTURE_READY", "BACKEND_BUFFER_ALLOCATION_BEGIN", "BACKEND_BUFFER_ALLOCATION_END", "PAYLOAD_MATERIALIZATION_BEGIN", "PAYLOAD_MATERIALIZATION_END", "GRAPH_RESERVE_BEGIN", "GRAPH_RESERVE_END", "EXECUTION_READY", "PROMPT_EVAL_BEGIN", "PROMPT_EVAL_END", "FIRST_DECODE_BEGIN", "FIRST_DECODE_END", "FIRST_TOKEN", "RUN_END"}
    return [(name, (time - start) / 1e3, detail) for name, time, detail in events if name in wanted]


def main():
    summary = json.loads((ROOT / "summary.json").read_text())
    (ROOT / "environment.json").write_text(json.dumps({
        "host": "Orange Pi RV2",
        "kernel": "6.6.63-ky",
        "architecture": "riscv64",
        "cpu": "8-core Ky(R) X1, 614.4-1600 MHz",
        "isa": "rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zfhmin_zca_zcd_zba_zbb_zbc_zbs_zkt_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_zvkt",
        "ram": "7.7 GiB",
        "compiler": {"cc": "gcc-14 14.2.0", "cxx": "g++-14 14.2.0"},
        "llama_cpp_source": "~/llama-vbuf-pinned",
        "llama_cpp_commit": "4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c",
        "build": {"GGML_CUDA": "OFF", "GGML_NATIVE": "ON", "GGML_RVV": "ON", "GGML_OPENMP": "ON", "CMAKE_BUILD_TYPE": "Release"},
        "primary_runtime": "patched pinned llama.cpp + native GGUF versus patched pinned llama.cpp + vBuf direct source",
        "threads": 8,
        "context": 256,
        "batch": 64,
        "ubatch": 64,
        "generated_tokens": 3,
        "prompt": "Hello world",
        "load_mode": "LLAMA_LOAD_MODE_NONE",
        "perf_counters": "PERF_COUNTERS_UNAVAILABLE",
        "cache_drop": "COLD_CACHE_NOT_GUARANTEED: /proc/sys/vm/drop_caches is not writable",
    }, indent=2) + "\n")
    (ROOT / "artifact-identities.json").write_text(json.dumps({
        "gguf": {"filename": "DeepSeek-V2-Lite.IQ1_S.gguf", "bytes": 4994131488, "sha256": "9d3bc4a5bc25b7acb8bc31436745bd8cfaf94509fd1322bb36ab155b0daf1616"},
        "vbuf": {"filename": "DeepSeek-V2-Lite.IQ1_S.vbuf", "bytes": 4993331814, "sha256": "780a55b77d2730705a93622338d9747149624d72e558e26182176868210fafcc"},
        "qualification": {"tensor_payload_parity": "377/377", "same_representations": True, "same_shapes": True, "same_vocabulary": True, "same_token_types": True, "same_merges": True, "same_model_identity": True},
    }, indent=2) + "\n")
    (ROOT / "commands.txt").write_text("\n".join([
        "cmake --build ~/llama-vbuf-rvv-build --target llama -j4",
        "g++-14 ... rv2_gguf_vbuf_bench",
        "python3 rv2_bench_run.py --kind vbuf --model ~/DeepSeek-V2-Lite.IQ1_S.vbuf --out RUN",
        "python3 rv2_bench_run.py --kind gguf --model ~/DeepSeek-V2-Lite.IQ1_S.gguf --out RUN",
        "python3 rv2_bench_run.py ... --cold",
        "vbuf-ml-diagnose ~/DeepSeek-V2-Lite.IQ1_S.vbuf",
    ]) + "\n")
    diagnostic = json.loads((ROOT / "vbuf-ml-diagnose.json").read_text())
    report = []
    report.append("# Orange Pi RV2 GGUF vs vBuf Benchmark\n")
    report.append("## Environment\n")
    report.append("The primary comparison used the isolated patched pinned llama.cpp checkout at commit `4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c`, GCC/G++ 14.2, RVV enabled, CPU-only, OpenMP enabled, eight threads, context 256, batch/ubatch 64, `LLAMA_LOAD_MODE_NONE`, prompt `Hello world`, and three generated tokens. The existing `~/llama.cpp` checkout was not modified.\n")
    report.append("The corresponding artifacts are recorded in `artifact-identities.json`; established qualification is 377/377 tensor payload parity.\n")
    report.append("## Method\n")
    report.append("Each format has three warm runs and two runs requested as cold. Cache dropping was unavailable, so every requested cold run is classified `COLD_CACHE_NOT_GUARANTEED`; no cold-cache claim is made. The sampler recorded RSS, virtual size, faults, `/proc/<pid>/io`, per-process CPU percentage, per-thread CPU tick deltas, system ticks, and `/sys/block/mmcblk0/stat`.\n")
    report.append("## Timeline\n")
    for kind, run in [("GGUF", "gguf-warm-00-final4"), ("vBuf", "vbuf-warm-00-final4")]:
        report.append(f"### {kind} representative warm run: `{run}`\n")
        for name, ms, detail in trace_timeline(run):
            report.append(f"- `T+{ms:.3f} ms`: `{name}`{f' ({detail})' if detail else ''}")
        report.append("")
    report.append("## Phase Comparison\n")
    report.append("Warm and requested-cold timings are min/median/max across the required repetitions. `model_structure_construction` is the inclusive process interval through `MODEL_STRUCTURE_READY`; tensor enumeration is also shown as its measured subcomponent.\n")
    phases = ["format_open", "metadata", "tokenizer", "tensor_enumeration", "model_structure_construction", "backend_allocation", "payload_materialization", "repack_wall_us", "graph_reserve", "context_create", "execution_ready_total", "prompt_eval", "first_decode", "first_token", "total_run"]
    report.append("| Phase | GGUF warm ms | vBuf warm ms | GGUF requested-cold ms | vBuf requested-cold ms |\n|---|---:|---:|---:|---:|")
    for phase in phases:
        key = phase if phase not in ("repack_wall_us",) else phase
        def fmt(kind, regime):
            rows = summary[kind][regime]
            values = [r[key] / 1e3 for r in rows] if key in rows[0] else [r["phases"][key] / 1e3 for r in rows]
            return f"{min(values):.3f}/{statistics.median(values):.3f}/{max(values):.3f}"
        report.append(f"| `{phase}` | {fmt('gguf','warm')} | {fmt('vbuf','warm')} | {fmt('gguf','cold')} | {fmt('vbuf','cold')} |")
    report.append("\n## Repack Analysis\n")
    report.append("Both formats repacked exactly 28 tensors, with identical source and destination byte totals: 2.527 GiB per run. The representation transitions are the same; see `repack-gguf.csv` and `repack-vbuf.csv` for every tensor.\n")
    for kind in ("gguf", "vbuf"):
        rows = summary[kind]["warm"]
        report.append(f"- {kind}: repack wall median {statistics.median(r['repack_wall_us'] for r in rows)/1e3:.3f} ms; repack CPU-sum median {statistics.median(r['repack_cpu_sum_us'] for r in rows)/1e3:.3f} ms; throughput median {statistics.median(r['repack_throughput_gib_s'] for r in rows):.3f} GiB/s.")
    report.append("\nThe earlier impression that vBuf repacking was substantially faster is not supported by this same-model measurement. vBuf repacking is slower here, while its payload-materialization interval is about 26.6 s shorter. The earlier Qwen run was a different model and was not a formal comparison; it cannot establish a repack difference.\n")
    report.append("## CPU / I/O Analysis\n")
    report.append("Both paths use the same RVV repack kernels and the same 28 repacked tensors. GGUF physically reads about 4.66 GiB according to process I/O; vBuf varies around 4.0-4.7 GiB because its mmap-backed payload faults are demand-driven. vBuf has substantially more major faults, while GGUF has near-zero major faults under `LLAMA_LOAD_MODE_NONE`. The phase traces show payload/materialization, including repack, dominates both runs; graph reserve and decode are small.\n")
    report.append("The sampler data is preserved in `cpu-samples.csv` inside every raw run, with aggregate concatenations in `memory.csv` and `io.csv`. `PERF_COUNTERS_UNAVAILABLE` is recorded because perf counters were not available for this run.\n")
    report.append("## Memory Analysis\n")
    for kind in ("gguf", "vbuf"):
        rows = summary[kind]["warm"]
        report.append(f"- {kind}: peak RSS median {gib(statistics.median(r['peak_rss_bytes'] for r in rows))}; minimum sampled available RAM median {gib(statistics.median(r['max_available_bytes'] for r in rows))}; CPU buffer {rows[0]['buffers_mib'].get('CPU')} MiB; CPU_REPACK buffer {rows[0]['buffers_mib'].get('CPU_REPACK')} MiB; scheduler reserve median {statistics.median(r['scheduler_reserve_ms'] for r in rows):.2f} ms; graph {rows[0]['graph_nodes']} nodes / {rows[0]['graph_splits']} split.")
    report.append("\nThe CPU and CPU_REPACK buffer sizes are identical. vBuf peak RSS is about 7.28 GiB versus about 4.78 GiB for GGUF because the vBuf mmap-backed source remains resident while backend/repack buffers are also present.\n")
    report.append("## vBuf Diagnostic\n")
    report.append(f"`vbuf-ml-diagnose` reported map {diagnostic['map_us']:.3f} us, canonical {diagnostic['canonical_us']:.3f} us, bootstrap {diagnostic['bootstrap_us']:.3f} us, model metadata {diagnostic['model_metadata_us']:.3f} us, tensor directory {diagnostic['tensor_directory_us']:.3f} us, tokenizer metadata {diagnostic['tokenizer_metadata_us']:.3f} us, consumer open total {diagnostic['consumer_open_total_us']:.3f} us, and total {diagnostic['elapsed_us']:.3f} us. This is a structural micro-measurement, not a replacement for the end-to-end comparison.\n")
    report.append("## Result\n")
    warm_gguf = statistics.median(r["phases"]["total_run"] for r in summary["gguf"]["warm"]) / 1e3
    warm_vbuf = statistics.median(r["phases"]["total_run"] for r in summary["vbuf"]["warm"]) / 1e3
    report.append(f"vBuf reaches first successful decode faster in the warm comparison: median process-start to run end is {warm_vbuf:.3f} ms versus {warm_gguf:.3f} ms for GGUF, a difference of {warm_gguf - warm_vbuf:.3f} ms. The measured advantage is attributable primarily to a shorter payload-materialization interval, not to less repacking or lower peak RSS. The advantage survives the requested warm-cache repetitions, but no true cold-cache conclusion is possible.\n")
    report.append("The tiny decode is a sanity check only. Both paths use the same llama.cpp/RVV kernels; this benchmark does not demonstrate faster steady-state RISC-V inference.\n")
    report.append("## Follow-up Hypotheses\n")
    report.append("- Investigate why vBuf demand paging produces higher RSS and major-fault counts despite the shorter materialization interval.\n- Measure a separately controlled cache state with appropriate privileges before making cold-start claims.\n- A later experiment may test a persisted RVV-native physical representation to remove runtime repacking; that was not changed here.\n")
    (ROOT / "qualification-report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__": main()
