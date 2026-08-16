# Orange Pi RV2 GGUF vs vBuf Benchmark

## Environment

The primary comparison used the isolated patched pinned llama.cpp checkout at commit `4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c`, GCC/G++ 14.2, RVV enabled, CPU-only, OpenMP enabled, eight threads, context 256, batch/ubatch 64, `LLAMA_LOAD_MODE_NONE`, prompt `Hello world`, and three generated tokens. The existing `~/llama.cpp` checkout was not modified.

The corresponding artifacts are recorded in `artifact-identities.json`; established qualification is 377/377 tensor payload parity.

## Method

Each format has three warm runs and two runs requested as cold. Cache dropping was unavailable, so every requested cold run is classified `COLD_CACHE_NOT_GUARANTEED`; no cold-cache claim is made. The sampler recorded RSS, virtual size, faults, `/proc/<pid>/io`, per-process CPU percentage, per-thread CPU tick deltas, system ticks, and `/sys/block/mmcblk0/stat`.

## Timeline

### GGUF representative warm run: `gguf-warm-00-final4`

- `T+0.000 ms`: `PROCESS_START` (gguf)
- `T+14.532 ms`: `MODEL_OPEN_BEGIN` (gguf)
- `T+14.635 ms`: `MODEL_METADATA_BEGIN`
- `T+14.703 ms`: `FORMAT_OPEN_BEGIN`
- `T+200.875 ms`: `FORMAT_OPEN_END`
- `T+201.468 ms`: `TENSOR_ENUMERATION_BEGIN`
- `T+216.756 ms`: `TENSOR_ENUMERATION_END` (format)
- `T+471.377 ms`: `MODEL_METADATA_END`
- `T+472.574 ms`: `TOKENIZER_BEGIN`
- `T+1751.230 ms`: `TOKENIZER_END`
- `T+1752.030 ms`: `TENSOR_ENUMERATION_BEGIN` (runtime)
- `T+1779.135 ms`: `TENSOR_ENUMERATION_END` (runtime)
- `T+1779.171 ms`: `MODEL_STRUCTURE_READY`
- `T+1779.783 ms`: `BACKEND_BUFFER_ALLOCATION_BEGIN`
- `T+1780.294 ms`: `BACKEND_BUFFER_ALLOCATION_END`
- `T+1780.431 ms`: `PAYLOAD_MATERIALIZATION_BEGIN`
- `T+101681.057 ms`: `PAYLOAD_MATERIALIZATION_END`
- `T+101772.656 ms`: `GRAPH_RESERVE_BEGIN`
- `T+101876.644 ms`: `GRAPH_RESERVE_END`
- `T+101877.529 ms`: `EXECUTION_READY`
- `T+101877.543 ms`: `PROMPT_EVAL_BEGIN`
- `T+102918.936 ms`: `PROMPT_EVAL_END` (success)
- `T+102919.747 ms`: `FIRST_DECODE_BEGIN`
- `T+103310.345 ms`: `FIRST_DECODE_END` (success)
- `T+103310.462 ms`: `FIRST_TOKEN`
- `T+104048.506 ms`: `RUN_END` (success)

### vBuf representative warm run: `vbuf-warm-00-final4`

- `T+0.000 ms`: `PROCESS_START` (vbuf)
- `T+14.514 ms`: `MODEL_OPEN_BEGIN` (vbuf)
- `T+14.636 ms`: `FORMAT_OPEN_BEGIN`
- `T+807.102 ms`: `FORMAT_OPEN_END`
- `T+807.209 ms`: `MODEL_METADATA_BEGIN` (vbuf)
- `T+807.749 ms`: `MODEL_METADATA_END` (vbuf)
- `T+807.764 ms`: `TOKENIZER_BEGIN` (vbuf)
- `T+807.774 ms`: `TOKENIZER_END` (vbuf)
- `T+807.783 ms`: `TENSOR_ENUMERATION_BEGIN` (vbuf)
- `T+807.890 ms`: `TENSOR_ENUMERATION_END` (vbuf)
- `T+809.316 ms`: `MODEL_METADATA_BEGIN`
- `T+816.549 ms`: `MODEL_METADATA_END`
- `T+817.542 ms`: `TOKENIZER_BEGIN`
- `T+2200.971 ms`: `TOKENIZER_END`
- `T+2201.609 ms`: `TENSOR_ENUMERATION_BEGIN` (runtime)
- `T+2227.891 ms`: `TENSOR_ENUMERATION_END` (runtime)
- `T+2227.923 ms`: `MODEL_STRUCTURE_READY`
- `T+2228.465 ms`: `BACKEND_BUFFER_ALLOCATION_BEGIN`
- `T+2228.948 ms`: `BACKEND_BUFFER_ALLOCATION_END`
- `T+2229.076 ms`: `PAYLOAD_MATERIALIZATION_BEGIN`
- `T+76533.892 ms`: `PAYLOAD_MATERIALIZATION_END`
- `T+76702.556 ms`: `GRAPH_RESERVE_BEGIN`
- `T+76843.491 ms`: `GRAPH_RESERVE_END`
- `T+76844.308 ms`: `EXECUTION_READY`
- `T+76844.320 ms`: `PROMPT_EVAL_BEGIN`
- `T+77894.587 ms`: `PROMPT_EVAL_END` (success)
- `T+77895.379 ms`: `FIRST_DECODE_BEGIN`
- `T+78271.137 ms`: `FIRST_DECODE_END` (success)
- `T+78271.243 ms`: `FIRST_TOKEN`
- `T+79007.210 ms`: `RUN_END` (success)

## Phase Comparison

Warm and requested-cold timings are min/median/max across the required repetitions. `model_structure_construction` is the inclusive process interval through `MODEL_STRUCTURE_READY`; tensor enumeration is also shown as its measured subcomponent.

| Phase | GGUF warm ms | vBuf warm ms | GGUF requested-cold ms | vBuf requested-cold ms |
|---|---:|---:|---:|---:|
| `format_open` | 186.172/190.190/192.317 | 752.609/764.460/792.466 | 189.236/190.072/190.909 | 757.502/761.210/764.919 |
| `metadata` | 456.742/460.483/462.984 | 0.519/0.540/0.586 | 458.815/459.726/460.638 | 0.522/0.525/0.528 |
| `tokenizer` | 1277.472/1278.656/1297.381 | 1363.405/1367.928/1383.429 | 1277.085/1277.899/1278.713 | 1366.675/1367.020/1367.365 |
| `tensor_enumeration` | 26.267/26.393/27.105 | 25.882/26.044/26.282 | 26.585/26.796/27.006 | 25.864/25.968/26.073 |
| `model_structure_construction` | 1764.639/1768.870/1786.118 | 2153.186/2169.482/2213.409 | 1765.808/1766.113/1766.419 | 2162.238/2165.460/2168.682 |
| `backend_allocation` | 0.486/0.511/0.526 | 0.483/0.483/0.503 | 0.485/0.488/0.492 | 0.478/0.493/0.508 |
| `payload_materialization` | 99737.171/99749.641/99900.626 | 71293.104/72626.880/74304.816 | 99731.887/99853.636/99975.386 | 72135.038/72740.067/73345.096 |
| `repack_wall_us` | 32473.628/32549.910/32623.149 | 39872.196/40991.643/41632.850 | 32508.560/32519.435/32530.309 | 39755.520/40200.536/40645.552 |
| `graph_reserve` | 103.988/106.037/107.257 | 116.660/139.904/140.935 | 102.329/104.133/105.937 | 122.962/132.583/142.204 |
| `context_create` | 188.990/191.628/192.774 | 236.632/263.322/271.707 | 187.579/188.745/189.910 | 253.451/257.949/262.447 |
| `execution_ready_total` | 101721.849/101750.605/101877.529 | 73786.968/75107.327/76844.308 | 101708.319/101831.079/101953.839 | 74619.184/75216.585/75813.986 |
| `prompt_eval` | 1030.421/1034.823/1041.393 | 1050.267/1116.665/1127.779 | 1020.434/1065.275/1110.117 | 1048.467/1056.107/1063.747 |
| `first_decode` | 368.351/376.657/390.598 | 375.758/378.932/400.064 | 391.508/391.802/392.095 | 393.346/436.661/479.976 |
| `first_token` | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 |
| `total_run` | 103878.478/103898.894/104048.506 | 76027.272/77366.161/79007.210 | 103945.774/104038.827/104131.880 | 76878.109/77489.128/78100.146 |

## Repack Analysis

Both formats repacked exactly 28 tensors, with identical source and destination byte totals: 2.527 GiB per run. The representation transitions are the same; see `repack-gguf.csv` and `repack-vbuf.csv` for every tensor.

- gguf: repack wall median 32549.910 ms; repack CPU-sum median 32549.910 ms; throughput median 0.078 GiB/s.
- vbuf: repack wall median 40991.643 ms; repack CPU-sum median 40991.643 ms; throughput median 0.062 GiB/s.

The earlier impression that vBuf repacking was substantially faster is not supported by this same-model measurement. vBuf repacking is slower here, while its payload-materialization interval is about 26.6 s shorter. The earlier Qwen run was a different model and was not a formal comparison; it cannot establish a repack difference.

## CPU / I/O Analysis

Both paths use the same RVV repack kernels and the same 28 repacked tensors. GGUF physically reads about 4.66 GiB according to process I/O; vBuf varies around 4.0-4.7 GiB because its mmap-backed payload faults are demand-driven. vBuf has substantially more major faults, while GGUF has near-zero major faults under `LLAMA_LOAD_MODE_NONE`. The phase traces show payload/materialization, including repack, dominates both runs; graph reserve and decode are small.

The sampler data is preserved in `cpu-samples.csv` inside every raw run, with aggregate concatenations in `memory.csv` and `io.csv`. `PERF_COUNTERS_UNAVAILABLE` is recorded because perf counters were not available for this run.

## Memory Analysis

- gguf: peak RSS median 4.780 GiB; minimum sampled available RAM median 2.499 GiB; CPU buffer 2171.14 MiB; CPU_REPACK buffer 2587.83 MiB; scheduler reserve median 105.94 ms; graph 1710 nodes / 1 split.
- vbuf: peak RSS median 7.282 GiB; minimum sampled available RAM median 2.523 GiB; CPU buffer 2171.14 MiB; CPU_REPACK buffer 2587.83 MiB; scheduler reserve median 139.81 ms; graph 1710 nodes / 1 split.

The CPU and CPU_REPACK buffer sizes are identical. vBuf peak RSS is about 7.28 GiB versus about 4.78 GiB for GGUF because the vBuf mmap-backed source remains resident while backend/repack buffers are also present.

## vBuf Diagnostic

`vbuf-ml-diagnose` reported map 66.498 us, canonical 675307.471 us, bootstrap 33.124 us, model metadata 29.332 us, tensor directory 1114.561 us, tokenizer metadata 43885.928 us, consumer open total 115857.180 us, and total 836328.013 us. This is a structural micro-measurement, not a replacement for the end-to-end comparison.

## Result

vBuf reaches first successful decode faster in the warm comparison: median process-start to run end is 77366.161 ms versus 103898.894 ms for GGUF, a difference of 26532.733 ms. The measured advantage is attributable primarily to a shorter payload-materialization interval, not to less repacking or lower peak RSS. The advantage survives the requested warm-cache repetitions, but no true cold-cache conclusion is possible.

The tiny decode is a sanity check only. Both paths use the same llama.cpp/RVV kernels; this benchmark does not demonstrate faster steady-state RISC-V inference.

## Follow-up Hypotheses

- Investigate why vBuf demand paging produces higher RSS and major-fault counts despite the shorter materialization interval.
- Measure a separately controlled cache state with appropriate privileges before making cold-start claims.
- A later experiment may test a persisted RVV-native physical representation to remove runtime repacking; that was not changed here.

