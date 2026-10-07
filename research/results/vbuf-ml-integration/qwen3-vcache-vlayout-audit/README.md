# Qwen3 V-cache / V-layout audit

## Decision

No production redesign was implemented. The canonical Qwen3-14B Q4_K_M
26/14 run, 1,032-token production capacity, chunk size 32, single-GPU default,
and `SHADOW` optimizer default remain unchanged. Only the qualification gate's
capacity sweep was extended to admit capacity 1,024.

The measured copy is a real large-capacity cost, but the tested visible-extent
attention graph is **not numerically equivalent** to canonical. Dual-layout
storage is too tight at 32K, and a native-layout attention consumer requires a
new CUDA operator. Thus there is no evidence for a small, low-risk production
fusion. If further research is authorized, prioritize a native-layout AV
consumer (Option C) as a research prototype; keep persistent dual-layout V
(Option B) as a capacity-1,032 fallback if its memory cost is acceptable. Do
not adopt the visible-extent form tested here (Option A).

## Artifact and conditions

- Model semantic SHA-256: `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- Model: qualified local Qwen3-14B Q4_K_M vBuf artifact; 40 layers,
  5,120 hidden, 40 query heads, 8 KV heads, 128 dimensions, F16 KV.
- Placement: 26 layers on RTX 3060, 14 on RTX 2080 SUPER; CUDA driver
  `550.163.01`; Nsight Systems `2023.4.4.54`.
- CUDA backend source used by the external qualification build:
  `/home/eugen/.cache/vbuf-agent-qualification/ggml`, commit
  `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`.
- Captures use the qualified range server and natural-repeat-token fixture.
  Every point is one instrumented capture, not a multi-run median.
- The checked-in raw `.qdrep` files are the converted Nsight traces; companion
  gate logs and CSV summaries are retained under `raw/`. Candidate numerical
  dumps are F32 outputs and sampled F16 KV rows.

GPU-kernel sums below are sums of individual kernel durations across both
GPUs, not wall-clock latency. Nsight-traced host/decode wall timings are
observer-distorted and are not used for speedup claims. Logical read/write
traffic is derived from tensor sizes, not a DRAM hardware-counter measurement.

## Source-backed hot path

1. `QwenCudaRuntimeState::create_session` allocates one device-local packed-V
   scratch tensor of shape `[capacity, 128, 8]` per device, alongside the
   per-layer K and V caches (`integrations/ggml/src/qwen3_cuda_core.cpp:515-549`).
   At capacity 32,768 the scratch is 64 MiB/device; it is reused layer by layer,
   not allocated once per layer.
2. `qwen3_cuda_build_layer` writes the current V rows into the canonical
   `[128, 8*capacity]` cache with `ggml_set_rows`, reshapes it to
   `[128, 8, capacity]`, permutes it to `[capacity, 128, 8]`, then `ggml_cpy`s
   the *entire capacity* into packed scratch before the AV matmul
   (`integrations/ggml/src/qwen3_cuda_core.cpp:201-228`).
3. The graph is built at capacity-sized extents for decode and prefill
   (`integrations/ggml/src/qwen3_generation.cpp:577-623`). For query rows,
   `run_step` fills a capacity-wide causal mask: visible keys receive zero and
   future keys `-inf`; scores, softmax, and AV retain the capacity dimension
   (`qwen3_generation.cpp:768-856`). The session length commits only after the
   graph and output readback complete.
4. The CUDA AV path loops over its `ncols2` sequence dimension and loads the
   packed V value for each column even when the corresponding softmax weight
   is zero (`ggml-cuda/mmvf.cu`, especially lines 136-158 and 159-185). Thus
   masking suppresses future contributions, not capacity-sized V traffic or
   attention work.
5. The qualified CUDA dependency dispatches the non-contiguous F16-to-F16
   permutation through generic scalar `ggml_cuda_cpy` (`ggml-cuda/cpy.cu:429-501`);
   the measured kernel is `cpy_scalar<&cpy_1_scalar<__half, __half>>`. The
   scalar launch uses 64 threads/block (`cpy.cuh:3`, `cpy.cu:205-216`).

The cache row layout is `[position, KV-head, dimension]` in physical order;
packed scratch makes position contiguous for AV. There is one such V-layout
copy per transformer layer for a one-row decode and one copy per layer for a
prefill graph, independent of query row count. Source inspection confirms this
for `query_count=1,16,32`; captures observed rows 1 and 32 at capacity 32,768,
and rows 16 at capacity 16,384 (the current 32K/chunk-16 scratch qualification
bound is insufficient). The rows-16 capture verifies the same 40-copy graph
structure, but is not a capacity-matched timing comparison.

## Capacity scaling at visible prefix 32

The decode appends position 32, so its attention extent is 33. Each copy moves
`capacity * 8 * 128 * 2` payload bytes; source reads plus destination writes are
2x the listed copy payload. Each row below is one Nsight capture.

| Capacity | Bytes/copy | 40-copy payload | Logical read + write | Copies / grid blocks | Copy sum (ms) | All kernel sum (ms) | Copy share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 2 MiB | 80 MiB | 0.156 GiB | 40 / 16,384 | 2.650 | 30.822 | 8.6% |
| 2,048 | 4 MiB | 160 MiB | 0.313 GiB | 40 / 32,768 | 4.548 | 34.049 | 13.4% |
| 4,096 | 8 MiB | 320 MiB | 0.625 GiB | 40 / 65,536 | 9.324 | 41.442 | 22.5% |
| 8,192 | 16 MiB | 640 MiB | 1.250 GiB | 40 / 131,072 | 21.244 | 58.911 | 36.1% |
| 16,384 | 32 MiB | 1,280 MiB | 2.500 GiB | 40 / 262,144 | 54.011 | 113.746 | 47.5% |
| 32,768 | 64 MiB | 2,560 MiB (2.5 GiB) | 5.000 GiB | 40 / 524,288 | 273.150 | 369.376 | 73.9% |

The 26/14 device copy sums at capacity 32,768 were 248.138 ms on the RTX 3060
(26 copies) and 25.011 ms on the RTX 2080 SUPER (14 copies). Capacity 1,024 is
the measured rung nearest the unchanged production capacity 1,032; capacity
1,032 itself was not profiled.

## Context scaling at fixed capacity 32,768

Only the visible prefix varies. The copy extent and grid remain at capacity;
`visible_extent_payload` is the payload that an ideal prefix-only pack would
need for the same 40 layers. The measured copy sum was effectively flat across
these contexts: 274.392–281.550 ms; copy share stayed 73.3–74.0%.

| Visible prefix | Attention extent | Ideal visible payload / 40 layers | Full / visible factor | Measured copy sum (ms) | All kernel sum (ms) | Copy share |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 33 | 2.578 MiB | 992.97x | 277.350 | 375.769 | 73.8% |
| 128 | 129 | 10.078 MiB | 254.02x | 278.178 | 378.996 | 73.4% |
| 512 | 513 | 40.078 MiB | 63.88x | 277.921 | 379.268 | 73.3% |
| 2,048 | 2,049 | 160.078 MiB | 15.99x | 276.807 | 377.319 | 73.4% |
| 8,192 | 8,193 | 640.078 MiB | 4.00x | 277.523 | 374.898 | 74.0% |
| 16,384 | 16,385 | 1,280.078 MiB | 2.00x | 281.550 | 382.489 | 73.6% |
| 24,576 | 24,577 | 1,920.078 MiB | 1.33x | 278.838 | 378.809 | 73.6% |
| 32,736 | 32,737 | 2,557.578 MiB | 1.001x | 274.392 | 373.101 | 73.5% |

The 40-copy payload is 2.5 GiB and the logical source-read plus destination-write
volume is 5 GiB at every row. That volume is a tensor-size derivation, not an
observed DRAM-byte counter.

## Temporary visible-extent probe (Option A)

A local-only graph probe used `extent = prefix + 1`, prefix views of K/V,
capacity-strided writes into the existing packed scratch, and matching
extent-sized score, mask, softmax, and AV tensors. It exercised a 32-row prefill
and one-row decode. The probe code was reverted after measurement; no runtime
change remains, and the optimizer and production defaults are unchanged. The
temporary override directly built
this graph outside optimizer selection while the optimizer remained `SHADOW`;
the `canonical_selected` diagnostics in those logs describe optimizer dispatch,
not whether the test-only shape override ran. Its raw profiles and comparison
vectors are retained in `raw/visible-extent/`.

| Prefix / extent | Canonical copy / total (ms) | Probe copy / total (ms) | Probe Cutlass kernel sum (ms) |
|---:|---:|---:|---:|
| 32 / 33 | 277.350 / 375.769 | 0.146 / 27.155 | 0.239 |
| 8,192 / 8,193 | 277.523 / 374.898 | 19.500 / 78.824 | 32.855 |
| 32,736 / 32,737 | 274.392 / 373.101 | 279.934 / 441.456 | 129.374 |

These are one-capture kernel sums, not repeatable speedup claims. The prefix
view's capacity stride also changed CUDA matrix dispatch: the probe introduced
52+28 Cutlass launches at the two larger extents. At near-full context that
added a 129.374 ms Cutlass sum while saving no V-copy time. A contiguous
prefix scratch may avoid that dispatch change, but was not qualified.

More importantly, the probe failed parity at all three points. Prompt and
appended sampled F16 KV rows differed; final hidden and logits were not
bitwise-equal, despite the repeated-token fixture returning the same one token. This
is not generation parity. The measured differences are in
`raw/visible-extent-numerics.csv`:

| Prefix | Token same | Hidden RMS diff / max abs | Logits RMS diff / max abs | Historical KV changed values | Progressive KV changed values | Appended KV changed values |
|---:|:---:|---:|---:|---:|---:|---:|
| 32 | yes | 0.361 / 2.304 | 0.058 / 0.287 | 10,143 / 24,576 | n/a | 10,129 / 12,288 |
| 8,192 | yes | 0.643 / 20.486 | 0.164 / 0.420 | 10,168 / 24,576 | 61,202 / 86,016 unique sampled values | 10,108 / 12,288 |
| 32,736 | yes | 0.896 / 34.326 | 0.218 / 0.673 | 10,148 / 24,576 | 91,681 / 122,880 unique sampled values | 10,172 / 12,288 |

Progressive samples are deduplicated by `(layer, position, K/V)` before diff
counts; the same start row was sampled at multiple checkpoints.

The repeated-token fixture can hide token-ID changes; the changed hidden/logit/KV
values are sufficient to reject this graph as equivalent. Reduced softmax and
AV extents change reduction/dispatch behavior. Do not enable this candidate.

## Options and recommendation

- **A — visible-extent packing plus reduced-shape attention:** empirically
  eliminates most copies at short context, but this implementation changes
  hidden/logit/KV values and becomes slower near capacity due matrix dispatch.
  **Rejected in its tested form.** Exact per-position graph shapes would also
  need rebuilds or a bucketed graph cache for autoregressive decode.
- **B — persistent dual-layout V:** preserves the canonical K/V representation
  and lets AV consume the existing packed format, so it is the conservative
  parity path. At 32K it adds 64 MiB per layer: 1,664 MiB on the 26-layer GPU
  and 896 MiB on the 14-layer GPU (2,560 MiB total). The lowest free-memory
  observations in this audit were about 2,539/1,273 MiB, leaving only
  875/377 MiB; the earlier conservative estimate of 2,568/1,222 MiB leaves
  904/326 MiB. Either leaves fragile 2080 SUPER headroom. At production
  capacity 1,032, the
  extra allocation is about 80.6 MiB total (52.4/28.2 MiB per device), which
  is materially more plausible, though its scatter/update path and value still
  need measurement.
- **C — native-layout AV consumer:** no duplicate persistent V and no capacity
  conversion, so it is the only route that can address 32K without the Option B
  footprint. It requires a Qwen-aware CUDA operator for the native
  `[position, KV-head, dimension]` source. Preserve the canonical reduction
  order if parity is required; a generic strided matmul may select a different
  kernel or silently repack.

If a next redesign study is authorized, **C is the primary research direction**
for 32K; **B is the fallback only at the unchanged 1,032 production capacity**
(or at 32K only after explicit memory-headroom qualification). This is a
research ranking, not approval to implement either. Under the current
small/guarded-change constraint, the decision is **no production change**.

## Reproduction and retained files

The capacity/context captures use the existing opt-in `VBUF_QWEN3_CUDA_PROFILE_CONTEXT`
range in the capacity gate. Representative command shape (one profile per
capacity/context):

```bash
MODEL=/home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.vbuf
SEMANTIC=/home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.semantic.vbuf
TOKENS=research/results/vbuf-ml-integration/qwen3-multigpu-26-14-capacity-qualification/raw/natural-repeat-token-ids.csv
GATE=/home/eugen/.cache/vbuf-agent-qualification/qwen3-cuda-q4k/build-residency/vbuf_qwen3_multigpu_capacity_gate
python3 scripts/range_server.py --file "$MODEL" --port 18783 &
CUDA_VISIBLE_DEVICES=0,1 VBUF_QWEN3_CUDA_PROFILE_CONTEXT=32 \
  nsys profile --trace=cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=/tmp/qwen-profile \
  "$GATE" "$SEMANTIC" http://127.0.0.1:18783 "$TOKENS" 32768 32 1 32 run 0
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter \
  -i /tmp/qwen-profile.qdstrm -o /tmp/qwen-profile.qdrep -f
nsys stats --report cuda_gpu_kern_sum --force-export=true /tmp/qwen-profile.qdrep
```

For the capacity sweep change only the capacity argument to 1,024, 2,048,
4,096, 8,192, 16,384, or 32,768. For context scaling change only prefix and
`VBUF_QWEN3_CUDA_PROFILE_CONTEXT` together to 32, 128, 512, 2,048, 8,192,
16,384, 24,576, or 32,736. The temporary Option A probe and prefill-row
capture hooks were reverted; their converted traces and output dumps remain as
evidence but are not a checked-in runnable mode.

`raw/capacity-sweep.csv`, `raw/context-sweep.csv`,
`raw/prefill-row-sweep.csv`, `raw/visible-extent-profile.csv`, and
`raw/visible-extent-numerics.csv` contain machine-readable summaries. The
`.qdrep` traces and original gate logs are next to those summaries or in the
corresponding subdirectories. No QDSTRM importer outputs or SQLite databases
are required for the retained results.
