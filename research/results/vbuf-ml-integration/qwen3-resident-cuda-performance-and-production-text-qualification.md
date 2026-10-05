# Qwen3 Resident CUDA Performance and Production Text Qualification

**Current disposition:** A qualification-only per-layer prefill graph now reuses one bounded scratch allocation across all 40 layers. The old all-layers-in-one-context score/probability lifetime was demonstrated to contribute 409,728 B/capacity-token; the new same-basis no-audit allocation fit is 189,060 B/token total, comprising 163,840 B/token persistent F16 KV and 25,220 B/token non-KV workspace (14,980 main-graph slope plus 10,240 reusable per-layer scratch slope). The prefill-specific score/probability scratch is 10,240 B/token rather than 409,600 B/token. The memory-scaling gate **PASSES for this qualification-only runtime**: the 512-token run plus eight appends reached KV length 520, and the 1024-token run plus eight appends reached 1032 with 3.20 GB free VRAM in the audited run. Same-capacity decode-only versus prefill comparisons and diagnostic-overhead accounting are documented under Phase D below.

This does **not** qualify `VbufGenerationSession`: Qwen3 remains explicitly rejected by its direct graph builder, and session integration has not yet been implemented. No HTTP, tools, Pi, dogfood, or llama.cpp benchmark work was performed. The pinned artifact SHA-256 remains `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.

The external llama.cpp `1e-5` gate remains **FAIL**. Previously classified numerical findings are retained without reopening: chunked versus full CUDA is **EXPECTED CUDA KERNEL-DISPATCH-DEPENDENT PREFILL DIVERGENCE**; persistent CUDA versus full CUDA is **EXPECTED SHAPE-DEPENDENT CUDA NUMERICAL DIVERGENCE**; prefix-25 CPU/CUDA is **UNRESOLVED CONTEXT-SENSITIVE ACCUMULATED DIVERGENCE** (about 61.8% final-logit relative RMS).

## PHASE A — PERFORMANCE

### Environment

- GPU: NVIDIA GeForce RTX 3060, 12,288 MiB; CUDA device 0; SM 8.6. A separate RTX 2080 SUPER was present but not used.
- Driver/runtime: NVIDIA driver 550.163.01; CUDA runtime/compiler 12.4.131.
- GGML: pinned commit `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`; Release build, CUDA enabled, `GGML_CUDA_COMPRESSION_MODE=size`, MMQ/CUBLAS overrides off.
- CPU: AMD Ryzen 7 5800X, 8 cores / 16 threads. Host RAM: 62 GiB.
- Artifact: Qwen3-14B Q4_K_M vBuf, SHA-256 `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- Backend: resident GGML CUDA backend on device 0; weights, embedding, workspaces, and incremental K/V use the qualification graph allocation. Source was the local range server over HTTP.

### Initialization and residency

Artifact metadata/bootstrap setup was 5.6–5.9 ms. For the 32-position full-batch resident graph, backend initialization was 6.3 ms, graph-buffer allocation 0.65 ms, and combined weight materialization plus upload enqueue 22.99 s; the subsequent backend upload barrier measured 0.54 ms. The staging interval includes fetching/materializing the 8.996 GB payload and enqueueing tensor copies, so the 0.54 ms barrier is not an independent measure of total DMA duration. Total measured resident setup was 23.06 s, excluding process startup and the separately reported metadata setup.

The model upload included the 437,575,680-byte embedding tensor within 8,995,793,920 model-weight bytes. The CUDA API audit recorded 8,995,798,272 H2D bytes during the earlier full-graph initialization (443 weights plus small graph inputs). The previous one-graph-per-position implementation is historical: capacity 64 allocated 10,947,079,168 bytes, and full-batch prefill capacity 128 allocated 12,321,615,872 bytes with only 146,931,712 bytes free. The redesigned single-token decode graph plus 32-position prefill graph allocated 9,838,812,928 bytes at capacity 72, 9,876,471,808 bytes at capacity 136, and 10,102,425,088 bytes at capacity 520. At capacity 520, 957,939,712 bytes were free after initialization in the timed run; the audit run measured 929,300,480 bytes free. Model weights plus embedding remain 8,995,793,920 bytes; F16 KV at capacity 520 is 85,196,800 bytes. Geometry resolves the earlier arithmetic confusion: 40 layers × 8 KV heads × 128 elements × 2 bytes × K/V = 163,840 KV bytes per token (2,048 bytes each for K and V per layer-token).

### Bounded chunked prefill

The redesigned path builds one reusable graph for 32 query positions, fills absolute position IDs/cache rows and a causal mask per chunk, and appends K/V into the persistent cache in place. Each chunk is completion-synchronized before its timing is reported. Length fixtures are deterministic repeating token IDs rather than natural-language prompts; these results qualify graph shape, memory, and execution, not user-facing text quality. Warmup run 1 is excluded from the averages below; timed rows are runs 2–3 in the same process. Initialization/upload is separate.

| Prompt / final context | Chunks | Mean timed prefill | Prompt throughput | Decode mean / median | Device graph allocation | Free after init |
|---|---:|---:|---:|---:|---:|---:|
| 64 / 72 | 2 × 32 | 111.684 ms | 573 tok/s | 29.926 / 29.869 ms/token (33.42 / 33.48 tok/s) | 9.839 GB | 2.661 GB |
| 128 / 136 | 4 × 32 | 347.554 ms | 368 tok/s | 48.122 / 48.282 ms/token (20.78 / 20.71 tok/s) | 9.876 GB | 1.142 GB |
| 512 / 520 | 16 × 32 | 1,547.944 ms | 331 tok/s | 50.804 / 50.792 ms/token (19.68 / 19.69 tok/s) | 10.102 GB | 0.958 GB |

#### Allocation-scaling measurement (new)

The same 32-token prefill graph was built for each prompt length, with capacity set to `prompt length + 8` for the decode appends. Model payload (weights plus embedding) was constant at 8,995,793,920 bytes: 8,558,218,240 bytes of non-embedding weights plus a 437,575,680-byte embedding. The CUDA allocation interposer observed one backing `cudaMalloc` for each process buffer, no alloc/free calls at chunk boundaries, and a process peak equal to the graph buffer.

| Prompt / capacity | Persistent KV bytes | Combined non-KV graph workspace | Total buffer / peak tracked | Free VRAM after init → completed chunks |
|---:|---:|---:|---:|---:|
| 32 / 40 | 6,553,600 | 817,635,968 | 9,819,983,488 | 1,187,971,072 → 1,160,708,096 |
| 64 / 72 | 11,796,480 | 831,222,528 | 9,838,812,928 | 1,165,557,760 → 1,138,294,784 |
| 128 / 136 | 22,282,240 | 858,395,648 | 9,876,471,808 | 1,127,219,200 → 1,099,956,224 |
| 256 / 264 | 43,253,760 | 912,741,888 | 9,951,789,568 | 1,051,459,584 → 1,024,196,608 |
| 512 / 520 | 85,196,800 | 1,021,434,368 | 10,102,425,088 | 900,464,640 → 873,201,664 |
| 1024 / 1032 | 169,082,880 | 1,238,819,328 | 10,403,696,128 | 585,367,552 → 553,910,272* |

`*` At capacity 1032, free VRAM fell by a further 2,097,152 bytes after chunk 2, then remained constant through chunk 32. Other capacities had a single ~26 MiB free-VRAM decrease at first execution, then a flat trace. The CUDA `cudaMalloc*` interposer did not attribute these first-use reserves; they are bounded backend/driver scratch behavior, not per-chunk growth. Free-VRAM values also varied with other GPU usage, so buffer/category accounting is the primary scaling evidence.

Post-chunk free-VRAM summaries below cover every `resident_cuda_prefill_memory` row present in the raw run logs for each prompt size (across repeated executions); “max” is the greatest observed free space and “min” the lowest. This captures within-run/backend reserve variation instead of treating the single final sample in the preceding table as a complete range.

| Prompt / capacity | Logged chunk samples | Min free VRAM | Median free VRAM | Max free VRAM |
|---:|---:|---:|---:|---:|
| 32 / 40 | 3 | 1,155,465,216 | 1,156,513,792 | 1,160,708,096 |
| 64 / 72 | 6 | 1,138,294,784 | 1,138,294,784 | 1,142,489,088 |
| 128 / 136 | 11 | 1,093,926,912 | 1,099,956,224 | 1,099,956,224 |
| 256 / 264 | 23 | 1,024,196,608 | 1,024,196,608 | 1,024,196,608 |
| 512 / 520 | 47 | 873,201,664 | 873,201,664 | 873,201,664 |
| 1024 / 1032 | 96 | 549,715,968 | 553,910,272 | 556,007,424 |

The operation breakdown isolates the shape-dependent arrays. Across 40 layers, prefill attention scores and probabilities each consume `204,800 B × capacity` (40 heads × 32 queries × F32 × 40 layers); the prefill mask adds `128 B × capacity`. The decode graph adds its measured `14,852 B × capacity` (scores, probabilities, packed-V scratch, and mask). The resulting non-KV slope is exactly **424,580 B/context-capacity-token**, with zero regression residual across all six capacities. Persistent KV adds 163,840 B/token, for a combined buffer slope of **588,420 B/capacity-token**. The fit is:

```text
non-KV workspace(capacity) = 800,652,768 B + 424,580 B × capacity
backend buffer(capacity)   = 9,796,446,688 B + 588,420 B × capacity

decode-only workspace(capacity) = 24,262,752 B + 14,852 B × capacity
prefill graph increment(capacity) = 776,390,016 B + 409,728 B × capacity
```

| Prompt / capacity | Decode workspace | Additional prefill graph workspace | Combined workspace |
|---:|---:|---:|---:|
| 32 / 40 | 24,856,832 | 792,779,136 | 817,635,968 |
| 64 / 72 | 25,332,096 | 805,890,432 | 831,222,528 |
| 128 / 136 | 26,282,624 | 832,113,024 | 858,395,648 |
| 256 / 264 | 28,183,680 | 884,558,208 | 912,741,888 |
| 512 / 520 | 31,985,792 | 989,448,576 | 1,021,434,368 |
| 1024 / 1032 | 39,590,016 | 1,199,229,312 | 1,238,819,328 |

The shared packed-V scratch is allocated once with the decode workspace and reused by the prefill graph; it is not duplicated in the incremental prefill column. The op breakdown plus these formulas accounts for the full retained slope; only a 564-byte constant backend-arena residual separates summed tensor allocations from the buffer size.

#### PHASE A — old prefill scratch lifetime proof

**ROOT CAUSE IDENTIFIED.** This is graph-context allocation lifetime, not per-chunk retention and not a CUDA allocator leak. The pinned GGML `ggml_backend_alloc_ctx_tensors_from_buft_impl` walks every non-view tensor in the context and allocates it sequentially in a backend buffer; it does not use graph last-use information to recycle an earlier node's storage. `build_batch_graph` creates all 40 transformer blocks in that one context before calling `ggml_backend_alloc_ctx_tensors`. Consequently every layer's standalone score and softmax output owns a distinct buffer range through graph construction/execution, even though each layer's output is consumed before the next layer's attention begins.

| Tensor | Shape / dtype | Copies simultaneously allocated | Bytes per layer/token | At capacity 32 | 128 | 512 | 1024 |
|---|---|---:|---:|---:|---:|---:|---:|
| Raw attention scores (`MUL_MAT`) | `[capacity, 32, 40]` F32 | 40 | 5,120 | 163,840 | 655,360 | 2,621,440 | 5,242,880 |
| Softmax probabilities | `[capacity, 32, 40]` F32 | 40 | 5,120 | 163,840 | 655,360 | 2,621,440 | 5,242,880 |
| Packed V | `[capacity, 128, 8]` F16 | 1 shared | 2,048 | 65,536 | 262,144 | 1,048,576 | 2,097,152 |
| Causal mask | `[capacity, 32]` F32 | 1 shared | 128 | 4,096 | 16,384 | 65,536 | 131,072 |

The score and probability values in the table are per-layer; the 40-layer graph multiplies each score/probability row by 40. No separate masked-score array exists: the causal mask is an input to softmax. The measured run capacities were prompt+8 (40, 72, 136, 264, 520, 1032); the exact tensor shape formula also gives the requested capacity values above. At capacity 520, each layer's score and probability tensor is 2,662,400 bytes; all 40 copies of each account for 106,496,000 bytes apiece. Packed V and the mask are each shared once, not 40 times.

**Slope accounting:** `40 × 5,120 × 2 + 128 = 409,728 B/capacity-token` of prefill-specific storage. The measured slope is 409,728 B/token, the explained slope is 409,728 B/token, and the unexplained slope is **0 B/token**. Decode's separate 14,852 B/token remains accounted for as previously documented. Other attention tensors (Q/K/V current-chunk values, context, projections) are fixed by the 32-query extent, not key capacity.

**Fixed prefill delta:** measured at capacity 520 is 989,448,576 bytes; subtracting the proven score/probability/mask extent (213,058,560 bytes) leaves 776,390,016 bytes. The per-op differential between the same-capacity prefill and decode-only buffers accounts for that fixed part:

| Op allocation class | Fixed prefill increment (bytes) |
|---|---:|
| `MUL_MAT` outputs other than context-shaped scores (including Q/K/V, output and FFN projections) | 313,049,088 |
| `MUL` | 173,670,400 |
| `UNARY` | 89,128,960 |
| `RMS_NORM` | 84,541,440 |
| `ADD` | 52,428,800 |
| `ROPE` | 31,457,280 |
| `CONT` | 26,214,400 |
| `CPY` | 5,242,880 |
| `GET_ROWS` | 655,360 |
| `NONE` residual after the causal mask | 1,280 |
| `ARGMAX` | 128 |
| **Total fixed increment** | **776,390,016** |

These are 32-query, 40-layer batched graph activations and associated outputs, including FFN intermediates; the table is a backend allocation-op ledger, not an assertion that every byte is live at the same graph execution instant. They remain approximately constant as capacity changes. The operation ledger and CUDA allocation audit show no additional prefill-specific quantization scratch or separately growing backend work allocation: the graph tensors reside in the tracked backing allocation, with no `cudaMalloc`/`cudaFree` at chunk boundaries. The large fixed part is therefore bounded chunk/model graph storage; the unbounded-with-context defect is specifically the 40-fold score/probability tensor lifetime.

A same-capacity-520 decode-only control allocated 9,112,976,512 bytes total (31,985,792 bytes non-KV workspace). Adding the 32-query prefill graph raised the buffer by **989,448,576 bytes**: 212,992,000 bytes of score outputs, 212,992,000 bytes of probability outputs, a 66,560-byte mask, and ~776,390,016 bytes of fixed-per-chunk graph activations/other prefill storage. Thus score/probability storage alone grows by **409,600 B per context token**, 2.5× the KV slope. This is explained and bounded per configured capacity, but it is not constant prefill workspace or a small O(context) addition relative to KV.

At 512 tokens, all 16 completed chunks reported identical graph/buffer addresses and the same 10,102,425,088-byte backing allocation; each later chunk had zero interposer alloc/free calls and the same post-chunk free-VRAM reading. At 1024 tokens, the graph/buffer addresses stayed fixed through 32 chunks. The fixed-chunk Q/K/V, hidden, and FFN graph tensors are allocated once and reused; no allocation proportional to the number of completed chunks was observed. Context-shaped attention score/probability tensors are allocated once at graph construction with key-capacity extents, explaining the linear workspace slope.

A separate non-performance 512-token run plus eight appends completed at KV length 520. Three same-shape executions produced bit-identical token trajectories and hidden/norm/logits at prefixes 512–520. The trajectory was `66,198,220,197,280,197,197,280`. Sampled historical K/V immutability checks passed at prefixes 32, 64, 128, 512, 513, and 520 for layers 0, 20, 29, and 39; K/V operator-to-cache checks passed at appended positions 512 and 519. Normal inference windows showed no historical KV, model-weight, or embedding re-upload. Test-only replay resets zero the allocated KV buffer, and cache-audit readbacks are excluded from normal transfer claims.

Timed 512-token runs 2–3 averaged 1.463 s prefill (~350 tok/s) and 20.703 tok/s decode after context 512, consistent with the previous ~19.7 tok/s reference. This was a performance sanity check, not a benchmark suite.

Decode summaries use the two timed generations after one warmup; each prompt appended eight tokens. The first 32-position prefill chunk includes a warmer first-run cost (about 166 ms at prompt 64, 415 ms total across four chunks at 128, and 1,602 ms across 16 chunks at 512); warm prefill measurements are the tabled runs 2–3. These are local range-server artifact runs on the same GPU; no CPU throughput comparison is implied. The former 32- and 128-position full-batch rows remain historical controls, not the path used here. Full-batch 512 prefill was not attempted.

### Persistent decode and repeatability

The new decode graph is constructed once and reused at every logical position. Performance runs at final contexts 72, 136, and 520 produced the rates in the table above. The 512-prompt non-performance audit run repeated full exported hidden/norm/logit outputs bit-identically across three graph/cache resets, and its eight generated tokens were identical. Performance-mode runs intentionally report `TOKENS_IDENTICAL_ONLY`; token agreement is not numerical parity. The 512 audit checked sampled K/V operator-to-cache writes at appended positions 512 and 519, and historical cache bytes at prefixes 32, 64, 128, 512, 513, and 520 for layers 0, 20, 29, and 39; all sampled prior K/V rows remained unchanged. For the 128-token prompt with four 32-token chunks, three repeated runs produced bit-identical exported outputs and identical eight-token appends; absolute position, cache-row, and mask checks passed for each chunk.

A same-input shape control at 32 tokens compared full-batch prefill with one 32-token chunk: final hidden, final norm, and final logits were bit-identical; top-1 was token 220. The fixed prompt token-ID SHA-256 was `b6fdefec6440be37af2896f68a1edc2c07e45a524327b8b070704a0fecad3263`. Splitting it into two 16-token chunks changed prefix-32 final hidden/norm/logit relative RMS by 2.016% / 3.081% / 7.353%; max absolute differences were 16.8507 / 1.74841 / 0.428336. Prefix-32 top-1 remained token 220, with all top-10 candidates shared (two positions swapped within top-5). After eight appends, prefix-40 relative RMS was 1.021% / 1.197% / 1.898%, top-1 was token 549 for both, and top-5 order matched. Both paths repeated the same eight-token sequence, but token agreement is not numerical parity.

**Numerical classification: EXPECTED CUDA KERNEL-DISPATCH-DEPENDENT PREFILL DIVERGENCE.** Three 16+16 replay runs had byte-identical captures for both layer-0 chunks. Their structural checks passed for absolute RoPE positions, absolute KV rows, causal visibility (`key <= query`), append writes, and sampled prior-row immutability. At layer 0, full-32 versus the second chunk's positions 16–31 had identical inputs through Q/K projection, Q/K normalization/RoPE, and cached K/V; the attention-score matmul was the first differing operator. Its score output differed by max absolute 0.375 and relative RMS 0.000793730.

A direct same-input CUDA score probe reconstructed the persistent K cache and replayed the captured Q operands at full-32 and chunk-16 shapes. It reproduced each captured score tensor bit-for-bit, and the same-input shape control independently reproduced the 0.375 max / 0.000793730 relative-RMS difference. Both GGML operations took `CUBLAS_FINAL` on SM 8.6 with F16 K, F32 Q converted to F16, cuBLAS `CUBLAS_COMPUTE_16F` with an F16 result temporary promoted to logical F32 scores, dimensions M=40, K=128, batch=40, and N=32 versus N=16. Nsight Systems traced cuBLAS's different internal kernels: full-32 selected `ampere_h16816gemm_128x64_ldg8_stages_32x6_tn` (grid 1×1×40); chunk-16 selected `gemmSN_TN_kernel_half<256,8,2,4,8>` (grid 3×2×40). This is shape-dependent cuBLAS kernel selection, not an unexplained cache/position defect. The larger downstream hidden/norm/logit differences remain real numerical divergence; matching greedy tokens does not establish parity.

The 16-position capture comparison's earlier V-attention script had a layout/shape error; it is not used as evidence. Cache-write and historical-row checks are sampled rather than exhaustive.

Additional structural audit: 16-position chunks over a 32-token prompt and 32-position chunks over 64-, 128-, and 512-token prompts (then eight decode appends, reaching context 520) were checked for absolute RoPE positions, absolute cache-row mapping, and causal visibility (`key <= query`). Sampled appended K/V rows were nonzero; sampled historical rows at the previously reported prefixes remained unchanged. These checks establish the sampled structure/cache invariants, not exhaustive byte-integrity for every cache row. At the first layer, input through RoPE and cached K/V for full-32 versus two 16-token chunks matched bit-identically; attention scores first differed (relative RMS ~0.000794), followed by later attention/FFN/block tensors. This audit therefore does not identify a semantic mapping defect or attribute the difference to a particular CUDA kernel. The comparison script used for V-attention had a layout/shape error, so its V-attention comparison is invalid. Repeated-run deterministic comparison of the 16-chunk path and actual dispatch-kernel attribution remain unresolved.

### Transfers, allocations, and synchronization

Historical transfer audit (the earlier context-32 per-position-graph run) recorded the following timed sequence windows; the new 64/128/512 chunked performance runs did not rerun CUDA API transfer interception:

- H2D: 32 calls / 128 bytes = **4 bytes per appended token**, the current token ID.
- D2H: 32 calls / 128 bytes = **4 bytes per selected output token** on average, the scalar argmax ID. This includes the first selected token read after prefill and 31 subsequent predictions; the last appended token is processed to leave KV fully populated but its next-token argmax is not part of this 32-token trajectory.
- Weight H2D/token: **0**. Embedding H2D/token: **0**. Historical-KV H2D/token: **0**. Historical-KV D2H/token: **0**.
- The 10 MiB cache clear and 32 input-slot reset IDs occur outside the timed decode window (additional reset H2D is reported in raw audit windows); they are not historical-KV transfers.
- Explicit stream synchronizations: 34 per run across 32 generated positions, including the prompt completion/first-token result boundary; effectively one graph/result synchronization per generated position plus prefill-boundary synchronization. No device-wide synchronization was observed in the measured window.
- CUDA runtime allocation audit on that historical path: one graph-buffer allocation and one free for the process; **zero `cudaMalloc`/`cudaFree` per timed token**. In the redesigned path the single decode graph and fixed-size prefill graph are also allocated once per process; the capacity-only measurements above still expose context-proportional attention workspace.

Raw logs and counters are under `~/.cache/vbuf-agent-qualification/qwen3-cuda-q4k/cuda-transfer-audit/`, including `chunked-prefill64-chunk32-perf/`, `chunked-prefill128-chunk32-perf/`, `chunked-prefill512-chunk32-perf/`, `chunked-prefill512-chunk32-audit/`, `chunked-prefill32-canonical-control/`, `chunked-prefill16x2-canonical32/`, and `reusable-workspace-final-capacities/`, plus the historical `resident-prefill32/`, `resident-prefill128/`, and `resident-perf-context32/` controls.

### CPU comparison and performance verdict

No directly comparable warmed Qwen3 CPU serial/parallel decode tok/s measurement was collected in this qualification phase. Existing nearby timing evidence is a different model or different diagnostic scope and is not presented as a matched ceiling. CPU serial and bounded-parallel comparison: **NOT TESTED** here.

**Historical memory-scaling verdict for the original all-layers graph: GATE FAILED.** The per-layer scratch reuse redesign and its passing qualification-only gate are recorded in Phase D below. This preceding section preserves the original failed measurements and root-cause evidence; it is not the current runtime allocation result. The original reusable decode graph was not strictly fixed-workspace: capacity-only allocation measurements separated persistent KV from an additional workspace slope. Model weights plus embedding are 8,995,793,920 bytes; persistent F16 KV is exactly 163,840 bytes/context-token (40 layers × 8 KV heads × 128 elements × 2 bytes × K/V). For capacities 64, 128, 512, 1024, and 4096, capacity-only total allocations were 9,031,492,864; 9,042,929,152; 9,111,546,880; 9,203,037,184; and 9,751,979,008 bytes. Subtracting model/embedding and KV leaves 25,213,184; 26,163,712; 31,866,880; 39,471,104; and 85,096,448 bytes of non-KV graph storage. The operation-level graph audit resolves this as 24,262,188 bytes fixed graph storage plus exactly 14,852 bytes/context-token; the backend allocation has an additional constant 468-byte alignment/arena difference in these samples. This decode-only slope is observed and attributed; KV remains dominant in the decode-only path, and the 14,852 B/context-token addition is accepted here as bounded small O(context) overhead. The chunked path is now separately measured: at capacity 520 its non-weight/non-KV workspace is 1,021,434,368 bytes, including 989,448,576 bytes introduced by the 32-query prefill graph over a same-capacity decode-only control. Prefill score and probability outputs each grow by 204,800 B/context token; the prefill mask adds 128 B/token. Thus the prefill-specific score/probability/mask slope is 409,728 B/token, measured rather than inferred. Capacity-only allocation at 4096 does not qualify execution at 4096. The operation-level allocation audit localizes the additional 14,852 B/context-token slope in the current reusable decode graph: raw attention-score `MUL_MAT` outputs contribute 6,400 B/token (40 layers × 40 heads × 4-byte F32 scores), `SOFT_MAX` probability outputs contribute another 6,400 B/token, the shared packed-V F16 scratch contributes 2,048 B/token (8 KV heads × 128 values × 2 bytes), and the F32 causal mask contributes 4 B/token. The 4096-slot audit recorded 26,214,400 bytes each for score outputs and softmax outputs, plus 8,388,608 bytes for the packed-V scratch; the mask is included in `NONE` op allocation. This accounts for the measured slope; no unidentified per-context graph allocation remains in that model. The chunk executor does not retain one copy of hidden/QKV/FFN activations per historical chunk; its graph and backing allocation are reused. Nevertheless, prefill workspace is not constant with prompt length: score/probability arrays contribute 409,600 B/context token (2.5× the KV slope), so the non-KV retained slope is larger than KV rather than small metadata. The 512- and 1024-token runs completed safely, but the requested memory-scaling gate **FAILED** its bounded-prefill-workspace and KV-dominated-retention criteria. This is not an unexplained leak or per-chunk accumulation. Do not redesign the already-understood decode path to zero its 14,852 B/token slope; the prefill score/probability extent is the blocker.

## PHASE D — bounded per-layer prefill scratch requalification

The per-layer path builds 40 separate 32-query block graphs, while retaining the decode graph and persistent K/V allocation. Each block graph binds its non-view tensors into the same largest-layer CUDA scratch buffer; hidden-state ping/pong and K/V remain device-resident. The final layer's committed output feeds norm/LM-head work. No host staging or weight re-upload is used between layers or chunks. This remains qualification-tool-only code.

The old graph's context-shaped score and probability arrays were each allocated once per layer because GGML's context allocator does not recycle storage at graph-node last use. Each was `capacity × 32 × 40 × F32 = 5,120 B/capacity-token/layer`; 40 copies of each plus one shared 128-B/token mask yielded the measured `40 × 5,120 × 2 + 128 = 409,728 B/token` prefill slope. In the new path, the maximum layer score/probability pair is reused: `2 × 5,120 = 10,240 B/token` in the shared scratch buffer, a 40× reduction of the score/probability portion. The audited decode graph retains its known bounded small O(context) overhead.

Capacity-only measurements without KV-audit capture give the following same-basis allocation fit. The main decode graph allocation slope is 178,820 B/token, including 163,840 B/token KV and 14,980 B/token graph workspace. The shared per-layer prefill scratch grows by 10,240 B/token. Combined allocation slope is therefore 189,060 B/token, with a non-KV slope of 25,220 B/token. The fixed per-layer scratch reservation is about 39.6 MB plus the one-layer context-shaped score/probability extent; it is allocated once and reused, not retained 40 times.

| Capacity | Main graph buffer | Reusable layer scratch | Combined buffer | Persistent KV | Non-weight/non-KV workspace |
|---:|---:|---:|---:|---:|---:|
| 40 | 9,028,521,472 | 40,042,624 | 9,068,564,096 | 6,553,600 | 66,216,576 |
| 72 | 9,034,243,712 | 40,370,304 | 9,074,614,016 | 11,796,480 | 67,023,616 |
| 136 | 9,045,688,192 | 41,025,664 | 9,086,713,856 | 22,282,240 | 68,637,696 |
| 264 | 9,068,577,152 | 42,336,384 | 9,110,913,536 | 43,253,760 | 71,865,696 |
| 520 | 9,114,355,072 | 44,957,824 | 9,159,312,896 | 85,196,800 | 78,322,176 |
| 1032 | 9,205,910,912 | 50,200,704 | 9,256,111,616 | 169,082,880 | 91,234,816 |

Values are bytes and include the constant 8.996-GB model/embedding payload. The 14,980-B/token main-graph workspace slope is consistent with the known 14,852-B/token decode workspace plus a 128-B/token arena residual; that small residual is retained explicitly rather than attributed to attention tensors. The layer scratch measurement's slope is exact across the six capacities. At capacity 520, the reused score/probability pair is 5,324,800 bytes, versus 212,992,000 bytes for the combined score-and-probability storage across all 40 layers in the old graph.

KV-integrity audit captures add a diagnostic-only 16,384 B/capacity-token to the main graph (4 captured layers × 2 cache-shaped tensors × 2,048 B/token/layer). The final 32/128/512/1024 correctness runs used this more memory-intensive audit configuration; the no-audit capacity table above is the production-shaped allocation basis. No inference is drawn from the audit-inflated slope as if it were runtime workspace.

Final-runtime checks used the pinned artifact and canonical 32-token fixture. Prompts 32, 64, 128, 256, 512, and 1024 with chunk size 32 plus eight decode appends completed; capacities were 40, 72, 136, 264, 520, and 1032. The 512-token run completed at context 520 and the 1024-token run at context 1032. Sampled historical K/V integrity and appended-row checks passed; repeated generated outputs were bit-identical. The graph and scratch addresses remained fixed across chunks, and no per-chunk CUDA allocations/frees were observed. Normal prefill chunk windows had no D2H copies; their small H2D inputs were control tensors, not weights or historical KV. Test-only cache resets/readbacks are excluded from normal transfer claims. At capacity 1032, the audited run retained 3,198,156,800 bytes free after completed chunks.

The 32-token full-batch control and the 16+16 replay were rerun against the final per-layer implementation: hidden, norm, and logits at prefixes 32 and 40 were byte-identical to their corresponding earlier qualification exports. This preserves the prior numerical classification; it does not claim CPU or llama.cpp parity. The memory gate **PASSES for the qualification-only graph**: workspace growth is bounded and small relative to the 163,840-B/token persistent KV, the 40-fold attention-scratch lifetime is removed, and the longest requested run completed with headroom. Performance samples showed no obvious severe regression (512-token prefill about 1.01 s; 1024 about 2.14 s in the recorded run), but these are sanity measurements, not a comparative benchmark.

Raw final logs are under `/tmp/vbuf-qwen3-layer-reuse/final-p*/`; capacity-only decomposition is under `/tmp/vbuf-qwen3-layer-reuse/breakdown-{40,1032}/` and `/tmp/vbuf-qwen3-layer-reuse/final-capacity-*/`.

## PHASE B — VbufGenerationSession

| Item | Status |
|---|---|
| Current canonical `VbufGenerationSession` audited | **TESTED**; Qwen3 still explicitly rejected as “direct graph builder is not implemented” |
| Qwen3 session implementation / experimental admission | **NOT IMPLEMENTED**; qualification-only gate passed, session integration is now the next required phase |
| Backend/device/KV-capacity production configuration | **NOT IMPLEMENTED** |
| Production prefill/decode, append, reset, destroy, isolation | **NOT TESTED** |
| Qualification-vs-production comparison / 16-token production generation | **NOT TESTED** |
| Shared immutable production residency / lifecycle | **NOT IMPLEMENTED** |

No production session/source/server code was changed. The performance instrumentation and capacity-64 extension are qualification-tool-only. The qualification graph has not been promoted or forked into a second production session abstraction.

## PHASE C — OpenAI-compatible HTTP

| Item | Status |
|---|---|
| `/v1/models` Qwen discovery | **NOT TESTED** |
| Non-streaming or SSE Qwen chat | **NOT TESTED** |
| tools omitted / empty / non-empty behavior with Qwen | **NOT TESTED** in this task; native tool support remains absent |
| Multi-turn, usage, repeated HTTP requests, memory stability | **NOT TESTED** |
| Real local text milestone | **NOT TESTED**; Phase C was not entered because Phase B did not pass |

## Known numerical status

- Persistent CUDA versus same-context full CUDA: **EXPECTED SHAPE-DEPENDENT CUDA NUMERICAL DIVERGENCE** (one-column MMVQ versus 32-column MMQ; cache checks passed at sampled boundaries).
- Prefix-25 CPU/CUDA: **UNRESOLVED CONTEXT-SENSITIVE ACCUMULATED DIVERGENCE**; about 61.8% final-logit relative RMS.
- Strict external llama.cpp tolerance `1e-5`: **FAIL**, unchanged.
- No CPU/CUDA or external numerical parity is claimed.

## Regressions and tests

- Qwen CPU qualification / exact persistent-KV equality: not rerun in this performance step.
- Q4_K/Q6_K regression controls: not rerun; no numerical logic was changed.
- DeepSeek: not rerun; production/runtime code was not changed.
- OpenAI protocol, HTTP, SSE: not rerun; server code was not changed.
- Pinned GGML CUDA qualification tool: **BUILT** successfully; final chunked runs completed at prompts 32, 64, 128, 256, 512, and 1024.
- Focused contracts: model architecture, Qwen execution policy, Qwen KV cache, Qwen query groups, residency, shared-residency concurrency, and device residency: **7/7 PASSED**.
- `git diff --check`: **PASS**.
- `ccc index`: **PASS**.
- Sanitizers: **NOT RUN**; changed code is qualification-only instrumentation.

## Repository and status

- Worktree was inspected before edits. This step changed only `integrations/ggml/tools/qwen3_block_qualification.cpp` and added this report, in addition to previously existing uncommitted qualification files.
- Commits: none. All investigation work remains uncommitted; no unrelated dirty state was committed.
- `stash@{0}` (`android: unverified vbuf_parallel_executor CMake hunk`): untouched.
- `stash@{1}` (`pre-merge local Step-32 reconciliation draft`): untouched.

| Capability | Status |
|---|---|
| GPU performance | **512- and 1024-token chunked prefill plus eight appends qualified in the qualification harness**; per-layer scratch memory gate passes |
| `VbufGenerationSession` Qwen3 | **NOT IMPLEMENTED / UNSUPPORTED**; promotion remains pending |
| Production persistent GPU KV | **NOT IMPLEMENTED** |
| OpenAI Qwen text | **NOT TESTED** |
| Native Qwen tools | **NOT TESTED** |
| Pi | **NOT TESTED** |
| Production Qwen3 default | **DISABLED** |

## Stop / next gate

The original single-context prefill graph failed its memory gate because 40 layers retained separate context-shaped score/probability tensors. The per-layer graph now passes the qualification-only memory gate through 1024-token prefill plus eight appends, with a measured 25,220-B/token non-KV allocation slope and preserved persistent F16 KV. The 32-versus-16+16 numerical result remains classified as expected cuBLAS kernel-dispatch-dependent divergence; deterministic replay and direct same-input attribution remain valid, and no structural position/cache/mask defect was demonstrated. The next required task is promotion into the canonical `VbufGenerationSession`, including its configuration and lifecycle tests. Until that separate phase is implemented and qualified, production Qwen3 remains disabled. Stop before HTTP/tools/Pi/dogfood/llama.cpp benchmarking. Native tool support, Pi integration, and dogfood remain unqualified.
