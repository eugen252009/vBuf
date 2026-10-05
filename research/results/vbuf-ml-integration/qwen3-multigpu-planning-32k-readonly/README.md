# Qwen3-14B 32K multi-GPU placement planning — read-only evidence

Status: **planning only; no multi-GPU inference or production qualification**. Model: Qwen3-14B Q4_K_M, semantic vBuf SHA-256 `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`. Baseline repository HEAD observed: `d06cc495b16b096eb3e67442601e0551641e81d7`. Production-qualified capacity remains 1,032 tokens; the 32,768 KV capacity here is a harness planning setting, not a production capacity claim.

## What matters most

1. **Correctness gate:** SMALL block 5 on RTX 3060, no intermediate captures, matched saved resident input and output fixture; output matches bit-for-bit and repeated execution is stable. LARGE block 0 has an earlier 32-row/capacity-40 control that matched its fixture bit-for-bit and replayed stably. Cross-GPU results are finite/repeatable but not bitwise equal: block 0 max absolute difference `9.54e-7`, relative RMS `8.59e-8`; block 5 max absolute `0.02129`, relative RMS `0.003986`, cosine `0.9999921`. The first captured block-5 cross-GPU discrepancy appears at Q projection (relative RMS `1.10e-7`) and grows downstream. This establishes a device-dependent numerical difference, not its hardware/compiler/kernel root cause, and not parity under any unstated tolerance.
2. **Representative timings:** raw harness measurements in `raw/qwen_block_32k_nocapture_timings.log`; these are class representatives (block 0 LARGE and block 5 SMALL), not an individual measurement of every layer. 30 repetitions, 3 warmups. At 1/16/32/64/128 rows, medians (ms) were RTX 3060 LARGE `12.187/14.797/18.207/24.373/36.423`, SMALL `12.001/14.655/17.907/24.123/36.174`; RTX 2080S LARGE `4.558/6.183/8.880/13.669/24.209`, SMALL `3.923/6.265/8.798/13.775/24.496`. The 1-row 2080S p95 is notably wider than its median; see raw data.
3. **Boundary transfer:** `raw/qwen_boundary_transfer.csv`, pinned staging transfer samples, not concurrent full-layer transfer. RTX 3060→2080S 1 row `20.84 us`, 32 rows `255.04 us`; reverse `20.77 us` and `288.64 us`.
4. **Memory feasibility dominates split choice:** planning assumptions and arithmetic below are estimates based on measured harness allocations plus reserved headroom; this is not a real dual-GPU allocation test. The 2080S is the limiting device.

## Correctness caveat: captures perturb the graph

An earlier block-5 mismatch occurred with intermediate graph outputs retained (`max abs 0.0212002`, relative RMS `0.00167619`). The same no-capture graph and fixtures match exactly. Capturing all intermediates adds graph roots and changes allocation/liveness, so those captured tensors are diagnostic, not a qualified reference for normal graph execution. The four device/block capture sets in `captures/` are retained strictly to locate cross-device divergence; they must not be used as proof that captured and no-capture execution are equivalent. K/V attention-input tensors make these captures comparatively large.

## Placement arithmetic (candidate planning only)

For direction A, embedding and early layers are on the RTX 3060; one hidden-state boundary crosses to the RTX 2080S, which owns late layers, final norm, and output head. LARGE/SMALL membership was counted by exact layer index from the observed class pattern. The 24/16 through 28/12 counts are:

| Cut | 3060 early L/S | 2080S late L/S | Projected decode ms | Projected 32-row prefill ms |
|---:|---:|---:|---:|---:|
| 24/16 | 11/13 | 9/7 | 360.185 | 577.383 |
| 25/15 | 11/14 | 9/6 | 368.263 | 586.492 |
| 26/14 | 12/14 | 8/6 | 375.892 | 595.819 |
| 27/13 | 12/15 | 8/5 | 383.970 | 604.928 |
| 28/12 | 12/16 | 8/4 | 392.048 | 614.037 |

Projections sum representative-block medians and available embedding, head, and boundary measurements. They omit final-norm compute; they are not measured end-to-end latency. The 24/16 cut is fastest in this simple sum but does not pass the conservative 2080S memory stress budget. 25/15 is a speed-oriented **primary planning candidate**; 26/14 is a **headroom-oriented fallback candidate**. Neither is a selected/qualified runtime placement.

Reverse direction B (early layers and embedding on the 2080S) is rejected in this estimate: even cut 24/16 exceeds its modeled budget by roughly 1,954 MiB.

Memory basis: hardware reported 12,288 MiB RTX 3060 and 8,192 MiB RTX 2080S. Budgets after existing device/background allowances and 512 MiB reserve: 11,405 MiB and 6,885 MiB. Measured maximum one-block graph buffer at 32 rows was about 863 MiB. Subtracting its 128 MiB KV allocation and including observed post-run workspace yielded a planning temporary allowance about 841 MiB. An extra 512 MiB overhead stress leaves the 2080S with modeled headroom ~136 MiB for 25/15 and ~465 MiB for 26/14. This narrow modeled margin, plus unmeasured full placement interactions, is why these remain candidates only. The 2080S's free VRAM was observed constant at 7,664 MiB over one 300-second desktop sample; that does not guarantee future availability.

Rows 64 and 128 have exploratory timing data but are **not** qualified placement chunk sizes: their graph memory increases substantially. No 32K inference claim is made.

## Archive contents

- `raw/qwen3_single_block_bench.cpp`: temporary diagnostic/timing harness source; not production code.
- `raw/qwen_block_32k_nocapture_timings.log`, `raw/qwen_head_timings.log`, `raw/qwen_embedding_timings.log`: raw timing output.
- `raw/qwen_boundary_transfer.csv`: boundary-transfer samples.
- `raw/qwen-capture-g{0,1}-b{0,5}.log`: capture-run metadata/output.
- `captures/qwen-capture-g{0,1}-b{0,5}/`: raw binary intermediate tensors, retained for the cross-device diagnostic only.

Temporary source, logs, CSV, and capture binaries were copied from `/tmp` into this directory so they survive temporary-file cleanup. Model payloads and resident-layer fixture files were not duplicated; their authoritative paths and model identity are given in the qualification record and earlier evidence.

## Repository/change boundary

No production behavior, model format, capacity, or multi-GPU ownership/runtime was changed for this work. The archive is a new research artifact. Existing worktree changes and stashes were preserved; no reset, clean, stash, staging, commit, or push was performed. The reported HEAD remains the baseline above.
