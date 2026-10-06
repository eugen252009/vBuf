# Qwen CUDA 26/14 staged multi-GPU implementation and qualification

Status: explicit experimental implementation present; **small-context qualification passed for the recorded test set only**. This does not alter the single-GPU default or the production-qualified 1,032-token capacity. The initial design work began from repository baseline `c67bd79ea66f2b88d24ba1942c55d861e2c84215`; this is historical, not the current HEAD. The subsequent sequential 2K–32K experimental capacity results and their qualification limits are documented in [`qwen3-multigpu-26-14-capacity-qualification/README.md`](qwen3-multigpu-26-14-capacity-qualification/README.md). No reset, clean, or stash operation was used.

Detailed small-context evidence and raw output: [`qwen3-multigpu-26-14-small-context-qualification.md`](qwen3-multigpu-26-14-small-context-qualification.md).

## Runtime ownership boundaries

- `QwenCudaPlacement` validates a contiguous 26/14 split. The explicit `VbufModelRuntime::prepare_qwen3_cuda_multigpu_26_14()` entry point opts into device 0/1; the default remains single-device.
- Runtime-owned per-device GGML backends/model allocations own each model tensor exactly once. The embedding and blocks 0–25 reside on device 0; blocks 26–39 and final norm/head reside on device 1. Residency keys include actual device IDs.
- One session owns per-device layer KV, scratch/graph contexts, global logical progress, and the CUDA-pinned host staging buffer. KV lookup rejects non-owners. Reset synchronizes each owned backend and clears logical progress.
- `Qwen3MultiDeviceGenerationExecutor` constructs per-layer decode/prefill graphs and preserves ordered execution. The activation crosses device boundary once per layer pass/chunk via D2H → synchronize → H2D → synchronize. Logical token progress commits only after final-stage output completes.
- The existing `Qwen3GenerationExecutor` and single-device production capacity remain the default path. Final-hidden download in the single-device reference is opt-in for the qualification harness only.

## Test gates executed

The real two-GPU harness compared prefixes 1, 2, 4, 8, 16, and 32 against the canonical single-GPU executor at context capacity 64. It verified tensor/KV ownership, pinned handoff byte identity, sampled F16 KV finiteness and append preservation, greedy parity, recorded hidden/logit tolerance, replay, sequential A/B/A isolation, injected failure atomicity, and fresh-session recovery. All recorded gates passed; details and numerical metrics are in the qualification report.

An early attempt failed with illegal CUDA memory access. Diagnosis showed auxiliary decode graph scratch reused the buffer holding device-local main graph controls and hidden state. Auxiliary decode layers now bind to layer scratch separate from the main decode/control allocation (decode and prefill auxiliary contexts are reused serially). The later clean complete qualification run is the retained pass.

## Explicit limits

- This design snapshot's 64-token test did not exercise 32K. The later capacity report linked above records a 32K experimental inference run; neither report establishes a production-capacity increase, actual multi-GPU production request, concurrency, natural-language quality, or extended stress.
- The capacity-64 run and 26/14 placement do not resolve the narrow modeled RTX 2080 SUPER margin at 32K. Existing 25/15 and 26/14 planning projections remain planning-only.
- Cross-GPU arithmetic is not bitwise equal. For the recorded synthetic token prefixes, all greedy tokens matched and hidden/logit metrics passed the test harness's explicit bound (relative RMS ≤ 0.02, cosine ≥ 0.9998). These limits apply only to this recorded test set.
- Failure tests establish logical progress atomicity, not physical rollback of K/V writes. Failed writes are beyond logical length and ignored by reset/causal masking.

## Build / rerun

Configure the GGML CUDA build for architectures 75 and 86 and build target `vbuf_qwen3_multigpu_qualification`. The harness takes semantic bootstrap, range-source URL, comma-separated token IDs (at least 32), and capacity (33–1032 for this default qualification path). The archived run used `CUDA_VISIBLE_DEVICES=0,1`, tokens `0..31`, capacity 64, semantic artifact SHA/payload identity `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`, and the local source URL documented in the raw run command context.
