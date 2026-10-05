# Qwen3 resident CUDA full-forward and incremental-KV qualification

> **Follow-up:** The original eight-position fixed-input KV probe below has been extended to 32 positions and true generated-token append/reuse. See [Qwen3 CUDA prefix-25 and generated-KV qualification](qwen3-cuda-prefix25-and-generated-kv-qualification.md) for the current results, including numerical differences against full recompute. The original measurements remain historical evidence; statements below that capacity beyond eight and generated updates were untested are superseded by that follow-up, not retroactively rewritten.

## Decision and boundary

The qualification-only CUDA probe now executes the real Qwen3-14B Q4_K_M artifact through all 40 blocks, final RMSNorm, output projection, and logits. Greedy top-1 matched the canonical CPU vBuf path at prefixes 1, 2, 4, 8, 16, 25, and 32. It also ran a separate, qualification-only eight-token CUDA incremental/KV experiment and compared its outputs directly with the resident full-recompute CUDA outputs for prefixes 1, 2, 4, and 8.

These are diagnostic GPU results, **not production Qwen3 enablement** and not a pass of the strict external llama.cpp `1e-5` numerical gate. The known external numerical failure remains unchanged. `VbufGenerationSession` still does not build Qwen3's direct graph. No HTTP endpoint, tools, or Pi integration was attempted. The device allocation is one monolithic GGML graph buffer; independent production weight eviction, long-context KV capacity, memory-pressure behavior, generated-token input updates, and production session lifecycle are not qualified.

## Artifact and environment

- Artifact: Qwen3-14B Q4_K_M vBuf, SHA-256 `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- GGML: pinned qualification checkout at revision `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`.
- Device: RTX 3060, CUDA device 0, SM 8.6, 12,037 MiB reported VRAM.
- Probe and implementation: `integrations/ggml/tools/qwen3_block_qualification.cpp`; execution remains behind the internal qualification-only mode.
- Full-forward CPU reference: `~/.cache/vbuf-agent-qualification/qwen3-cuda-q4k/cuda-transfer-audit/cpu-oracle-canonical-prefix32/`.
- Full-forward resident CUDA logs, exports, and transfer audit: `~/.cache/vbuf-agent-qualification/qwen3-cuda-q4k/cuda-transfer-audit/resident-canonical-fullhead-prefix32-allprobes/`.
- Incremental/KV logs and transfer audit: `~/.cache/vbuf-agent-qualification/qwen3-cuda-q4k/cuda-transfer-audit/resident-incremental-prefix8-final/`.

## Full resident forward

The resident graph used all 40 blocks and 443 weight tensors (8,995,793,920 payload bytes), including embedding and LM-head tensors. The graph buffer was 9,787,915,520 bytes, with 2,677,276,672 bytes reported free after initialization. Three runs completed with finite outputs; runs 2 and 3 were bit-identical to run 1. Execution-window CUDA API audits recorded zero H2D transfers during each run; the run-1 diagnostic readbacks included all layer checkpoints. The resident weights were released and the CUDA allocation freed at teardown.

| Prefix | CPU/GPU greedy top-1 | Top-5 | Top-10 | CPU margin | CUDA margin |
|---:|---|---:|---:|---:|---:|
| 1 | 25 / 25 | 5/5 | 9/10 | 0.9170 | 0.9264 |
| 2 | 220 / 220 | 5/5 | 8/10 | 1.0526 | 1.0625 |
| 4 | 13 / 13 | 5/5 | 9/10 | 0.4866 | 0.3809 |
| 8 | 198 / 198 | 5/5 | 10/10 | 1.1904 | 1.3869 |
| 16 | 13 / 13 | 5/5 | 10/10 | 4.5744 | 4.4564 |
| 25 | 15 / 15 | 5/5 | 10/10 | 3.6834 | 3.5025 |
| 32 | 220 / 220 | 5/5 | 9/10 | 6.3311 | 6.3080 |

Top-token agreement is not numerical parity. The full-vocabulary comparisons are retained in the run log. Prefixes 1, 2, 4, 8, 16, and 32 had CPU-vBuf/CUDA relative-RMS logit differences of approximately 5.65%, 3.59%, 5.39%, 5.33%, 7.47%, and 6.86%, respectively. Prefix 25 is a material sequence-local outlier: relative RMS 61.8%, cosine 0.9242, and logit norm ratio 1.4097, despite top-5/top-10 agreement and a matching top-1.

A same-GPU-hidden CPU-head control localized most of the prefix-25 discrepancy upstream of the output-head kernel: CPU and CUDA final norm on the identical CUDA hidden state had relative RMS `8.42e-8`; CPU-vs-CUDA head logits on that same hidden state had relative RMS 3.17%, cosine 0.99950, and the same top-1. The final hidden row at token position 25 differed from CPU vBuf by 14.38% relative RMS (cosine 0.98965). Per-layer row-level drift grew across many layers, peaked at layer 29 (51.16% relative RMS, cosine 0.89064, norm ratio 1.1252), then partially recovered to layer 39. This is a substantial position-specific accumulated numerical divergence, not a single-layer shape/payload failure; its source is unresolved. It must remain visible in later numerical qualification and is not waived by token agreement.

The prefix-32 same-hidden head control was also stable: final norm relative RMS `7.61e-8`; same-hidden CPU/CUDA head logits relative RMS 4.51%, cosine 0.99899; greedy top-1 matched.

## Incremental replay and device KV probe

The separate eight-token probe constructs one CUDA graph step per token. Each step writes the layer's F16 K/V rows into a shared device cache using GGML `SET_ROWS`, then attends over the cache with a causal mask. It reuses the same resident model weights and cache allocation for three repeated eight-token replays. The graph/token inputs are fixed for this diagnostic; this is not autoregressive generation or a runtime session API.

- Cache geometry: 40 layers, 8 KV heads, head dimension 128, F16 K and V; 1,310,720 bytes for eight positions (160 KiB/token).
- Resident weight payload: 8,995,793,920 bytes.
- Monolithic graph buffer: 9,197,267,968 bytes; reported free after initialization: 3,268,673,536 bytes.
- Initialization H2D: 555 calls / 8,997,105,216 bytes (weights, one-time cache zeroing, and small fixed graph inputs/indices).
- Each replay execution window: **zero H2D**, 12 D2H calls / 2,594,816 bytes for selected hidden/norm/logit outputs, one stream synchronization. No historical KV D2H/H2D was observed.
- All selected hidden, final-norm, and logit outputs were finite. Three replays were bit-identical. Allocation and residency entries were released at teardown.

Against the full-recompute CUDA path, incremental-vs-full relative-RMS logit differences were approximately 4.25%, 2.29%, 3.09%, and 5.01% at prefixes 1, 2, 4, and 8; cosine remained at least 0.9989. Incremental and full-recompute top-1 agreed at every prefix, with top-5 overlap 5/5 and top-10 overlap 10/10 at prefix 1, 5/5 and 9/10 at prefixes 2 and 4, and 5/5 and 10/10 at prefix 8. CPU-vBuf top-1 also matched at all four prefixes. These are measured backend/graph-geometry differences, not bitwise equivalence.

## KV capacity and qualification boundary

For this model, raw F16 KV storage is `40 layers × 8 KV heads × 128 dimensions × 2 bytes × 2 (K,V) = 163,840 bytes/token` (160 KiB/token): 1,310,720 bytes at 8 tokens, 160 MiB at 1,024, 640 MiB at 4,096, and 2.5 GiB at 16,384. These are payload-only estimates; allocator/runtime overhead is additional. With approximately 3.27 GB free after the eight-token probe's graph-buffer allocation, this does not establish a viable long-context GPU KV budget. No capacity above eight, eviction, memory pressure, cache reuse across different prompts, or generated-token updates was tested.

The probe's device-residency store accounts for tensor identity, leases, allocation ownership, and teardown, but all weights/cache/intermediates share a single backend graph buffer. It does not establish independently evictable production allocations or a production residency planner.

## Qualification status

| Gate | Status |
|---|---|
| 40-block resident CUDA forward, final norm, LM head, logits | **EXECUTED; finite and repeatable; diagnostic qualification only** |
| Greedy token comparison at prefixes 1/2/4/8/16/25/32 | **Top-1 matched; top-k and margins recorded** |
| Prefix-25 full-logit numerical behavior | **SUBSTANTIAL POSITION-SPECIFIC DRIFT; source unresolved; retain as follow-up** |
| Eight-token incremental CUDA replay with device F16 KV | **EXECUTED three times; stable; no execution-window H2D or historical KV D2H/H2D** |
| GPU KV capacity, pressure/eviction, generated-token updates, production session | **NOT TESTED** |
| Strict external llama.cpp `1e-5` parity | **FAIL remains; unchanged** |
| Production Qwen3, HTTP/tools/Pi integration | **DISABLED / NOT STARTED** |
