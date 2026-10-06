# Qwen3-14B 26/14 staged two-GPU small-context qualification

Status: **small-context qualification passed for the recorded test set; experimental only**. This does not qualify 32K, increase production capacity, or establish production multi-GPU behavior. Production-qualified capacity remains **1,032 tokens**. No multi-GPU path was enabled by default.

## Identity and environment

- Model semantic payload SHA-256: `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- Payload file size: `9,000,232,144` bytes. Sum of 443 tensor payload lengths uploaded once: `8,995,793,920` bytes (the difference is non-tensor source layout/metadata space).
- Placement: RTX 3060 / CUDA device 0 owns embedding, blocks 0–25, and 26 layers' KV; RTX 2080 SUPER / CUDA device 1 owns blocks 26–39, final norm/head, and 14 layers' KV. The one boundary is F32 hidden state staged through a session-owned pinned host allocation. No P2P/NVLink is used.
- CUDA 12.4; GGML source commit `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d` (local checkout reported dirty, so this is not a pristine-source qualification).
- Context capacity 64 for this run; prompt prefixes 1, 2, 4, 8, 16, 32; one generated token per request. Inputs were token IDs `0..31` (synthetic qualification tokens, not natural-language quality tests). The local range endpoint was `http://127.0.0.1:18946/payload`.
- Command and complete raw output are retained in `raw/qualification.log` (SHA-256 `59a12c97629970dfde88f26ac69d77beb2b9dbe44a0814e3593dc1987b3131e7`).

## Results

- All 443 admitted model tensors were assigned exactly once: device 0 held 287 tensors / `5,561,288,704` payload bytes; device 1 held 156 tensors / `3,434,505,216` bytes. Per-device totals reconcile exactly to tensor payload lengths.
- All 40 K/V pairs were checked for the expected owner, shape, and F16 type; every non-owner lookup returned null. All sampled layer-25/layer-26 KV rows were finite and remained byte-identical across a subsequent decode append.
- The staged buffer was CUDA-pinned host memory, `655,360` bytes (`32 × 5,120 × sizeof(F32)`). Every handoff's received activation bytes matched the pinned source bytes. Measured handoffs were 2–17 per case, according to tokenwise prompt rows or one 32-row prefill chunk, plus decode.
- Greedy token IDs matched the single-device canonical Qwen executor at every prompt prefix; post-decode logits argmax also matched.
- Numeric acceptance used an explicit test-set bound of **relative RMS ≤ 0.02 and cosine similarity ≥ 0.9998** for both final hidden and logits. All six prefixes passed. Worst observed hidden relative RMS was `0.0182232` (cosine `0.9998367`); worst logits relative RMS was `0.0164695` (cosine `0.9998716`). Maximum absolute differences were `5.64246` hidden and `0.341287` logits. These are cross-GPU numerical differences, not bitwise parity or a universal model acceptance threshold.
- Same-session reset/replay was bitwise repeatable for logits and final hidden. Sequential A/B/A session isolation passed.
- Injected failures before boundary transfer, after D2H, after H2D, before late-device execution, and before the last transformer block all left logical session length at zero. A fresh session recovered and completed successfully. Physical K/V writes from a failed request are not rolled back; logical progress remains uncommitted and reset/masking prevents them being treated as live context.
- Capacity, placement, model loading, and execution were exercised end-to-end on both GPUs for these small contexts. This is not a long-run, natural-language, stress-memory, concurrency, or production qualification.

Rerun command shape:

```bash
CUDA_VISIBLE_DEVICES=0,1 vbuf_qwen3_multigpu_qualification \
  /path/to/Qwen_Qwen3-14B-Q4_K_M.semantic.vbuf \
  http://127.0.0.1:18946/payload \
  0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  64
```

The endpoint was served by the local range server over the matching Q4_K_M `.vbuf` payload.

## Memory observation

At capacity 64, session K/V allocation was `6,946,816` bytes on device 0 and `3,801,088` bytes on device 1; each device also allocated 64 MiB prefill/layer scratch and 32 MiB main decode/control scratch. Immediately after executor graph allocation, reported free VRAM was about `6,815,875,072` bytes on the RTX 3060 and `4,358,144,000` bytes on the RTX 2080 SUPER. These measurements do not establish the 32K memory envelope or guarantee availability under other desktop workloads.

## Implementation and qualification boundary

The opt-in 26/14 placement is represented in Qwen CUDA runtime/session/executor ownership; canonical one-GPU placement remains the default. Device-local tensor residency and KV ownership are explicit. The staged executor runs layers in order, synchronizes around pinned D2H/H2D, and commits logical position only after a complete step. The generic vBuf format and backend ownership boundary were not changed.

An initial run exposed illegal device memory access because auxiliary decode layer graph scratch aliased the main graph-context control/hidden allocation. That allocation alias was corrected by keeping auxiliary layer scratch separate from main decode/control scratch (decode and prefill auxiliary graphs reuse the larger layer scratch sequentially). The complete successful run was then repeated and archived; the failed run is not treated as qualification evidence.

The qualification harness is `integrations/ggml/qualification/qwen3_multigpu_qualification.cpp`, built as `vbuf_qwen3_multigpu_qualification`. It captures final hidden from the single-device reference only through an opt-in qualification flag; ordinary single-device execution does not add that capture transfer.

At the time of this 64-token run, no 32K test had been attempted and no experimental-capacity opt-in was enabled. This small-context report remains limited to its recorded test set. A later, separately scoped 2K–32K experimental capacity qualification is documented in [`qwen3-multigpu-26-14-capacity-qualification/README.md`](qwen3-multigpu-26-14-capacity-qualification/README.md); production capacity remains 1,032 and the planning candidates remain distinct from qualified placements.
