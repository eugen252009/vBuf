# Qwen3 CUDA Prefix-25 and Generated-KV Qualification

**Disposition:** The prefix-25 CPU/CUDA difference remains a context-sensitive accumulated numerical divergence whose exact contributors are unresolved; no structural GPU/runtime defect was found in the checked prefix-25 controls. A separate persistent-KV versus same-context full-recompute audit localizes the first incremental/full CUDA difference to quantized Q/K matrix products taking different shape-selected CUDA kernels, with cache writes, historical rows, and effective attention views verified correct at sampled layers/positions. Phase B exercised device-resident incremental K/V through 32 positions and appended 24 GPU-selected tokens from an eight-token prompt. The generated sequence was repeatable and matched same-context GPU full-recompute and CPU vBuf top-1 at all 24 appended positions. These are isolated diagnostic results, **not numerical parity or production qualification**. Production Qwen3 remains disabled; the external llama.cpp `1e-5` gate remains **FAIL**, unchanged.

## Scope and identities

- Model: Qwen3-14B Q4_K_M vBuf artifact, SHA-256 `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`; source reports 443 tensors and 9,000,232,144 bytes.
- GPU: RTX 3060, CUDA device 0, SM 8.6. CUDA qualification uses the pinned GGML revision `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`.
- BF16 context: sibling BF16 artifact from immutable snapshot `bd080f768a6401c2d5a7fa53a2e50cd8218a9ce2`, SHA-256 `9677a58b0fa8da7771a4d8cc8080208ce02a32ab21305ded10a3798766132d3a`. Its 32-position CPU capture is a practical higher-precision comparison, **not mathematical ground truth**. The BF16 capture used a different llama.cpp/GGML revision than the pinned vBuf/CUDA Q4 probe; representation and runtime differences are confounded.
- Canonical teacher-forced sequence for Phase A and static Phase B:
  `0,25,220,16,13,15,13,15,198,262,549,0,220,16,13,15,13,15,198,262,549,0,220,16,13,15,13,15,198,262,549,0`.
- Phase A exact-prefix-25 and full-sequence probes were repeated; arrays were finite. The CPU/CUDA numerical comparisons are diagnostic observations, not a waiver of the fixed parity gate.

## Phase A — prefix-25 classification

### Neighbourhood and BF16 context

The following relative-RMS errors use the first column's path as denominator. Jensen-Shannon divergence (JS) compares the CPU and CUDA output distributions. All top-1 IDs agree across CPU Q4, CUDA Q4, and BF16 for prefixes 22–27; top-50 CPU/CUDA overlap is shown separately.

| Prefix | CPU Q4 vs CUDA Q4 rel-RMS | CPU/CUDA cosine | CPU/CUDA JS | CPU Q4 vs BF16 rel-RMS | CUDA Q4 vs BF16 rel-RMS | top-1 CPU/CUDA/BF16 | top-50 overlap |
|---:|---:|---:|---:|---:|---:|---|---:|
| 22 | 4.45% | 0.999344 | 0.0005618 | 4.72% | 5.12% | 220 / 220 / 220 | 48/50 |
| 23 | 11.36% | 0.993557 | 0.0000457 | 33.45% | 26.05% | 16 / 16 / 16 | 46/50 |
| 24 | 6.38% | 0.997963 | 0.0000249 | 12.46% | 11.59% | 13 / 13 / 13 | 48/50 |
| 25 | **61.78%** | **0.924173** | 0.0001238 | **36.57%** | **14.22%** | 15 / 15 / 15 | 44/50 |
| 26 | 14.61% | 0.992959 | 0.0000060 | 11.27% | 13.42% | 13 / 13 / 13 | 47/50 |
| 27 | 19.67% | 0.991787 | 0.0001571 | 14.31% | 28.87% | 15 / 15 / 15 | 47/50 |

The spike is strongly position-specific. At prefix 25 the GPU logits are closer to the BF16 capture than CPU vBuf logits are; at neighboring positions the ordering alternates. This argues against treating either Q4 path as an oracle and does not establish that CUDA is generally more accurate. Small JS divergence and matching top-1 do not imply numerical parity.

At prefix 25, final hidden relative-RMS error against BF16 is 13.67% for CPU vBuf and 5.44% for CUDA; final-normalized hidden errors are 18.11% and 7.38%. The logits values in the table are the direct full-vector comparison. These BF16 comparisons are contextual only and include different GGML revisions.

### Layer growth and operator checks

For the same teacher-forced 32-token sequence, per-layer CPU/CUDA hidden relative-RMS errors at prefixes 24–27 were:

| Layer output | Prefix 24 | Prefix 25 | Prefix 26 | Prefix 27 |
|---:|---:|---:|---:|---:|
| 18 | 6.7% | 10.5% | 5.9% | 5.0% |
| 20 | 6.2% | 9.5% | 5.6% | 5.0% |
| 21 | 6.0% | 12.5% | 5.2% | 5.4% |
| 22 | 5.9% | 16.4% | 5.3% | 5.7% |
| 23 | 6.0% | 21.6% | 5.3% | 5.9% |
| 24 | 5.7% | 28.5% | 5.4% | 5.7% |
| 27 | 5.4% | 47.7% | 5.2% | 6.6% |
| 29 | 5.5% | **51.2%** | 5.2% | 7.1% |
| 30 | 5.4% | 47.6% | 4.9% | 7.2% |
| 39 | 3.2% | 14.4% | 5.2% | 5.6% |

The prefix-25 excess becomes conspicuous around layers 21–23, grows to a layer-29 peak, and partially recovers. It is not a one-layer output discontinuity; the source of the accumulated difference is not localized to one operator.

A layer-21 capture at query position 24 compared the CPU canonical-prefix-25 path (logical 25 keys, compute extent 32) with the CUDA exact-prefix-25 path (logical and physical extent 25):

| Captured boundary | CPU/CUDA relative RMS | Cosine |
|---|---:|---:|
| layer input | 9.53% | 0.995510 |
| Q / K / V projection | 8.41% / 12.58% / 19.71% | 0.996730 / 0.992062 / 0.980475 |
| attention scores | 6.75% | 0.998057 |
| attention probabilities | 14.22% | 0.989965 |
| attention context / output projection | 20.13% / 21.98% | 0.980784 / 0.976351 |
| FFN SwiGLU / down projection | 30.46% / 30.05% | 0.952584 / 0.953896 |
| layer output | 12.51% | 0.992164 |

This capture shows substantial differences in intermediate values, not a single corrupt or non-finite tensor. It is consistent with an accumulated backend-numerics sensitivity; it does not by itself identify the contribution of each quantized projection or prove that all differences are expected.

### Attention extent, masking, and same-score softmax

- CPU capture metadata: `queries=25`, logical keys 25, padded compute `keys=32`. CUDA exact-prefix capture metadata: attention score/probability tensor extent `25 x 25 x 40`. The full 32-position CUDA replay also exposes a 32-key physical extent.
- At query position 24, the CPU's seven padded future probabilities are exactly zero. Both paths have visible probability mass between `0.9999998` and `1.0000001`; the GPU has no physical slots beyond its 25 keys. The differing physical extents therefore preserve the same checked causal visibility for this row.
- CPU/GPU attention-score differences feed different probabilities, but a same-score control isolates softmax: CPU vs FP64 relative RMS `1.32e-7`, CUDA vs FP64 `1.47e-7`, CPU vs CUDA `7.77e-8`; maximum absolute differences are at most `1.79e-7`. Mean entropy is 1.827/1.867, mean max probability 0.503/0.492, and only 3/40 heads change argmax key. This does not implicate a softmax numerical defect.

### Phase-A verdict

**Classification:** context-sensitive, accumulated CPU/CUDA numerical divergence; **no structural GPU/runtime defect identified** in the checked source identity, tensor payloads, layer graph, attention extent/masking, softmax control, device residency, or finite outputs. The precise operator-level numerical cause remains unresolved. The existing Q4/Q6 backend-specific activation-quantization differences are relevant context, not a complete causal proof for this position-25 event.

The full-prefix CPU/CUDA difference is not dismissed: 61.78% relative-RMS logits error is material. BF16 helps contextualize it but does not waive it. This classification allowed the separate diagnostic Phase B; it does not clear production numerical qualification.

## Phase B — 32-position incremental K/V and true generated append

### Qualification-only implementation

`integrations/ggml/tools/qwen3_block_qualification.cpp` now supports 1–32 position device-resident incremental graphs and a qualification-only `resident_cuda_generate` mode. Each one-position graph updates the shared F16 K/V cache with `ggml_set_rows`; a device `ggml_argmax` selects the next token. Only the selected 32-bit token is read to the host and uploaded into the next position. The complete history of K/V remains in the CUDA allocation. The mode is internal-only and does not enable or modify `VbufGenerationSession`.

The fixed teacher-forced 32-position incremental replay ran three times, produced bit-identical outputs, and tore down all device residency. In comparison to the same-context monolithic full-forward CUDA graph, its logits relative-RMS error ranges from 1.44% to 11.54% over prefixes 1–32 (median 3.15%). On that *different, fixed teacher-forced token sequence*, prefix 10 has incremental top-1 549 versus full-recompute top-1 829. This is an important backend-path numerical difference; it is not concealed by the generated-sequence result below.

### Persistent-KV versus same-context full-recompute audit

The 32-token generated history was replayed both as one full CUDA batch and as one-token-at-a-time resident execution. At each selected layer/position, the incremental graph had the **same exact input row and attention RMSNorm output** as the corresponding full-batch row at layer 0, but the first quantized projections already differed: at positions 7, 8, 23, and 31, Q relative-RMS was 0.452%, 3.159%, 3.047%, and 0.328%; K was 0.413%, 2.276%, 2.291%, and 0.269%. The layer-0 Q6_K V projection differed by only 0.007–0.011% at those positions. Later attention scores/probabilities and layer outputs diverge further.

This matches the pinned GGML CUDA dispatch in `ggml_cuda_mul_mat`: on SM 8.6, a one-column quantized matmul (`ne11=1`) selects MMVQ, while the 32-column batch exceeds the MMVQ batch limit and selects MMQ. Layer-0 Q/K weights are Q4_K; V is Q6_K. Thus the same weights and input row encounter different activation quantization/kernel arithmetic in incremental and full-batch execution. This is a **shape-dependent CUDA numerical-path difference**, not evidence that either result is a ground-truth reference. The fixed `1e-5` parity criterion is not met.

Cache/runtime controls passed at sampled layers 0, 20, 29, and 39 and sampled append boundaries spanning positions 8–32: newly computed K/V exactly matched their written cache rows; prior K/V bytes remained immutable after append; and the effective K/V tensors consumed by attention exactly matched the corresponding resident cache view. Compared with full-batch effective attention inputs, K relative-RMS ranges from 0.45% to 3.00% and V from 0.02% to 6.11% across those samples. The numerical differences are therefore present in the computed K/V and grow with upstream hidden-state divergence; they are not caused by cache row placement, mutation, or stale/misread history.

**Classification:** no persistent-KV semantic/runtime defect was found in the exercised 32-position path. Incremental-versus-full-recompute divergence is localized at its first observed point to shape-selected CUDA quantized-matmul paths, then accumulates through attention and later layers. This does not resolve the separate prefix-25 CPU/CUDA accumulated drift, clear the external parity gate, or qualify production CUDA execution.

### Generated-token append/reuse result

Starting with prompt `0,25,220,16,13,15,13,15`, the 32-slot cache performed 24 genuine greedy token appends. Across three cache-reset replays, generated token IDs and exported hidden/norm/logit outputs were bit-identical:

```text
198,220,829,25,330,2408,33696,698,220,2319,25,330,16,13,15,13,15,698,220,4008,25,330,5050,2390
```

A separate CPU full-sequence Q4 reference and a separate monolithic CUDA full-recompute graph were run on the exact resulting 32-token history. At every generated step, all 24 selected tokens matched the corresponding incremental logits argmax, CPU full-sequence vBuf top-1, and CUDA full-recompute top-1 (**24/24** for both references). After the final 32-token context, incremental and full-recompute CUDA both predict 698; CPU full-sequence Q4 predicts 4008 (top-1/top-2 margin 0.0650 versus CUDA's 0.1345), a near-tie divergence beyond the 24-token matched segment. This is a bounded token-trajectory result only.

Incremental-versus-full-recompute logit relative-RMS over the 25 exported contexts from prefix 8 through 32 has minimum 1.76%, median 2.90%, mean 3.44%, and maximum 9.41%; cosine similarity remains high, but the fixed `1e-5` numerical criterion is not met. Incremental-versus-CPU-vBuf logit relative-RMS over those contexts has median 5.27% and maximum 9.12%. The same-context distributions can select the same token while differing numerically; do not describe this as numerical parity.

### Residency and transfer accounting

For the 32-slot generated run on CUDA 0:

- Model resident payloads: 8,995,793,920 bytes; allocated GGML graph buffer: 9,874,439,168 bytes; F16 K/V capacity: 5,242,880 bytes; free VRAM after initialization: 2,591,293,440 bytes.
- After initialization, each generated replay recorded zero weight upload. Its execution window recorded 5,243,072 H2D bytes: 5,242,880 bytes to reset the K/V cache, 96 bytes of generated token IDs, and 96 bytes to reset generated input slots. These are cache reset/new-token controls, **not historical-KV transfers**.
- Each replay recorded 96 D2H bytes for 24 scalar argmax token IDs and no historical-KV D2H. A separate diagnostic readback window recorded 16,217,600 D2H bytes for hidden/norm/logit comparison exports across prefixes 8–32; this intentionally reads logits and is not part of token/KV feedback.
- Each replay completed with the 32-position cache populated; cache and residency allocations were freed at teardown. The qualification used one monolithic GGML graph allocation and does not qualify independently evictable production allocations or memory-pressure behavior.

The eight-append capacity-16 probe also generated the first eight IDs of the 24-token sequence on three identical replays. The capacity-32 run is the primary generated-append evidence.

## Limitations and gate status

- Qwen3 remains disabled in production; the production builder still rejects it as unimplemented.
- External llama.cpp Q4 `1e-5` parity remains **FAIL**, unchanged. No result here waives that gate.
- This is one base-model prompt/trajectory, one GPU, one fixed 32-position capacity, and the qualification graph; it does not qualify longer contexts, eviction, reuse across prompts, streaming service, generated assistant tasks, or production lifecycle.
- The unexplained prefix-25 CPU/CUDA accumulated drift remains documented. BF16 is context, not ground truth.
- HTTP, native tools, Pi round trips, and coding-agent dogfood remain **NOT TESTED**. Android remains unverified.

## Reproduction evidence

Raw qualification output is retained under:

`~/.cache/vbuf-agent-qualification/qwen3-cuda-q4k/cuda-transfer-audit/`

Key directories:

- `resident-neighborhood22-27/`, `resident-exact-prefix25-repro/`, `cpu-canonical-prefix25-layer21-position24/`, `resident-exact-prefix25-layer21-position24/`, `bf16-A32-final/`
- `resident-canonical-fullhead-prefix32-phaseB/`, `resident-incremental-prefix32-final/`
- `resident-generated-prefix8-plus8/`, `resident-generated-prefix8-plus24/`, `resident-generated-prefix8-plus24-cache-operators-final/`, `cpu-generated-prefix8-plus24/`, `resident-generated-full-recompute-prefix32/`, `resident-generated-full-kv-captures-v3/`, `resident-generated-prefix8-plus24-compared-final/`

These are isolated qualification fixtures and do not alter the external parity record or production behavior.
