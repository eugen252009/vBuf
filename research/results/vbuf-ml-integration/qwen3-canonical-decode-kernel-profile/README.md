# Qwen3 canonical decode kernel profile and fusion decision

## Result

**No small fusion was implemented.** The canonical 26/14 decode was profiled with the optimizer in `SHADOW`; canonical execution remained selected. The strongest measured opportunity is the full-capacity V-cache layout copy, but addressing it requires a new strided/fused attention-value kernel or a changed KV layout. That is outside the requested small-fusion scope and risks changing numerical behavior or the KV representation. The best remaining small local candidates account for at most about 0.06% of summed kernel time, below a credible end-to-end measurement floor. Keep the guarded prebound-dispatch experiment performance-rejected and disabled; no production defaults or placement policy changed.

## Identity and conditions

- Model: Qwen3-14B Q4_K_M, semantic SHA-256 `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- GGML: pinned revision `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`.
- GPUs: RTX 3060 / SM 8.6 owns embedding and blocks 0–25; RTX 2080 SUPER / SM 7.5 owns blocks 26–39 and final norm/head.
- Profiled gate: experimental capacity 32,768, chunk 32, one decode at context 32. The application emitted `capture=PASS`; optimizer mode was `SHADOW`, guards were eligible, the candidate was unvalidated and **not** selected, and canonical execution was selected.
- The production capacity remains 1,032, single-GPU remains the default, chunk remains 32, and the 26/14 split remains qualification-only. No 25/15 placement, value/range specialization, JIT, or KV-format change was made.

The opt-in CUDA profiler range is confined to `vbuf_qwen3_multigpu_capacity_gate`: set `VBUF_QWEN3_CUDA_PROFILE_CONTEXT` to the gate prefix and run exactly one appended token under Nsight Systems with its CUDA-profiler-API capture range. The default gate path is unchanged. The captured report, gate output, and kernel summary are preserved in `raw/context-32.qdrep`, `raw/context-32-capture.log`, and `raw/context-32-nsys-kernel-summary.txt`. Reproduce with the same qualified semantic artifact and source used by the gate:

```bash
VBUF_QWEN3_CUDA_PROFILE_CONTEXT=32 nsys profile \
  --capture-range=cudaProfilerApi --stop-on-range-end=true \
  --output=/tmp/qwen-context-32 \
  <vbuf_qwen3_multigpu_capacity_gate> <semantic-artifact> <source-endpoint> \
  <token-ids-file> 32768 32 1 32 run 0
nsys stats --report cuda_gpu_kern_sum /tmp/qwen-context-32.qdrep
```

## Context-range observations

The earlier canonical/SHADOW capacity-32K host profiles covered these prefix positions (32 decode samples each):

| Prefix context | Step median | Step p95 |
|---:|---:|---:|
| 32 | 374.752 ms | 378.866 ms |
| 1,024 | 383.720 ms | 386.025 ms |
| 8,192 | 383.712 ms | 389.097 ms |
| 16,384 | 379.237 ms | 385.783 ms |
| 24,576 | 380.563 ms | 385.123 ms |
| 32,736 | 378.117 ms | 379.832 ms |

These are host-observed timings from separate runs, not a controlled cross-context benchmark; desktop activity and the earlier optimizer qualification can affect them. They show no material monotonic context trend at this fixed capacity. The six source logs are retained under `../qwen3-first-guarded-specialization/raw/context-*-profile.log`.

Only context 32 received a new Nsight kernel capture. This is sufficient to identify the capacity-sized work, not to claim a kernel-timing sweep over contexts. In `qwen3_cuda_build_layer`, K/V views, causal-mask geometry, and packed-V scratch are all built with the fixed `capacity`; changing the current position changes mask contents, not those tensor extents. Thus the profiled layout-copy grid and byte extent are capacity-bound at every context for this run. A separate high-context Nsight capture was not performed.

The profiled one-token gate's host `decode_ns` was 4.114 s, versus roughly 0.375–0.384 s in the unprofiled host samples. Treat that traced wall interval as observer-distorted, not as a performance result. The CUDA kernel durations below are instrumented observations and their sum across devices is not an exact end-to-end critical-path timer. A later inspection found desktop-compositor utilization on the RTX 2080 SUPER; whether it overlapped the capture is unknown.

## Kernel inventory and ranking

Nsight reported 377.951 ms total summed GPU-kernel duration across both devices for the captured decode. The leading entries were:

| Operation | Instances | Summed duration | Share of summed kernel time | Assessment |
|---|---:|---:|---:|---|
| F16 `cpy_scalar` for V layout conversion | 40 | 278.529 ms | 73.7% | Dominant opportunity, but not a small local fusion. |
| KQ `mul_mat_f` | 40 | 38.908 ms | 10.3% | Part of a larger attention rewrite. |
| AV `mul_mat_vec_f` | 40 | 30.189 ms | 8.0% | Fusing with V conversion would require a custom attention-value kernel. |
| softmax | 40 | 3.562 ms | 0.9% | A full attention fusion would change reduction structure. |
| SiLU | 40 | 0.072 ms | 0.02% | Potential SiLU×up fusion is too small. |
| elementwise multiply | 40 | 0.093 ms | 0.02% | Together with SiLU, only 0.165 ms / 0.044%. |
| F32→F16 cast + KV `set_rows` | 80 each | 0.221 ms combined | 0.06% | A possible local fusion, but not enough measurable headroom. |

The `cpy_scalar` kernel used grid `(524288,1,1)` for each copy. At capacity 32,768 the packed-V allocation is 67,108,864 bytes (64 MiB) per device, and the copy occurs once for every transformer layer. This implies 2.5 GiB of source plus 2.5 GiB of destination tensor traffic per decode across 40 layers (derived from tensor geometry; not a hardware-counter measurement). Device-0's 26 copies accounted for 248.116 ms and device-1's 14 for 30.412 ms in this trace.

The copy is built from `ggml_cpy(ggml_permute(v_reshaped, ...), packed_value_scratch)` in `integrations/ggml/src/qwen3_cuda_core.cpp`. Replacing it with a fused AV kernel would require non-contiguous V reads and a new attention kernel; changing the cache to avoid the transpose would change the KV layout. The RMSNorm kernels already include their scale multiply (`do_multiply=true` in the captured CUDA kernel specialization), so that obvious pair is not an unfused candidate in this trace. Neither is a small, low-risk local fusion. The separate SiLU and elementwise multiply are a more local candidate, but their combined measured kernels are only 0.044% of the summed kernel time. That ceiling is far below the run-to-run variation already observed, so implementing it would not support a credible end-to-end speedup claim. The cast/scatter pair is similarly not compelling.

A FlashAttention-style KQ/softmax/AV replacement has more potential work to remove, but is a broader alternate attention kernel with different reduction behavior, not the one small fusion requested. It was not implemented or enabled. No candidate was selected for a follow-up experiment.

## Verification and disposition

- Rebuilt `vbuf_qwen3_multigpu_capacity_gate` with the opt-in CUDA profiler API enabled.
- Captured exactly one canonical decode step at context 32; `optimizer_mode=SHADOW`, `candidate_selected=NO`, `canonical_selected=YES`.
- Nsight kernel aggregation produced 40 V-layout copies, 40 KQ matmuls, 40 softmax kernels, and 40 AV matvecs, matching the 40-layer graph.
- No production decode operation, KV tensor layout, cache lifetime, placement, capacity default, or optimizer default was changed.
- No fusion is proposed for implementation absent new evidence that a small local sequence contributes materially more end-to-end time.
