# Step 32K-A: Generic CUDA Real GLM Block

Date: 2026-08-28
Starting commit: `463345c`
Result: **PASS_FULL_DEVICE_BLOCK**

## Scope

The vBuf-ML runtime executed one complete real layer-23
GLM-4.5-Air-FP8 MoE transformer block on CUDA device 0. The run used the
canonical vBuf payload and semantic sidecar, bounded host materialization, a
backend-neutral device tensor contract, and a CUDA adapter implemented without
GGML or llama.cpp. It did not execute the full stack, GPU prefill/decode,
generation, or multi-GPU work.

## Model And Device

| Field | Measured value |
|---|---|
| Model | `zai-org/GLM-4.5-Air-FP8` |
| Revision | `f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc` |
| Persistent model bytes | `112563538898` |
| Canonical artifact | `.step32c/glm-4.5-air-fp8.vbuf` |
| Semantic sidecar | `.step32c/glm-4.5-air-fp8.semantic.vbuf` |
| CUDA runtime / driver | `12.4` / `550.163.01` |
| CUDA device count | `2` |
| Qualification device | `0: NVIDIA GeForce RTX 3060` |
| Compute capability | `8.6` |
| Total VRAM | `12622168064` bytes |
| Free VRAM before run | `12466913280` bytes |
| PCIe state | Gen1 x4 current, Gen3 x16 maximum |

The RTX 2080 SUPER was not used. No second GPU execution was performed.

## Architecture Audit

| Portable op | CPU support | Generic device contract | CUDA support | Required | Gap class |
|---|---|---|---|---|---|
| RMSNorm | yes | yes | yes | yes | none |
| Dense MatMul | yes | yes | yes, cuBLAS F32 | yes | none |
| Bias/add/multiply | yes | yes | yes, CUDA kernels | yes | none |
| SiLU/sigmoid | yes | yes | yes, CUDA kernels | yes | none |
| Reshape and RoPE | yes | yes | yes, CUDA kernels/copies | yes | none |
| Causal GQA attention | yes | yes | yes, bounded qualification kernel | yes | none |
| Router MatMul | yes | yes | yes | yes | none |
| Top-K | yes | yes | host control only | yes | qualification boundary |
| Selected indexed expert MatMul | yes | yes | yes, selected expert only | yes | none |
| Request KV state reuse | yes | yes | not in this prefill-only scope | no | qualification gap |
| F16/BF16/native FP8 device path | no requirement | dtype-neutral contract | not used | no | deferred |

Before this step, no generic device identity, device tensor lifetime, CUDA
adapter, or CUDA operation implementation existed. The portable graph,
lowering, materializer, semantic bindings, and CPU state remain the canonical
runtime authority.

## Device Boundary

The generic contract defines `DeviceId`, `DeviceKind`, `DeviceCapabilities`,
`DeviceDType`, and an opaque `DeviceTensor` backed by an owned storage trait.
The CUDA adapter owns CUDA context, cuBLAS handle, device allocation, transfer,
kernel invocation, synchronization, and release. Portable graph code sees no
CUDA runtime, stream, cuBLAS, or NVIDIA types.

`CUDA_TYPE_LEAKAGE_COUNT=0` was verified by
`scripts/verify_portable_graph_neutrality.py`. CUDA is linked only when the
`cuda` feature is enabled, using `nvcc`, GCC 13, CUDA runtime 12.4, and cuBLAS.
The local GGML checkout and GGML execution were not used.

## Data Path And Dtypes

```text
canonical vBuf TensorId
  -> bounded vBuf-ML materialization and FP8 scale resolution
  -> host F32 staging for one demanded tensor
  -> CUDA F32 DeviceTensor
  -> CUDA operation
  -> device-resident activation
  -> bounded readback for comparison/checkpoints
```

| Field | Value |
|---|---|
| Persistent dtype | `F8_E4M3` weights, BF16 bias/norm values |
| Device weight dtype | `F32` |
| Device activation dtype | `F32` |
| Accumulation dtype | `F32` |
| TF32 | disabled with `CUBLAS_PEDANTIC_MATH` |
| CUDA streams | one default stream |
| Synchronization | explicit `cudaDeviceSynchronize` at qualification boundaries |

The implementation uses bounded host F32 staging rather than a full converted
model. Persistent FP8 provenance remains in the sidecar and is not rewritten.
The direct-to-device future seam remains the materialized tensor upload call;
GPUDirect Storage was not implemented.

## Real Block

- Layer: `23`
- Input shape: `[1, 4, 4096]`
- Input SHA-256: `5eb4f1bfdd34f9a76ea0189484d387cf1a047c27d2c49d3f2783211bb1ec604b`
- Total lowered operations: `48`
- CUDA operations: `46`
- Host control operations: `1` Top-K selection
- CPU fallback operations: `0`
- Undeclared CPU fallbacks: `0`
- Router: CUDA projection and sigmoid, host-only control Top-K
- Selected experts: `[1, 6, 11, 19, 27, 52, 59, 62, 89]`
- Selected expert set parity: `PASS`
- Selection order mismatches: `0`
- Expert ID/weight association: `PASS`

The block order was input RMSNorm, Q/K/V projection and bias, head reshape,
partial RoPE, causal GQA attention, output projection and residual,
post-attention RMSNorm, router, host Top-K control, selected routed experts,
shared expert, MoE combine, and final residual. Routing completed before any
selected expert payload acquisition.

## Sparse Acquisition And Residency

| Field | Measured value |
|---|---:|
| Persistent bytes read | `283519488` |
| Selected expert persistent bytes | `173299712` |
| Unselected expert persistent bytes read | `0` |
| Unselected expert device transfers | `0` |
| Host staging peak | `402653184` bytes |
| Converted host weight peak | `201326592` bytes |
| Host to device total | `1130603008` bytes |
| Weight H2D | bounded per demanded persistent tensor; included above |
| Activation H2D | input and graph inputs; included above |
| Control H2D | `2304` bytes |
| Device to host total | `466944` bytes |
| Activation/control D2H | bounded checkpoints and router control; included above |
| Peak logical device allocation | `202817536` bytes |
| Peak device weight allocation | `201326592` bytes |
| Device peak/model ratio | `0.001801805` |

Only the current demanded persistent tensor was uploaded. The complete model,
complete layer expert bank, and full host F32 model were never preloaded.
Activations remained as opaque device tensors between CUDA operations; the
only host activation movement was input upload and bounded qualification
readback.

## Parity And Timing

| Field | Value |
|---|---:|
| Output elements | `16384` |
| Maximum absolute error | `8.010864258e-05` |
| Maximum relative error | `3.848298940e-02` |
| RMS error | `2.504991820e-06` |
| NaN / Inf count | `0 / 0` |
| Maximum error stage | `moe_output`, `1.029968262e-04` absolute |
| Attention output error | `3.814697266e-06` absolute |
| Post-attention residual error | `2.288818359e-05` absolute |
| Router score error | `2.115964890e-06` absolute |
| Router corrected error | `1.907348633e-06` absolute |
| MoE output error | `1.029968262e-04` absolute |
| CPU reference block time | `29962.599 ms` |
| CUDA block wall time | `15369.377 ms` |
| Transfer time fraction | `0.0252` |
| Compute time fraction | `0.9748` |

The final post-cleanup-fix rerun reported persistent acquisition `14769.652 ms`,
FP8 conversion `14746.687 ms` inclusive of acquisition, H2D `386.363 ms`,
device compute `14982.736 ms`, and D2H `0.277 ms`. Timing is host/load-state
dependent and is recorded as qualification evidence, not a transport ceiling
or production performance claim.

The CPU path executed the same real layer after device cleanup and served as
the existing Step32E-B semantic reference. This is correctness evidence, not a
production performance claim.

## Cleanup And Failure Semantics

- Device live tensors after cleanup: `0`.
- Device live model/request bytes after cleanup: `0`.
- Device VRAM returned to its pre-run baseline after cleanup.
- Upload/download and operation failures return errors without silent CPU
  fallback.
- Cross-device tensor combinations are rejected; implicit peer copies are not
  performed.
- Default-stream work is synchronized before owned storage release.
- CUDA micro-tests cover upload/download, MatMul, RMSNorm, activation,
  attention, selected expert dispatch, cleanup, and device mismatch.
- Generic device contract tests cover mocked out-of-memory failure and shape
  validation.

## Source Independence And Boundary

```text
HF_ACCESS_DURING_EXECUTION=NO
SAFETENSORS_ACCESS_DURING_EXECUTION=NO
CONFIG_JSON_ACCESS_DURING_EXECUTION=NO
TOKENIZER_JSON_ACCESS_DURING_EXECUTION=NO
REMOTE_ACCESS_DURING_EXECUTION=NO
SOURCE_NAME_RUNTIME_AUTHORITY=NO
GGML_EXECUTION_USED=NO
LLAMA_CPP_EXECUTION_USED=NO
```

The CUDA runner discovered the real layer through semantic tensor identities
and the persisted sidecar catalog. The adapter did not parse source tensor
names or model metadata.

## Qualification Boundary

```text
DEVICE_BLOCK_CLASSIFICATION=PASS_FULL_DEVICE_BLOCK
FULL_STACK_GPU_EXECUTED=NO
GPU_PREFILL_EXECUTED=NO
GPU_DECODE_EXECUTED=NO
GPU_GENERATION_EXECUTED=NO
MULTI_GPU_EXECUTED=NO
```

This qualifies one complete real GLM block on the generic CUDA backend. It does
not establish full-model GPU inference, full-stack GPU residency, GPU
generation, multi-GPU execution, production GPU performance, or GGML parity.

## Verification

- `cargo test --manifest-path rust/Cargo.toml --workspace`: passed.
- `cargo test --manifest-path rust/Cargo.toml -p vbuf-ml`: passed as part of
  the workspace run.
- `cargo fmt --all -- --check`: passed.
- `cargo test -p vbuf-runtime --features cuda --test cuda -- --nocapture`:
  4 CUDA tests passed on device 0.
- CUDA backend build with `nvcc -ccbin /usr/bin/g++-13`: passed.
- Real CUDA block qualification and CPU comparison: passed.
- `python scripts/verify_portable_graph_neutrality.py`: passed with
  `FORBIDDEN_LEAKAGE_COUNT=0` and `CUDA_TYPE_LEAKAGE_COUNT=0`.
- Step32E-B semantics, FP8 provenance, expert-bank, semantic-sidecar, and
  Step32J CPU regressions: passed through the workspace and portable contract
  tests; Step32J was not rerun physically.
- `git diff --check`: passed.
- `ccc index`: required after the final edits.

## Next Step

Recommend exactly one next step: **Step32K-B: progressive 2 -> 4 -> 8 real
CUDA layers**. The one-block seam is qualified, but cross-layer activation and
KV/state residency have not been measured on CUDA. Do not implement 32K-B as
part of this qualification.
