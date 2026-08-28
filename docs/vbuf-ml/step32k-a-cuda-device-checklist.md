# Step 32K-A CUDA Device Block Checklist

Evidence: `research/results/vbuf-ml-integration/step32k-a-cuda-device-block.md`.

- [x] Audit portable graph, lowering, materialization, leases, attention state, and indexed expert dispatch.
- [x] Add backend-neutral device identity and capability contracts.
- [x] Add opaque backend-neutral device tensor ownership.
- [x] Add bounded device allocation, use, release, and cleanup accounting.
- [x] Keep CUDA runtime, cuBLAS, stream, and kernel types below the adapter boundary.
- [x] Build the CUDA adapter with CUDA 12.4 and GCC 13.
- [x] Qualify upload/download and explicit synchronization.
- [x] Qualify dense MatMul, RMSNorm, activation, residual, reshape, and RoPE operations.
- [x] Qualify causal GQA attention for the Step32K-A prefill-only geometry.
- [x] Keep persistent vBuf tensor identity separate from disposable device representation.
- [x] Use bounded FP8-to-F32 host staging without a model-sized host copy.
- [x] Execute router projection and sigmoid on CUDA.
- [x] Perform host Top-K only as explicit small control work.
- [x] Route before expert acquisition.
- [x] Transfer only selected real expert tensors.
- [x] Qualify selected expert gate/up/down execution on CUDA.
- [x] Keep activation values device-resident between device operations.
- [x] Instrument operation execution location and reject undeclared CPU fallback.
- [x] Verify device mismatch rejection and mocked allocation failure semantics.
- [x] Execute one complete real layer-23 GLM block.
- [x] Compare intermediate checkpoints and final output with the CPU/reference path.
- [x] Verify zero unselected expert device transfers.
- [x] Verify device-owned model/request allocations return to zero.
- [x] Verify source independence and CUDA neutrality.
- [x] Preserve Step32E-B and Step32J portable CPU qualification.
- [ ] Qualify retained-KV CUDA state reuse.
- [ ] Qualify progressive multi-layer CUDA execution.
- [ ] Qualify GPU prefill, decode, or generation.
- [ ] Qualify GGML parity or multi-GPU execution.
