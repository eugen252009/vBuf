# Step 32K-B Progressive CUDA Layers Checklist

Evidence: `research/results/vbuf-ml-integration/step32k-b-progressive-cuda-layers.md`.

- [x] Reproduce Step32K-A real layer-23 CUDA baseline.
- [x] Use one canonical consecutive range for all gates: layers `23..30`.
- [x] Use one deterministic `[1, 4, 4096]` F32 input and shared input hash.
- [x] Build every layer graph from canonical vBuf identities and the semantic sidecar.
- [x] Pass layer N's actual CUDA `DeviceTensor` to layer N+1.
- [x] Keep synthetic activation reinjection at zero.
- [x] Keep reference activation injection at zero.
- [x] Keep inter-layer device-to-host and host-to-device activation traffic at zero.
- [x] Release layer weights before the next layer retains them.
- [x] Release selected expert weights at each layer boundary.
- [x] Avoid whole-range and full expert-bank device preload.
- [x] Perform host Top-K only as explicit control work.
- [x] Transfer selected expert tensors only.
- [x] Record persistent traversal separately from peak device residency.
- [x] Record weight H2D separately from activation and router-control transfers.
- [x] Record device weight, activation, scratch, total, and physical VRAM telemetry.
- [x] Record bounded host staging and converted-weight peaks.
- [x] Compare every layer checkpoint to the independent CPU/reference path.
- [x] Compare selected expert membership and ordered decisions per layer.
- [x] Account for CUDA, host control, explicit no-op, and unclassified operations.
- [x] Qualify Gate 2: layers `23..24`.
- [x] Qualify Gate 4: layers `23..26`.
- [x] Qualify Gate 8: layers `23..30`.
- [x] Run partial second-layer failure cleanup test.
- [x] Run mocked multi-layer OOM/rejection cleanup test.
- [x] Verify every gate returns device ownership to zero.
- [x] Commit compact Gate 1/2/4/8 manifests without model or activation dumps.
- [x] Preserve the Step32K-A, Step32E-B, and Step32J CPU qualification boundaries.
- [ ] Qualify full 46-layer CUDA execution.
- [ ] Qualify GPU text prefill, retained-KV decode, or generation.
- [ ] Qualify multi-GPU, tensor parallelism, pipeline parallelism, or GGML parity.
