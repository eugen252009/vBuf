# Step 32F Progressive Execution Checklist

Evidence: `research/results/vbuf-ml-integration/step32f-progressive-real-multilayer-execution.md`

## Portable Execution

- [x] Semantic TensorId layer catalogs resolve without runtime source-name lookups.
- [x] Generic layer runner accepts an arbitrary consecutive layer window.
- [x] Partial RoPE, GQA attention, GLM router correction, Top-8 routing, routed experts, shared expert, and residual handoff execute through the portable graph.
- [x] Only the initial deterministic hidden state is synthetic.
- [x] Later layer inputs are actual preceding layer outputs.
- [x] No GGML, device backend, Safetensors, Hugging Face, or config access occurs during execution.

## Residency And Lifetime

- [x] FP8 payload remains persistent and read-only.
- [x] FP8-to-F32 conversion is layer-scoped and released at each boundary.
- [x] Selected expert acquisition is measured per layer.
- [x] Unselected expert tensor and byte touches are zero.
- [x] Activation values are released at layer boundaries.
- [x] Layer-local KV state uses a distinct opaque state identity.
- [x] Transient lease count remains zero in the qualification provider.
- [x] Request and layer execution state are reset and destroyed after teardown.
- [x] File-backed mmap/page-cache RSS is reported separately from internal converted-weight residency.

## Progressive Gates

- [x] Gate A: layers 23..24.
- [x] Gate B: layers 23..26.
- [x] Gate C: layers 23..30.
- [x] Independent per-layer numerical checkpoints pass.
- [x] Independent per-layer router IDs match exactly.
- [x] Working set remains bounded from 2 to 4 to 8 layers.
- [x] Three repeated 2-layer runs remain deterministic.
- [x] Sequential cross-run state isolation passes.
- [ ] Full transformer stack execution.
- [ ] Output head and logits qualification.
- [ ] Decode or token generation.
