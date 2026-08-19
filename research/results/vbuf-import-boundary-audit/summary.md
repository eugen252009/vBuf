# vBuf-ML Portable Model Audit Summary

## Result

Current vBuf-ML storage semantics are sufficient for selected Qwen3 and
DeepSeek-derived tensor artifacts. Current execution semantics are not
sufficient for a neutral runtime because behavior, state, and dynamic routing
are not imported as portable semantic truth.

## Smallest Portable Boundary

The smallest justified boundary is a typed importer-owned program profile:

```text
TensorRefs and semantic bindings
typed operation/dataflow nodes
StateRefs and state transitions
dynamic selector-to-TensorRef alternatives
tokenizer pipeline semantics
aliases/shared-storage relations
```

The profile remains separate from vBuf v0.6. A runtime derives schedules,
source reads, residency, backend buffers, device placement, and lookup indexes.

## Evidence

- Qwen3.6 contains both attention and SSM tensor families, so tensor storage
  alone does not describe the hybrid schedule or recurrent state.
- PoC22 demonstrates generic bounded residency and dynamic expert acquisition,
  but its graph lowering is still a DeepSeek-specific importer/lowering.
- The native region audit identifies generic `StateRead/StateWrite`,
  `IndexedMatMul`, `TopK`, `Gather/Scatter`, and explicit region dependencies
  as sufficient concepts for the demonstrated MoE behavior.
- The v0.6 specification keeps the base contract limited to canonical blocks,
  checked ranges, and generic extensions; it does not authorize model graph or
  backend state in vBuf Core.

## Single Next Step

Define one typed program/state sidecar for the current Qwen3.6 artifact that can
express one attention layer, one SSM layer, one MoE routing region, and their
state transitions without vendor names or backend types. Do not implement
kernels in that step.

## P0.1 Falsification Result

Phi-4-mini-instruct was audited from the pinned Hugging Face revision
`cfbefacb99257ffa30c83adab238a50856ac3083` using configuration, tokenizer,
model-index, and safetensors-header evidence only. The dense model confirmed
that the Qwen-derived vocabulary needs generic storage views for fused QKV and
gate/up tensors, alias/shared-storage relations for tied embeddings, richer
partial-LongRoPE position attributes, and structured KV lifecycle fields.

No Phi-specific portable type, runtime vendor branch, execution tensor-name
parse, or backend state was needed. Dense layers naturally omit SSM and MoE.
The result is `ABSTRACTION_SURVIVES_WITH_GENERIC_REFINEMENTS`; one more model
family audit is recommended before PoC22 lowering. Full evidence is in
`p0-1-phi-abstraction-falsification.md` and the metadata-only prototype is
`phi4-mini-typed-program.json`.

## P0.2 Gemma Result

`google/gemma-3-270m-it` was audited at official revision
`ac82b4e820549b854eebf28ce6dedaf9fdfa17b3` using official Hub metadata/file
identity plus the public Gemma3 implementation; model weights were not
downloaded. Its 5-local/1-global attention pattern, local/global RoPE bases,
Gemma norm parameterization, hybrid KV state, and text-only/multimodal boundary
fit with generic attributes. The result is
`GEMMA_FITS_WITH_GENERIC_REFINEMENTS`.

## P1 Gate Result

The qualified DeepSeek-V2-Lite manifest was imported into the same generic
sidecar vocabulary. A minimal model-agnostic lowerer now maps the real layer-1
router prefix (`RmsNorm -> MatMul -> TopK`) into the architecture-neutral Rust
`ExecutionGraph` seed. Gate 2 still stops before execution because no adapter
connects that graph to the existing PoC22/GGML primitives; the active C++ path
remains DeepSeek-shaped. No vendor-specific workaround was added. DeepSeek
parity and Qwen lowering remain pending the backend adapter.

The existing runtime also still parses `blk.N.` names in `parse_layer`; this
pre-existing leakage remains to be removed or isolated behind importer-owned
region metadata before the runtime can satisfy the no-name-parsing invariant.

Evidence: `p0-2-gemma-falsification.md`, `portable-semantic-v1.md`,
`p1-deepseek-portable-lowering.md`, `p1-qwen-portable-lowering.md`, and
`deepseek2-typed-program.json`.
