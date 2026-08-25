# Step 32E: Portable Graph Operation Contract

Date: 2026-08-25

## Scope

This qualification extends the existing model-agnostic vBuf runtime graph and
GGML adapter boundary. It does not add GLM tensor names, model detection,
loader callbacks, payload copies, or a dependency from vBuf-ML to
`llama_model_loader`.

The extension preserves the existing 64-byte ABI layout and adds generic graph
operation exposure for activation, indexed matmul/expert dispatch, attention,
and residual addition. `Silu` is an explicit activation attribute carried in a
previously reserved ABI byte. Activation and residual operations execute over
validated value buffers; tensor-backed matmul and indexed matmul use the
existing vBuf-owned GGML tensor-wave executor.

## Real GLM Mapping Boundary

The selected real artifact remains:

- Model: `zai-org/GLM-4.5-Air-FP8`
- Revision: `f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc`
- Layer candidate: `23`
- Hidden width: `4096`
- Attention geometry: `96` query heads, `8` KV heads, head dimension `128`
- Routed experts: `128`, active experts: `8`, shared experts: `1`

The layer inventory requires attention sequence/state geometry, RoPE
attributes, FP8-to-F32 materialization, BF16 bias handling, router score
correction, selected expert dispatch, and ordered residual composition. The
current generic ABI does not yet define those attention/state attributes, so
attention is exposed and rejected fail-closed with an explicit contract error.
No real GLM block was executed.

## Qualification

Passed:

- `cargo fmt --all -- --check`
- `cargo test --workspace`
- Local GGML build of `vbuf_portable_graph_adapter_contract`
- Direct adapter contract execution covering TopK, `Silu`, residual addition,
  and attention fail-closed behavior
- `git diff --check`

The native build used the local diagnostic GGML revision
`a97123e497968f3440264c0464a7adc7c999c027`. It is not the pinned canonical
revision. The repository's canonical GGML target remains
`2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`, which is unavailable locally.

## Boundary

This is a generic operation-contract qualification, not real-model backend
qualification. It provides no evidence of GLM logits, numerical parity,
generation, full-block execution, or canonical GGML compatibility. The next
generic step is to define a model-independent attention/state descriptor and
backend implementation, then repeat the qualification with the pinned GGML
revision before claiming real GLM execution.
