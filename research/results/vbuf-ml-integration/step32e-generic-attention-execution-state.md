# Step 32E: Generic Attention Execution State

## Scope

This qualification defines the first backend-neutral attention execution
contract for vBuf-ML. It covers request-local key/value state, multi-token
causal prefill, ordinary multi-head attention (MHA), grouped-query attention
(GQA), explicit tensor geometry, stable softmax, and fail-closed validation.
It does not implement a full GLM block, decode scheduling, a persistent-format
change, or a llama.cpp loader path.

The immutable GLM payload and semantic sidecar were not modified. The generic
adapter consumes already-resolved validated tensor payloads; it does not
resolve sources, inspect model names, or choose a backend KV layout.

## Contract

The semantic graph carries Q, K, and V operation inputs, batch size, query and
KV head counts, head dimension, query length, current KV length, explicit
scale, causal/none mask, state-length position semantics, and an opaque
request state ID. Lowering requires three operands, valid nonzero geometry,
query-head divisibility by KV-head count, finite positive scale, a declared
request state, and supported mask/position semantics. MHA is represented by
equal query/KV head counts; GQA uses a larger query-head count with an integer
group ratio.

The C ABI v2 appends the attention descriptor to the existing operation
descriptor. ABI v1 remains fail-closed for attention rather than silently
guessing geometry or state behavior. The native adapter supports both value
operands and already-resolved rank-4 F32 tensor operands for Q/K/V.

State is request-local and opaque to the graph. It records binding identity,
capacity, current length, geometry, and owned key/value storage. Execution
checks capacity, exact prior state payload size, geometry, and the requested
new length before committing. Invalid input does not partially update state.
Reset clears all state; separate state objects are isolated.

The CPU reference implementation uses logical `[batch, sequence, head,
component]` indexing, maps query heads to KV heads by integer group, applies
causal visibility through the prior state length, and computes softmax by
subtracting the row maximum before exponentiation. Shape products and host
allocation sizes use checked arithmetic.

## Qualification

The Rust runtime tests pass for ABI layout, attention attribute preservation,
MHA/GQA lowering, missing state, invalid GQA geometry, and missing/invalid
attributes. The native contract test builds and passes ten deterministic GQA
prefill repetitions and an independent-reference comparison for both GQA and
MHA. It also verifies state length, state byte accounting, causal masking, and
scratch accounting.

Commands:

```text
cargo test -p vbuf-runtime
cmake --build /tmp/opencode/glmml-build --target vbuf_portable_graph_adapter_contract -j2
LD_LIBRARY_PATH=/tmp/opencode/glmml-build/ggml/src:/tmp/opencode/glmml-build/ggml/src/ggml-cpu /tmp/opencode/glmml-build/vbuf_portable_graph_adapter_contract
```

All commands completed successfully on the local diagnostic GGML checkout at
`a97123e497968f3440264c0464a7adc7c999c027`. This is a native adapter contract
test, not canonical GGML qualification. The canonical GGML pin
`2d191b5dee1a591c41ee8a653ce42bfcd9c8716d` was unavailable locally.

## Boundaries and Open Work

- The independent reference covers deterministic CPU F32 MHA/GQA prefill, not a full real-model attention projection path.
- Q/K/V in the test are resolved rank-4 F32 tensors; production projection outputs still need an end-to-end graph fixture.
- One-step decode, paged KV storage, device execution, real GLM attention materialization, and canonical GGML parity remain open.
- No payload, semantic sidecar, generic wire format, or llama.cpp loader ownership changed.
