# P1 DeepSeek Portable Backend Adapter

## 1. Objective

Connect the selected portable `RmsNorm -> MatMul -> TopK` region to the
existing GGML/TensorDependencyExecutor capabilities without moving semantic
authority into C++.

## 2. Why Rust FFI Was Selected

The Rust lowerer is the canonical semantic source. A versioned C ABI exposes
its lowered graph without exposing Rust layout or requiring JSON.

## 3. Semantic Authority Boundary

```text
PortableProgram -> Rust lower_region -> ExecutionGraph -> C ABI descriptors
                 -> generic C++ TensorDependencyExecutor/TopK capabilities
```

C++ dispatches operation kind and consumes resolved payloads. It does not
interpret model identity, source tensor names, or architecture metadata.

## 4. Missing Lowering Attributes

The graph seed previously discarded RMSNorm epsilon and TopK k. MatMul also
lacked an explicit weight operand/orientation contract. These are now typed
operation attributes.

## 5. epsilon Preservation

`OperationAttributes.epsilon` is copied into `ExecutionGraph::Operation` and
exported as an ABI `f32`. Missing epsilon is rejected by Rust lowering.

## 6. top_k Preservation

`OperationAttributes.top_k` is copied into the graph and exported as an ABI
`u32`. Missing k, ordering, or tie policy is rejected by Rust lowering.

## 7. TensorBinding / Payload Boundary

ExecutionGraph retains semantic TensorId bindings only. It does not duplicate
source offsets or lengths. The C++ adapter accepts a resolver-produced ready
`PersistentTensorRef` view; source resolution and materialization remain caller
and vBuf-ML responsibilities.

## 8. FFI ABI Design

`rust/vbuf-runtime/src/ffi.rs` defines explicit `repr(C)` descriptors and an
opaque graph handle. The matching declarations are in
`integrations/ggml/include/vbuf_portable_graph_ffi.h`.

## 9. Ownership and Lifetime

The Rust graph handle owns graph strings and descriptors until close. The
backend receives a retained execution lease and checks that each resolved
payload is ready before compute. C++ does not retain source pointers after the
operation call.

## 10. Error Propagation

The ABI returns status codes for invalid arguments, invalid UTF-8, invalid
graphs, unsupported operations, missing bindings, and invalid attributes.
Rust entry points catch panics and never unwind across the ABI.

## 11. Generic C++ Adapter

`integrations/ggml/tools/portable_graph_poc22_adapter.cpp` consumes only the
Rust graph descriptors. It has no model-family branch, source-name lookup, or
`parse_layer` call.

## 12. RmsNorm Mapping

RMSNorm maps to `TensorWaveOpKind::RmsNorm` with the ABI-supplied epsilon.
There is no C++ default epsilon.

## 13. RmsNorm Parity

Not executed. No runnable real DeepSeek fixture and linked PoC22 adapter
qualification binary is available in this worktree.

## 14. MatMul Mapping

MatMul maps to `TensorWaveOpKind::MulMat`. The Rust graph supplies the weight
operand and explicit non-transposed weight orientation required by this
selected router projection. Unsupported transposition is rejected.

## 15. Router Logit Parity

Not executed for the new path. Numeric error is therefore not measured.

## 16. TopK Mapping

TopK maps to the existing `deterministic_top_k` helper. Rust supplies k,
descending order, and lower-index tie semantics. No expert count is used to
derive k; expert count is only the score vector length passed to the helper.

## 17. TopK Parity

Not executed for the new path. Index and value parity are not measured.

## 18. Source-Name Independence

Rust lowering tests use semantic names unrelated to `blk.N.` and verify that
renaming the region does not alter the graph. The new C++ adapter contains no
source-name strings.

## 19. Architecture Leakage Audit

The new FFI and adapter files contain no DeepSeek, Qwen, Phi, Gemma, GGUF,
`blk.`, model-family, or `parse_layer` execution logic. Existing legacy PoC22
and runtime convenience parsing remain preserved outside this path.

## 20. Gate 2B Result

`GATE_2B_FFI_IMPLEMENTED_EXECUTION_BLOCKED`

Rust semantic completeness, ABI layout, fail-closed validation, and generic C++
adapter compilation are implemented. Real RMSNorm, router-logit, and TopK
parity cannot be claimed without the real fixture and linked execution run.

## 21. Gate 2C Readiness

`GATE_2C_READY: NO`. Gate 2C remains blocked until a real router-prefix run
crosses the ABI and matches the legacy oracle.

## Validation

```text
RUST_LOWERING_AUDITED: YES
RMSNORM_EPSILON_PRESERVED: YES
TOPK_K_PRESERVED: YES
MATMUL_SEMANTICS_COMPLETE: YES for selected non-transposed router projection
EXECUTION_GRAPH_ATTRIBUTES_COMPLETE_FOR_SELECTED_SLICE: YES
FFI_IMPLEMENTED: YES
FFI_ABI_VERSION: VBUF_PORTABLE_EXEC_ABI_V1
CPP_GENERIC_ADAPTER_IMPLEMENTED: YES
CPP_SYNTAX_CHECK: PASS
REAL_ROUTER_PREFIX_PARITY: NOT_RUN
```
