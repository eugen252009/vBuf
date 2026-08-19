# P1 DeepSeek Portable Lowering

## 1. Objective

Implement and validate the smallest generic lowering boundary from imported
portable semantics toward the existing PoC22 execution target, without adding
a DeepSeek runtime branch or changing vBuf v0.6.

## 2. Gate 2 Starting State

DeepSeek-V2-Lite was already imported into the generic sidecar vocabulary from
the qualified IQ1_S manifest. The existing legacy PoC22 control is known-correct
on x86 for four positions, router selection, zero maximum logit error,
generated-token feedback, and zero-resource teardown.

## 3. Legacy PoC22 DeepSeek Call Graph

The legacy path is:

```text
autoregressive_poc22 main
  -> load_metadata / vbuf_ml_consumer_open
  -> make_plan / LayerPlan
  -> lookup token_embd, output_norm, output
  -> run_sequence
       -> attention_tensors / compute_token
       -> RuntimeStateSlot KV append/read
       -> run_layer / DeepSeek MoE router
       -> execute_selected / indexed expert graphs
       -> output graph / logits
  -> greedy token feedback / teardown
```

Relevant symbols are in `autoregressive_poc22.cpp`, `multi_layer_poc16.cpp`,
`attention_poc14.cpp`, `full_moe_layer_poc13.cpp`, and
`router_driven_moe_poc11.cpp`.

## 4. Existing Runtime Name Leakage

`rust/vbuf-runtime/src/lib.rs::parse_layer` parses `blk.N.` names while opening
models and groups tensor ranges. The legacy C++ path additionally resolves
roles through `lookup(metadata, "blk.1....")`, rewrites names in `make_plan`,
and fixes DeepSeek dimensions/expert counts in execution helpers.

These remain in the legacy control path. The new lowering path does not call
`parse_layer`, `lookup`, or inspect source names.

## 5. Selected Vertical Slice

```text
SELECTED_DEEPSEEK_REGION: layer.1.moe.router_prefix
WHY_SELECTED: real PoC13/P0C22 layer prefix; nontrivial norm, projection, and dynamic selection; small enough for isolated validation
PORTABLE_OPS_REQUIRED: RmsNorm, MatMul, TopK
PORTABLE_TENSORS_REQUIRED: layer.1.mlp.input_norm, layer.1.moe.router_weight
PORTABLE_STATE_REQUIRED: none; selector output is transient
```

The legacy equivalent is `full_moe_layer_poc13.cpp::run_layer`: normalize the
real layer input, compute router logits, select six experts, and derive routing
weights. The selected slice stops before expert materialization so it does not
pretend that the backend bridge is already generic.

## 6. Portable Semantic Inputs

The lowerer consumes `PortableProgram`, `PortableRegion`, `PortableOperation`,
`TensorBinding`, `SemanticTensorKey`, and generic operation attributes. Binding
resolution uses semantic keys to `TensorId`; source names are not present in
the lowering input. `StateRef` is part of the interface for later stateful
regions, but this slice requires no state.

## 7. Generic Lowering Boundary

Added `rust/vbuf-runtime/src/lowering.rs` with the narrow interface:

```text
PortableProgram + PortableRegion
    -> validate semantic bindings/attributes
    -> map portable operation kind
    -> ExecutionGraph
```

The existing graph seed is the current architecture-neutral target. The
lowerer maps `RmsNorm`, `MatMul`, and `TopK` to `RmsNorm`, `MatMul`, and
`TopKRouter`. It does not select a GGML type, buffer, device, schedule, or
source range plan.

## 8. Operation Mapping

| Portable construct | Semantic meaning | Existing PoC22 capability | Generic mapping possible? |
| --- | --- | --- | ---: |
| `RmsNorm` | normalize layer activation | `build_norm_graph` / `TensorWaveOpKind::RmsNorm` | Yes |
| `MatMul` | router projection | router graph / `TensorWaveOpKind::MulMat` | Yes |
| `TopK` | dynamic expert selector | `deterministic_top_k` / router selection | Yes |
| `IndexedMatMul` | selected expert projections | `execute_selected` expert graphs | Boundary identified; not yet bridged |
| `StateRead/Write` | KV/sequence state | `RuntimeStateSlot` | Boundary identified; not needed in slice |

This is semantic equivalence, not source-name equivalence. The Rust tests use
semantic names unrelated to `blk.N.` and still produce the same graph.

## 9. Tensor Binding

The slice resolves two semantic bindings to `TensorId(10)` and `TensorId(11)`.
No tensor-name parsing, GGUF architecture, or GGML type ID participates. The
current `ExecutionGraph` still stores a diagnostic semantic name and zero
physical metadata in this seed lowering; canonical payload/range resolution
remains the next backend adapter responsibility.

## 10. Layer/Region Identity

`PortableRegion.id` is an imported semantic region label. The lowerer does not
derive it from source names or use numeric layer parsing. A regression test
renames the region ID to an unrelated importer label and obtains identical
operation and acquisition structure.

## 11. `parse_layer` Removal From Portable Path

`parse_layer` was not removed because legacy `VBufRuntime::open` still uses it
for its convenience cache. The new `lower_region` path never calls it and
requires only explicit semantic bindings and region operations. The targeted
tests prove lowering succeeds with no `blk.N.` source strings. Full runtime
removal requires a later explicit region-indexed acquisition change and is not
part of this narrow lowerer.

## 12. Selected-Region Parity

The semantic lowerer passes structural tests, including binding and attribute
failure cases and renamed-source independence. The Rust FFI and C++ adapter
contract tests now cover attribute transfer, fail-closed operation handling,
and deterministic TopK behavior. No numeric real selected-region parity is
claimed because the fixture-backed native run remains unavailable.

```text
SELECTED_REGION_EXECUTION: NOT_REACHED
SELECTED_REGION_PARITY: NOT_RUN
SELECTED_REGION_MAX_NUMERIC_ERROR: NOT_MEASURED
```

## 13. Incremental Expansion

Expansion stops after the semantic graph-seed slice. Expert `IndexedMatMul`,
DeepSeek compressed-KV attention, state allocation, materialization, and output
logits remain unlowered. This prevents a second DeepSeek-specific graph from
being hidden behind a generic function name.

## 14. Router Parity

The legacy control has router parity evidence. The portable graph structurally
represents the same norm/projection/top-k sequence, but the portable path has
not executed router values.

## 15. Logits Parity

Not run. The selected slice ends before expert composition and output logits.

## 16. Token Parity

Not run for the portable path. Legacy four-token evidence remains preserved in
`research/results/vbuf-autoregressive-generation-poc22-x86/report.md`.

## 17. Four-Token Generation

Not attempted. Gate 2 requires the generic C++/GGML adapter and full region
coverage first.

## 18. Teardown

Not attempted for the portable path. No new materializer, residency, or state
ownership code was introduced.

## 19. Architecture Leakage Audit

The new `lowering.rs` contains no `DeepSeek`, `Qwen`, `Phi`, `Gemma`, `GGUF`,
`blk.`, model-family, or source-name behavior. Existing leakage remains in the
legacy C++ graph and in `vbuf-runtime::parse_layer`; both are explicitly
outside the new lowerer but must be isolated before full neutrality.

## 20. Gate 2 Result

`GATE_2_FFI_IMPLEMENTED_EXECUTION_BLOCKED`.

The semantic lowerer now retains selected-slice operation attributes, a
versioned Rust C ABI exports the graph, and a generic C++ adapter consumes the
ABI using ready payload views and existing backend helpers. The real parity run
is still blocked because the worktree has no runnable DeepSeek fixture and
linked qualification binary. Therefore numeric DeepSeek execution parity
cannot honestly be claimed.

## 21. Gate 3 Readiness

`QWEN_GATE_3_READY: NO`.

The next step is a real fixture-backed router-prefix parity run through the new
ABI. Qwen remains untouched.

## 22. Focused Validation

```text
RUST_FFI_TESTS: PASS
CPP_ADAPTER_SYNTAX: PASS
CPP_ADAPTER_CONTRACT_TEST: ADDED/CTest-registered, NOT RUN without configured pinned GGML build
SOURCE_LEAKAGE_SCAN: PASS, FORBIDDEN_LEAKAGE_COUNT=0
REAL_RMSNORM_PARITY: NOT_RUN
REAL_MATMUL_PARITY: NOT_RUN
REAL_TOPK_PARITY: NOT_RUN
```
