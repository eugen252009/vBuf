# P1 DeepSeek Portable Real Execution

Date: 2026-08-20

## 1. Objective

Close Gate 2B by executing the real selected DeepSeek router prefix through the
portable path and comparing its intermediate outputs with the existing legacy
PoC22/GGML path.

Selected region:

```text
layer.1.moe.router_prefix
    RmsNorm -> MatMul -> TopK
```

## 2. Starting Integrated Baseline

```text
BRANCH: vbuf-ml
STARTING_HEAD: 6c6570d
ANDROID_QUALIFICATION_COMMIT: 6c6570d
MERGE_COMMIT: ee57377
POST_MERGE_ANDROID_D2_3: PASS
GATE_2B_STARTING_RESULT: GATE_2B_FFI_IMPLEMENTED_EXECUTION_BLOCKED
```

The Android qualification report was committed separately before this work.
No Android rerun was required because this work is opt-in qualification wiring
and does not change production runtime ownership or behavior.

## 3. Existing Gate 2B Implementation

The merged branch already provided:

- model-neutral Rust `PortableProgram`/`PortableRegion` lowering;
- `ExecutionGraph` and versioned `VBUF_PORTABLE_EXEC_ABI_V1` descriptors;
- generic C++ `execute_portable_graph` adapter;
- ready-payload-only `PersistentTensorRef` input and retained lease boundary;
- deterministic generic TopK semantics.

The missing seam was a real-fixture caller connecting those pieces to the
existing DeepSeek/PoC22 control path.

## 4. Real Fixture Audit

```text
DEEPSEEK_IQ1_S_FIXTURE_AVAILABLE: YES, research-models/DeepSeek-V2-Lite.IQ1_S.vbuf
DEEPSEEK_IQ2_XXS_FIXTURE_AVAILABLE: YES, /tmp/opencode/deepseek-v2-lite-imat/
PINNED_GGML_BUILD_AVAILABLE: YES
LEGACY_POC22_BINARY_AVAILABLE: YES
LEGACY_ROUTER_PREFIX_ENTRY_AVAILABLE: YES via existing RouterGraph/full_moe code
REAL_HIDDEN_STATE_FIXTURE_AVAILABLE: NO
REAL_ROUTER_REFERENCE_AVAILABLE: YES, existing legacy RouterGraph/reference_scores path
PORTABLE_ADAPTER_LINKABLE: YES
RUST_FFI_LINKABLE_FROM_QUALIFICATION_BINARY: YES
MISSING_EXECUTION_SEAM: closed by opt-in vbuf_gate2b_real_deepseek target
```

The selected real model was the existing DeepSeek-V2-Lite IQ2_XXS semantic
bootstrap and external vBuf payload. No model or payload was added to the
repository.

```text
MODEL: DeepSeek-V2-Lite
QUANTIZATION: IQ2_XXS
MODEL_ARTIFACT: /tmp/opencode/deepseek-v2-lite-imat/DeepSeek-V2-Lite.IQ2_XXS.vbuf
SEMANTIC_ARTIFACT: /tmp/opencode/deepseek-v2-lite-imat/DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf
SELECTED_LAYER: layer.1
SELECTED_OPS: RmsNorm -> MatMul -> TopK
```

## 5. Qualification Harness

Added an explicit opt-in native target:

```text
integrations/ggml/tests/portable_deepseek_router_prefix.cpp
vbuf_gate2b_real_deepseek
```

The target is enabled only with:

```text
-DVBUF_BUILD_REAL_GATE2B=ON
-DVBUF_ML_LIBRARY=<libvbuf_ml.so>
-DVBUF_RUNTIME_LIBRARY=<libvbuf_runtime.so>
```

Invocation used:

```text
LD_LIBRARY_PATH=/tmp/opencode/vbuf-gate2b-real-build/ggml/src:/home/eugen/projekte/vBuf/rust/target/release \
  /tmp/opencode/vbuf-gate2b-real-build/vbuf_gate2b_real_deepseek \
  /tmp/opencode/deepseek-v2-lite-imat/DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf \
  http://127.0.0.1:18124
```

The harness is qualification-only. It does not add a loader, scheduler,
source policy, backend abstraction, or production runtime path.

## 6. Legacy Reference Path

The legacy side uses the existing `full_moe_layer_poc13.cpp` library lineage:

```text
build_norm_graph
    -> TensorDependencyExecutor RmsNorm
build_router_graph
    -> TensorDependencyExecutor MatMul
deterministic_top_k
    -> legacy router selection
```

Both legacy and portable sides use separate canonical range materializers over
the same external IQ2_XXS source endpoint. The legacy side uses the existing
`OffsetMaterializer` readiness path.

## 7. Portable Execution Path

```text
fixture semantic metadata
    -> test-harness TensorBinding selection
    -> PortableProgram / PortableRegion descriptors
    -> rust lower_region
    -> ExecutionGraph
    -> VBUF_PORTABLE_EXEC_ABI_V1
    -> execute_portable_graph
    -> ready PersistentTensorRef resolver
    -> TensorDependencyExecutor / GGML
    -> deterministic_top_k
```

The generic adapter receives only ready payload views and the shared retained
lease. It has no materializer, source, endpoint, path, model-family, or source
name input.

## 8. TensorBinding Resolution

The fixture harness maps the semantic imported bindings to the actual validated
fixture records:

```text
layer.1.mlp.input_norm  -> blk.1.ffn_norm.weight
layer.1.moe.router_weight -> blk.1.ffn_gate_inp.weight
```

The mapping is fixture/oracle code only. The portable graph carries the
semantic keys and actual canonical Tensor IDs; the generic adapter never reads
the fixture names.

```text
REAL_TENSOR_BINDING_RESOLUTION: PASS
NORM_PHYSICAL_SHAPE: 2048
ROUTER_PHYSICAL_SHAPE: 2048 x 64
ROUTER_REPRESENTATION: F32
ROUTER_PHYSICAL_BYTES: 524288
```

## 9. Ready Payload / Lease Boundary

Both real persistent tensors were requested through
`ResidentTensorMaterializer`, waited to `Ready`, obtained, and held in a
lease-owned bundle until portable execution completed.

```text
READY_PAYLOAD_RESOLUTION: PASS
REAL_PAYLOAD_POINTER_CROSSED_FFI: YES
REAL_LEASE_RETAINED_DURING_COMPUTE: YES
REAL_PAYLOAD_STILL_VALID_AFTER_RMSNORM: YES
REAL_PAYLOAD_STILL_VALID_AFTER_MATMUL: YES
DANGLING_POINTER_OBSERVED: NO
```

## 10. Identical Input Proof

No captured real hidden-state fixture exists at this boundary. The existing
legacy router-prefix qualification control supplies the deterministic input,
and the harness passes that same input vector to both paths:

```text
INPUT_SOURCE: existing legacy PoC13 deterministic one-hot input
INPUT_SHAPE: 2048 x 1
INPUT_ELEMENT_TYPE: f32
INPUT_ELEMENT_COUNT: 2048
INPUT_IDENTITY_VERIFIED: YES
```

This is real DeepSeek tensor/backend execution with the existing qualification
input, not a random-weight or mock-tensor test.

## 11. RMSNorm Execution and Parity

The portable graph supplies epsilon explicitly:

```text
RMSNORM_EPSILON: 1e-6
RMSNORM_EXECUTION: PASS
RMSNORM_PARITY: EXACT
RMSNORM_MAX_ABSOLUTE_ERROR: 0
RMSNORM_MAX_RELATIVE_ERROR: 0
```

No epsilon default is supplied by the C++ adapter.

## 12. MatMul Execution and Router-Logit Parity

The portable operation supplies RHS weight operand and non-transposed weight
orientation through the ABI attributes:

```text
MATMUL_WEIGHT_ORIENTATION: RHS, non-transposed
MATMUL_EXECUTION: PASS
ROUTER_LOGITS_PARITY: EXACT
ROUTER_LOGITS_MAX_ABSOLUTE_ERROR: 0
ROUTER_LOGITS_MAX_RELATIVE_ERROR: 0
```

The legacy and portable router output counts were both 64.

## 13. TopK Execution and Parity

```text
TOPK_K: 6
TOPK_ORDERING: descending
TOPK_TIE_SEMANTICS: lower index
TOPK_EXECUTION: PASS
TOPK_INDEX_PARITY: PASS
TOPK_VALUE_PARITY: EXACT
TOPK_MAX_VALUE_ERROR: 0
TOPK_IDS: 37,21,31,54,6,33
```

TopK `k` is supplied by the portable graph and is not derived from model
identity or expert count.

## 14. Numeric Error Analysis

All three compared boundaries were exact for this run:

```text
RMSNorm: EXACT
MatMul: EXACT
TopK values: EXACT
TopK indices: identical
```

No tolerance widening was required. The pass threshold remains `1e-5` for
numeric comparisons and exact index equality for TopK.

## 15. Source/Model Neutrality Audit

The fixture harness is permitted to select DeepSeek names. The generic
portable implementation remains model-neutral:

```text
MODEL_FAMILY_READ_BY_GENERIC_PATH: NO
SOURCE_TENSOR_NAMES_READ_BY_GENERIC_PATH: NO
BLK_N_PARSED_BY_GENERIC_PATH: NO
PARSE_LAYER_USED_BY_GENERIC_PATH: NO
GGUF_ARCHITECTURE_READ_BY_GENERIC_PATH: NO
FORBIDDEN_PORTABLE_PATH_LEAKAGE_COUNT: 0
```

The existing `scripts/verify_portable_graph_neutrality.py` passed. The legacy
PoC22 path retains its documented model-specific lookup behavior; that is the
reference side, not the portable generic path.

## 16. Legacy Runtime Regression

The legacy norm/router/TopK control executed successfully on the same real
DeepSeek IQ2_XXS fixture and input. The prior physical Pixel D2.3 result is
unchanged.

```text
LEGACY_POC22_REGRESSION: PASS for selected router prefix
D2_3_RUNTIME_REGRESSION: NO
PRODUCTION_RUNTIME_CHANGED: NO
PORTABLE_RUNTIME_CHANGED: NO
BACKEND_ADAPTER_CHANGED: NO
ANDROID_RERUN_REQUIRED: NO
ANDROID_RERUN_PERFORMED: NO
```

## 17. Regression Suites

```text
CARGO_TEST_WORKSPACE: PASS
RUST_FFI_TESTS: PASS
NATIVE_GGML_BUILD: PASS
CTEST: PASS 19/19
PORTABLE_ADAPTER_CONTRACT_TEST: PASS
NEUTRALITY_GUARD: PASS
POC22_TESTS: PASS, legacy selected slice and existing targets built
JSON_VALIDATION: PASS, 979 repository JSON files; ts/tsconfig.json is JSONC
CCC_INDEX: PASS
GIT_DIFF_CHECK: PASS
CARGO_FMT_CHECK: FAIL on pre-existing unrelated vbuf-ml formatting
```

No unrelated formatting changes were introduced.

## 18. Gate 2B Verdict

```text
GATE_2B_RESULT: GATE_2B_PASS
```

The real DeepSeek `layer.1.moe.router_prefix` crossed the complete portable
chain, used real canonical tensors and ready payloads, retained leases through
backend access, matched the legacy path exactly at RMSNorm, router logits, and
TopK, and produced zero forbidden generic-path architecture leakage.

## 19. Gate 2C Readiness

```text
GATE_2C_READY: YES
```

This only authorizes a future incremental expansion. It does not implement
Gate 2C in this task.

## 20. Recommended Next Step

Incrementally expand the same portable path toward full DeepSeek execution,
preserving intermediate parity at each newly added semantic region. Do not
start that expansion, D3, or Qwen Gate 3 as part of this qualification.

```text
D3_READY: YES
QWEN_GATE_3_READY: NO
```

## 21. Git Result

```text
IMPLEMENTATION_COMMIT: pending
PUSH_PERFORMED: NO
```

The intended Gate 2B changes are limited to the opt-in CMake target, the
qualification harness, and this report. No model, APK, native binary, or log
dump is tracked.
