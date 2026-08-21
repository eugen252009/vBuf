# Step 31M GGML `mul_mat_id` Microqualification

Date: 2026-08-21
Starting commit: `0292286`
Status: **HOST-NATIVE MICROQUALIFICATION COMPLETE; NO RUNTIME CHANGE**

## Question and Scope

Step 31L selected a separately authorized host/native microqualification of raw
GGML `ggml_mul_mat_id` using the actual DeepSeek-V2-Lite IQ2_XXS routed-expert
geometry. This experiment compares one graph containing six ordinary rank-2
`ggml_mul_mat` nodes with one graph containing a single `ggml_mul_mat_id` node.

The probe is opt-in and diagnostic only. It does not modify the vBuf TensorWave
adapter, add a grouped operation kind, change source resolution, alter
materialization or residency, change TopK or reduction order, or qualify an
Android backend path.

## Evidence Classification

- **PHYSICALLY_MEASURED:** native CPU execution of the actual GGUF tensor bytes
  for IQ2_XXS gate/up and IQ4_NL down weights.
- **CODE_AUDITED:** the baseline uses six rank-2 views into one `[K, N, 64]`
  expert tensor; the grouped path uses raw GGML `ggml_mul_mat_id` with one
  broadcast `[K, 1, 1]` activation tensor and six I32 IDs.
- **DERIVED:** speedup ratios and working-set/output byte counts.
- **NOT_QUALIFIED:** Android or TensorWave backend behavior, full routed-layer
  scheduling, residency impact, and end-to-end model performance.

## Artifact and Geometry

```text
SOURCE: /tmp/opencode/deepseek-v2-lite-imat/DeepSeek-V2-Lite.IQ2_XXS.gguf
SOURCE_SHA256: 3b7da33584bebf89afcdbdd2e7a8e3e47e11092971559371f13b510f475e3c0c
SOURCE_SIZE_BYTES: 5640619552
GGML_COMMIT: 2d191b5dee1a591c41ee8a653ce42bfcd9c8716d
EXPERT_COUNT: 64
SELECTED_IDS: [3, 11, 17, 29, 41, 53]
SELECTED_COUNT: 6
INPUT_TOKENS: 1
```

| Case | Weight shape | Type | GGUF source offset | Payload bytes | Grouped output bytes |
|---|---|---|---:|---:|---:|
| Gate/up | `[2048, 1408, 64]` | IQ2_XXS | 348,536,352 | 47,579,136 | 33,792 |
| Down | `[1408, 2048, 64]` | IQ4_NL | 244,727,328 | 103,809,024 | 49,152 |

The source ranges and shapes are the real layer-1 expert tensors recorded in
the checked-in DeepSeek manifest. The probe maps the GGUF read-only and binds
the full rank-3 tensors as borrowed CPU backend buffers; it does not generate
synthetic quantized values or repack the payload.

## Compared Graphs

The ordinary baseline has six independent operations, one per TopK rank:

```text
expert_view(rank 0) -> ggml_mul_mat(view, input) -> output[0]
...
expert_view(rank 5) -> ggml_mul_mat(view, input) -> output[5]
```

The grouped graph has one indirect operation:

```text
ggml_mul_mat_id(full_experts, input[K,1,1], ids[6,1])
    -> output[N,6,1]
```

`input[K,1,1]` is broadcast across the six selected IDs. The output is compared
rank-by-rank against the six baseline outputs, so backend expert grouping does
not become reduction-order grouping. The final weighted TopK merge is not part
of this kernel microqualification.

## Measurement Method

```text
BACKEND: raw GGML CPU backend
WARMUP: 2 graph executions per case and path
MEASURED_ITERATIONS: 100 per case and path
TIMING: synchronous graph-compute wall time averaged over measured iterations
THREAD_COUNTS: 1 and 8
BASELINE: six ggml_mul_mat nodes in one graph
GROUPED: one ggml_mul_mat_id node in one graph
```

The two paths use the same mapped weight ranges, activation values, selected
IDs, backend, and graph-compute timing boundary. This isolates raw host
operator behavior; it does not model the current TensorWave per-operation
submission boundary or Android transport/materialization overhead.

## Results

### One CPU thread

| Case | Separate average | Grouped average | Grouped / separate | Grouped speedup | Max abs diff | Parity |
|---|---:|---:|---:|---:|---:|---|
| Gate/up IQ2_XXS | 1.139 ms | 1.073 ms | 0.941x | 1.062x | 0 | PASS |
| Down IQ4_NL | 0.790 ms | 0.790 ms | 1.000x | 1.000x | 0 | PASS |

### Eight CPU threads

| Case | Separate average | Grouped average | Grouped / separate | Grouped speedup | Max abs diff | Parity |
|---|---:|---:|---:|---:|---:|---|
| Gate/up IQ2_XXS | 0.438 ms | 0.458 ms | 1.046x | 0.956x | 0 | PASS |
| Down IQ4_NL | 0.278 ms | 0.253 ms | 0.910x | 1.098x | 0 | PASS |

All compared output elements were finite and bit-identical in this run:

```text
GATE_UP_MAX_ABS_DIFF: 0
GATE_UP_MAX_REL_DIFF: 0
DOWN_MAX_ABS_DIFF: 0
DOWN_MAX_REL_DIFF: 0
```

## Interpretation

The host CPU result proves that the pinned raw GGML `ggml_mul_mat_id` path can
consume the actual IQ2_XXS and IQ4_NL expert encodings at the required geometry
and preserve the six rank output slots. It does not prove a stable speedup:

- Gate/up was 6.2% faster at one thread but 4.4% slower at eight threads.
- Down was neutral at one thread and 9.8% faster at eight threads.
- The grouped path's full rank-3 tensor descriptor remains approximately 45.4
  MiB for gate/up and 99.0 MiB for down before backend temporaries.
- The probe does not measure full-tensor materialization, TensorWave readiness,
  residency pressure, or Android device execution.

The result therefore qualifies **capability and output parity**, not a production
optimization decision. The host measurements are evidence for a future backend
qualification only. They do not justify adding `MUL_MAT_ID` to the vBuf adapter
or moving grouped scheduling ownership into GGML.

## Implementation Boundary

```text
NEW_OPT_IN_CMAKE_OPTION: VBUF_BUILD_MOE_GROUPING_MICROQUALIFICATION
NEW_PROBE: integrations/ggml/tests/moe_grouping_microqualification.cpp
REGISTERED_AS_CTEST: NO
RUNTIME_SOURCE_CHANGED: NO
TENSORWAVE_ADAPTER_CHANGED: NO
GGML_SOURCE_CHANGED: NO
MODEL_FORMAT_CHANGED: NO
RESIDENCY_CHANGED: NO
TOPK_CHANGED: NO
REDUCTION_ORDER_CHANGED: NO
ANDROID_QUALIFICATION: NOT_PERFORMED
```

## Verification

```text
OPT_IN_NATIVE_TARGET_BUILD: PASS
NATIVE_CTEST: 20/20 PASS
RUST_CARGO_TEST_WORKSPACE: PASS (from rust/)
MICROQUALIFICATION_THREADS_1: PASS; both cases parity PASS
MICROQUALIFICATION_THREADS_8: PASS; both cases parity PASS
GIT_DIFF_CHECK: PASS
```

The next decision boundary is an explicitly authorized Android/backend
qualification if grouped execution is still considered worthwhile. Until that
qualification exists, the current vBuf TensorWave path remains unchanged and
the host result is retained as bounded raw-GGML evidence only.
