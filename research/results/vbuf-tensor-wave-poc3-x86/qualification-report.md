# Tensor Wave POC3: x86 Qualification

## Result

`PASS` on x86-64. The dependency graph, not a Region or Layer boundary,
determines persistent tensor acquisition and release.

The runtime implementation is `TensorDependencyExecutor`. It derives
consumer counts from operation input edges, selects runnable operations in
topological order, acquires only persistent inputs needed by the selected
operation, and releases them when the derived remaining-consumer count reaches
zero. `ExecutionRegion` remains unchanged as the POC2 qualification scope.

## Generic Graph

```text
ffn_inp -> rms_norm -> norm_out
                         |-> gate_matmul -> gate_out -\
                         |-> up_matmul   -> up_out   -> swiglu -> mul_out
                                                               -> down_matmul -> ffn_out
```

Operation semantics are the already-qualified direct ggml operations:

- `rms_norm`: `mul(rms_norm(ffn_inp, 1.0e-6), ffn_norm.weight)`
- `gate_matmul`: `mul_mat(ffn_gate.weight, norm_out)`
- `up_matmul`: `mul_mat(ffn_up.weight, norm_out)`
- `swiglu`: `swiglu_split(gate_out, up_out)`
- `down_matmul`: `mul_mat(ffn_down.weight, mul_out)`

Persistent tensor consumers are derived as follows:

| Persistent tensor | Consumer count | First consumer | Last consumer |
|---|---:|---|---|
| `blk.0.ffn_norm.weight` | 1 | `rms_norm` | `rms_norm` |
| `blk.0.ffn_gate.weight` | 1 | `gate_matmul` | `gate_matmul` |
| `blk.0.ffn_up.weight` | 1 | `up_matmul` | `up_matmul` |
| `blk.0.ffn_down.weight` | 1 | `down_matmul` | `down_matmul` |

Runtime value consumers:

| Value | Consumer count |
|---|---:|
| `ffn_inp` | 1 |
| `norm_out` | 2 |
| `gate_out` | 1 |
| `up_out` | 1 |
| `mul_out` | 1 |
| `ffn_out` | 0, external output |

## Execution Trace

The complete trace is in `acquisition-trace.log`. The deterministic order
was:

1. `rms_norm`
2. `gate_matmul`
3. `up_matmul`
4. `swiglu`
5. `down_matmul`

The important residency timeline from the trace is:

| Step | Operation | Acquired | Active weight bytes during op | Released |
|---:|---|---|---:|---|
| 1 | `rms_norm` | `ffn_norm.weight` | 8,192 | `ffn_norm.weight` |
| 2 | `gate_matmul` | `ffn_gate.weight` | 4,377,600 | `ffn_gate.weight` |
| 3 | `up_matmul` | `ffn_up.weight` | 4,377,600 | `ffn_up.weight` |
| 4 | `swiglu` | none | 0 | none |
| 5 | `down_matmul` | `ffn_down.weight` | 12,607,488 | `ffn_down.weight` |

`ffn_down.weight` is acquired only at step 5, when `down_matmul` becomes
runnable. No persistent weight survives its last consumer. The graph reports
zero persistent leases after teardown. Internal values are also released at
their last consumer; `ffn_out` is retained as an external output and remains
valid after the executor object is reset.

## Working Set

- Model artifact: `DeepSeek-V2-Lite.IQ1_S.vbuf`
- Model bytes: 4,993,331,814
- Total FFN weight bytes: 21,370,880
- Peak simultaneously active persistent weight bytes: 12,607,488
- Sum of acquired weight bytes: 21,370,880
- Full-model weight allocation: none
- CUDA: disabled
- `CPU_REPACK`: `OFF`
- ggml revision: `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`
- Model SHA-256: `780a55b77d2730705a93622338d9747149624d72e558e26182176868210fafcc`

The peak is determined by dependency overlap and is lower than the 21,370,880
bytes effectively scoped together by POC1/POC2.

## Parity

Intermediate activation parity against `ffn_swiglu-0`:

- elements: 21,888
- max absolute error: `4.47035e-08`
- max relative error: `2.41219e-03`
- mean absolute error: `7.10122e-10`
- result: `PASS` under the established tolerance

Final output parity against `ffn_out-0`:

- elements: 4,096
- max absolute error: `1.78814e-07`
- max relative error: `1.96622e-03`
- mean absolute error: `1.39676e-08`
- result: `PASS`

The native substrate CTest suite passed 4/4, including the dependency-contract
guard that rejects a persistent tensor with no real consumer; output is preserved in
`ctest.log`.

## RV2

`gcc-14` and `g++-14` are available on RV2 as GCC 14.2.0. The native pinned
ggml substrate rebuild was attempted with that toolchain, but compilation
failed before the POC3 executable was built because the compiler headers do
not provide the RVV FP16 intrinsic types/functions used by pinned ggml,
including `vfloat16m2_t` and `__riscv_vle16_v_f16m2`.

Classification:

```text
RV2_BUILD: BLOCKED_BY_TOOLCHAIN
RV2_RUNTIME_RESULT: NOT_EXECUTED
```

No ggml patch or runtime weakening was made.

## Architecture Search

The generic runtime files `integrations/ggml/include/vbuf_tensor_wave.h` and
`integrations/ggml/src/vbuf_tensor_wave.cpp` contain no DeepSeek, Qwen, Llama,
Phi, layer, or expert-specific logic. Model tensor names occur only in the
qualification fixture `tensor_wave_poc3.cpp`, where source-model lowering is
explicitly allowed.

Layer and Region are not referenced by `TensorDependencyExecutor` and own no
leases. Persistent residency is represented only by graph persistent inputs
and runtime resident entries.

## Required Classification

```text
TENSOR_DEPENDENCY_EXECUTION: PASS
JUST_IN_TIME_WEIGHT_ACQUISITION: PASS
LAST_CONSUMER_WEIGHT_RELEASE: PASS
EAGER_FULL_REGION_ACQUISITION: NO
FULL_MODEL_WEIGHT_ALLOCATION: NO
PEAK_ACTIVE_WEIGHT_BYTES: 12607488
TOTAL_FFN_WEIGHT_BYTES: 21370880
EXTERNAL_OUTPUT_SURVIVES_GRAPH_RELEASE: PASS
REFERENCE_PARITY_X86: PASS
RV2_RUNTIME_RESULT: NOT_EXECUTED
ARCHITECTURE_SPECIFIC_RUNTIME_LOGIC: NO
VBUF_FORMAT_CHANGE_REQUIRED: NO
READY_FOR_ASYNC_PREFETCH_POC: NO
```

## Files Changed

- `integrations/ggml/include/vbuf_tensor_wave.h`
- `integrations/ggml/src/vbuf_tensor_wave.cpp`
- `integrations/ggml/tools/tensor_wave_poc3.cpp`
- `integrations/ggml/tests/tensor_wave_dependency_contract.cpp`
- `integrations/ggml/CMakeLists.txt`

## Evidence

- `acquisition-trace.log`
- `ctest.log`
- `metadata.txt`
- `model-sha256.txt`
- `ffn_inp.f32`
- `ffn_swiglu.f32`
- `ffn_out.f32`
