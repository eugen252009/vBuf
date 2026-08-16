# Lookahead / Prefetch Planning POC4: x86 Qualification

## Result

`PASS` on x86-64. The planner observes graph state and produces bounded,
deterministic tensor-level demand without acquiring weights or performing I/O.
The existing POC3 executor remains the only component that acquires/releases
weights.

## Minimal API

Added:

- `TensorWaveGraphView`: read-only graph snapshot.
- `TensorWavePlannerState`: completed operations, available values, resident
  persistent tensor indices, and current active-byte/lease counts.
- `PrefetchCandidate`: tensor reference, consumer op, dependency distance, byte
  size, and residency flag.
- `PrefetchPlan PrefetchPlanner::plan(graph, state, max_distance, byte_budget)`.
- `TensorWavePlanningObserver`: synchronous observation callback invoked before
  each normal executor step.

The planner returns `visible_candidates` for all graph-visible persistent
requirements and `candidates` for nonresident candidates selected under the
byte budget. Resident entries remain observable but are excluded from transfer
planning.

## Dependency Distance

Distance is the maximum unresolved producer depth from the current graph state
to the consumer operation:

- `0`: all value inputs are available and the consumer is runnable.
- `1`: the consumer becomes runnable after one unresolved producer operation.
- Larger values recursively count the deepest unresolved value dependency.

Persistent inputs do not add dependency depth. This definition uses only value
producer edges and availability, never layer, region, or source tensor order.

## Graph And Ordering

The exact qualified POC3 graph was reused:

```text
ffn_inp -> rms_norm -> norm_out
                         |-> gate_matmul -> gate_out -\
                         |-> up_matmul   -> up_out   -> swiglu -> mul_out
                                                               -> down_matmul -> ffn_out
```

Candidate ordering is deterministic and generic:

1. dependency distance ascending
2. persistent tensor ID ascending
3. required operation ID lexicographically
4. persistent tensor index ascending

When applying the byte budget, candidates are considered in that order.
Already-resident candidates are skipped. An oversized candidate is skipped,
not allowed to overflow the remaining budget, and later smaller candidates are
still considered.

## Planning Trace

The full deterministic trace is in `planning-execution-trace.log`.

Important states:

| State | Runnable/visible demand | Down distance |
|---|---|---:|
| Initial | `norm` 0; `gate`, `up` 1; `down` 3 | 3 |
| After `rms_norm` | `gate`, `up` 0; `down` 2 | 2 |
| After `gate_matmul` | `up` 0; `down` 2 | 2 |
| After `up_matmul` | `down` 1 | 1 |
| After `swiglu` | `down` 0 | 0 |

Therefore `ffn_down.weight` first becomes visible at the initial graph state
with dependency distance 3, well before `down_matmul` is runnable. It is not
acquired by planning and is acquired only by the normal executor at
`down_matmul`.

The initial visible candidates were:

| Tensor | Required by | Distance | Bytes | Resident |
|---|---|---:|---:|---|
| `ffn_norm.weight` | `rms_norm` | 0 | 8,192 | no |
| `ffn_gate.weight` | `gate_matmul` | 1 | 4,377,600 | no |
| `ffn_up.weight` | `up_matmul` | 1 | 4,377,600 | no |
| `ffn_down.weight` | `down_matmul` | 3 | 12,607,488 | no |

The equal-distance gate/up candidates demonstrate independent branch demand.
Their ordering is determined by tensor ID, not FFN or model-specific rules.

## Byte Budget Qualification

Using the real FFN weight sizes:

| Budget | Selected candidates | Planned bytes |
|---:|---|---:|
| 0 | none | 0 |
| 8,191 | none | 0 |
| 8,192 | `ffn_norm.weight` | 8,192 |
| 4,385,792 | norm + gate | 4,385,792 |
| 21,370,880 | all four visible tensors | 21,370,880 |

The contract test additionally verifies oversized-candidate skipping and
continuation to later smaller candidates.

## No Acquisition Or I/O During Planning

For every planning callback, the trace records equal before/after values for:

- `active_weight_bytes`
- `active_lease_count`
- storage-provider call count

At the initial state all are zero. Later callbacks occur after prior normal
execution has completed and released its weights; the planner itself leaves
the counts unchanged. The planner accepts only graph/state data and has no
storage provider, file, network, or lease API.

## Execution Regression And Parity

Normal execution remained the POC3 path:

- JIT acquire before the current op
- execute synchronously
- release at last consumer
- zero persistent leases after graph teardown
- external `ffn_out` survives executor release

Final lifecycle metrics remained:

- total model bytes: 4,993,331,814
- total FFN weight bytes: 21,370,880
- peak active weight bytes: 12,607,488
- sum acquired weight bytes: 21,370,880

Intermediate activation parity:

- elements: 21,888
- max absolute error: `4.47035e-08`
- max relative error: `2.41219e-03`
- mean absolute error: `7.10122e-10`
- result: `PASS`

Final output parity:

- elements: 4,096
- max absolute error: `1.78814e-07`
- max relative error: `1.96622e-03`
- mean absolute error: `1.39676e-08`
- result: `PASS`

CTest passed 5/5, including the planner contract and dependency contract
tests. The complete output is in `ctest.log`.

## Architecture And Environment

Runtime search found no DeepSeek, Qwen, Llama, Phi, layer, region, or expert
logic in `vbuf_tensor_wave.*` or `vbuf_prefetch_planner.*`. Model tensor names
remain confined to the qualification fixture.

The planner is source-independent. vBuf, NVMe, RAM, LAN, device placement,
bandwidth, and latency are not represented in this POC.

ggml revision: `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`.

Model SHA-256:
`780a55b77d2730705a93622338d9747149624d72e558e26182176868210fafcc`.

RV2 remains blocked by the previously identified pinned-ggml/RVV FP16
toolchain issue. GCC/G++ 14.2 is present, but the native build fails before
the runtime executes because headers lack `vfloat16m2_t` and related
`__riscv_vle16_v_f16m2` intrinsics.

```text
RV2_BUILD: BLOCKED_BY_TOOLCHAIN
RV2_RUNTIME_RESULT: NOT_EXECUTED
```

## Required Classification

```text
GENERIC_LOOKAHEAD_PLANNER: PASS
TENSOR_LEVEL_PREFETCH_CANDIDATES: PASS
DEPENDENCY_DISTANCE: PASS
BYTE_BUDGET_ENFORCEMENT: PASS
PLANNER_PERFORMS_IO: NO
PLANNER_ACQUIRES_WEIGHT_LEASES: NO
EXECUTION_LIFETIME_REGRESSION: NO
REFERENCE_PARITY_X86: PASS
ARCHITECTURE_SPECIFIC_RUNTIME_LOGIC: NO
VBUF_FORMAT_CHANGE_REQUIRED: NO
RV2_RUNTIME_RESULT: NOT_EXECUTED
READY_FOR_ASYNC_MATERIALIZATION_POC: NO
```

## Files Changed

- `integrations/ggml/include/vbuf_tensor_wave.h`
- `integrations/ggml/src/vbuf_tensor_wave.cpp`
- `integrations/ggml/include/vbuf_prefetch_planner.h`
- `integrations/ggml/src/vbuf_prefetch_planner.cpp`
- `integrations/ggml/tools/tensor_wave_poc3.cpp`
- `integrations/ggml/tests/prefetch_planner_contract.cpp`
- `integrations/ggml/CMakeLists.txt`

## Evidence

- `planning-execution-trace.log`
- `ctest.log`
- `metadata.txt`
- `model-sha256.txt`
- `ffn_inp.f32`
- `ffn_swiglu.f32`
- `ffn_out.f32`
