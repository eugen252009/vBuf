# Async Materialization / Wave Loading POC5: x86 Qualification

## Result

`PASS` on x86-64. The POC5 path performs real tensor-level asynchronous
materialization while the unchanged synchronous executor computes other ops.
The planner remains pure and the POC3 JIT/lifetime path remains the fallback
and correctness path.

## API And State Machine

Added `TensorMaterializer` with the minimal operations:

- `request(tensor_ref, tensor, byte_budget)`
- `state(tensor_ref)`
- `wait(tensor_ref)`
- `obtain_ready_tensor(tensor_ref)`
- `release(tensor_ref)`
- byte and trace inspection

States are:

```text
NOT_REQUESTED -> IN_FLIGHT -> READY -> RELEASED
                         \-> FAILED
```

The materialization state and owned buffer belong to the persistent tensor
reference. No Layer or Region object participates in ownership.

`TensorDependencyExecutor` behavior is unchanged when no materializer is
provided. With a materializer, a required tensor is consumed from READY,
waited for when IN_FLIGHT, or acquired through the existing borrowed vBuf JIT
path when NOT_REQUESTED or FAILED.

## Source And Range

The qualification source is `LocalVbufRangeMaterializer`. It copies exactly
the validated `VbufTensorView.payload` range for one persistent tensor into an
owned, 64-byte-aligned RAM buffer. For the selected tensor:

- tensor: `blk.0.ffn_down.weight`
- bytes: 12,607,488
- source: mapped vBuf tensor range
- destination: owned RAM buffer
- unrelated tensors copied: none
- vBuf format changed: no

The executor is source-independent and sees only the `TensorMaterializer`
interface.

## Selected Prefetch

The existing POC4 planner was used with the same graph and horizon. The
qualification policy selects the farthest visible nonresident candidate to
make the first real overlap unambiguous. This is a generic dependency-distance
policy, not an FFN or tensor-name rule.

`ffn_down.weight` was requested at the initial state:

```text
planner distance: 3
request budget: 12607488 bytes
request accepted: yes
```

It then progressed through distance 2, distance 1, and distance 0 while the
normal executor advanced through RMSNorm, gate matmul, up matmul, and SwiGLU.

## Timing Trace

The full timestamped trace is in `prefetch.log`.

Measured timestamps from the same run, in monotonic nanoseconds:

| Event | Timestamp |
|---|---:|
| prefetch request / IN_FLIGHT | 15,055,207,131,791 |
| `rms_norm` start | 15,055,211,479,440 |
| `rms_norm` end | 15,055,212,869,472 |
| `gate_matmul` start | 15,055,214,255,334 |
| `gate_matmul` end | 15,055,217,448,264 |
| `up_matmul` start | 15,055,219,058,260 |
| `up_matmul` end | 15,055,222,542,554 |
| materialization READY | 15,055,223,928,156 |
| `swiglu` start | 15,055,224,238,901 |
| `down_matmul` start | 15,055,226,832,753 |
| materialization RELEASED | 15,055,230,162,435 |

Materialization duration:

```text
16.796365 ms
```

Overlap with current compute before the consumer:

```text
8.067256 ms
```

This is the sum of overlap with RMSNorm, gate matmul, and up matmul. The
materialization started before current compute ended and became READY before
`down_matmul` started.

Consumer wait:

```text
0.000000 ms
```

The result is `PREFETCH_HIT`, not `PREFETCH_PARTIAL`.

Hidden fraction is defined as:

```text
overlap_hidden_materialization_time / total_materialization_time
= 8.067256 / 16.796365
= 0.480298
```

## Baseline Comparison

Both runs used the same graph, model artifact, references, pinned ggml
revision, CPU backend, `CPU_REPACK=OFF`, and CUDA-disabled build.

| Run | Total execute time | Materialization | Peak active weights |
|---|---:|---:|---:|
| synchronous baseline | 40.459 ms | implicit borrowed JIT | 12,607,488 bytes |
| async prefetch | 25.901 ms | 16.796 ms | 12,607,488 bytes |

The fixture is small and timing variance is expected. This report does not
claim a general throughput improvement. The architectural result is the
measured overlap and READY-before-consumer behavior.

## Memory And Budget

Prefetch budget was 12,607,488 bytes, equal to the largest persistent tensor.
The materializer reported:

- in-flight bytes at request: 12,607,488
- ready bytes at READY: 12,607,488
- bytes after RELEASED: 0
- maximum prefetched-but-not-consumed bytes: 12,607,488
- budget overflow: none

RSS snapshots from the trace:

- baseline before execution: 28,220 KiB
- prefetch before request: 28,492 KiB
- prefetch during RMSNorm: 36,180 KiB to 40,408 KiB
- prefetch during gate/up compute: 49,856 KiB to 60,856 KiB
- at READY: 63,176 KiB
- after final execution: 51,248 KiB
- after materializer release: 63,564 KiB in the final trace snapshot

The final snapshot includes process allocator effects and is not treated as
an OS page-eviction measurement. In-flight, ready, active persistent, and
post-release byte categories are reported separately in `prefetch.log`.

## Lifetime And Fallback Proof

The normal executor trace remains the POC3 trace:

- persistent weights are acquired only for the current op
- `ffn_down.weight` is released at the last consumer
- persistent leases after graph teardown: `0`
- external output survives executor teardown
- no full-model allocation occurs

The materialized tensor transitions READY to RELEASED at the same final
consumer boundary. The materializer destructor joins any worker before
discarding remaining buffers, so no in-flight worker, buffer, or lease
survives teardown. The baseline run exercises the unmodified JIT fallback and
produces the same numerical results.

## Numerical Qualification

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

CTest passed 6/6, including planner, dependency-lifetime, and materializer
contract tests.

## RV2 And Architecture Search

RV2 remains blocked by the pinned ggml/RVV FP16 toolchain issue. GCC/G++ 14.2
is available, but the pinned ggml build fails before runtime execution because
the compiler headers lack `vfloat16m2_t` and related RVV FP16 intrinsics.

```text
RV2_BUILD: BLOCKED_BY_TOOLCHAIN
RV2_RUNTIME_RESULT: NOT_EXECUTED
```

No ggml pin change was made.

Search of `vbuf_tensor_wave.*`, `vbuf_prefetch_planner.*`, and
`vbuf_materializer.*` found no DeepSeek, Qwen, Llama, Phi, layer, region, or
expert-specific runtime logic. Model tensor names remain in the qualification
fixture only.

ggml revision:
`2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`

Model SHA-256:
`780a55b77d2730705a93622338d9747149624d72e558e26182176868210fafcc`

## Required Classification

```text
ASYNC_TENSOR_MATERIALIZATION: PASS
REAL_COMPUTE_IO_OVERLAP: PASS
PREFETCH_BEFORE_CONSUMER: PASS
PREFETCH_HIT: YES
CONSUMER_WAIT_MS: 0.000000
MATERIALIZATION_MS: 16.796365
OVERLAP_MS: 8.067256
HIDDEN_FRACTION: 0.480298
PREFETCH_BUDGET_RESPECTED: PASS
JIT_FALLBACK_PRESERVED: PASS
LAST_CONSUMER_RELEASE: PASS
INFLIGHT_WORK_AFTER_TEARDOWN: 0
REFERENCE_PARITY_X86: PASS
ARCHITECTURE_SPECIFIC_RUNTIME_LOGIC: NO
VBUF_FORMAT_CHANGE_REQUIRED: NO
RV2_RUNTIME_RESULT: NOT_EXECUTED
READY_FOR_REMOTE_RANGE_SOURCE_POC: NO
```

## Files Changed

- `integrations/ggml/include/vbuf_materializer.h`
- `integrations/ggml/src/vbuf_materializer.cpp`
- `integrations/ggml/include/vbuf_tensor_wave.h`
- `integrations/ggml/src/vbuf_tensor_wave.cpp`
- `integrations/ggml/tools/tensor_wave_poc3.cpp`
- `integrations/ggml/tests/materializer_contract.cpp`
- `integrations/ggml/CMakeLists.txt`

## Evidence

- `baseline.log`
- `prefetch.log`
- `ctest.log`
- `metadata.txt`
- `model-sha256.txt`
- `ffn_inp.f32`
- `ffn_swiglu.f32`
- `reference_ffn_out.f32`
- `ffn_out.f32`
