# Phase D0.1 Instrumentation

## 1. Objective

Phase D0.1 adds diagnostic-only timing and accounting to the selected C1 remote
model-open path. It does not optimize the loader or change transport policy,
tensor ordering, request count, `EAGER_ALL`, persistent formats, or model
semantics.

The reference is the Phase C selected median of `44,451 ms`, with 310 requests,
one persistent connection, and `633,495,552` transferred bytes.

The Android device matrix was completed using the manually packaged native
artifact because Gradle cannot run without `javac`/an Android SDK. The device
was a Pixel 7 Pro (Android 17). The reference repeat opened the model in
`13,580 ms`; Instrumented B opened it in `13,256 ms`; Instrumented A opened it
in `39,843 ms` because per-tensor `smaps_rollup` sampling was enabled.

## 2. Instrumentation Changes

Added:

- `integrations/ggml/include/vbuf_d0_1_diagnostics.h`
- `integrations/ggml/src/vbuf_d0_1_diagnostics.cpp`
- aggregate counters in `vbuf_materializer.cpp`;
- HTTP first-byte, body, mutex, and memcpy accounting in `vbuf_range_source.cpp`;
- callback lookup and pointer-bind timing in `vbuf_remote_source.cpp`;
- top-level Android open and post-generation boundaries in
  `vbuf_android_chat.cpp`;
- descriptor-loop and backend-allocation weak hooks in the already-prepared
  `/tmp/llama.cpp-step21` checkout.

The diagnostic session is enabled only with:

```text
VBUF_D0_1_TRACE=1
```

The runtime emits one aggregate `VBUF_D0_1_JSON {...}` line at model-open
completion and another after successful generation. Coarse memory snapshots are
emitted as `VBUF_D0_1_SNAPSHOT {...}` lines.

The existing per-event RSS sampling is now diagnostic-gated. With D0.1 enabled,
it remains enabled by default to characterize current behavior. Set
`VBUF_D0_1_DISABLE_SMAPS=1` for the controlled observer-overhead comparison.

The instrumentation is opt-in and does not alter payload bytes, request
scheduling, ownership, backend placement, or model construction decisions.

## 3. Observer-Effect Validation

Required run matrix:

| Run | Environment | Purpose | Status |
|---|---|---|---|
| Reference | no D0.1 environment | Existing C1 behavior | Completed: `13,580 ms` |
| Instrumented A | `VBUF_D0_1_TRACE=1` | Current smaps diagnostics plus lightweight counters | Completed: `39,843 ms` |
| Instrumented B | `VBUF_D0_1_TRACE=1 VBUF_D0_1_DISABLE_SMAPS=1` | Isolate smaps observer cost | Completed: `13,256 ms` |
| Reference repeat | no D0.1 environment | Estimate run variance | Same as Reference |

The native Android target was built successfully with direct CMake/Ninja. Gradle
remains unavailable because `javac` and the Android SDK are not configured.

## 4. Complete Model-Open Timeline

The following boundaries are implemented:

```text
MODEL_OPEN_BEGIN              vbuf_d0_1_begin(), from JNI remote open
BOOTSTRAP_BEGIN               before remote source construction
BOOTSTRAP_COMPLETE            after VbufRemoteSource construction
DESCRIPTOR_LOOP_BEGIN         pinned semantic source-loader hook
DESCRIPTOR_LOOP_COMPLETE      pinned semantic source-loader hook
FIRST_PAYLOAD_REQUEST_BEGIN   first accepted materializer request
HALF_PAYLOADS_READY           coarse memory snapshot
FINAL_PAYLOAD_READY           last retained payload inserted
LLAMA_MODEL_CREATE_BEGIN      pinned model-create hook
LLAMA_MODEL_CREATE_COMPLETE   pinned model-create hook
BACKEND_ALLOCATION_BEGIN      pinned backend allocation hook
BACKEND_ALLOCATION_COMPLETE   pinned backend allocation hook
REMOTE_POINTER_BIND_BEGIN     first llama remote callback
REMOTE_POINTER_BIND_COMPLETE  final callback aggregate update
MODEL_OPEN_COMPLETE           after llama context creation in JNI
POST_GENERATION               after successful remote generation
```

The most important derived interval is explicitly emitted as
`post_final_payload_ns`:

```text
FINAL_PAYLOAD_READY -> MODEL_OPEN_COMPLETE
```

## 5. Serialized Payload Phase

`serialized_payload_ns` is measured from the first accepted payload request to
the final retained payload. It includes the current one-request-at-a-time
critical path and is an inclusive wall-clock interval, not a sum of per-request
subtimers.

Result B: `12,751.550 ms` serialized payload wall-clock, with `310` requests and
`633,495,552` bytes. Descriptor loop was `12,858.900 ms`; bootstrap was
`7.807 ms`; post-final-payload was `468.206 ms`.

## 6. HTTP Latency vs Body Transfer

For each successful HTTP request the instrumentation records:

```text
read_range entry -> first response body byte
first response body byte -> complete response body
send completion -> first response body byte
```

The first interval includes mutex wait and request setup. The second interval
includes receive and destination memcpy. Header/response setup is reported as
the third interval and is not claimed to be a pure network-latency measure.

The instrumentation retains 310 timing values for median and p95 calculation.

Result B: request-to-first-byte total `4,133.346 ms`, median `4.910 ms`, p95
`50.382 ms`. Body receive total `5,819.662 ms`, median `11.715 ms`, p95
`31.046 ms`. Header/response setup total was `4,040.828 ms`.

## 7. Worker and Synchronization Cost

Recorded counters:

- request setup before worker construction;
- `std::thread` construction duration;
- worker start delay from request acceptance;
- `join()` duration;
- request finalization from post-hash publication through ready state recording.

`WORKER_WAIT_TOTAL` is inclusive and may contain HTTP, memcpy, hash, smaps, and
finalization time. It must not be added to those nested costs in the final model.

Result B: worker creation `125.100 ms`, worker start delay `250.204 ms`, worker
wait `12,513.421 ms`, mutex wait `0.099 ms`, mutex held `9,964.159 ms`, and
finalization `2.704 ms`.

## 8. Copy Cost

The HTTP source confirms the fixed receive buffer size is 4 KiB. Each successful
request aggregates destination memcpy calls, bytes, and nanoseconds. The initial
body bytes already present in the header buffer are included.

Result B: `156,897` memcpy calls covering `633,495,552` bytes in `1,065.032 ms`
(approximately `595 MB/s`).

## 9. Hash Cost

The complete FNV-1a pass remains enabled. The materializer records call count,
bytes, and duration around the existing hash operation. No hash behavior was
removed or replaced.

Result B: `156,897` FNV-1a calls covering `633,495,552` bytes in `2,278.201 ms`
(approximately `278 MB/s`).

## 10. Diagnostic / smaps Cost

Existing `smaps_rollup` calls are timed individually and summarized as count,
total, median, and p95. Coarse phase snapshots use the same file only at:

```text
BOOTSTRAP_COMPLETE
HALF_PAYLOADS_READY
FINAL_PAYLOAD_READY
BACKEND_ALLOCATION_COMPLETE
MODEL_OPEN_COMPLETE
POST_GENERATION
```

The controlled B run disables per-tensor smaps sampling while retaining all
lightweight counters and timers. This is an observer-effect characterization,
not a production optimization.

Result A: `1,240` smaps calls consumed `33,535 ms` and inflated model open to
`39,843 ms`. Result B disabled these calls and completed in `13,256 ms`.

## 11. Callback Lookup Cost

`set_vbuf_remote_tensor_data()` records:

- callback count;
- source-list iterations;
- time spent searching and validating the source descriptor;
- pointer assignment time.

The source list and tensor order are unchanged. Tied-output handling remains
unchanged.

Result B: `310` callbacks, `48,205` source-list iterations, `5.526 ms` lookup
time, and `0.014 ms` pointer assignment time.

## 12. Backend Allocation Cost

The prepared pinned llama.cpp checkout has weak diagnostic hooks around the actual
`ggml_backend_alloc_ctx_tensors_from_buft()` call. The project implementation
records allocation count, returned buffer size, and call duration when the
remote diagnostic session is active. The hook is a no-op for standalone builds
that do not link the vBuf diagnostics implementation.

This measures allocation, not physical residency. PSS/RSS snapshots are required
to determine whether allocated pages are resident or merely reserved.

Result B: one allocation of `633,495,552` bytes in `0.082 ms`. This is an
allocation call measurement, not a residency claim.

## 13. Memory Residency Observations

Only coarse snapshots are collected by the new instrumentation. The JSON and
snapshot lines expose total process RSS, PSS, and SwapPss from
`/proc/self/smaps_rollup`.

Native-heap-only PSS is not synthesized when the platform does not expose a
stable native-heap counter. Existing Phase C values remain the reference:

```text
native heap PSS: 735,592 KiB
```

Result B snapshots: RSS was `160,076 KiB` after bootstrap, `404,764 KiB` at
half payloads, `795,200 KiB` at final payload, `836,392 KiB` after backend
allocation, `896,856 KiB` at model open, and `911,540 KiB` after generation.
PSS and SwapPss were unavailable and reported as zero by the device.

## 14. Post-Final-Payload Cost

This is the primary unresolved boundary from D0.1:

```text
FINAL_PAYLOAD_READY -> MODEL_OPEN_COMPLETE
```

It includes backend allocation, pointer binding, llama model finalization,
context creation, and any other post-payload work in the Android open call. It
is explicitly emitted as `post_final_payload_ns`.

Result B: `468.206 ms` from final payload ready to model-open completion.

## 15. Measured Critical-Path Model

Instrumented B accounts for the measured model-open boundary as follows:

```text
MODEL_OPEN_TOTAL
├─ bootstrap
├─ descriptor loop
├─ serialized payload wall-clock interval
│  ├─ request -> first byte
│  ├─ first byte -> body complete
│  │  └─ destination memcpy
│  ├─ hash
│  ├─ smaps diagnostics
│  └─ request bookkeeping
└─ final payload -> model open complete
   ├─ model create
   ├─ backend allocation
   ├─ callback lookup and pointer binding
   └─ remaining finalization/context work
```

Worker wait, body receive, and request-to-first-byte are inclusive timers. They
must not be added together as exclusive costs.

## 16. Confirmed Bottlenecks

Confirmed before measurement:

- requests are serialized by immediate wait/join;
- HTTP uses one mutex-protected persistent connection;
- C1 still performs 310 logical request/response exchanges;
- all payloads are retained under `EAGER_ALL`;
- FNV-1a scans every payload;
- the callback performs a linear source-list walk.

The dominant measured interval is serialized payload materialization. The A/B
comparison confirms that smaps diagnostics, rather than loader semantics, cause
the large observed slowdown: `(39,843 - 13,256) / 13,256 = 200.6%`.

## 17. Eliminated Suspects

The following are not valid conclusions yet:

- the full 23.2-second difference from the C0 reference is materialization time;
- the payload is necessarily copied a second time into llama tensor data;
- the O(n^2) lookup is a meaningful bottleneck because it is algorithmically
  unattractive;
- hashing is expensive because it scans 633 MB;
- backend allocation is resident merely because its size is large.

Phase C already eliminated positive-gap coalescing and bounded concurrency as
selected next experiments.

## 18. Remaining Unknowns

- physical residency of backend buffers, because device PSS is unavailable;
- exact payload parity and direct-source regression for this final packaged APK.

## 19. Phase D Architectural Implication

No architectural change is justified from this diagnostic pass alone. The valid
instrumented C1 run accounts for the current `13.256 s` device wall-clock path,
while the Phase C `44.451 s` reference used a different runtime/setup. The instrumentation preserves the
current selected semantics and is designed to distinguish:

- request latency from body throughput;
- body receive from memcpy;
- payload work from post-payload llama/backend work;
- real memory residency from allocation size;
- runtime costs from diagnostics overhead.

## 20. Recommended Next Experiment

Complete qualification on the same C1 setup:

1. payload parity against the Phase C capture;
2. remote generation correctness on the final packaged APK;
3. direct-source regression;
4. repeat B if tighter variance bounds are required.

The run artifacts are preserved in `/tmp/opencode/d0-a4-logcat.txt`,
`/tmp/opencode/d0-b-logcat.txt`, and `/tmp/opencode/d0-b-generation3.txt`.
Only after parity and regression qualification should Phase D select one
architectural seam.

## Summary Table

| Component | Calls | Bytes | Total Time | Median | P95 | Exclusive/Inclusive | Percent of Model Open | Confidence |
|---|---:|---:|---:|---:|---:|---|---:|---|
| Bootstrap | 1 | N/A | 7.807 ms | N/A | N/A | Exclusive boundary | 0.06% | Device B |
| Descriptor loop | 310 | 633,495,552 | 12,858.900 ms | N/A | N/A | Exclusive boundary | 97.01% | Device B |
| Request to first byte | 310 | N/A | 4,133.346 ms | 4.910 ms | 50.382 ms | Inclusive per request | 31.18% | Device B |
| Body receive | 310 | 633,495,552 | 5,819.662 ms | 11.715 ms | 31.046 ms | Inclusive per request | 43.90% | Device B |
| Destination memcpy | 156,897 | 633,495,552 | 1,065.032 ms | N/A | N/A | Nested in body receive | 8.03% | Device B |
| FNV-1a hash | 156,897 | 633,495,552 | 2,278.201 ms | N/A | N/A | Nested in worker wait | 17.19% | Device B |
| Worker creation | 310 | N/A | 125.100 ms | N/A | N/A | Partially exclusive | 0.94% | Device B |
| Worker wait | 310 | N/A | 12,513.421 ms | N/A | N/A | Inclusive | 94.40% | Device B |
| HTTP mutex | 310 | N/A | 9,964.159 ms held | N/A | N/A | Nested in request | 75.17% | Device B |
| smaps diagnostics | 1,240 | N/A | 33,535 ms | N/A | N/A | Nested in trace events | 84.16% of A | Device A |
| Callback lookup | 310 | N/A | 5.526 ms | N/A | N/A | Nested in binding | 0.04% | Device B |
| Backend allocation | 1 | 633,495,552 | 0.082 ms | N/A | N/A | Exclusive call boundary | <0.01% | Device B |
| Final payload to model open | 1 | N/A | 468.206 ms | N/A | N/A | Exclusive wall-clock boundary | 3.53% | Device B |

## Verification

```text
RUST_TESTS: passed (`cargo test --workspace`, Rust workspace)
ANDROID_LOCAL_BUILD: passed via direct CMake/Ninja; Gradle blocked by missing javac/SDK
ANDROID_REMOTE_BUILD: passed via direct CMake/Ninja and manual APK repackaging
REMOTE_PAYLOAD_PARITY: not run for final D0.1 APK
REMOTE_GENERATION: model open and POST_GENERATION boundary passed; output qualification pending
DIRECT_SOURCE_REGRESSION: not run for final D0.1 APK
JSON_VALIDATION: passed for captured `VBUF_D0_1_JSON` and `VBUF_D0_1_SNAPSHOT` records
CCC_INDEX: passed (`ccc index`)
GIT_DIFF_CHECK: passed (`git diff --check`)

PERSISTENT_FORMAT_CHANGED: NO
RESIDENCY_POLICY_CHANGED: NO
EAGER_ALL_CHANGED: NO
TRANSPORT_POLICY_CHANGED: NO
REQUEST_COUNT_POLICY_CHANGED: NO
PRODUCTION_SEMANTICS_CHANGED: NO
```
