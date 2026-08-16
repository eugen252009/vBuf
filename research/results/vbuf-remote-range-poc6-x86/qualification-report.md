# Remote Range Source POC6: x86 Qualification

## Result

`PASS` on x86-64. The same real tensor demand was fulfilled through both a
local exact-range source and an HTTP exact-range source. The graph, planner,
consumer counts, materializer state machine, ggml descriptors, execution order,
and last-consumer release path were unchanged.

## RangeSource Contract

Added the minimal source contract:

```text
RangeSource::read_range(offset, length, destination, result)
```

Implementations:

- `LocalVbufRangeSource`: copies the validated mapped artifact interval.
- `HttpRangeSource`: sends an HTTP/1.1 `Range: bytes=start-end` request over a
  POSIX socket and accepts only a matching `206 Partial Content` response.

`LocalVbufRangeMaterializer` now accepts a `RangeSource`. The destination,
state machine, worker, budget, and executor integration remain shared.

## Exact Tensor Range

The Rust vBuf consumer ABI's authoritative physical-range API supplied:

- tensor: `blk.0.ffn_down.weight`
- tensor ID: unchanged POC3/POC4 ID
- offset: `83,518,776`
- length: `12,607,488`
- requested end: `96,126,263`

The server received no model semantics, graph information, or tensor names.
It served only the artifact byte range.

## Artifact Identity

Qualification-time identity verification used SHA-256 of the local artifact and
the artifact served by the local HTTP server:

```text
780a55b77d2730705a93622338d9747149624d72e558e26182176868210fafcc
```

This proves the local and HTTP qualification artifact files were identical.
It is qualification-time verification, not a runtime distributed integrity
protocol.

## Payload Identity

The materializer FNV-1a payload hash was identical for both sources:

```text
0c7bdf85b162206b
```

The HTTP trace recorded:

```text
status=206
returned_bytes=12607488
Content-Range: bytes 83518776-96126263/4993331814
```

The remote response payload exactly matched the requested tensor length. The
HTTP client rejected status 200, wrong ranges, truncated payloads, and wrong
byte counts. A non-range response was rejected from its headers without
accepting or buffering the full artifact.

## Timing Comparison

Both runs used the same graph, references, model, backend, ggml revision,
`CPU_REPACK=OFF`, and CUDA-disabled build.

| Metric | Local source | HTTP source |
|---|---:|---:|
| materialization duration | 24.065682 ms | 44.229870 ms |
| effective payload rate | 523.878 MB/s | 285.045 MB/s |
| compute overlap | 5.318377 ms | 3.815832 ms |
| consumer wait | 15.572810 ms | 38.879683 ms |
| total execution time | 31.469 ms | 51.770 ms |

The HTTP source was served from a loopback qualification server in this run.
The measured result is a fresh POC-path measurement, not the earlier LAN
context measurement. Both runs reached READY before `down_matmul` started, but
both had a nonzero wait while resolving the required tensor. They are reported
as partial prefetch hits rather than claiming zero-wait hits.

The materialization interval overlapped unrelated current computation before
the consumer. Source substitution changed timing only; it did not change
execution semantics.

## State And Lifetime Trace

The complete local and HTTP traces are in `local.log` and `remote.log`.
The remote state sequence was:

```text
NOT_REQUESTED -> IN_FLIGHT -> READY -> RELEASED
```

HTTP failure paths transitioned to `FAILED`, joined the worker, released all
temporary resources, and explicitly used the configured local JIT fallback:

- unreachable server
- HTTP 500 status
- ignored Range request / HTTP 200
- truncated payload
- mismatched Content-Range

Failure traces are preserved as `failure-*.log`.

The normal executor trace remained identical in dependency order and
last-consumer release. Persistent leases after graph teardown were zero. No
in-flight worker, socket, or materialized buffer survived teardown.

## Memory

The remote materializer buffered only the requested 12,607,488-byte range.
It did not allocate or receive a full-artifact buffer.

Representative RSS snapshots:

| State | Local | HTTP |
|---|---:|---:|
| baseline before execution | 28,392 KiB | 28,300 KiB |
| request / in-flight event | 28,656 KiB | 28,564 KiB |
| ready event | 63,688 KiB | 67,252 KiB |
| after execution report | 51,436 KiB | 54,996 KiB |
| post-release event | 63,752 KiB | 67,312 KiB |

RSS values include allocator and process effects and are not interpreted as
OS page-eviction measurements. Active in-flight and ready bytes are explicitly
recorded in every materialization event.

## Numerical Qualification

Local and HTTP paths both matched the established reference tolerances.

Intermediate activation:

- elements: 21,888
- max absolute error: `4.47035e-08`
- max relative error: `2.41219e-03`
- result: `PASS`

Final output:

- elements: 4,096
- max absolute error: `1.78814e-07`
- max relative error: `1.96622e-03`
- result: `PASS`

CTest passed 7/7, including range-source, materializer, planner, and lifetime
contract tests.

## Architecture And RV2

The graph and executor receive only the generic `TensorMaterializer` API. They
do not branch on local versus HTTP sources. Source code search found no
DeepSeek, Qwen, Llama, Phi, layer, region, or expert-specific logic in the
range source or materializer runtime. Model names remain in the qualification
fixture only.

RV2 remains blocked by the existing pinned ggml/RVV FP16 toolchain issue.
GCC/G++ 14.2 is available, but the pinned ggml build fails before runtime
execution because the compiler headers lack `vfloat16m2_t` and related RVV
FP16 intrinsics.

```text
RV2_BUILD: BLOCKED_BY_TOOLCHAIN
RV2_RUNTIME_RESULT: NOT_EXECUTED
```

ggml revision:
`2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`

## Required Classification

```text
GENERIC_RANGE_SOURCE: PASS
LOCAL_RANGE_SOURCE: PASS
HTTP_RANGE_SOURCE: PASS
EXACT_REMOTE_RANGE_TRANSFER: PASS
FULL_ARTIFACT_REMOTE_DOWNLOAD: NO
LOCAL_REMOTE_PAYLOAD_IDENTITY: PASS
SOURCE_INDEPENDENT_EXECUTION: PASS
REMOTE_ASYNC_PREFETCH: PASS
REMOTE_PREFETCH_HIT: PARTIAL
LOCAL_MATERIALIZATION_MBPS: 523.878
REMOTE_MATERIALIZATION_MBPS: 285.045
LOCAL_CONSUMER_WAIT_MS: 15.572810
REMOTE_CONSUMER_WAIT_MS: 38.879683
REFERENCE_PARITY_LOCAL: PASS
REFERENCE_PARITY_REMOTE: PASS
MATERIALIZATION_RESOURCES_AFTER_TEARDOWN: 0
ARCHITECTURE_SPECIFIC_RUNTIME_LOGIC: NO
VBUF_FORMAT_CHANGE_REQUIRED: NO
READY_FOR_SOURCE_SELECTION_POLICY_POC: NO
```

## Files Changed

- `integrations/ggml/include/vbuf_range_source.h`
- `integrations/ggml/src/vbuf_range_source.cpp`
- `integrations/ggml/include/vbuf_materializer.h`
- `integrations/ggml/src/vbuf_materializer.cpp`
- `integrations/ggml/include/vbuf_tensor_wave.h`
- `integrations/ggml/tools/tensor_wave_poc3.cpp`
- `integrations/ggml/tests/range_source_contract.cpp`
- `integrations/ggml/CMakeLists.txt`
- `scripts/range_server.py`

## Evidence

- `local.log`
- `remote.log`
- `failure-*.log`
- `ctest.log`
- `metadata.txt`
- `model-sha256.txt`
- `artifact-identity-local.txt`
- `artifact-identity-http-served.txt`
- `ffn_inp.f32`
- `ffn_swiglu.f32`
- `reference_ffn_out.f32`
- `range_server.py`
