# Post-Merge Android D2.3 Hardware Qualification

Date: 2026-08-19

## 1. Objective

Qualify the merged `vbuf-ml` direct PoC22 runtime on the connected physical
Android device using the existing DeepSeek-V2-Lite IQ2_XXS semantic-bootstrap,
external HTTP range, bounded-residency, GGML, four-position configuration.

## 2. Merged Branch / Commits

```text
CURRENT_BRANCH: vbuf-ml
CURRENT_HEAD: 345bee3c7b6d36cf663a932a1fde8cb45131b4bc
MERGE_COMMIT_PRESENT: YES (ee57377)
QUALIFICATION_COMMIT_PRESENT: YES (345bee3)
WORKTREE_CLEAN_BEFORE_RUN: YES
```

No repository production files were changed for this run. The ARM64 Rust and
native artifacts were built into `/tmp/opencode` and deployed separately.

## 3. Device Identity

```text
ADB_DEVICE_VISIBLE: YES
ADB_DEVICE_SERIAL: adb-35231FDH3002TT-vJazBJ._adb-tls-connect._tcp
ADB_DEVICE_STATE: device
DEVICE_MODEL: Pixel 7 Pro
ANDROID_VERSION: 17
ANDROID_API_LEVEL: 37
DEVICE_ABI: arm64-v8a
DEVICE_CODENAME: cheetah
```

This was a physical Pixel device, not an emulator.

## 4. Build Environment

The canonical standalone PoC22 executable path was used, not the separate
Qwen/llama JNI Phase A app. The executable was built from the merged checkout:

```text
ANDROID_BUILD: PASS
ANDROID_NATIVE_BUILD: PASS
BUILD_HEAD: 345bee3c7b6d36cf663a932a1fde8cb45131b4bc
ANDROID_NDK: 27.1.12297006
ANDROID_PLATFORM: android-29
GGML_COMMIT: 2d191b5dee1a591c41ee8a653ce42bfcd9c8716d
APK_OR_BINARY_PATH: /tmp/opencode/vbuf-ml-postmerge-android-build/vbuf_autoregressive_poc22
```

The ARM64 binary was verified as an Android `aarch64` ELF. The first Cargo
attempt used host `cc` and was rejected as an environment linker configuration
error; retrying with the installed NDK `aarch64-linux-android29-clang` linker
passed without source changes.

## 5. Android Deployment

```text
DEPLOYMENT_METHOD: adb push standalone executable and shared libraries
DEPLOYMENT_SUCCESS: YES
REMOTE_BINARY_OR_PACKAGE: /data/local/tmp/vbuf-postmerge-d2.3/
```

The device-side SHA-256 values matched the newly built executable, Rust
library, and semantic bootstrap. The run was executed with the device-side
`LD_LIBRARY_PATH` and the standalone PoC22 binary.

## 6. Payload Endpoint

```text
PAYLOAD_SERVER_AVAILABLE: YES
PAYLOAD_ARTIFACT_AVAILABLE: YES
SEMANTIC_BOOTSTRAP_AVAILABLE: YES
ADB_REVERSE_OR_FORWARD_CONFIGURED: YES, reverse tcp:18124 -> tcp:18124
REMOTE_SOURCE_REACHABLE_FROM_DEVICE: YES
```

The canonical `scripts/range_server.py` served
`DeepSeek-V2-Lite.IQ2_XXS.vbuf` on `127.0.0.1:18124` with range mode and
logging enabled. The endpoint was recreated after stale reverse mappings and
the prior server process were removed.

## 7. Model / Quantization

```text
MODEL: DeepSeek-V2-Lite lineage
QUANTIZATION: IQ2_XXS
SEMANTIC_BOOTSTRAP: DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf
REMOTE_PAYLOAD_SOURCE: DeepSeek-V2-Lite.IQ2_XXS.vbuf
RESIDENCY_CAP: 268435456 bytes
RESIDENCY_POLICY: cost-aware
BLOCKS: 27
SEED: 0
STEPS: 4
```

The full payload was not copied to the device; only the semantic bootstrap and
the executable/runtime libraries were deployed.

## 8. Semantic Bootstrap and Physical TensorRef Range

```text
ANDROID_SEMANTIC_DISCOVERY: PASS
PHYSICAL_RANGE_LENGTH_VALID: YES
ZERO_BYTE_EXTERNAL_REQUEST_OBSERVED: NO
```

The first observed HTTP range was `bytes=2982064-2982735`, returned as 672
bytes. Subsequent persistent ranges were also non-zero; the first relevant
payload requests included 8192-byte and 1622016-byte responses. The run
therefore preserved the distinction between absent inline semantic payload and
validated non-zero physical source ranges.

## 9. Materialization and Readiness

```text
ANDROID_REMOTE_CONNECTED: PASS
ANDROID_FIRST_MATERIALIZATION_REQUEST: PASS
ANDROID_FIRST_READY_PAYLOAD: PASS
ANDROID_EMBEDDING_PAYLOAD_READY: PASS
PENDING_PAYLOAD_CONSUMED: NO
REQUIRED_READINESS_WAIT_OBSERVED: YES by ready-before-consumer evidence;
    individual wait calls remain explicitly not instrumented
GLOBAL_SYNCHRONOUS_MATERIALIZATION: NO
EMBEDDING_SPECIAL_CASE: NO
```

The first runtime line was an embedding parity result, followed by payload
ready timestamps before first consumer timestamps. The existing runtime log
labels routed `consumer_wait` as `not_instrumented`; this run does not claim a
separate wait-call counter. It does establish that pending payloads were not
consumed and that all consumed persistent inputs were ready.

## 10. Residency / Lease

```text
LEASE_RETAINED_DURING_BACKEND_ACCESS: YES, acquire/release recorded
ACTIVE_PAYLOAD_EVICTED_DURING_USE: NO
DANGLING_POINTER_OBSERVED: NO
```

The run completed all four positions with repeated materialization and
eviction activity, while the runtime recorded acquire/release evidence and no
failure or invalid access. Teardown reported zero resident bytes, execution
leases, materialization resources, and runtime-state resources.

## 11. Execution Milestones

```text
ANDROID_EMBEDDING_COMPUTE: PASS
ANDROID_GRAPH_CONSTRUCTION: PASS
ANDROID_GGML_EXECUTION: PASS
ANDROID_FIRST_TOKEN: PASS
ANDROID_FOUR_TOKEN_GENERATION: PASS
ANDROID_TEARDOWN: PASS
```

The four positions all completed with runtime token `59685` and reference token
`59685`.

## 12. Parity

```text
ROUTER_PARITY: PASS
LOGITS_PARITY: PASS, max absolute error 0
TOKEN_PARITY: PASS, 4/4 sequence
REFERENCE_SEQUENCE: 59685, 59685, 59685, 59685
RUNTIME_SEQUENCE: 59685, 59685, 59685, 59685
GENERATED_TOKEN_FEEDBACK: PASS
RUNTIME_USES_REFERENCE_FUTURE: NO
```

The device-side capture report generated by the existing executable contains a
stale IQ1_S artifact label, but its authoritative stdout, endpoint, deployed
bootstrap, deployed hashes, and IQ2_XXS source identify this run as IQ2_XXS.

## 13. Request / Residency Observations

The endpoint log recorded 5,398 HTTP 206 range responses and 5,460,460,160
returned bytes. No zero-byte response was observed.

Aggregated from the four runtime position records:

```text
ANDROID_PHYSICAL_REQUEST_COUNT: 5398 HTTP 206 responses
ANDROID_REQUESTED_BYTES: not separately exposed by the server log
ANDROID_RETURNED_BYTES: 5460460160
ANDROID_MATERIALIZATION_COUNT: not exposed as one aggregate
ANDROID_RESIDENCY_HITS: 16979
ANDROID_RESIDENCY_MISSES: 8305
ANDROID_EVICTIONS: 5316
ANDROID_REMATERIALIZATIONS: reflected in reload bytes
ANDROID_RELOAD_BYTES: 3967938880
PEAK_RESIDENT_BYTES: 267798528
PEAK_ACTIVE_PERSISTENT_BYTES: 12607488
```

The endpoint had one ADB-reversed keep-alive path. These values are correctness
run observations, not a performance comparison or optimization result.

## 14. Inefficiency Observations

```text
REQUEST_FRAGMENTATION_OBSERVED: YES
RESIDENCY_RELOAD_AMPLIFICATION_OBSERVED: YES
PREFETCH_OPPORTUNITY_OBSERVED: YES, existing prefetch records
```

No scheduler, batching, transport, concurrency, residency-capacity, eviction,
prefetch, or HTTP behavior was changed.

## 15. Gate 2B Boundary Sanity Check

```text
GENERIC_CPP_ADAPTER_RECEIVES_READY_PAYLOAD_ONLY: YES
GENERIC_CPP_ADAPTER_RECEIVES_MATERIALIZER: NO
GENERIC_CPP_ADAPTER_CAN_REACQUIRE_SOURCE: NO
ANDROID_PATH_EXECUTES_GATE_2B_ADAPTER: NO
REAL_GATE_2B_ANDROID_EXECUTION_AVAILABLE: NO
```

The merged Android direct path remained the canonical PoC22 source/materializer/
readiness path. The portable adapter has contract and neutrality coverage but
no existing Android command that executes the real DeepSeek router-prefix
fixture through the portable ABI.

## 16. Verdict

```text
POST_MERGE_ANDROID_RESULT: POST_MERGE_ANDROID_D2_3_PASS
D2_POST_MERGE_CONFIRMED: YES
D3_READY: YES
GATE_2B_RESULT: GATE_2B_FFI_IMPLEMENTED_EXECUTION_BLOCKED
GATE_2C_READY: NO
QWEN_GATE_3_READY: NO
```

The merged branch reproduced the previously qualified direct-runtime behavior
on real Android hardware: semantic discovery, non-zero external physical
ranges, materialization, ready-payload consumption, bounded residency, lease
lifetime, embedding, GGML execution, four-token generation, parity, and clean
teardown all passed.

## 17. Change / Git Status

```text
PRODUCTION_CODE_CHANGED: NO
SCHEDULER_CHANGED: NO
RESIDENCY_POLICY_CHANGED: NO
HTTP_BEHAVIOR_CHANGED: NO
TRANSPORT_CHANGED: NO
VBUF_0_6_CHANGED: NO
PERSISTENT_FORMAT_CHANGED: NO
RESEARCH_REPORT: research/results/vbuf-ml-integration/post-merge-android-hardware-qualification.md
COMMIT_PERFORMED: NO
PUSH_PERFORMED: NO
FINAL_WORKTREE_STATUS: one untracked qualification report only
```

Evidence retained outside the repository:

```text
/tmp/opencode/postmerge-d2.3-android-iq2-full.log
/tmp/opencode/postmerge-d2.3-range-server-18124.log
/tmp/opencode/postmerge-d2.3-capture-report.md
/tmp/opencode/postmerge-d2.3-recurrence-trace.json
```
