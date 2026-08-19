# Phase D2: PoC22 Android/ARM64 Qualification

Date: 2026-08-19

## 1. Objective

Run the existing direct PoC22/vBuf-ML runtime on the Pixel ARM64 environment
with semantic bootstrap and remote payload ownership outside llama.cpp. The
bounded target was the existing four-position deterministic recurrence, not a
chat or scheduler redesign.

## 2. Architectural Constraint

The direct path remains:

```text
vBuf-ML discovery/source/materialization/residency
    -> validated borrowed/materialized tensors
    -> GGML CPU backend
    -> PoC22 execution
```

No `llama_model_loader`, llama remote source callback, or llama-owned
materialization policy was added to the PoC22 path. llama.cpp remains an oracle
only.

## 3. D1 Starting Point

D1 qualified PoC22 on x86 against DeepSeek-V2-Lite with local HTTP ranges:

- Four generated positions passed autoregressive sequence and logits parity.
- Router selection parity passed.
- Teardown reported zero resident bytes, active leases, materialization
  resources, and runtime-state resources.
- The graph is hard-coded to the DeepSeek-V2-Lite tensor naming and shape
  contract, including `output.weight` and routed expert tensors.

The current Pixel Android proof uses Qwen3-0.6B, not DeepSeek-V2-Lite. This
model-family distinction was the decisive D2 finding.

## 4. Android/ARM64 Porting Seam

The existing Android JNI library is coupled to llama.cpp:

```text
NativeInference -> llama_model_load_vbuf_direct/remote -> llama/GGML
```

It was not reused for PoC22. Instead, the existing direct executable was
cross-compiled as a standalone ARM64 native qualification target and launched
on the Pixel through ADB. This isolates the direct runtime from the llama JNI
entry point without introducing a new llama-shaped callback seam.

The existing app's semantic-bootstrap path was used as the metadata fixture:

```text
files/models/Qwen3-0.6B-Q8_0.semantic.vbuf
```

The full Qwen artifact was absent from the device. No full DeepSeek or Qwen
payload artifact was copied to the device for D2.

## 5. Build and ABI Qualification

Target and toolchain:

- Target ABI: `arm64-v8a`, `aarch64`
- Android API: 29 minimum platform
- NDK: `27.1.12297006`
- Compiler: Android Clang 18.0.2
- GGML: pinned commit `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`
- Rust library: `aarch64-linux-android` `libvbuf_ml.so`
- Backend: GGML CPU, no CUDA, no OpenMP
- Pointer width: 64-bit

The direct executable compiled and linked for Android ARM64. The first device
launch exposed an absolute Rust-library `DT_NEEDED` path from the generic CMake
import. The smallest fix was setting `IMPORTED_NO_SONAME TRUE` on the imported
`vbuf_ml` target. The rebuilt executable carries `libvbuf_ml.so` by name and
launches on Android.

No wire-format, alignment, pointer-arithmetic, or runtime API redesign was
needed.

The Gradle Android app build was not completed because the environment's
installed Java 21 toolchain has no `javac` capability. This is a host build
environment blocker, not an ARM64 native compile failure.

## 6. Local SELF Qualification

`NOT_YET_QUALIFIED` for PoC22. The available local Android model is Qwen3 and
the direct PoC22 graph requires DeepSeek-V2-Lite. The D2 run therefore cannot
construct model tensors after metadata discovery. No full artifact was copied
to Android to bypass this mismatch.

The pre-existing Phase A Qwen llama path remains separate evidence and is not a
PoC22 qualification.

## 7. Remote Source Integration

The direct PoC22 executable uses its existing `HttpRangeSource`,
`LocalVbufRangeMaterializer`, and `TensorResidencyStore`. The Pixel run was
given the existing semantic bootstrap and an HTTP endpoint, but it stopped
before the first payload request because the Qwen graph did not contain the
required PoC22 tensor `output.weight`.

Therefore:

- Semantic bootstrap open: PASS.
- Source metadata discovery: PASS up to the model-graph lookup boundary.
- Remote payload acquisition by PoC22: NOT_YET_QUALIFIED.
- Physical HTTP requests by PoC22: `0` for the failed Qwen run.

The existing Phase B Qwen llama path transferred 633,495,552 payload bytes in
310 HTTP requests, but those are oracle/reference-path measurements and must
not be attributed to PoC22.

## 8. Runtime Ownership

The direct ownership model remains unchanged:

- vBuf-ML owns semantic discovery, source IDs, offsets, lengths, and metadata
  lifetime.
- `HttpRangeSource` owns transport reads.
- `LocalVbufRangeMaterializer` owns aligned payload allocations and publishes
  shared-owner-backed borrowed views.
- `TensorResidencyStore` owns resident payload retention and leases.
- GGML consumes the validated borrowed storage.
- `RuntimeStateSlot` owns copied bounded KV history.

The failed Android run reached semantic metadata validation without invoking a
llama loader or acquiring payload ranges.

## 9. Correctness / Oracle Comparison

The direct PoC22 x86 regression was rerun after the semantic-bootstrap fallback
change. It passed the four-position recurrence, router parity, logits parity,
generated-token feedback, and teardown cleanup.

Android correctness is classified as follows:

- Tensor metadata parity: `NOT_YET_QUALIFIED` for a PoC22-compatible model.
- Tensor type/shape parity: `NOT_YET_QUALIFIED`.
- Source offset/length parity: `NOT_YET_QUALIFIED`.
- Payload hashes: `NOT_YET_QUALIFIED`; no PoC22 payload request occurred.
- Router parity: `NOT_YET_QUALIFIED`.
- Logits parity: `NOT_YET_QUALIFIED`.
- Selected-token parity: `NOT_YET_QUALIFIED`.

The failure is not a numerical mismatch. It is an explicit model graph lookup
failure: `POC22_FAILURE=missing tensor: output.weight`.

## 10. Generation Result

The Android direct runtime did not generate tokens. The x86 direct regression
generated the expected four-token sequence:

```text
94761, 94761, 86711, 86711
```

The same generation loop is therefore still functional on the supported
DeepSeek PoC22 graph, but not yet connected to the current Pixel Qwen3 model
family.

## 11. Timing Result

PoC22 Android timing is `NOT_YET_QUALIFIED` because execution stopped before
payload acquisition and runtime construction:

```text
POC22_MODEL_OPEN_MS: NOT_YET_QUALIFIED
POC22_REMOTE_ACQUISITION_MS: 0 ms measured for this failed run
POC22_RUNTIME_CONSTRUCTION_MS: NOT_YET_QUALIFIED
POC22_TTFT_MS: NOT_YET_QUALIFIED
POC22_TIME_TO_4_TOKENS_MS: NOT_YET_QUALIFIED
```

No synchronous `/proc/self/smaps_rollup` tracing was introduced.

## 12. Request / Transport Shape

The failed direct Android run issued no HTTP payload requests. Logical and
physical request metrics for a successful PoC22/Qwen-compatible run remain
unresolved. The existing Qwen llama path's 310-request/633,495,552-byte result
is retained as a separate comparison baseline only.

## 13. Teardown and Lifetime

The successful x86 direct regression again reported:

```text
RESIDENT_BYTES_AFTER_TEARDOWN=0
EXECUTION_LEASES_AFTER_TEARDOWN=0
MATERIALIZATION_RESOURCES_AFTER_TEARDOWN=0
RUNTIME_STATE_RESOURCES_AFTER_TEARDOWN=0
```

The Android failed run stopped before materializer/residency construction, so
Android payload teardown is `NOT_YET_QUALIFIED`. The process exited without a
native crash or retained worker from the direct executable.

## 14. Remaining Tokenizer/Chat Gaps

Tokenizer, sampler, chat-template, and UI JNI integration were intentionally
not redesigned in D2. The current app's tokenizer/chat path remains on the
llama oracle path and is not evidence for direct PoC22.

## 15. Remaining Scheduler Gap

No scheduler redesign was attempted. PoC22 still has up-front tensor metadata
and a simple materializer/residency path. Request coalescing, batching,
concurrency, and Android production scheduling remain future work.

## 16. Comparison Against llama Oracle

| Property | PoC22 Android runtime | llama.cpp oracle |
| --- | --- | --- |
| Runtime owner | vBuf-ML direct runtime | Reference only |
| Materialization owner | vBuf-ML `LocalVbufRangeMaterializer` | llama integration path |
| Residency owner | vBuf-ML `TensorResidencyStore` | llama integration path |
| Request scheduling owner | vBuf-ML runtime/materializer | llama integration path |
| Uses llama_model_loader | No | Yes, reference path |
| GGML backend | GGML CPU | GGML CPU |
| Remote source | `HttpRangeSource`, attempted; no payload request | Existing Qwen HTTP Range source |
| Tensor count | NOT_YET_QUALIFIED for compatible graph | 311 metadata / 310 payload tensors |
| Physical requests | 0 in failed direct run | 310 historical Phase B baseline |
| Connections | 0 in failed direct run | 1 in retained C1 path |
| Payload bytes | 0 in failed direct run | 633,495,552 historical Phase B baseline |
| Requested bytes | 0 in failed direct run | 633,495,552 historical Phase B baseline |
| Overfetch | 0 in failed direct run | 0 historical Phase B baseline |
| Model open | NOT_YET_QUALIFIED | 53,203 ms Phase B baseline |
| TTFT | NOT_YET_QUALIFIED | NOT separately recorded in Phase B |
| Bounded generation | NOT_YET_QUALIFIED on Android; PASS x86 | PASS, 16 tokens in Phase B |
| Router/logits parity | NOT_YET_QUALIFIED on Android; PASS x86 | Generation/reference qualification PASS |
| Teardown | NOT_YET_QUALIFIED for Android payloads; PASS x86 | Existing app cleanup path |

Clean llama model-open reference is approximately `13.58 s`; it is not a
PoC22 measurement. D0.3 matched controls are `6,146.521 ms` single-span and
`10,159.064 ms` exact 310-range.

## 17. Selected Runtime Direction

Keep the direct PoC22/vBuf-ML runtime as the selected architecture. The ARM64
build and semantic bootstrap launch prove that the direct boundary is portable
through native startup and discovery. Do not route it through llama.cpp.

The immediate corrective experiment is narrowly scoped: provide a
DeepSeek-V2-Lite semantic bootstrap and remote source fixture to the Pixel, or
qualify a separate existing direct Qwen execution graph if one exists. Do not
generalize PoC22 into a multi-architecture model executor in D2.

## 18. Next Phase

The requested D3 scheduler phase is not yet authorized by the D2 result. First
qualify a PoC22-compatible model family on Android/ARM64 through payload
materialization and four-token generation. After that, the project can move to:

```text
Phase D3 — canonical vBuf-ML load-plan / scheduler qualification
```

## Classification

```text
POC22_ANDROID_MAJOR_RUNTIME_GAP
```

This classification means the current canonical PoC22 graph cannot consume
the current Android Qwen3 fixture. It does not mean ARM64 portability failed,
and it does not invalidate the direct runtime architecture.

## Validation

```text
RUST_TESTS: PASS (cargo test --workspace)
POC22_BUILD: PASS (x86_64 and Android arm64-v8a)
POC22_TESTS: PASS (x86 four-position direct recurrence)
GGML_ADAPTER_TESTS: PASS (17/17 CTest contracts)
ANDROID_ARM64_BUILD: PASS (standalone direct PoC22 executable)
ANDROID_APP_BUILD: NOT_YET_QUALIFIED (host lacks javac capability)
POC22_ANDROID_LOCAL_SELF: NOT_YET_QUALIFIED (available fixture is Qwen3)
POC22_ANDROID_REMOTE_OPEN: PARTIAL (semantic bootstrap PASS; graph lookup fails)
POC22_ANDROID_REMOTE_GENERATION: NOT_YET_QUALIFIED
LLAMA_ORACLE_REFERENCE: EXISTING PHASE B/C EVIDENCE ONLY
ROUTER_PARITY: PASS x86; NOT_YET_QUALIFIED Android
LOGITS_PARITY: PASS x86; NOT_YET_QUALIFIED Android
PAYLOAD_PARITY: NOT_YET_QUALIFIED Android direct path
TEARDOWN_CLEANUP: PASS x86; NOT_YET_QUALIFIED Android payload path
JSON_VALIDATION: PASS (7 JSON files)
CCC_INDEX: PASS
GIT_DIFF_CHECK: PASS
```
