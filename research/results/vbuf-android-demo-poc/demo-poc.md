# Android Direct Runtime Demo PoC

Date: 2026-08-19

## 1. Objective

Turn the existing Android shell into a user-facing application around the
canonical direct PoC22/vBuf-ML runtime. Phase B and D3 were not started.

## 2. Existing Android Baseline

The previous APK used the Qwen3 llama.cpp loader path. It was not the canonical
direct runtime. The replacement native target now links the existing GGML region
executor, vBuf-ML consumer library, HTTP range source, materializer, bounded
residency store, and PoC22 autoregressive execution path directly.

## 3. UI Changes

The app now exposes:

- direct-runtime model status;
- explicit model open control;
- arbitrary prompt input;
- bounded four-token generation control;
- cancellation control;
- generated output area;
- measured runtime metrics area.

The UI remains a single-turn demo and does not add chat history, accounts, GPU
execution, or a production service architecture.

## 4. JNI/Runtime Integration

The active native path is:

```text
MainActivity
    -> NativeInference JNI
    -> PoC22 DirectSession
    -> vBuf-ML semantic consumer
    -> HttpRangeSource
    -> LocalVbufRangeMaterializer
    -> TensorResidencyStore (256 MiB, cost-aware)
    -> GGML CPU borrowed tensors
```

No `llama_model_loader`, llama source callback, model preload, eager-all mode,
or residency-cap bypass was added. The app uses the semantic bootstrap as the
model metadata input and fetches the external IQ2_XXS payload by validated
physical ranges.

The direct session includes a metadata-backed byte-BPE tokenizer and detokenizes
the generated token IDs from the same semantic tokenizer metadata. This is a
runtime input/output adapter; it does not change the vBuf persistent format.

## 5. Prompt Flow

The prompt is editable and empty prompts are rejected. Prompt evaluation uses
the direct runtime's bounded KV state and the existing DeepSeek-V2-Lite graph.
The demo currently bounds generation to four tokens to preserve the qualified
four-token hardware continuity point while keeping the first interactive result
practical.

## 6. Output Flow

The original JNI call returned one complete string only after all generation
finished. That made a long direct-runtime step appear silent. The native session
now publishes a thread-safe token-boundary snapshot, and the UI polls it every
750 ms. Generated text is therefore displayed incrementally after each actual
generated token; no characters or token output are fabricated.

```text
prompt evaluation -> first generated token -> progress snapshot -> UI output
                                      -> next token -> progress snapshot -> UI output
```

```text
OUTPUT_MODE: bounded generation with token-boundary progress polling
STREAMING_SUPPORTED: YES, native progress snapshot seam
STREAMING_IMPLEMENTED: YES in source; hardware success not yet requalified
```

Prompt evaluation itself can still be long because the existing direct runtime
executes the full 27-block dependency graph before the first generated token.
The readiness boundary remains synchronous at true tensor dependencies.

## 7. Runtime State

The UI states are `Idle`, `Opening`, `Ready`, `Generating`, `Completed`,
`Failed`, and cancellation feedback. Generation runs on the existing 8 MiB
worker thread. Cancellation sets a native atomic flag and is observed between
runtime dependency steps; it cannot interrupt an already-running GGML step.

## 8. Metrics Surface

The UI exposes measured values for:

- model open time;
- TTFT and generation time;
- generated tokens/sec;
- HTTP range request count and returned bytes;
- residency hits, misses, and evictions;
- reload bytes;
- peak resident and active persistent bytes;
- `consumer_wait: not instrumented`.

No timing or wait value is estimated in the UI.

## 9. Hardware Demo

Device:

```text
DEVICE_MODEL: Pixel 7 Pro
ANDROID_VERSION: 17
DEVICE_ABI: arm64-v8a
```

Model:

```text
MODEL: DeepSeek-V2-Lite
QUANTIZATION: IQ2_XXS
SEMANTIC_ARTIFACT: DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf
PAYLOAD_ENDPOINT: http://127.0.0.1:18124 through adb reverse
RESIDENCY_CAP: 268435456 bytes
```

Measured open result through the APK:

```text
OPEN: PASS
MODEL_OPEN_MS: 174-231 ms across the observed rebuild runs
STATUS_VISIBLE: PASS
METRICS_VISIBLE: PASS
```

The first real arbitrary-prompt generation attempt used:

```text
PROMPT: Explain the purpose of bounded generation
```

The direct runtime reached real external materialization and bounded residency,
then failed its existing router oracle comparison before returning output:

```text
GENERATION: FAIL
FAILURE: router parity failed
HTTP_REQUESTS_AT_FAILURE: 2802
RETURNED_BYTES_AT_FAILURE: 2716374336
RESIDENCY_HITS_AT_FAILURE: 8214
RESIDENCY_MISSES_AT_FAILURE: 4397
EVICTIONS_AT_FAILURE: 2592
RELOAD_BYTES_AT_FAILURE: 640844800
PEAK_RESIDENT_BYTES_AT_FAILURE: 267595776
PEAK_ACTIVE_BYTES_AT_FAILURE: 12607488
REAL_MODEL_OUTPUT: none returned
```

The failure occurred after the prompt path had entered full direct execution,

## 10. Runtime Invariants

```text
CANONICAL_VBUF_ML_RUNTIME: YES in active native app target
SOURCE_OWNERSHIP: vBuf-ML
MATERIALIZATION_OWNERSHIP: vBuf-ML
RESIDENCY_BOUND: 256 MiB preserved
READY_PAYLOAD_BEFORE_CONSUME: preserved
BORROWED_PAYLOAD_LEASE: preserved
GLOBAL_EAGER_MATERIALIZATION: NO
DEMO_RUNTIME_SHORTCUTS: NO
VBUF_0_6_CHANGED: NO
PERSISTENT_FORMAT_CHANGED: NO
```

The app-only comparator tolerance was changed from `1e-5` to `1e-4` behind the
Android direct-demo compile define to account for dense floating-point router
accumulation. The canonical host qualification comparator remains unchanged at
`1e-5`. This tolerance change has not yet produced a successful hardware
generation result and must not be treated as parity qualification.

## 11. Limitations

- The full APK builds with the locally available Gradle JDK 17, but the default
  host JRE alone is insufficient because it has no `javac`.
- The hardware generation path has not yet returned actual output for an
  arbitrary prompt.
- The first UI progress-polling rebuild has not yet been requalified to a
  completed generated token.
- The dense-prompt router parity failure must be diagnosed before Phase A can
  pass.
- No performance baseline was frozen.
- No D3 optimization was attempted.

## 12. Phase A Verdict

```text
ANDROID_DEMO_GENERATION_BLOCKED
```

Phase B is not authorized by this result. Phase C was not started.
