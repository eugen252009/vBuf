# Android Application Baseline

Date: 2026-08-20
Branch: `vbuf-ml`
Status: physical app baseline captured and ready for the Android app milestone commit

## 1. Objective

Turn the Android demo into a thin user-facing harness over the already
qualified direct vBuf-ML runtime. The app baseline measures cold app/model open,
batched prompt prefill, first-token latency, bounded four-token decode, source
and residency counters, and truthful UI progress without changing model
semantics or runtime ownership.

## 2. App Architecture

```text
MainActivity
    -> NativeInference JNI facade
    -> DirectSession
    -> vBuf-ML semantic consumer
    -> HttpRangeSource
    -> LocalVbufRangeMaterializer
    -> TensorResidencyStore (256 MiB, cost-aware)
    -> existing layer-major batched prefill
    -> existing single-position decode
    -> GGML CPU backend
```

`DirectSession` remains the existing direct-runtime implementation from
`vbuf_android_direct.cpp`. The Java layer supplies prompt/UI/lifecycle adapters;
it does not parse vBuf, resolve TensorRefs, materialize payloads, manage
residency, or implement inference semantics.

## 3. JNI / Native Call Graph

```text
app launch
  -> MainActivity.onCreate()
  -> Generate/open worker
  -> NativeInference.open(modelPath, endpoint, false)
  -> DirectSession(..., RuntimeMode::NormalInference)
  -> load semantic bootstrap and build source/materializer/residency
  -> NativeInference.generate(prompt, 4)
  -> ByteBpeTokenizer::encode()
  -> PromptBatch / run_sequence_batched()
  -> output head / first decode token
  -> run_step() for four generated positions
  -> tokenizer decode / UI output
  -> NativeInference.metrics() / closeModel()
```

The app explicitly requests `NormalInference`. The existing compile-time
qualification selection remains available to direct qualification callers; the
app does not remove or weaken it.

## 4. Runtime Path Reuse

The app confirmed the same qualified runtime path as the direct probe:

```text
APP_RUNTIME_MODE: NORMAL_INFERENCE
APP_PREFILL_MODE: BATCHED
REFERENCE_WORK: NOT_EXECUTED
POSITION_LEVEL_PARALLELISM: NO
ORDERED_CAUSAL_ATTENTION: YES
DECODE_PATH: existing run_step single-position path
```

No Android-specific inference implementation, source policy, materialization
policy, residency policy, or model semantic path was created.

## 5. RuntimeMode

`NormalInference` performs actual computation only. Reference/oracle graphs and
parity work were not executed in the app run. Qualification mode remains serial,
reference-enabled, and fail-closed in the existing runtime.

## 6. Prompt / Model / Device

```text
DEVICE: Pixel 7 Pro
ANDROID_VERSION: 17
ABI: arm64-v8a
MODEL: DeepSeek-V2-Lite
QUANTIZATION: IQ2_XXS
PROMPT: Explain the purpose of bounded generation
PROMPT_CHAR_COUNT: 41
PROMPT_TOKEN_COUNT: 7
GENERATED_TOKEN_LIMIT: 4
PAYLOAD_BYTES: 5639819878
RESIDENCY_CAP: 268435456 bytes (256 MiB)
```

## 7. Source Configuration

The app used a debug build property for the development endpoint:

```text
SEMANTIC_BOOTSTRAP_HOST_SOURCE: /tmp/opencode/DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf
APP_SEMANTIC_BOOTSTRAP: app private files/models/DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf
PAYLOAD_SOURCE: local HTTP RangeSource serving DeepSeek-V2-Lite.IQ2_XXS.vbuf
APP_ENDPOINT: http://127.0.0.1:18125
ADB_MAPPING: adb reverse tcp:18125 tcp:18125
GGML_SOURCE: /tmp/opencode/vbuf-batch-full/_deps/ggml_source-src
```

The endpoint and port are development-run configuration, not hardcoded generic
runtime policy. IQ1_S was not used. The semantic bootstrap contains metadata
and TensorRefs; the app did not preload the 5.64 GB payload.

## 8. App Lifecycle

The final run completed this lifecycle:

```text
APK build -> adb install -> launch -> Open
  -> Ready · direct vBuf-ML runtime
  -> prompt entry
  -> Generate 4 tokens
  -> Prefill · batched prompt rows
  -> Decode · autoregressive
  -> Completed
  -> output and metrics visible
```

The model was opened once before generation. `APP_START_TO_READY_MS` includes
the observed app launch-to-ready boundary; `MODEL_OPEN_MS` is the native model
session-open interval.

## 9. Progress / Output Semantics

The UI displays real native phase snapshots:

```text
TOKENIZING
PREFILL prompt_tokens=7
DECODE generated_tokens=0..4
COMPLETE generated_tokens=4
```

No fake token animation or synthetic progress was added. The final visible
generated output was:

```text
----
```

The first generated continuation was `-`, matching the prior direct-probe first
continuation. Four real decoded token pieces were displayed.

## 10. Cold Model Open

```text
APP_START_TO_READY_MS: 17172
MODEL_OPEN_MS: 193
```

The app-start interval includes process/UI launch and the user-triggered open
boundary. It is not a pure model-open benchmark. The APK was built with the
installed JDK 25 and Gradle wrapper and installed successfully on the Pixel.

## 11. Prompt Prefill

```text
APP_PREFILL_MODE: BATCHED
APP_PREFILL_MS: 126792
APP_PREFILL_LAYER_SEQUENCE_MS: 123131
DIRECT_PROBE_PREFILL_MS: 82231
APP_VS_DIRECT_PREFILL_DELTA_MS: 44561
APP_VS_DIRECT_PREFILL_DELTA_PERCENT: 54.19%
```

The direct-probe comparison is a separate single-run observation. The app and
probe used the same model/source architecture, but source/server cache state
was not independently reset. The app result is therefore an app integration
baseline, not a transport ceiling or a replacement for the direct-probe number.

## 12. TTFT

TTFT is defined here as the first real generated token becoming available after
prompt prefill. Model open is reported separately:

```text
PREFILL_MS: 126792
FIRST_DECODE_TOKEN_MS: 51597
TTFT_MS: 51598
```

The one-millisecond difference is boundary-clock overhead between the first
decode interval and the TTFT snapshot.

## 13. Decode Timing

```text
DECODE_TOKEN_1_MS: 51597
DECODE_TOKEN_2_MS: 51981
DECODE_TOKEN_3_MS: 61047
DECODE_TOKEN_4_MS: 55652
DECODE_MS_TOTAL: 220284
TOTAL_GENERATION_MS: 347088
MEAN_DECODE_TOKEN_MS: 55069.25
```

The four token values are measured native token-boundary intervals. The total
generation interval includes tokenization, batched prefill, and decode; decode
total is reported separately.

## 14. Four-Token Generation

```text
GENERATED_TOKEN_LIMIT: 4
GENERATED_TOKEN_COUNT: 4
TOKEN_1: -
TOKEN_2: -
TOKEN_3: -
TOKEN_4: -
VISIBLE_OUTPUT: ----
```

The direct runtime completed the bounded generation and returned real decoded
text to the UI. The app did not extend the generation bound for this baseline.

## 15. Source / Residency Metrics

The app uses cumulative counter snapshots at each decode-token boundary and
subtracts the preceding snapshot. The measured deltas are:

| Token | Time ms | Source requests | Source bytes | Hits | Misses | Evictions | Reload bytes | Peak resident bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 51,597 | 665 | 819,151,520 | 869 | 1,217 | 672 | 640,765,952 | 267,424,768 |
| 2 | 51,981 | 664 | 806,544,032 | 870 | 1,213 | 759 | 643,703,456 | 267,452,416 |
| 3 | 61,047 | 767 | 879,644,320 | 767 | 1,319 | 767 | 710,586,016 | 267,452,416 |
| 4 | 55,652 | 692 | 759,750,304 | 842 | 1,094 | 700 | 593,800,864 | 267,452,416 |

Global final counters were:

```text
SOURCE_REQUEST_COUNT: 4287
SOURCE_BYTES: 4745501920
RESIDENCY_HITS: 6107
RESIDENCY_MISSES: 7898
EVICTIONS: 4217
RELOAD_BYTES: 2791685088
PEAK_RESIDENT_BYTES: 267452416
RESIDENCY_WITHIN_256_MIB_CAP: YES
```

The per-token source deltas sum to `3,265,090,176` bytes. The remaining global
source activity belongs to prompt prefill and other non-decode work. Source
bytes, materialized bytes, resident bytes, reload bytes, and compute bytes are
distinct quantities. The local HTTP RangeSource serves source ranges; these
counters do not prove that every source byte was transferred over a network or
that each byte represents a unique model weight.

## 16. App vs Direct-Probe Comparison

```text
DIRECT_PROBE_BATCHED_PREFILL_MS: 82231
ANDROID_APP_BATCHED_PREFILL_MS: 126792
APP_PREFILL_OVERHEAD_MS: 44561
APP_PREFILL_OVERHEAD_PERCENT_OF_DIRECT: 54.19%
```

The difference is attributable only as an observed app/probe delta. Source
cache state, process lifecycle, UI/JNI boundary, and endpoint history were not
controlled tightly enough to assign the entire difference to JNI or UI work.

## 17. Correctness Evidence

```text
PROMPT_TOKEN_COUNT: 7
BATCHED_PREFILL_CONFIRMED: YES, UI/native phase and metrics
ORDERED_CAUSAL_ATTENTION: existing runtime path preserved
KV_CONTINUATION: existing state path preserved; decode began at position 7
FIRST_DECODE_PARITY: PASS, app token 1 '-' matches direct-probe continuation
NORMAL_REFERENCE_WORK: NOT_EXECUTED
RESIDENCY_CAP: preserved; peak 267452416 <= 268435456
TOPK_ACCUMULATION_ORDER: existing original TopK-rank path preserved
```

This app result is not a claim of full hidden-state or final-logit hash parity
for every prompt row. It is an end-to-end app integration result with the
qualified batching contracts and matching first continuation.

## 18. Decode Token Cost Attribution

The primary attribution calculations are:

```text
MEAN_DECODE_MS: 55069.25
MEAN_SOURCE_BYTES_PER_DECODE_TOKEN: 816272544
MEAN_SOURCE_REQUESTS_PER_DECODE_TOKEN: 697
MEAN_SOURCE_FRACTION_OF_MODEL: 0.1447338 (14.47%)
EFFECTIVE_SOURCE_BYTES_PER_SECOND: 14822656 (14.82 MB/s)
```

Per-token source fractions of the `5,639,819,878`-byte payload were:

```text
TOKEN_1_SOURCE_FRACTION_OF_MODEL: 0.145244 (14.52%)
TOKEN_2_SOURCE_FRACTION_OF_MODEL: 0.143009 (14.30%)
TOKEN_3_SOURCE_FRACTION_OF_MODEL: 0.155970 (15.60%)
TOKEN_4_SOURCE_FRACTION_OF_MODEL: 0.134712 (13.47%)
```

Existing inclusive actual compute intervals were also captured by subtracting
the runtime timing counters:

```text
TOKEN_1_ACTUAL_COMPUTE_INTERVAL_MS: 51487
TOKEN_2_ACTUAL_COMPUTE_INTERVAL_MS: 51875
TOKEN_3_ACTUAL_COMPUTE_INTERVAL_MS: 60927
TOKEN_4_ACTUAL_COMPUTE_INTERVAL_MS: 55486
MEAN_ACTUAL_COMPUTE_INTERVAL_MS: 54943.75
SOURCE_MATERIALIZATION_MS_PER_TOKEN: UNINSTRUMENTED
```

The actual compute intervals are inclusive runtime timing scopes and are not an
exclusive decomposition of token wall time. No smaps, smaps_rollup, or dumpsys
memory polling was used during the timing window.

## 19. Observer Effect / Instrumentation

The attribution metric uses existing source and residency counters plus bounded
snapshot scans and subtraction at token boundaries. It does not reset global
counters, alter source policy, add polling, or change the residency cap.
Materialization wait remains explicitly `UNINSTRUMENTED`.

## 20. Current UX Limitations

- Native decode remains long, with approximately `55.07 s` mean per token on this device.
- Progress is phase- and token-boundary based; current-layer progress is not exposed.
- The app is a single-turn bounded demo, not a production chat application.
- The endpoint is a development HTTP Range configuration and requires the local adb reverse mapping.
- The Android build emits a device warning about 16 KiB ELF page-size compatibility for bundled native libraries; APK install and runtime execution still passed on this Pixel.

## 21. Next Evidence-Driven Runtime Target

The measured result is classified as:

```text
DECODE_BOTTLENECK_CLASSIFICATION: MIXED
```

Primary evidence:

```text
SOURCE_RELOAD_EVIDENCE: 759750304-879644320 source bytes and
                         593800864-710586016 reload bytes per token;
                         664-767 source requests per token
COMPUTE_EVIDENCE:        51487-60927 ms inclusive actual compute intervals
UNRESOLVED_BOUNDARY:     exclusive source/materialization wait is uninstrumented
```

Both source/reload activity and compute intervals are substantial. The result
does not justify a source-only or compute-only conclusion. No warmup, cache,
prefetch, residency, or compute optimization was started by this task. Any next
investigation must first rank source/reload and compute contributions with
non-overlapping instrumentation.
