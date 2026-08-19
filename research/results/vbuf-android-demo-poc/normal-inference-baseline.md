# Normal Inference Baseline

## 1. Objective

Separate the expensive qualification/reference oracle from the normal direct
vBuf-ML Android inference path, without changing actual runtime math,
materialization, source resolution, residency, transport, or model semantics.

## 2. Existing Qualification Execution Structure

The direct path is:

```text
JNI.generate()
  -> DirectSession::generate()
    -> run_embedding()
    -> run_sequence()
      -> compute_token()                 actual attention
      -> compute_token()                 reference attention
      -> run_layer()/run_dense_layer()  actual FFN/router
      -> run_layer()/run_dense_layer()  reference FFN/router
    -> run_output_head()                actual output
    -> run_output_head()                reference output
    -> greedy()
```

The reference operations were qualification-only:

| Operation | Classification | Failure/comparison |
| --- | --- | --- |
| Duplicate embedding graph | Qualification-only | Embedding output parity |
| Reference attention/KV state | Qualification-only | Attention output parity |
| Scalar RMSNorm | Qualification-only | Normalized activation parity |
| Scalar router logits | Qualification-only | `router_score_parity`; fail closed in `route_activation()` |
| Reference TopK/weights | Qualification-only | Selected-weight parity |
| Reference selected-expert FFN | Qualification-only | Expert and routed-merge parity |
| Reference shared FFN | Qualification-only | Shared-expert parity |
| Reference residual composition | Qualification-only | Final layer parity |
| Duplicate output head | Qualification-only | Logits parity and token comparison |
| Actual GGML attention, FFN, output head, sampling | Required for inference | Runtime failure on execution error |
| Materialization, leases, residency, source reads | Required for inference | Existing readiness/failure contract |

Relevant implementation boundaries are in
`integrations/ggml/tools/multi_layer_poc16.cpp:379-667`,
`full_moe_layer_poc13.cpp:175-330`, and
`multi_expert_moe_poc12.cpp:60-215`.

## 3. Runtime Mode Boundary

`integrations/ggml/include/vbuf_runtime_mode.h` defines the explicit
`RuntimeMode` seam:

```text
NormalInference -> actual computation only
Qualification   -> existing actual + reference + parity behavior
```

The Android demo defaults to `NormalInference`. A qualification Android native
build is selected with `VBUF_ANDROID_QUALIFICATION`; reusable PoC callers retain
qualification as their default. This leaves the Java/JNI API unchanged.

Normal mode skips reference graph execution, scalar reference calculations,
reference materialization requests, reference KV updates, and parity checks.
Actual graph construction and all vBuf-ML source/materialization/residency calls
remain unchanged.

## 4. Correctness / Qualification Preservation

- Qualification position 0 emitted repeated `expert_parity`, attention parity,
  and block parity passes with zero reported error.
- The existing D2.3 full four-position qualification remains the authoritative
  full-generation control: router parity, logits parity, token parity, and
  teardown passed.
- The new mode contract test passes and asserts the mode gate directly.
- No parity tolerance or parity definition changed.
- Actual computation, tensor IDs, physical ranges, readiness waits, leases,
  residency capacity/policy, source endpoint, tokenization, KV state, sampling,
  and GGML operation definitions were not changed.

## 5. Physical Pixel Configuration

```text
DEVICE: Pixel 7 Pro, Android 17, arm64-v8a
MODEL: DeepSeek-V2-Lite
QUANTIZATION: IQ2_XXS
SEMANTIC_ARTIFACT: DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf
PROMPT: Explain the purpose of bounded generation
PROMPT_TOKEN_COUNT: 7
PROMPT_TOKEN_COUNT_SOURCE: MEASURED
FULL_PROMPT_PREFILL_MEASURED: NO
MAX_GENERATED_TOKENS: 4
RESIDENCY_CAP: 268435456 bytes (256 MiB)
ENDPOINT: http://127.0.0.1:18124 via adb reverse
GGML_CPU: enabled; OpenMP disabled as existing Android configuration
```

## 6. Qualification Control Timing

One complete prompt position was measured in qualification mode:

| Component | Time | Scope |
| --- | ---: | --- |
| Position total | 73,588 ms | Inclusive |
| Actual attention | 7,943 ms | Inclusive actual operation interval |
| Reference attention | 5,704 ms | Inclusive reference operation interval |
| Actual FFN | 39,124 ms | Inclusive actual FFN/router interval |
| Reference FFN | 17,972 ms | Inclusive reference FFN/router interval |
| Actual output head | 2,711 ms | Inclusive |
| Reference output head | 96 ms | Inclusive |
| Actual compute subtotal | 49,778 ms | Derived sum of actual components |
| Reference compute subtotal | 23,772 ms | Derived sum of reference components |

Qualification cumulative counters at that position:

```text
HTTP requests: 1507
HTTP bytes: 1530495648
Residency hits/misses/evictions: 4121/2383/1428
Reload bytes: 650851328
Peak resident bytes: 267595776
```

The reference subtotal excludes no measured reference component, but all
component timings are inclusive and should not be added to transport or nested
worker counters as exclusive time.

## 7. Normal Inference Timing

Two complete prompt positions were measured with `NormalInference`:

| Position | Total | Actual attention | Reference attention | Actual FFN | Reference FFN | Output head |
| ---: | ---: | ---: | --- | ---: | --- | ---: |
| 0 | 35,397 ms | 7,407 ms | `NOT_EXECUTED` | 25,038 ms | `NOT_EXECUTED` | 2,759 ms |
| 1 | 40,060 ms | 9,206 ms | `NOT_EXECUTED` | 28,422 ms | `NOT_EXECUTED` | 2,696 ms |

Normal position-0 cumulative counters:

```text
HTTP requests: 767
HTTP bytes: 879644320
Residency hits/misses/evictions: 767/1319/696
Reload bytes: 0
Peak resident bytes: 267354112
```

Normal position-1 cumulative counters:

```text
HTTP requests: 1533
HTTP bytes: 1746681152
Residency hits/misses/evictions: 1535/2634/1455
Reload bytes: 582949888
Peak resident bytes: 267354112
```

`MATERIALIZATION_WAIT_MS_TOTAL` remains `NOT_INSTRUMENTED`; the existing
readiness boundary and worker progress are preserved, but no new wait timer was
introduced into the generic materializer.

## 8. Timing Comparison

Position 0 is the directly comparable control:

| Metric | Qualification | Normal inference | Delta |
| --- | ---: | ---: | ---: |
| Position total | 73,588 ms | 35,397 ms | -38,191 ms |
| Layer sequence | 70,743 ms | 32,445 ms | -38,298 ms |
| Actual attention | 7,943 ms | 7,407 ms | -536 ms |
| Reference attention | 5,704 ms | `NOT_EXECUTED` | -5,704 ms |
| Actual FFN | 39,124 ms | 25,038 ms | -14,086 ms |
| Reference FFN | 17,972 ms | `NOT_EXECUTED` | -17,972 ms |
| Output head | 2,711 ms actual + 96 ms reference | 2,759 ms actual | comparable actual |
| Materialization wait | `NOT_INSTRUMENTED` | `NOT_INSTRUMENTED` | unresolved |
| HTTP requests | 1,507 | 767 | configuration/cache-sensitive |
| HTTP bytes | 1,530,495,648 | 879,644,320 | configuration/cache-sensitive |
| Residency hits | 4,121 | 767 | qualification includes reference path |
| Residency misses | 2,383 | 1,319 | qualification includes reference path |
| Evictions | 1,428 | 696 | qualification includes reference path |
| Reload bytes | 650,851,328 | 0 | qualification includes reference path |

Derived from measured position totals:

```text
POSITION_SPEEDUP: 2.079x
POSITION_TIME_REDUCTION: 51.9%
MEASURED_REFERENCE_COMPONENT_SUBTOTAL_MS: 23772
OBSERVED_POSITION_0_WALL_CLOCK_DELTA_MS: 38191
UNATTRIBUTED_OR_NOT_SEPARATELY_INSTRUMENTED_DELTA_MS: 14419
```

The total-time difference is not attributed exclusively to reference compute.
The measured reference subtotal is `23,772 ms`; the observed wall-clock delta is
`38,191 ms`, leaving `14,419 ms` unattributed or not separately instrumented.
Materialization and residency behavior also differ because qualification makes
additional source and residency accesses. No cause is assigned to the
remainder without a direct measurement.

The earlier `~109.7 s` result is historical diagnostic evidence, not part of
this controlled A/B comparison. The controlled values are:

```text
HISTORICAL_DIAGNOSTIC_POSITION_MS: ~109700
CONTROLLED_QUALIFICATION_POSITION_MS: 73588
CONTROLLED_NORMAL_POSITION_0_MS: 35397
CONTROLLED_NORMAL_POSITION_1_MS: 40060
```

### Counter Semantics

`HttpRangeSource::metrics().requests` counts completed HTTP range requests and
`.bytes` counts returned/requested range bytes accumulated by the source. They
are not residency events and are not expected to equal `hits + misses`.

The Android residency surface counts trace events of kind `Hit`, `Miss`, and
`Evict`. `reload_bytes` is derived from inserted resident payload bytes whose
tensor identity was previously observed by the runtime. An eviction alone does
not imply a reload; therefore `reload_bytes=0` can coexist with evictions for a
position when no previously observed identity is reinserted in that interval.

The normal peak resident value was `267,354,112` bytes versus the unchanged
`268,435,456` byte cap. It is within the cap with approximately `1,081,344`
bytes, or `1.03 MiB`, of headroom. This is vBuf-ML residency, not process RSS.

## 9. Remaining Bottlenecks

Normal mode's measured position-0 subtotal is dominated by actual FFN
(`25,038 ms`), followed by actual attention (`7,407 ms`) and output head
(`2,759 ms`). Materialization wait is not separately measured. The next primary
optimization target is therefore **prompt prefill batching**, not another
reference-path change: the current normal single-position cost is now the
relevant baseline and the prompt has seven positions.

## 10. Prompt Prefill Structure Audit

`DirectSession::generate()` encodes seven prompt tokens and runs:

```cpp
for (uint32_t prompt_token : prompt_tokens) {
    input = prompt_token;
    run_step(input, position++);
}
```

This is `vbuf_android_direct.cpp:297-326`. Therefore:

```text
PREFILL_BATCHED: NO
PROMPT_POSITIONS_EXECUTED_SERIALLY: YES
GGML_MULTI_TOKEN_SHAPES: NO
CURRENT_PREFILL_SHAPE: effectively 2048 x 1 per position
```

No batching was implemented in this task.

## 11. Existing Parallelism Audit

```text
POSITION_LEVEL_PARALLELISM: NO
LAYER_LEVEL_PARALLELISM: NO
INTRA_GGML_PARALLELISM: UNKNOWN/PARTIAL
MATERIALIZATION_COMPUTE_OVERLAP: PARTIAL
REFERENCE_ACTUAL_OVERLAP: NO
```

Positions and layers are ordinary serial loops. Materializer workers can run
asynchronously, but the dependency executor waits when a persistent input is
consumed; no compute/prefetch overlap was added. The Android CMake configuration
disables OpenMP, and the application does not set a normal-mode GGML thread
count. GGML CPU backend internal scheduling may still use backend threads, so
intra-GGML parallelism is not claimed as a measured fact. The only explicit
`ggml_backend_cpu_set_n_threads` call is behind the existing audit environment
variable in `vbuf_tensor_wave.cpp:398-400`.

## 12. Expected Demo Runtime

The prompt contains seven tokens. Using the measured normal position mean
`(35,397 + 40,060) / 2 = 37,728.5 ms`:

```text
EXPECTED_PROMPT_PREFILL_TIME: ~264,103 ms (7 x mean; extrapolated)
EXPECTED_4_TOKEN_DECODE_TIME: ~150,916 ms (4 x mean; extrapolated)
EXPECTED_TOTAL_DEMO_TIME: ~415,019 ms (~6.9 min; extrapolated)
```

Classification: prompt token count and position timings are `MEASURED`;
speedup, reduction, component subtotals, wall-clock delta, and the seven-token
projection are `DERIVED`; materialization wait is `UNINSTRUMENTED`; full
seven-token prefill and complete normal four-token generation are `NOT MEASURED`.

Generated positions may differ from prompt positions because KV history and
source/residency state differ. No complete normal four-token result was claimed
in this baseline run.

## 13. Next Optimization Target

Implement **prompt prefill batching** next. It is the highest-value target
because normal inference now costs approximately 37.7 seconds per serial prompt
position, actual FFN is the largest measured component, and the current seven
position prompt implies roughly 4.4 minutes of prefill before decode. Do not
implement that optimization as part of this baseline.

## 14. Limitations

- The full APK build was blocked by the environment: Android SDK was initially
  unset, and the installed JDK 21 has no `javac`; no JDK 17 is installed.
- Measurements used a temporary native probe containing the same
  `DirectSession`; the probe was outside the repository and was not retained.
- Qualification and normal counters are cumulative at position boundaries and
  are affected by the bounded residency/source cache state; they are not
  exclusive per-layer transport counters.
- Materialization wait was not separately instrumented.
- The complete four-token normal demo was not run to terminal output, so the
  expected demo runtime is extrapolated.
- Existing CTest artifacts were stale/incomplete: 18 tests were `Not Run` due
  missing executables. The neutrality test passed. The new mode contract passed
  as a standalone compile/run; a fresh native CMake configure was blocked by
  the available GGML checkout being commit `4c1a0af40`, not the pinned
  `2d191b5dee1a...` required by the repository CMake file.

## Final Status

```text
VBUF_ML_NORMAL_INFERENCE_BASELINE_COMPLETE: YES, bounded baseline complete

BRANCH: vbuf-ml
HEAD: c006130cd8e32eb3ea60cf6c03ad62d9eb2a4cfe

DEVICE: Pixel 7 Pro, Android 17, arm64-v8a
MODEL: DeepSeek-V2-Lite
QUANTIZATION: IQ2_XXS
RESIDENCY_CAP: 256 MiB
MEASURED_PEAK_RESIDENT_BYTES: 267354112
RESIDENCY_WITHIN_CAP: YES; approximately 1.03 MiB headroom
PROMPT: Explain the purpose of bounded generation
PROMPT_TOKEN_COUNT: 7

RUNTIME_MODE_MECHANISM: RuntimeMode plus VBUF_ANDROID_QUALIFICATION native build seam

QUALIFICATION_REFERENCE_PATH: actual + reference embedding/attention/FFN/output, scalar controls, parity
NORMAL_REFERENCE_PATH: not invoked; reference fields report NOT_EXECUTED

QUALIFICATION_CORRECTNESS_PRESERVED: YES for measured position and existing D2.3 full control
PARITY_CONDITION_CHANGED: NO
MODEL_SEMANTICS_CHANGED: NO
MATERIALIZATION_CHANGED: NO
RESIDENCY_CHANGED: NO
TRANSPORT_CHANGED: NO

QUALIFICATION_POSITION_MS: 73588
NORMAL_POSITION_MS: 35397 (position 0); 40060 (position 1)
POSITION_SPEEDUP: 2.079x (position 0)
POSITION_TIME_REDUCTION_PERCENT: 51.9%

QUALIFICATION_LAYER_SEQUENCE_MS: 70743
NORMAL_LAYER_SEQUENCE_MS: 32445 (position 0)

MEASURED_REFERENCE_COMPONENT_SUBTOTAL_MS: 23772
OBSERVED_WALL_CLOCK_DELTA_MS: 38191
UNATTRIBUTED_DELTA_MS: 14419
NORMAL_REFERENCE_COMPUTE: NOT_EXECUTED

NORMAL_ACTUAL_ATTENTION_MS: 7407 (position 0)
NORMAL_ACTUAL_FFN_MS: 25038 (position 0)
NORMAL_OUTPUT_HEAD_MS: 2759 (position 0)
NORMAL_MATERIALIZATION_WAIT_MS: NOT_INSTRUMENTED

HTTP_REQUEST_COUNT: 767 (normal position 0); 1507 (qualification position 0)
HTTP_BYTES: 879644320 (normal position 0); 1530495648 (qualification position 0)

RESIDENCY_HITS: 767 normal / 4121 qualification
RESIDENCY_MISSES: 1319 normal / 2383 qualification
EVICTIONS: 696 normal / 1428 qualification
RELOAD_BYTES: 0 normal / 650851328 qualification

PROMPT_PREFILL_IMPLEMENTATION: serial run_step per tokenizer output
PREFILL_BATCHED: NO
PROMPT_POSITIONS_EXECUTED_SERIALLY: YES

POSITION_LEVEL_PARALLELISM: NO
LAYER_LEVEL_PARALLELISM: NO
INTRA_GGML_PARALLELISM: UNKNOWN/PARTIAL
MATERIALIZATION_COMPUTE_OVERLAP: PARTIAL

EXPECTED_PROMPT_PREFILL_TIME: ~264103 ms, extrapolated
EXPECTED_4_TOKEN_DECODE_TIME: ~150916 ms, extrapolated
EXPECTED_TOTAL_DEMO_TIME: ~415019 ms, extrapolated
DERIVED_SERIAL_PREFILL_RANGE: ~247779-280420 ms (~4.13-4.67 min)
DERIVED_SERIAL_PREFILL_MEAN: ~264100 ms (~4.40 min)
HISTORICAL_DIAGNOSTIC_POSITION_MS: ~109700 ms

PRIMARY_REMAINING_BOTTLENECK: actual FFN within serial prompt prefill
RECOMMENDED_NEXT_OPTIMIZATION: prompt prefill batching

CARGO_TEST_WORKSPACE: PASS
ANDROID_BUILD: BLOCKED_ENVIRONMENT_MISSING_JAVAC
ANDROID_NATIVE_BUILD: PASS, normal and qualification variants
CTEST: UNAVAILABLE_FROM_EXISTING_STALE_BUILD; 18 tests Not Run
NEUTRALITY_GUARD: PASS, FORBIDDEN_LEAKAGE_COUNT=0

RESEARCH_REPORT: research/results/vbuf-android-demo-poc/normal-inference-baseline.md
REPORT_JNI_TIMING_UPDATED: YES, historical diagnostic explicitly separated
REPORT_ROUTER_AUDIT_UPDATED: YES, later serial-position explanation added

PRODUCTION_CODE_CHANGED: YES, explicit mode seam and low-overhead timing metrics
TEMPORARY_INSTRUMENTATION_REMOVED: YES, temporary native probe/receiver removed; intentional bounded baseline metrics retained

COMMIT_PERFORMED: NO
PUSH_PERFORMED: NO
FINAL_WORKTREE_STATUS: expected implementation/report changes plus pre-existing untracked router-parity audit
```

### Direct Answers

1. The qualification/reference path cost `23,772 ms` of directly measured
   reference component time in position 0; the end-to-end position reduction was
   `38,191 ms` or `51.9%`.
2. Normal direct runtime measured `35,397 ms` for position 0 and `40,060 ms`
   for position 1 under the unchanged 256 MiB design.
3. Yes. Prompt prefill is token-by-token serial and executes one full ordinary
   position per token.
4. Macro-level position/layer parallelism is absent. Materialization workers
   are asynchronous at the source layer, and GGML may use internal CPU
   scheduling, but the application does not explicitly configure or measure a
   normal-mode thread count.
5. Prompt prefill batching is the single highest-value next optimization based
   on the new normal baseline; it was not implemented here.
