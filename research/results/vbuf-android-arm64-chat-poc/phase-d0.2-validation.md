# Phase D0.2 Validation

## 1. Objective

This experiment separates the old synchronous `smaps_rollup` observer cost from
the selected C1 remote loader and decomposes the remaining serialized payload
interval. No loader, transport, residency, format, tensor-order, request-count,
hashing, or llama.cpp behavior was changed.

## 2. D0.1 Initial Finding

Phase C selected C1 model open was `44.451 s`. The D0.1 smaps-off result was
`13.256 s`. The controlled A/B/A matrix below used the same Pixel 7 Pro,
artifact, server, model, semantic bootstrap, C1 keep-alive, 310 requests,
`633,495,552` bytes, and EAGER_ALL policy. The stable D0.1 instrumentation build
was not rebuilt between these three runs.

## 3. Controlled A/B/A Method

The sequence was:

1. A1: `debug.vbuf.d0_1.trace=1`, `debug.vbuf.d0_1.disable_smaps=1`.
2. B: `debug.vbuf.d0_1.trace=1`, smaps sampling enabled.
3. A2: `debug.vbuf.d0_1.trace=1`, `debug.vbuf.d0_1.disable_smaps=1`.

Each run completed model open and reached the `POST_GENERATION` diagnostic
boundary. The device thermal service was observable, but no stable thermal
normalization was available; BIG/LITTLE temperatures were recorded before B
and A2 in `/tmp/opencode/d0-2-b-thermal-before.txt` and
`/tmp/opencode/d0-2-a2-thermal-before.txt`.

## 4. Diagnostic Observer Effect

| Metric | A1 smaps off | B smaps on | A2 smaps off |
|---|---:|---:|---:|
| Model open | 13,616.145 ms | 41,663.965 ms | 13,543.247 ms |
| Serialized payload | 13,118.148 ms | 41,277.479 ms | 13,091.658 ms |
| Post-final-payload | 459.294 ms | 355.374 ms | 413.964 ms |
| smaps calls | 0 | 1,240 | 0 |
| smaps total | 0 ms | 34,695.475 ms | 0 ms |
| Requests | 310 | 310 | 310 |
| Bytes | 633,495,552 | 633,495,552 | 633,495,552 |
| Generation | PASS | PASS | PASS |
| Payload parity | NOT RUN | NOT RUN | NOT RUN |

The off-run mean is `13,579.696 ms`. A1/A2 absolute variance is `72.898 ms`,
or `0.537%` of the off-run mean. B adds `28,084.269 ms`, or `206.811%`, over
that mean. The result is therefore a confirmed benchmark-distorting observer
effect. It explains most of the difference between Phase C C1 and D0.1 off,
but not every millisecond of the historical result.

## 5. Serialized Payload Decomposition

The decomposition below uses the D0.1 B smaps-off run with
`MODEL_OPEN_TOTAL_NS=13,255,603,645` and
`SERIALIZED_PAYLOAD_NS=12,751,550,137`. Timers are nested and are not additive.

| Component | Calls | Bytes | Total | Median | P95 | Max | Inclusive/Exclusive |
|---|---:|---:|---:|---:|---:|---:|---|
| Request -> first body byte | 310 | N/A | 4,133.346 ms | 4.910 ms | 50.382 ms | NOT RECORDED | Inclusive per request |
| Body receive | 310 | 633,495,552 | 5,819.662 ms | 11.715 ms | 31.046 ms | NOT RECORDED | Inclusive per request |
| memcpy | 156,897 | 633,495,552 | 1,065.032 ms | N/A | N/A | N/A | Nested in body receive |
| hash | 310 | 633,495,552 | 2,278.201 ms | N/A | N/A | N/A | Nested in worker wait |
| worker wait | 310 | N/A | 12,513.421 ms | N/A | N/A | N/A | Inclusive |
| mutex wait | 310 | N/A | 0.099 ms | N/A | N/A | N/A | Nested in request |

Additional B counters were request setup `11.671 ms`, worker creation
`125.100 ms`, worker start delay `250.204 ms`, request finalization `2.704 ms`,
mutex held `9,964.159 ms`, and header setup `4,040.828 ms`.

## 6. HTTP First-Byte Analysis

`REQUEST_TO_FIRST_BODY_BYTE` is measured from `read_range` entry to the first
body byte accepted by the response callback. It is not a headers-received or
status-line boundary. The current instrumentation passes the body callback's
first-byte timestamp into the aggregate counter. A maximum was not serialized
by this stable build and is therefore not reported.

## 7. HTTP Body Transfer Analysis

Body receive is measured from first body byte to complete response body and
includes receive-side copying. The exact transfer total was
`5,819.662 ms` across 310 requests and `633,495,552` bytes. The server and
Android metrics confirm one persistent connection, `unique_bytes` equal to
the received bytes, and zero overfetch.

## 8. Worker and Mutex Analysis

The actual nesting is:

```text
serialized payload phase
└─ worker wait
   ├─ request setup and worker creation
   ├─ request -> first body byte
   ├─ body receive
   │  └─ memcpy
   ├─ hash
   └─ request finalization
```

`WORKER_WAIT_TOTAL` includes nested I/O, copying, hashing, and finalization.
The `9,964.159 ms` mutex-held total is substantial but is itself nested in the
serialized request path; mutex wait was only `0.099 ms`.

## 9. memcpy Cost

The receiver copied `633,495,552` bytes in `156,897` calls and
`1,065.032 ms`, an effective `0.595 GB/s` using decimal bytes and seconds.

## 10. Hash Cost

The existing FNV-1a validation scanned `633,495,552` bytes in 310 calls and
`2,278.201 ms`, an effective `0.278 GB/s`. Hash semantics were unchanged.

## 11. Descriptor and Callback Cost

The descriptor loop was `12,858.900 ms`; serialized payload was `12,751.550 ms`.
The approximately `107.349 ms` difference is small relative to the total and
does not support descriptor processing outside payload acquisition as the main
bottleneck.

Callback lookup performed `48,205` iterations in `5.526 ms` across 310 binds.
The linear walk remains measurable but negligible for this critical path.

## 12. Post-Payload Construction Cost

The B run measured `468.206 ms` from final payload ready to model-open
completion. Model creation was `0.038 ms`, backend allocation was `0.082 ms`,
and pointer binding was `0.026 ms`. Post-I/O llama.cpp construction is not a
dominant target in this run.

## 13. Backend Allocation and RSS Interpretation

The backend made one logical allocation of `633,495,552` bytes in `0.082 ms`.
RSS snapshots from the B smaps-off run were:

```text
BOOTSTRAP_COMPLETE          160,076 KiB
HALF_PAYLOADS_READY         404,764 KiB
FINAL_PAYLOAD_READY         795,200 KiB
BACKEND_ALLOCATION_COMPLETE 836,392 KiB
MODEL_OPEN_COMPLETE         896,856 KiB
POST_GENERATION             911,540 KiB
```

PSS was not sampled in the smaps-off run. The numeric zero in the legacy JSON
is not a measured zero-PSS value. The approximately `41,192 KiB` RSS increase
from final payload to backend allocation is much smaller than the logical
633,495,552-byte allocation. This is consistent with, but does not prove, that
the backend allocation did not immediately create an additional fully resident
633 MB copy. RSS alone cannot establish exact physical residency.

## 14. Phase C Measurement Reinterpretation

Phase C remains historical evidence for the loader as measured with its then
current diagnostics. The A/B/A result shows those diagnostics materially
inflated absolute timing. The Phase C baseline `53.203 s` and C1 `44.451 s`
remain useful for comparing connection reuse because they used equivalent
diagnostic conditions, but their absolute values should not be treated as clean
production-loader timing. C1 is not invalidated automatically; its relative
keep-alive conclusion remains conditionally useful.

## 15. Transport-Rate Consistency Check

Using decimal MB and the A1/A2 off-run mean:

```text
633.495552 MB / 13.579696 s = 46.650 MB/s model-open rate
633.495552 MB / 13.104903 s = 48.340 MB/s serialized-payload rate
```

These are not directly comparable to the historical C0 large-range figure of
`29.763 MB/s`: the D0.2 values use total 310-request loader timing or its
payload-phase boundary, while historical C0 used Pixel toybox `netcat` and a
different direct HTTP Range benchmark. A separate ARM64 source-transfer snapshot
used ADB reverse HTTP. The scopes, request
pattern, byte accounting, device/client path, server/cache state, and timing
boundaries are not equivalent. The apparent rate discrepancy therefore remains
unresolved rather than being treated as a real transport improvement.

## 16. Confirmed Bottlenecks

- Synchronous per-request `smaps_rollup` sampling is a confirmed observer effect.
- Serialized payload acquisition remains the dominant clean runtime interval.
- Worker wait is the dominant inclusive nested timer.
- The persistent HTTP mutex is held for `9.964 s`, while mutex acquisition wait is negligible.
- The run performs exactly 310 serialized C1 requests with no overfetch.

## 17. Eliminated Bottlenecks

- Callback lookup is not material at `5.526 ms` total.
- Backend allocation latency is not material at `0.082 ms`.
- Post-final-payload construction is not material at `0.468 s`.
- Descriptor work outside payload acquisition is not a dominant explanation.
- memcpy and hashing are measurable nested costs, but neither alone explains the clean `12.752 s` phase.

## 18. Remaining Unknowns

- First-byte and body-transfer maxima were not serialized by the stable build.
- Payload parity against the Phase C capture was not run.
- Direct-source regression was not run on the final APK.
- Physical PSS/residency remains unavailable without a valid smaps sample.
- The C0 transport-rate discrepancy requires a matched-scope rerun.

## 19. Recommended Next Experiment

Run one smaller, matched-scope transport-control experiment: repeat the C0
large-range characterization and the selected C1 310-request loader back to
back with the same server, device, model, byte accounting, cache state, and
lightweight diagnostics, without architectural changes. This directly targets
the only unresolved dominant interpretation after smaps is controlled: why the
clean 310-request payload rate differs from the historical C0 transport rate.

## Validation

```text
RUST_TESTS: PASSED (cargo test --workspace)
ANDROID_LOCAL_BUILD: PASSED (direct CMake/Ninja; Gradle remains unavailable)
ANDROID_REMOTE_BUILD: PASSED (manual APK packaging and device execution)
REMOTE_PAYLOAD_PARITY: NOT RUN
REMOTE_GENERATION: PASS (POST_GENERATION reached in A1, B, and A2)
DIRECT_SOURCE_REGRESSION: NOT RUN
JSON_VALIDATION: PASSED for captured stable-build JSON records
CCC_INDEX: PASSED
GIT_DIFF_CHECK: PASSED

PERSISTENT_FORMAT_CHANGED: NO
RESIDENCY_POLICY_CHANGED: NO
EAGER_ALL_CHANGED: NO
REQUEST_COUNT_POLICY_CHANGED: NO
TRANSPORT_POLICY_CHANGED: NO
PRODUCTION_SEMANTICS_CHANGED: NO
```

PHASE_D0_2_VALIDATION_COMPLETE:

A1_MODEL_OPEN_MS: 13616.145
B_MODEL_OPEN_MS: 41663.965
A2_MODEL_OPEN_MS: 13543.247

A_RUN_VARIANCE_PERCENT: 0.537
SMAPS_ON_PENALTY_MS: 28084.269
SMAPS_ON_PENALTY_PERCENT: 206.811
SMAPS_OBSERVER_EFFECT_CONFIRMED: YES

SERIALIZED_PAYLOAD_MS: 12751.550

REQUEST_COUNT: 310
HTTP_BYTES_RECEIVED: 633495552

REQUEST_TO_FIRST_BYTE_TOTAL_MS: 4133.346
REQUEST_TO_FIRST_BYTE_MEDIAN_MS: 4.910
REQUEST_TO_FIRST_BYTE_P95_MS: 50.382
REQUEST_TO_FIRST_BYTE_MAX_MS: NOT_RECORDED

HTTP_BODY_RECEIVE_TOTAL_MS: 5819.662
HTTP_BODY_RECEIVE_MEDIAN_MS: 11.715
HTTP_BODY_RECEIVE_P95_MS: 31.046
HTTP_BODY_RECEIVE_MAX_MS: NOT_RECORDED

WORKER_CREATE_TOTAL_MS: 125.100
WORKER_WAIT_TOTAL_MS: 12513.421

HTTP_MUTEX_WAIT_TOTAL_MS: 0.099
HTTP_MUTEX_HELD_TOTAL_MS: 9964.159

MEMCPY_CALLS: 156897
MEMCPY_BYTES: 633495552
MEMCPY_TOTAL_MS: 1065.032
MEMCPY_EFFECTIVE_GBPS: 0.595

HASH_CALLS: 310
HASH_BYTES: 633495552
HASH_TOTAL_MS: 2278.201
HASH_EFFECTIVE_GBPS: 0.278

CALLBACK_LOOKUP_ITERATIONS: 48205
CALLBACK_LOOKUP_TOTAL_MS: 5.526

BACKEND_ALLOCATED_BYTES: 633495552
BACKEND_ALLOCATION_TOTAL_MS: 0.082

FINAL_PAYLOAD_TO_MODEL_OPEN_COMPLETE_MS: 468.206

EFFECTIVE_MODEL_OPEN_MBPS: 46.650
EFFECTIVE_SERIALIZED_PAYLOAD_MBPS: 48.340
C0_LARGE_RANGE_MBPS: 29.763
TRANSPORT_RATE_DISCREPANCY_EXPLAINED: NO; timing scopes and cache/path conditions are not matched

DOMINANT_CONFIRMED_COST: serialized payload acquisition / inclusive worker wait
SECONDARY_CONFIRMED_COST: HTTP mutex-held serialized request path
ELIMINATED_BOTTLENECKS: smaps observer cost as production-loader cost; callback lookup; backend allocation latency; post-payload construction
REMAINING_UNKNOWN: matched C0/C1 transport-rate cause; payload parity; maxima; physical PSS

PHASE_C_ABSOLUTE_TIMINGS_DISTORTED: YES
PHASE_C_KEEPALIVE_RELATIVE_RESULT_STILL_VALID: CONDITIONALLY YES

RECOMMENDED_NEXT_EXPERIMENT: matched-scope C0 large-range versus C1 transport-control rerun

RUNTIME_CODE_CHANGED: NO
DIAGNOSTIC_CODE_CHANGED: YES
PERSISTENT_FORMAT_CHANGED: NO
RESIDENCY_POLICY_CHANGED: NO
TRANSPORT_POLICY_CHANGED: NO
PRODUCTION_SEMANTICS_CHANGED: NO
WORKTREE_STATUS: diagnostic/report changes plus pre-existing unrelated untracked IDE files; nothing committed or pushed
