# Phase D0.3 Transport Control

## 1. Why This Rerun Was Necessary

Phase C compared a historical single-range Pixel `netcat` probe at
`29.763 MB/s` with a complete 310-request Android model-open path. Phase D0.2
then showed that synchronous smaps sampling distorted the latter by more than
`28 s`. Those scopes and clients were not equivalent. D0.3 measures a
single-span control and the exact 310-range Keep-Alive plan through the same
current Android `HttpRangeSource` and artifact.

## 2. Historical C0 Scope

The historical evidence is `phase-c/transport-ceiling.json`. It used toybox
`netcat` on the Pixel over direct HTTP Range against `scripts/range_server.py`, one contiguous
`268,435,456`-byte (256 MiB) range, three samples, and selected the `9,019 ms`
median. Its calculation was:

```text
268,435,456 / 9.019 / 1,000,000 = 29.763 MB/s
268,435,456 / 9.019 / 1,048,576 = 28.385 MiB/s
```

The historical client, byte volume, timing scope, cache state, and diagnostic
state were not fully matched to D0.3. Classification: `NOT_DIRECTLY_COMPARABLE`.
The repository's separate ARM64 performance snapshot used ADB reverse HTTP;
that topology must not be conflated with this 29.763 MB/s record.
The historical record is retained unchanged.

## 3. Matched Experimental Setup

- Device: Pixel 7 Pro, Android 17, arm64-v8a.
- Artifact: the same semantic bootstrap and Qwen3 payload used by C1.
- Server: the existing `scripts/range_server.py` process on port 18765.
- Client: the current Android `HttpRangeSource` socket/HTTP implementation.
- Instrumentation: D0.1 lightweight counters, smaps disabled.
- Network: the same direct Wi-Fi path.
- Sequence: C0-1, C1-1, C0-2, C1-2.
- Cache state: warm-ish/unknown. The server process was kept alive and no
  destructive cache flush was performed. Interleaving limits order drift but
  does not prove cold-cache equivalence.

C0 is one contiguous span from the minimum to maximum payload extent. The
TensorRef layout has `4,040` bytes of gaps, so one exact contiguous request is
not possible without overfetch. C0 therefore requested `633,499,592` bytes to
carry `633,495,552` payload bytes. C1 requested the 310 exact ranges with zero
overfetch. Both use one persistent connection.

## 4. C0 Results

| Run | Ranges | Connections | Requested bytes | Payload bytes | Overfetch | Transport ms | MB/s | MiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0-1 | 1 | 1 | 633,499,592 | 633,495,552 | 4,040 | 6,179.243 | 102.520 | 97.772 |
| C0-2 | 1 | 1 | 633,499,592 | 633,495,552 | 4,040 | 6,113.800 | 103.617 | 98.810 |
| Median | 1 | 1 | 633,499,592 | 633,495,552 | 4,040 | 6,146.521 | 103.066 | 98.291 |

C0 request-to-first-byte total median across the two runs was `403.361 ms`.
Body receive total median was `5,441.360 ms`. The same HTTP receive path copied
`633,499,592` bytes in a median `162.866 ms`.

## 5. C1 Results

| Run | Requests | Connections | Requested bytes | Payload bytes | Overfetch | Transport ms | MB/s | MiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1-1 | 310 | 1 | 633,495,552 | 633,495,552 | 0 | 10,141.137 | 62.468 | 59.574 |
| C1-2 | 310 | 1 | 633,495,552 | 633,495,552 | 0 | 10,176.992 | 62.248 | 59.364 |
| Median | 310 | 1 | 633,495,552 | 633,495,552 | 0 | 10,159.064 | 62.358 | 59.469 |

C1 request-to-first-byte total median was `4,164.596 ms`; per-request median
and p95 were `5.695 ms` and `50.226 ms` using the two run medians. Body receive
total median was `5,852.766 ms`; per-request median and p95 were `11.714 ms`
and `31.249 ms`. Memcpy total median was `194.305 ms`.

## 6. Scope-Normalized Throughput Comparison

| Metric | Historical C0 | Matched C0 | Matched C1 |
|---|---:|---:|---:|
| Requests | 1 | 1 | 310 |
| Connections | 1 | 1 | 1 |
| Payload bytes | 268,435,456 synthetic range | 633,495,552 | 633,495,552 |
| Requested bytes | 268,435,456 | 633,499,592 | 633,495,552 |
| Overfetch bytes | N/A to model payload | 4,040 | 0 |
| Scope | toybox netcat transport | HttpRangeSource transport | HttpRangeSource transport |
| Transport time | 9,019 ms median | 6,146.521 ms median | 10,159.064 ms median |
| MB/s | 29.763 | 103.066 | 62.358 |
| MiB/s | 28.385 | 98.291 | 59.469 |
| smaps enabled | N/A/unknown | false | false |
| Cache state | unknown | warm-ish/unknown | warm-ish/unknown |

C1/C0 transport throughput ratio was `0.605`. C1 transport penalty was
`4,012.543 ms`, or `65.282%`, relative to matched C0. This is a meaningful
transport headroom signal, but it is not an exact causal decomposition of every
request boundary. The C1 aggregate request-to-first-byte total was `4.165 s`
versus `0.403 s` for C0, making request/response fragmentation the largest
measured transport difference.

## 7. Critical Scope Separation

The measurements have three distinct scopes:

```text
TRANSPORT_CONTROL:
  C0 6.147 s median; C1 10.159 s median

PAYLOAD_MATERIALIZATION:
  D0.2 C1 serialized payload 12.752 s
  includes transport, retained-buffer memcpy, and FNV-1a validation

FULL_MODEL_OPEN:
  D0.2 clean C1 off-run mean 13.580 s
  includes post-final-payload construction of 0.468 s
```

The C1 control deliberately performed no hashing and no GGML construction.
`C1_HASH_TOTAL_MS` remains `2,278.201 ms` from D0.2, and must not be added to
the pure C1 transport control. The C1 full model-open observations after the
control probe were `14,057 ms` and `14,256 ms`; those are contextual only
because the preceding control probe warmed the same artifact path.

## 8. Request-Fragmentation Cost

The matched single-span control is approximately `4.013 s` faster than the
exact 310-range control. The C1 request-to-first-byte aggregate is approximately
`3.762 s` above C0. Body transfer totals are close (`5.853 s` C1 versus
`5.441 s` C0), so the dominant transport difference is request/response
boundary and serialized per-range latency rather than raw body throughput.

This supports request fragmentation as a meaningful remaining transport cost,
but does not authorize changing scheduling, concurrency, batching, or protocol
semantics in this phase.

## 9. Cache and Timing Considerations

The server stayed running across the interleaved sequence. No filesystem cache,
proxy cache, kernel cache, or Android cache was flushed. The run state is
therefore warm-ish/unknown, not cold. The same server, client, artifact, device,
and instrumentation were used in all four runs, which makes the C0/C1 result
more controlled than the historical comparison but does not explain all
historical cache differences.

## 10. Historical C0 Reconciliation

The historical `29.763 MB/s` is not a clean ceiling for this loader. It used a
different client (`netcat`), a 256 MiB synthetic range rather than the complete
633.5 MB payload, and an incompletely documented cache/timing state. The separate
ARM64 source-transfer record used ADB reverse HTTP, but is a different historical
measurement. The matched
current single-span control reached `103.066 MB/s` with the current HTTP client,
while matched C1 reached `62.358 MB/s`. The historical discrepancy is therefore
explained as a scope/client/cache-control mismatch, not as evidence that C1
exceeds a validated transport ceiling. The historical record remains useful as
context but is only partially comparable.

## 11. Architectural Implication

There is meaningful transport headroom between matched large-range transport
and the selected 310-range Keep-Alive path. The evidence supports a request
fragmentation cost, but it does not justify implementing an optimization yet.
The correct next step is one bounded load-plan/request-scheduling experiment,
with exact 310 logical ranges, one connection, zero overfetch, EAGER_ALL, and
unchanged payload semantics.

## 12. Correctness and Validation

```text
REMOTE_GENERATION: PASS in D0.2 qualification; D0.3 control itself did not generate
PAYLOAD_PARITY: PASS in Phase C byte-for-byte qualification; D0.3 validates exact lengths/ranges
DIRECT_SOURCE_REGRESSION: NOT RUN in D0.3
RUST_TESTS: PASSED (cargo test --workspace)
ANDROID_LOCAL_BUILD: PASSED (direct CMake/Ninja)
ANDROID_REMOTE_BUILD: PASSED (manual APK packaging and device execution)
JSON_VALIDATION: PASSED for D0.3 control records
CCC_INDEX: PASSED (`ccc index`)
GIT_DIFF_CHECK: PASSED (`git diff --check`)

PERSISTENT_FORMAT_CHANGED: NO
RESIDENCY_POLICY_CHANGED: NO
EAGER_ALL_CHANGED: NO
PRODUCTION_TRANSPORT_POLICY_CHANGED: NO
PRODUCTION_SEMANTICS_CHANGED: NO
```

The evidence correction status is:

```text
README_INTERPRETATION_CORRECTED: YES
PHASE_C_INTERPRETATION_CORRECTED: YES
PHASE_D0_2_INTERPRETATION_CORRECTED: YES
PHASE_D0_3_SCOPE_CORRECTED: YES
HISTORICAL_ADB_PATH_VERIFIED: YES, for the separate ARM64 source-transfer snapshot; not the 29.763 MB/s Phase C C0 record
HISTORICAL_C0_CLASSIFICATION: NOT_DIRECTLY_COMPARABLE
RAW_HISTORICAL_EVIDENCE_PRESERVED: YES
PRODUCTION_LOADING_CHANGED: NO
RUNTIME_CODE_CHANGED: NO
DIAGNOSTIC_CODE_CHANGED: YES
PERSISTENT_FORMAT_CHANGED: NO
RESIDENCY_POLICY_CHANGED: NO
PRODUCTION_TRANSPORT_POLICY_CHANGED: NO
```

PHASE_D0_3_TRANSPORT_CONTROL_COMPLETE:

HISTORICAL_C0_CLASSIFICATION: NOT_DIRECTLY_COMPARABLE

MATCHED_C0_RUNS: 2
MATCHED_C0_RANGE_COUNT: 1
MATCHED_C0_CONNECTIONS: 1
MATCHED_C0_PAYLOAD_BYTES: 633495552
MATCHED_C0_REQUESTED_BYTES: 633499592
MATCHED_C0_OVERFETCH_BYTES: 4040
MATCHED_C0_MEDIAN_MS: 6146.521
MATCHED_C0_MBPS: 103.066
MATCHED_C0_MIBPS: 98.291

MATCHED_C1_RUNS: 2
MATCHED_C1_REQUESTS: 310
MATCHED_C1_CONNECTIONS: 1
MATCHED_C1_PAYLOAD_BYTES: 633495552
MATCHED_C1_REQUESTED_BYTES: 633495552
MATCHED_C1_OVERFETCH_BYTES: 0
MATCHED_C1_TRANSPORT_MEDIAN_MS: 10159.064
MATCHED_C1_MBPS: 62.358
MATCHED_C1_MIBPS: 59.469

C1_TO_C0_THROUGHPUT_RATIO: 0.605
C1_TRANSPORT_PENALTY_MS: 4012.543
C1_TRANSPORT_PENALTY_PERCENT: 65.282

C1_REQUEST_TO_FIRST_BYTE_TOTAL_MS: 4164.596
C1_BODY_RECEIVE_TOTAL_MS: 5852.766
C1_MEMCPY_TOTAL_MS: 194.305
C1_HASH_TOTAL_MS: 2278.201
C1_SERIALIZED_PAYLOAD_MS: 12751.550
C1_FULL_MODEL_OPEN_MS: 13579.696 clean D0.2 baseline; 14156.5 post-control contextual median

TRANSPORT_HEADROOM_CONFIRMED: YES
HISTORICAL_C0_DISCREPANCY_EXPLAINED: YES, historical client/scope/cache state was not matched
CACHE_STATE_CONTROLLED: PARTIALLY, interleaved warm-ish/unknown with no flush
TIMING_SCOPE_MATCHED: YES for matched C0/C1 transport controls

REMOTE_GENERATION: PASS inherited from D0.2 same artifact qualification
PAYLOAD_PARITY: PASS inherited from Phase C byte-for-byte qualification
DIRECT_SOURCE_REGRESSION: NOT RUN

DOMINANT_REMAINING_TRANSPORT_COST: serialized request/response first-byte latency across 310 ranges
TRANSPORT_ARCHITECTURE_CHANGE_JUSTIFIED: NOT YET; controlled follow-up is required

RECOMMENDED_NEXT_EXPERIMENT: bounded load-plan/request-scheduling experiment

RESEARCH_SUMMARY_PATH: research/results/vbuf-android-arm64-chat-poc/phase-d0.3-transport-control.md

RUNTIME_CODE_CHANGED: NO
DIAGNOSTIC_CODE_CHANGED: YES
PERSISTENT_FORMAT_CHANGED: NO
RESIDENCY_POLICY_CHANGED: NO
PRODUCTION_SEMANTICS_CHANGED: NO
COMMIT_PERFORMED: NO
PUSH_PERFORMED: NO
WORKTREE_STATUS: diagnostic-only transport harness and report changes; unrelated IDE files preserved
