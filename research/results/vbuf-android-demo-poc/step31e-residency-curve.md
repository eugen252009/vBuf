# Step 31E Physical Residency Curve Qualification

Date: 2026-08-20
Starting commit: `83c954306727645629516c4b745d61f62725f8b7`
Status: **IMPLEMENTED / PHYSICALLY MEASURED; CURVE INCOMPLETE ABOVE 256 MiB**

## Objective

Measure whether the remaining decode/FFN cost is dominated by active-RAM
residency churn or by GGML/MoE execution. The persistent sparse mirror and
source acquisition policy were held constant; only the active residency budget
was varied.

## Evidence Classification

- **CODE-AUDITED:** `TensorResidencyStore` already accepts a numeric budget and
  keeps the same cost-aware replacement algorithm for every point. The Android
  qualification seam now supplies only that numeric value.
- **HOST-MEASURED:** Release CTest, the focused Debug progressive-source
  contract, Rust tests, and neutrality checks are reported separately below.
- **PHYSICALLY-MEASURED:** Pixel 7 Pro runs, source locality, residency events,
  timing, and output are recorded in the curve table.
- **DERIVED:** mean decode values and relative comparisons are calculated only
  from completed decode tokens.
- **UNINSTRUMENTED:** the exact internal `AdapterError` behind the failed
  larger-cap first decode was not surfaced by the existing direct-session error
  path. No OOM or Android process-kill event was observed.
- **HISTORICAL:** Step 31C/31D values remain historical controls and are not
  substituted for the new curve measurements.

## Step 31D Result Read

The committed Step 31D report records both physical runs as completed:

| Metric | Cold | Warm |
|---|---:|---:|
| Remote bytes | 1,380,311,040 | 0 |
| Upstream requests | 1,850 | 0 |
| Prefill | 179,800 ms | 67,329 ms |
| Decode total | 109,632 ms | 102,465 ms |
| FFN | 234,585 ms | 133,824 ms |
| Peak resident | 267,452,416 bytes | 267,452,416 bytes |
| Coverage after | 336,990 chunks / 1,380,311,040 rounded bytes | same |

The persistent coverage bitmap was `172,114` bytes; the sidecar file including
its header was `172,194` bytes. Both runs produced `----`.

## Minimal Qualification Seam

`vbufResidencyBudgetBytes` is a positive decimal Gradle property. It defaults to
`268435456` and becomes the native `VBUF_RESIDENCY_BUDGET_BYTES` definition.
No eviction, retention, source, materialization, backend, or inference policy
was changed.

The generic residency store now exposes aggregate counters:

- `materialization_count`: every materialization recorded at the existing
  residency boundary;
- `reacquisition_count`: a repeated materialization for a tensor reference
  that did not hit the active store. This is a residency/materialization
  counter, not a remote-source counter.

No per-tensor log spam or persistent-source changes were added.

## Fixed Workload

```text
Device: Pixel 7 Pro, Android 17, arm64-v8a
Model: DeepSeek-V2-Lite IQ2_XXS
Payload: 5,639,819,878 bytes
RuntimeMode: NormalInference
Prompt: Explain the purpose of bounded generation
Prompt tokens: 7
Prefill: batched
Decode: 4 bounded tokens
GGML/thread/runtime configuration: unchanged from Step 31D
```

Each point used a fresh app process. The payload mirror and coverage sidecar
were retained. The measured workload was already covered before the first
point and every run reported zero remote bytes and zero remote requests.

## Residency Curve

| Residency cap | Remote bytes | Local bytes | Evictions | Reacquisitions | Prefill | Mean decode | FFN | Peak resident |
|---------------|--------------|-------------|-----------|----------------|---------|-------------|-----|---------------|
| 256 MiB | 0 | 4,745,501,920 | 4,217 | 3,025 | 69,699 ms | 25,279 ms | 134,436 ms | 267,452,416 |
| 512 MiB | 0 | 1,565,833,472 | 1,035 | 308 | completed, 68,803 ms | UNINSTRUMENTED | UNINSTRUMENTED | 536,003,584 |
| 1 GiB | 0 | 1,283,370,240 | 289 | 2 | completed, 66,367 ms | UNINSTRUMENTED | UNINSTRUMENTED | 1,073,377,280 |
| 2 GiB | 0 | 1,277,583,616 | 0 | 0 | completed, 66,056 ms | UNINSTRUMENTED | UNINSTRUMENTED | 1,277,582,944 |

The 256 MiB point completed all four decode tokens. Its per-token decode times
were `23,647`, `24,274`, `27,026`, and `26,172` ms. The 512 MiB, 1 GiB, and
2 GiB points completed batched prefill, then failed on the first decode at the
existing `direct runtime execution failed` boundary. They therefore have no
valid decode mean, FFN total, output, or parity result.

| Residency cap | Materializations | Consumer requests | Failure/output |
|---|---:|---:|---|
| 256 MiB | 4,287 | 4,287 | completed, `----` |
| 512 MiB | 1,490 | 1,490 | first decode failed; `GEN_FAIL direct runtime execution failed` |
| 1 GiB | 1,160 | 1,160 | first decode failed; `GEN_FAIL direct runtime execution failed` |
| 2 GiB | 1,158 | 1,158 | first decode failed; `GEN_FAIL direct runtime execution failed` |

All points reported `REMOTE_BYTES=0`, `REMOTE_REQUESTS=0`, unchanged
`336,990` covered chunks, and `172,114` coverage bytes. The source locality
condition was therefore satisfied and did not contaminate the comparisons.

## Interpretation

The partial residency counters fall sharply as the budget grows: 4,217 to
1,035 to 289 to 0 evictions, and 3,025 to 308 to 2 to 0 repeated
materializations. This proves that the larger budgets change active residency
behavior, but the larger points do not complete the same workload. Their
latency cannot be compared as successful inference runs.

The result is **INCONCLUSIVE / DEVICE-CONSTRAINED**, not Class A-D. No working-
set threshold was observed. The only completed point is 256 MiB, so no larger
cap is a best-performing cap or production recommendation. Step 31D's
completed warm run still measured FFN/MoE as the dominant inclusive generation
scope, while Step 31E cannot determine whether eliminating residency churn would
reduce that scope because the larger-cap decode path fails first.

The immediate next attribution target is the existing GGML execution/error
boundary for the first decode, including the underlying `AdapterError` and
backend allocation result. This is an instrumentation experiment, not a GGML
tuning or residency-policy change.

## Scope Boundary

```text
RESIDENCY_POLICY_CHANGED: NO
RESIDENCY_ALGORITHM_CHANGED: NO
ONLY_BUDGET_VARIED: YES
PERSISTENT_MIRROR_REUSED: YES
MEASURED_WORKLOAD_FULLY_LOCAL: YES
REMOTE_TRAFFIC_CONTAMINATION: NO
TENSORREF_CHANGED: NO
MATERIALIZER_SEMANTICS_CHANGED: NO
PERSISTENCE_CHANGED: NO
ACQUISITION_POLICY_CHANGED: NO
BACKEND_CHANGED: NO
INFERENCE_SEMANTICS_CHANGED: NO
PRODUCTION_RESIDENCY_DEFAULT_CHANGED: NO
```

No prefetch, retention heuristic, source redesign, persistence redesign, GGML
tuning, tensor-semantic change, or production default change was made.
