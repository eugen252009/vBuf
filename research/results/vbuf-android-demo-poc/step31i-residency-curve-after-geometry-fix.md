# Step 31I Post-Fix Physical Residency Curve

Date: 2026-08-21
Starting commit: `452d2a7`
Status: **PHYSICALLY MEASURED; ALL FOUR CURVE POINTS COMPLETED AFTER STEP 31H**

## Objective

Resume the active-RAM residency curve after the Step 31H cached geometry
ownership repair. Only the numeric `TensorResidencyStore` budget was varied.
Persistent source state, source identity, materialization behavior, backend,
thread/runtime configuration, model, prompt, batched prefill, and four-token
decode workload were held constant.

This is a qualification result, not a residency-policy optimization. Retention,
offline completion, and production-default selection remain separate work.

## Evidence Classification

- **PHYSICALLY-MEASURED:** all four Pixel 7 Pro runs completed batched prefill,
  four decode tokens, and output `----`.
- **PHYSICALLY-MEASURED:** every run used 7 prompt tokens, zero remote bytes,
  zero remote requests, the complete retained mirror, and the same coverage
  sidecar.
- **DERIVED:** mean decode is calculated from the four completed per-token
  decode timers. The table reports the existing inclusive runtime counters.
- **HISTORICAL:** the earlier Step 31E curve and the Step 31H 512 MiB
  confirmation remain separate runs and are not substituted for these points.

## Fixed Workload and Environment

```text
Device: Pixel 7 Pro, Android 17, arm64-v8a
Model: DeepSeek-V2-Lite IQ2_XXS
Runtime mode: NORMAL_INFERENCE
Prefill mode: BATCHED
Prefill batch size: 7
Prompt: Explain the purpose of bounded generation
Prompt chars: 41
Prompt tokens: 7
Generated tokens: 4
Output: ----
Payload size: 5639819878 bytes
Payload SHA-256: 2ef0cdde67154ecc68bd82558007a4ea9da36262c4405007cb83306f68456e47
Coverage: 336990 chunks / 172114 coverage bytes
Coverage sidecar: 172194 bytes including its header
Remote requests: 0 at every point
Remote bytes: 0 at every point
```

Each point used a fresh app process while retaining the complete same-offset
payload mirror and coverage sidecar. The APK was rebuilt with only
`-PvbufResidencyBudgetBytes` changed between points. The endpoint and source
mechanism were unchanged.

## Residency Curve

| Residency cap | Prefill | Decode total | Total generation | Local source bytes | Evictions | Reacquisitions | Peak resident |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256 MiB | 71,515 ms | 104,615 ms | 176,141 ms | 4,745,501,920 | 4,217 | 3,025 | 267,452,416 |
| 512 MiB | 69,452 ms | 74,159 ms | 143,623 ms | 3,667,185,888 | 2,806 | 1,930 | 536,817,664 |
| 1 GiB | 65,927 ms | 44,107 ms | 110,045 ms | 2,388,389,088 | 952 | 580 | 1,073,737,376 |
| 2 GiB | 67,219 ms | 33,252 ms | 100,482 ms | 1,953,816,832 | 0 | 0 | 1,953,816,832 |

| Residency cap | Materializations | Residency hits/misses/evictions | Consumer requests | Reload bytes | Tokens/sec |
|---|---:|---:|---:|---:|---:|
| 256 MiB | 4,287 | 6,107/7,898/4,217 | 4,287 | 2,791,685,088 | 0.022709 |
| 512 MiB | 3,192 | 7,202/5,959/2,806 | 3,192 | 1,713,369,056 | 0.027851 |
| 1 GiB | 1,842 | 8,552/3,544/952 | 1,842 | 434,572,256 | 0.036349 |
| 2 GiB | 1,262 | 9,132/2,918/0 | 1,262 | 0 | 0.039808 |

The corresponding reacquisition counts were `3,025`, `1,930`, `580`, and `0`.
Peak active bytes were `12,607,488` at every point. The 2 GiB run reached
`1,953,816,832` resident bytes without an eviction; it did not need to fill the
entire configured 2 GiB budget.

## Result

The Step 31H geometry repair removed the prior first-decode failure boundary for
the larger residency caps. All points completed the same workload and produced
the same four-character output. Increasing the active budget reduced local
source bytes, materializations, reacquisitions, and evictions in this workload,
while the inclusive decode total decreased from `104,615` ms at 256 MiB to
`33,252` ms at 2 GiB.

This is evidence of a successful post-fix residency curve, not a production
residency recommendation. The points are single physical runs, so they do not
establish statistical variance, a universal working-set threshold, or an
optimal default. Retention policy and persistent storage budgets must be chosen
from reuse and coverage evidence separately from this active-RAM experiment.

## Scope Boundary

```text
ONLY_BUDGET_VARIED: YES
RESIDENCY_POLICY_CHANGED: NO
RESIDENCY_ALGORITHM_CHANGED: NO
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
RETENTION_POLICY_IMPLEMENTED: NO
OFFLINE_COMPLETION_IMPLEMENTED: NO
```
