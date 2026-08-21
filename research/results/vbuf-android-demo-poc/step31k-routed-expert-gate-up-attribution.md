# Step 31K Routed-Expert Gate/Up Attribution

Date: 2026-08-21
Starting commit: `077dae4`
Status: **PHYSICALLY MEASURED; ATTRIBUTION COMPLETE; NO OPTIMIZATION IMPLEMENTED**

## Question and Scope

Step 31J identified routed-expert gate/up matmuls as the largest measured routed
component, but its aggregate timer did not distinguish logical expert work from
the two backend matmul submissions per selected expert. Step 31K adds bounded
operation counters, shape/representation signatures, and fixed-capacity compute
samples for the routed gate/up and down paths.

The experiment did not change source resolution, persistence, residency policy,
materialization, GGML configuration, thread count, batching, expert ordering,
TopK semantics, reduction order, quantization, graph reuse, prefetching, or
model behavior. No grouped GEMM or other optimization was implemented.

## Evidence Classification

- **CODE-AUDITED:** each selected routed expert records one logical gate/up
  invocation and one logical down invocation.
- **CODE-AUDITED:** the routed expert graph contains separate gate and up
  matmul operations; each is counted as one backend submission. The down matmul
  is counted separately.
- **CODE-AUDITED:** signatures are restricted to `AttributionStage::RoutedExpert`
  and group identical dimensions, ranks, representations, and payload sizes.
- **CODE-AUDITED:** fixed-capacity samples record compute duration per gate/up
  or down submission. No dynamic per-operation trace or percentile machinery is
  used.
- **HOST-MEASURED:** native build, 20-test CTest suite, runtime timing contract,
  tensor-wave phase contract, and Rust workspace tests.
- **PHYSICALLY-MEASURED:** the canonical 2 GiB Pixel 7 Pro run below.
- **DERIVED:** averages, ratios, overhead, and MAC/weight-byte totals derived
  from the measured counters and signatures.

## Physical Workload

```text
Device: Pixel 7 Pro, Android 17, arm64-v8a
Model: DeepSeek-V2-Lite IQ2_XXS
Residency cap: 2147483648 bytes
Prompt: Explain the purpose of bounded generation
Prompt tokens: 7
Prefill: batched, batch size 7
Generated tokens: 4
Output: ----
Payload mirror: complete canonical local mirror
Remote requests: 0
Remote bytes: 0
Local source bytes: 1953816832
Residency hits/misses/evictions: 9132/2918/0
Residency materializations/reacquisitions: 1262/0
Peak resident bytes: 1953816832
Peak active bytes: 12607488
```

## Operation Counts

```text
ROUTED_MOE_LAYERS: 104
ROUTED_EXPERT_INVOCATIONS: 624
ROUTED_EXPERT_UNIQUE_IDS: 60
SELECTED_EXPERTS_PER_LAYER: 6 (derived)
GATE_UP_LOGICAL_INVOCATIONS: 624
GATE_MATMUL_CALLS: 624
UP_MATMUL_CALLS: 624
GATE_UP_BACKEND_SUBMISSIONS: 1248
DOWN_LOGICAL_INVOCATIONS: 624
DOWN_BACKEND_SUBMISSIONS: 624
```

One selected expert therefore produces one logical gate/up unit, two gate/up
backend submissions, and one down backend submission. The gate/up path has twice
the individual matmul submission count of down; it is not a single fused
backend operation in this path.

## Operation Signatures

| Path | Count | Input | Weight | Output | Weight type | Weight payload | Logical MACs | Logical weight bytes |
|---|---:|---|---|---|---|---:|---:|---:|
| Gate/up | 1,248 | `2048x1` F32 | `2048x1408` | `1408x1` F32 | IQ2_XXS | 743,424 | 3,598,712,832 | 927,793,152 |
| Down | 624 | `1408x1` F32 | `1408x2048` | `2048x1` F32 | IQ4_NL | 1,622,016 | 1,799,356,416 | 1,012,137,984 |

The aggregate logical MACs are exactly 2:1 because gate/up has two operations
per selected expert. Each individual operation has 2,883,584 logical MACs;
the aggregate signature count does not imply a fused gate/up kernel.

## Compute Attribution

| Category | Total | Count | Average | Min | Max |
|---|---:|---:|---:|---:|---:|
| Gate matmul compute | 5,024 ms | 624 | 8.05 ms | not separately retained | not separately retained |
| Up matmul compute | 5,139 ms | 624 | 8.24 ms | not separately retained | not separately retained |
| Gate/up matmul compute | 10,163 ms | 1,248 | 8.14 ms | 5 ms | 14 ms |
| Down matmul compute | 4,888 ms | 624 | 7.83 ms | 4 ms | 16 ms |

The min/max values are rounded to integer milliseconds by the Android report.
No p50 or p90 was added because the bounded sample set is used only for the
small-scope min/max distribution check. The per-submission averages are close;
the aggregate gate/up dominance is primarily the 2:1 submission count.

## Phase Attribution

| Path | Phase total | Ready | Descriptor setup | Graph build | Allocation | Compute | Result |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gate/up | 11,289 ms | 951 ms | 29 ms | 20 ms | 3 ms | 10,163 ms | 121 ms |
| Down | 5,840 ms | 848 ms | 13 ms | 9 ms | 1 ms | 4,888 ms | 77 ms |

Phase totals are nested within the decode and routed-expert wall scopes and must
not be added to those scopes as independent time. The gate/up phase is about
1.93x the down phase; its backend compute is about 2.08x the down compute.
Descriptor setup plus graph build plus allocation is small relative to compute
for both paths.

## Four-Token Runtime Result

```text
DECODE_TOTAL: 35546 ms
ACTUAL_ATTENTION: 16646 ms
ACTUAL_FFN: 78384 ms
ACTUAL_ROUTER_MOE: 52535 ms
ACTUAL_OUTPUT_HEAD: 5819 ms
ATTRIBUTION_ROUTED_EXPERT_TOTAL: 17661 ms
ATTRIBUTION_ROUTED_GATE_UP: 10163 ms
ATTRIBUTION_ROUTED_DOWN: 4888 ms
ATTRIBUTION_BACKEND_COMPUTE: 26796 ms
ATTRIBUTION_READY: 5015 ms
ATTRIBUTION_GRAPH_BUILD: 61 ms
ATTRIBUTION_GRAPH_ALLOCATION: 9 ms
ATTRIBUTION_RESULT_HANDLING: 422 ms
TOP_LEVEL_ACCOUNTED: 35399 ms
TOP_LEVEL_UNACCOUNTED: 146 ms
TOP_LEVEL_ACCOUNTED_PERCENT: 99.586820%
```

## Instrumentation Comparison

```text
STEP31I_2G_CONTROL_DECODE: 33252 ms
STEP31J_INSTRUMENTED_DECODE: 33704 ms
STEP31K_INSTRUMENTED_DECODE: 35546 ms
STEP31K_DELTA_VS_STEP31I_CONTROL: +2294 ms (+6.90%)
STEP31K_DELTA_VS_STEP31J_INSTRUMENTED: +1842 ms (+5.47%)
```

The Step 31K delta is not an isolated instrumentation-overhead control: it
includes run-to-run variance and the additional operation counters, signature
aggregation, and fixed-capacity samples. It is reported as an observation cost,
not as a backend or runtime performance regression.

## Classification

```text
DOMINANT_ROUTED_COMPONENT: GATE_UP_MATMUL
GATE_UP_DOMINANCE: COUNT_DRIVEN; 1248 submissions versus 624 down submissions
PER_SUBMISSION_COMPARISON: GATE_UP 8.14 ms versus DOWN 7.83 ms average
SETUP_DOMINANT: NO; descriptor setup and graph build are small versus compute
RESIDENCY_CONTROL: VALID; zero remote bytes, zero evictions, zero reacquisitions
OUTPUT_PARITY: VALID; output remained ----
ATTRIBUTION_CONFIDENCE: HIGH for measured counts and shape classification;
  moderate for distribution due to one physical run and rounded min/max output
```

The result supports the Step 31J classification that routed-expert compute is
the dominant remaining FFN bottleneck. It refines that result: gate/up takes
roughly twice the aggregate compute because the current path submits separate
gate and up matmuls, while the average cost of an individual gate/up submission
is close to down. This is evidence for a future separately authorized backend
experiment, not an implementation decision in Step 31K.

## Regression and Scope

```text
RUNTIME_BEHAVIOR_CHANGED: NO
RESIDENCY_POLICY_CHANGED: NO
PERSISTENCE_CHANGED: NO
ACQUISITION_CHANGED: NO
TENSORREF_CHANGED: NO
MATERIALIZER_CHANGED: NO; only bounded observation callbacks were added
BACKEND_SEMANTICS_CHANGED: NO
INFERENCE_SEMANTICS_CHANGED: NO
EXPERT_ORDER_CHANGED: NO
TOPK_OR_REDUCTION_ORDER_CHANGED: NO
OPTIMIZATION_IMPLEMENTED: NO
```

The operation signatures confirm the expected cached tensor geometry: gate/up
uses the rank-2 `2048x1408` IQ2_XXS expert weight and down uses the rank-2
`1408x2048` IQ4_NL expert weight. The run completed the canonical workload
without malformed geometry, failure, or output divergence.

## Verification

```text
NATIVE_CMAKE_BUILD: PASS
NATIVE_CTEST: 20/20 PASS
RUST_CARGO_TEST_WORKSPACE: PASS
ANDROID_ARM64_ASSEMBLE_DEBUG: PASS
ANDROID_PHYSICAL_2G_RUN: PASS; output ----
```

No source, residency, backend, or inference optimization is authorized by this
measurement. The next source work remains retention/offline completion; any
gate/up implementation alternative requires its own qualification experiment.
