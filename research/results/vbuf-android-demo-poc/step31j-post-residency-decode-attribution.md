# Step 31J Post-Residency Decode / MoE Bottleneck Attribution

Date: 2026-08-21
Starting commit: `4f0ba8e`
Status: **PHYSICALLY MEASURED; ATTRIBUTION COMPLETE; NO OPTIMIZATION IMPLEMENTED**

## Question and Scope

Step 31I established that increasing active residency materially reduced decode
time and residency churn. Step 31J asks what remains at the high-residency point
after source traffic and residency churn are removed.

Only bounded attribution instrumentation was added. The experiment did not
change residency policy, source persistence, acquisition, TensorRef,
materialization semantics, GGML configuration, thread count, batching, expert
ordering, TopK semantics, reduction order, or model behavior.

## Evidence Classification

- **CODE-AUDITED:** existing stage timers enclose wall scopes around embedding,
  attention, FFN, and output-head calls; the existing FFN timer includes norm,
  router, selected expert execution, shared expert execution, and CPU merges.
- **CODE-AUDITED:** new phase markers measure readiness/materializer access,
  graph construction, graph allocation, backend graph compute plus synchronize,
  and result extraction. These are bounded aggregate callbacks, not per-tensor
  log records.
- **HOST-MEASURED:** the native build, 20-test CTest suite, timing aggregation
  contract, tensor-wave phase contract, and existing lifetime/residency tests.
- **PHYSICALLY-MEASURED:** the 2 GiB Pixel 7 Pro run below.
- **DERIVED:** percentages, routed totals, top-level accounting, and the
  instrumentation-overhead comparison.
- **UNINSTRUMENTED:** no independent 256 MiB attribution rerun was performed;
  Step 31I's existing 256 MiB counters were retained as the residency control.

## Step 31I Baseline

All four Step 31I points completed the same canonical workload after the Step
31H cached-geometry ownership repair. They produced `----`, used the complete
local mirror, and recorded zero remote bytes. Decode totals were `104,615 ms`,
`74,159 ms`, `44,107 ms`, and `33,252 ms` at 256 MiB, 512 MiB, 1 GiB, and 2 GiB.
The 2 GiB point had configured cap `2,147,483,648`, actual peak resident bytes
`1,953,816,832`, materializations `1,262`, residency hits `9,132`, evictions `0`,
and reacquisitions `0`. The curve had not reached an obvious flat plateau:
2 GiB remained faster than 1 GiB.

## Timing Boundaries

The existing top-level timers are mutually ordered wall scopes in one decode
step:

```text
embedding -> attention -> FFN -> output head
```

`actual_attention_ns`, `actual_ffn_ns`, and `actual_output_head_ns` include
their complete caller scopes, including CPU setup and result handling. They are
not pure backend-compute timers. In particular, `actual_ffn_ns` encloses:

```text
FFN norm -> router graph -> TopK/weight normalization ->
routed expert graphs -> routed weighted merge -> shared expert graph ->
residual composition
```

The new tensor-wave phase markers are nested within those scopes:

```text
ready: materializer state/request/wait/ready-view access
graph_build: ggml context, tensor/op construction, graph expansion
graph_allocation: backend buffer allocation
compute: backend graph compute and synchronize
result: backend result extraction and bounded output handling
```

The phase totals must not be added to the top-level wall scopes as independent
time. Backend compute, graph preparation, result handling, and readiness are
subcomponents of the attention/FFN/output scopes. Top-level accounting uses
the four ordered wall scopes; nested phase totals explain those scopes.

## Physical Workload

```text
Device: Pixel 7 Pro, Android 17, arm64-v8a
Model: DeepSeek-V2-Lite IQ2_XXS
Residency cap: 2147483648 bytes
Prompt: Explain the purpose of bounded generation
Prompt tokens: 7
Prefill: batched
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

## Per-Token Attribution

Times below are rounded existing wall/phase timers in milliseconds. The
operation phase columns are nested within the token total and are not additive
with the token's top-level stage total.

| Token | Total | Attention compute | Router compute | Routed gate/up | Routed activation | Routed down | Routed accumulation | Shared compute | Output compute | Graph build | Graph allocation | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,433 | 1,023 | 31 | 2,396 | 106 | 1,182 | 2 | 1,137 | 332 | 56 | 2 | 100 |
| 2 | 8,298 | 1,040 | 34 | 2,402 | 103 | 1,166 | 2 | 1,146 | 334 | 56 | 2 | 98 |
| 3 | 8,536 | 1,040 | 33 | 2,401 | 100 | 1,185 | 2 | 1,130 | 345 | 58 | 2 | 94 |
| 4 | 8,433 | 1,054 | 34 | 2,416 | 100 | 1,163 | 2 | 1,144 | 332 | 57 | 2 | 96 |

## Four-Token Attribution

| Category | Measured time | Share of 33,704 ms decode wall |
|---|---:|---:|
| Embedding total | 10 ms | 0.03% |
| Attention total | 5,505 ms | 16.33% |
| FFN total | 23,666 ms | 70.22% |
| Router total | 149 ms | 0.44% |
| Routed expert total | 16,821 ms | 49.91% |
| Shared expert total | 4,647 ms | 13.79% |
| Dense FFN total | 629 ms | 1.87% |
| FFN other | 1,417 ms | 4.20% |
| Output head total | 4,381 ms | 13.00% |
| Top-level accounted | 33,564 ms | 99.59% |
| Top-level unaccounted | 139 ms | 0.41% |

The router total includes router readiness, router graph phases, router backend
compute, TopK/weight normalization, and result handling. Routed expert total
includes readiness, graph phases, gate/up matmuls, activation, down matmuls,
weighted accumulation, other routed compute, and result handling. Shared expert
and output totals use the same boundary rule.

### Nested phase counters

```text
READINESS_TOTAL: 4718 ms (14.00% of decode wall)
GRAPH_BUILD_TOTAL: 228 ms (0.68%)
GRAPH_ALLOCATION_TOTAL: 9 ms (0.03%)
BACKEND_COMPUTE_TOTAL: 25550 ms (75.80%)
RESULT_HANDLING_TOTAL: 390 ms (1.16%)

ATTENTION_READY: 3 ms
ATTENTION_BACKEND_COMPUTE: 4158 ms

ROUTER_READY: 0 ms
ROUTER_BACKEND_COMPUTE: 133 ms
ROUTER_SELECTION: 5 ms

ROUTED_EXPERT_READY: 1704 ms
ROUTED_EXPERT_GATE_UP: 9617 ms
ROUTED_EXPERT_ACTIVATION: 410 ms
ROUTED_EXPERT_DOWN: 4698 ms
ROUTED_EXPERT_ACCUMULATION: 11 ms

SHARED_EXPERT_READY: 1 ms
SHARED_EXPERT_BACKEND_COMPUTE: 4559 ms

DENSE_FFN_TOTAL: 629 ms
DENSE_FFN_READY: 0 ms

OUTPUT_HEAD_READY: 0 ms
OUTPUT_HEAD_BACKEND_COMPUTE: 1345 ms
```

Readiness is not zero merely because evictions and reacquisitions are zero.
This run still performed first-use materialization/readiness work for tensors
not touched by the batched prefill. Conversely, the readiness counter does not
claim that every resident lookup is expensive materialization; it measures the
bounded ready-view/state path at the tensor-wave boundary.

## Accounting and Overhead

```text
BASELINE_2G_DECODE_TOTAL: 33252 ms
INSTRUMENTED_2G_DECODE_TOTAL: 33704 ms
INSTRUMENTATION_DELTA: +452 ms
INSTRUMENTATION_OVERHEAD: +1.36%; modest single-run overhead
ACCOUNTED_TOP_LEVEL: 33564 ms
UNACCOUNTED_TOP_LEVEL: 139 ms
ACCOUNTED_PERCENT: 99.59%
```

The measured positive delta is modest relative to the full decode wall and is
not a reason to simplify the bounded instrumentation. No optimization
conclusion is drawn from the baseline comparison.

## Classification

```text
DOMINANT_COMPONENT: ROUTED_EXPERT_GATE_UP_MATMUL
BOTTLENECK_CLASSIFICATION: ROUTED_EXPERT_COMPUTE_DOMINANT_MIXED
GGML_MOE_COMPUTE_DOMINANT: YES; backend compute is 25550 ms and routed expert compute is the largest FFN component
RESIDENCY_OVERHEAD_STILL_MATERIAL: YES; 4718 ms readiness, including 1704 ms routed-expert readiness
GRAPH_BUILD_OR_BACKEND_SETUP_DOMINANT: NO; graph build plus allocation is 237 ms
ATTENTION_DOMINANT: NO
OUTPUT_HEAD_DOMINANT: NO
ATTRIBUTION_CONFIDENCE: HIGH for measured category ordering; single physical run
```

Routed expert gate/up matmuls are the largest measured routed component at
`9,617 ms`, ahead of routed down matmuls at `4,698 ms`. The result supports
calling actual GGML/MoE compute the dominant remaining bottleneck, while also
showing that readiness/materialization overhead remains measurable and should
not be conflated with backend compute.

## Regression and Scope

The former Step 31H target `blk.1.ffn_down_shexp.weight` remained valid at the
canonical rank-2 geometry `[2816, 2048]` and `1,486,848` payload bytes. The
physical run completed the former block-7 boundary without malformed geometry,
failure, or output divergence.

```text
RUNTIME_BEHAVIOR_CHANGED: NO
RESIDENCY_POLICY_CHANGED: NO
PERSISTENCE_CHANGED: NO
ACQUISITION_CHANGED: NO
TENSORREF_CHANGED: NO
MATERIALIZER_CHANGED: NO; only observation callbacks were added
BACKEND_SEMANTICS_CHANGED: NO
INFERENCE_SEMANTICS_CHANGED: NO
EXPERT_ORDER_CHANGED: NO
TOPK_OR_REDUCTION_ORDER_CHANGED: NO
OPTIMIZATION_IMPLEMENTED: NO
```

No 256 MiB attribution rerun was performed. The existing Step 31I 256 MiB
point remains sufficient for the residency/source control, while this Step 31J
experiment isolates the near/no-churn 2 GiB point.

## Next Isolated Experiment

The result justifies a separately authorized routed-expert gate/up matmul
qualification experiment. That future experiment may measure the gate/up path
in isolation and compare an explicitly chosen implementation alternative. It
must not be implemented as part of Step 31J. No grouped GEMM, expert pinning,
kernel change, quantization change, thread tuning, graph reuse, prefetch, or
residency change was made here.
