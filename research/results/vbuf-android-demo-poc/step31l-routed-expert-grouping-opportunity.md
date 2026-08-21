# Step 31L Routed-Expert Grouping Opportunity

Date: 2026-08-21
Starting commit: `393a77c`
Status: **HOST-QUALIFIED OPPORTUNITY ANALYSIS; NO GROUPED EXECUTION IMPLEMENTED**

## Question and Scope

Step 31K measured 1,248 gate/up and 624 down routed-expert matmul-like backend
submissions over the canonical four-token decode. Step 31L determines which
submission boundaries are structurally groupable without changing routing,
TopK semantics, expert weights, or reduction order.

This is an architecture and attribution analysis only. It does not combine
graphs, batch experts, fuse gate/up, change execution order, modify GGML, change
residency or materialization, or change the qualified inference path.

## Evidence Classification

- **PHYSICALLY_MEASURED:** Step 31K's Pixel 7 Pro run supplied the current
  counts, signatures, phase timings, output parity, and source/residency state.
- **CODE_AUDITED:** `run_layer` performs normalization, routing, selected-expert
  execution, ordered weighted merge, shared-expert execution, and residual
  composition.
- **CODE_AUDITED:** `execute_selected` iterates `selection.ids` in TopK rank
  order and creates separate gate, up, and down expert slices.
- **CODE_AUDITED:** `build_expert_graph` contains independent gate and up
  matmuls, SwiGLU, and down matmul. The current TensorWave executor submits one
  GGML graph for each TensorWave operation.
- **CODE_AUDITED:** the checked-in GGML API exposes both higher-rank/broadcast
  `ggml_mul_mat` and indirect multi-expert `ggml_mul_mat_id`. The CPU source has
  a `GGML_OP_MUL_MAT_ID` implementation and IQ2_XXS/IQ4_NL dot-product traits.
- **HOST_MEASURED:** native source/API inspection and the existing Step 31K
  contract/build evidence. No grouped inference was executed.
- **DERIVED:** grouping counts, reductions, setup bounds, memory estimates, and
  per-layer/per-token rates below.
- **HYPOTHESIS:** any compute-efficiency improvement from grouping.

## Current Execution Map

The actual current normal-inference route is:

```text
autoregressive decode position
  -> full_moe_layer_poc13::run_layer
  -> RMSNorm
  -> router graph: router_matmul
  -> deterministic_top_k: 6 IDs, descending score, ID tie-break
  -> normalized_selected_weights: weights in selection.ids order
  -> multi_expert_moe_poc12::execute_selected
       for rank = 0 .. 5:
         selected expert ID
         -> make_expert: rank-2 direct slice of gate/up/down rank-3 tensor
         -> router_driven_moe_poc11::build_expert_graph
              expert_gate_matmul(input, gate_slice) -> gate_out
              expert_up_matmul(input, up_slice) -> up_out
              expert_swiglu(gate_out, up_out) -> mul_out
              expert_down_matmul(mul_out, down_slice) -> expert_output
         -> floats(expert_output)
  -> weighted_merge(expert_outputs, routed_weights)
       for values in original selection.ids / TopK rank order
  -> shared expert
  -> residual composition
  -> next sequential MoE layer
```

`TensorDependencyExecutor` does retain the four operations in one graph
description, but its execution loop selects one runnable operation, constructs
one GGML context and cgraph, calls `ggml_backend_graph_compute`, synchronizes,
extracts the result, and releases values before continuing. Therefore the
current graph description is not one current backend submission.

Multiple transformer layers cannot be grouped across their dependency boundary:
the next layer consumes the completed residual output of the previous layer,
and each layer has its own normalization, router result, selected IDs, and
weights.

## Actual Step 31K Signatures

There is one distinct gate/up signature and one distinct down signature in the
qualified run. Gate and up have the same geometry and representation, so the
Step 31K signature aggregation intentionally reports them as one gate/up class;
their compute totals remained separately counted.

| Path | Count | Input | Weight | Output | M/N/K | Input/output | Weight | Payload per slice |
|---|---:|---|---|---|---|---|---|---:|
| Gate | 624 | `2048x1` | `2048x1408` | `1408x1` | `1/1408/2048` | F32/F32 | IQ2_XXS | 743,424 |
| Up | 624 | `2048x1` | `2048x1408` | `1408x1` | `1/1408/2048` | F32/F32 | IQ2_XXS | 743,424 |
| Down | 624 | `1408x1` | `1408x2048` | `2048x1` | `1/2048/1408` | F32/F32 | IQ4_NL | 1,622,016 |

The operation signatures report canonical contiguous GGML-compatible tensor
strides. The current adapter binds each selected slice directly through
`BorrowedGgmlTensor::bind_cpu`; Step 31K recorded no repacking.

The actual semantic manifest records the full expert tensors as:

```text
gate full: [2048, 1408, 64], IQ2_XXS, 47,579,136 bytes
up full:   [2048, 1408, 64], IQ2_XXS, 47,579,136 bytes
down full: [1408, 2048, 64], IQ4_NL, 103,809,024 bytes
```

`make_expert` creates a rank-2 direct slice at:

```text
full_source_offset + expert_id * (full_payload_bytes / 64)
```

Thus slices within each expert tensor are fixed-stride and individually
contiguous, but selected IDs are generally disjoint ranges. Gate and up are
separate tensor objects and separate weight regions. The source manifest shows
the full gate and up tensors adjacent in the original payload, but their expert
planes are stored as all gate experts followed by all up experts, not as
per-expert gate/up pairs. A zero-copy `[K, 2N, 64]` stacked gate/up view is
therefore not available from the existing layout.

## Current Submission Structure

```text
DECODE_TOKENS: 4
MOE_INVOCATIONS: 104 (26 sequential MoE layers per decode token)
SELECTED_EXPERTS: 624
SELECTED_EXPERTS_PER_MOE_INVOCATION: 6
SELECTED_EXPERTS_PER_TOKEN: 156
GATE_SUBMISSIONS: 624
UP_SUBMISSIONS: 624
GATE_UP_COMBINED_SUBMISSIONS: 1248
DOWN_SUBMISSIONS: 624
CURRENT_TOTAL_EXPERT_MATMUL_SUBMISSIONS: 1872
MATMUL_SUBMISSIONS_PER_SELECTED_EXPERT: 3
MATMUL_SUBMISSIONS_PER_MOE_INVOCATION: 18
MATMUL_SUBMISSIONS_PER_DECODE_TOKEN: 468
```

The full current TensorWave expert path also submits 624 SwiGLU operations,
so it has 2,496 routed-expert TensorWave operation executions. The required
1,872 figure counts only gate, up, and down matmul-like submissions.

## Grouping Domains

### Same expert gate plus up

The gate and up operations use the exact same normalized activation value, have
identical geometry and quantization, use independent weight tensors, and produce
independent outputs consumed by SwiGLU. There is no semantic synchronization
between the two matmuls. A GGML graph can contain both branches and expand both
into one backend graph submission.

```text
same input
  |\
  | +-- gate matmul -> gate_out --+
  +---- up matmul   -> up_out   --+-> SwiGLU
```

This is semantically groupable and graph-representable. It is not one matrix
multiplication: the weights remain separate and the outputs remain separate.
The current TensorWave executor does not use this capability because it
executes each operation independently.

### Multiple selected experts in one token and layer

All six selected experts in one invocation have the same measured geometry and
representation classes. Their weight slices are disjoint but fixed-stride.
They can be represented as independent same-shape branches in one GGML graph,
with one branch per TopK rank, or by the raw GGML `ggml_mul_mat_id` operation
using a rank-3 expert tensor plus an I32 ID matrix.

The current source's `ggml_mul_mat_id` contract explicitly supports one matrix
per expert, an expert ID list, and broadcast input columns. The CPU
implementation groups rows by selected expert internally while writing results
back to the selected-rank positions. This is backend/API capability evidence,
not vBuf Android qualification evidence.

For a zero-copy `ggml_mul_mat_id` representation, the full rank-3 expert tensor
must be available as one descriptor. The current vBuf path deliberately
materializes rank-2 selected slices, so using `ggml_mul_mat_id` would either
materialize the full tensor or require a new pointer-array/grouped API. Full
one-layer weights are approximately 47.6 MB gate, 47.6 MB up, and 103.8 MB down
before temporary outputs and graph metadata.

### Multiple decode tokens

The canonical decode path processes one autoregressive position at a time. No
cross-token grouping is analyzed or proposed. Prompt prefill batching is a
separate existing path and is outside this Step 31L decode opportunity.

### Multiple layers

Not groupable across sequential layers. The layer output, residual, KV state,
and next router input create real data dependencies. A whole-model grouped graph
would violate the current runtime dependency and lifetime boundaries.

## Capability Classification

| Candidate/domain | Semantic feasibility | Raw GGML representation | Current vBuf adapter | Layout/copy result |
|---|---|---|---|---|
| Same-expert gate/up branches | YES | `ggml_mul_mat` nodes in one graph | NOT AVAILABLE; executor is per-op | Zero-copy separate views; no repack |
| Multiple experts, independent branches | YES if rank mapping is retained | Multiple `ggml_mul_mat` nodes in one graph | NOT AVAILABLE; executor is per-op | Zero-copy rank-2 slices; larger simultaneous working set |
| Multiple experts, `ggml_mul_mat_id` | YES if IDs preserve rank slots | AVAILABLE in checked-in GGML | NOT AVAILABLE in TensorWave op enum/adapter | Full rank-3 weight view or new pointer-array API |
| True fused gate/up kernel | YES semantically | NOT AVAILABLE as current generic op | NOT AVAILABLE | Custom kernel/API; stacked layout would require repack |
| Across transformer layers | NO for current dependencies | NOT APPLICABLE | NOT APPLICABLE | Reject |

The raw checked-in GGML capability audit is:

```text
GGML_MULTI_OP_GRAPH: AVAILABLE_NOW at raw GGML graph level
GGML_BATCHED_MATMUL: AVAILABLE_NOW through higher-rank/broadcast ggml_mul_mat
GGML_GROUPED_EXPERT_MATMUL: AVAILABLE_NOW as ggml_mul_mat_id, CPU source audited
VBUF_TENSOR_WAVE_MULTI_OP_SUBMISSION: NOT AVAILABLE in current executor
VBUF_TENSOR_WAVE_MUL_MAT_ID: UNSUPPORTED; no op kind, I32 ID binding, or full-rank path
STACKED_GATE_UP_WEIGHT: REQUIRES_LAYOUT_CHANGE or separate custom pointer API
TRUE_FUSED_GATE_UP_KERNEL: REQUIRES_CUSTOM_KERNEL and backend API
```

`ggml_mul_mat_id` is an existing indirect grouped-expert primitive, not proof
that the current Android vBuf path can consume it without changing materializer
requests, tensor lifetimes, or the execution representation.

## TopK and Reduction Correctness

Every candidate is acceptable only if it preserves this mapping:

```text
selection.ids[rank] -> expert output slot[rank] -> selection.weights[rank]
```

`deterministic_top_k` produces descending score order with lower-ID tie breaks.
`normalized_selected_weights` emits weights in that same `selection.ids` order.
`execute_selected` currently executes in rank order, and `weighted_merge` loops
the values and weights in vector order. A grouped backend may schedule experts
by expert ID or completion order, but it must write each result to its original
TopK-rank slot and the final accumulation must still iterate rank `0..5`.

Backend completion order must never become floating-point reduction order.
Candidates that cannot preserve this slot mapping are rejected. Multiple expert
branches and `ggml_mul_mat_id` can preserve it conceptually; a completion-order
accumulator cannot.

## Theoretical Submission Bounds

These are structural backend-execution bounds, not measured performance claims.
They count a grouped graph submission as one execution even when that graph
contains multiple matmul nodes or a grouped matmul node.

| Case | Bound | Reduction from 1,872 | Interpretation |
|---|---:|---:|---|
| Current separate matmul submissions | 1,872 | 0% | 624 gate + 624 up + 624 down |
| A: same-expert gate/up graph, down separate | 1,248 | 33.33% | 624 gate/up graphs + 624 down graphs |
| B: one multi-expert graph per MoE invocation | 104 | 94.44% | One graph for all six experts and all routed operations |
| B conservative gate/up + down graphs per invocation | 208 | 88.89% | Two grouped graph submissions per invocation |
| C: true fused gate/up per expert, down separate | 1,248 | 33.33% | Same submission bound as A; compute effect unproven |

The current 1,872 matmul nodes do not disappear merely because they are placed
in one graph. The 104 bound is a graph-submission bound. The `ggml_mul_mat_id`
variant would reduce matmul nodes to three grouped matmul nodes plus SwiGLU
inside each graph, but it still requires a new vBuf execution representation.

At least 624 current submission boundaries are directly justified as removable
by the same-expert independent-branch graph opportunity. Up to 1,768 boundaries
are conditionally reducible under a multi-expert one-graph representation;
that larger number is not yet qualified for the Android vBuf path.

## Measured Submission Overhead

Step 31K measured these nested phase totals:

```text
GATE_UP_TOTAL: 11289 ms
GATE_UP_READY: 951 ms
GATE_UP_DESCRIPTOR_SETUP: 29 ms
GATE_UP_GRAPH_BUILD: 20 ms
GATE_UP_GRAPH_ALLOCATION: 3 ms
GATE_UP_COMPUTE: 10163 ms
GATE_UP_RESULT: 121 ms

DOWN_TOTAL: 5840 ms
DOWN_READY: 848 ms
DOWN_DESCRIPTOR_SETUP: 13 ms
DOWN_GRAPH_BUILD: 9 ms
DOWN_GRAPH_ALLOCATION: 1 ms
DOWN_COMPUTE: 4888 ms
DOWN_RESULT: 77 ms
```

The directly submission-bound graph/setup/result categories, excluding readiness,
are:

```text
GATE_UP_DIRECT_SETUP_RESULT: 173 ms / 1248 = 0.139 ms per submission
GATE_UP_COMPUTE: 10163 ms / 1248 = 8.14 ms per submission
DOWN_DIRECT_SETUP_RESULT: 100 ms / 624 = 0.160 ms per submission
DOWN_COMPUTE: 4888 ms / 624 = 7.83 ms per submission
ALL_MATMUL_DIRECT_SETUP_RESULT: 273 ms / 1872 = 0.146 ms per submission
ALL_MATMUL_COMPUTE: 15051 ms / 1872 = 8.04 ms per submission
```

Readiness is separately reported because it includes per-tensor materializer
state/request/wait/ready-view work. Grouping two operations does not remove the
need to make both distinct weight tensors ready. Gate/up readiness averages
`0.762 ms` per submission and down readiness averages `1.359 ms`; these are not
directly eliminable submission overhead.

Conservative upper bounds, assuming the directly measured categories scale
linearly with backend submission count, are:

```text
CANDIDATE_A_DIRECT_SETUP_SAVINGS: 173 ms / 2 = 86.5 ms
CANDIDATE_B_DIRECT_SETUP_SAVINGS: 273 ms * (1 - 104 / 1872) = 257.8 ms
```

These are upper bounds only. They exclude any new graph construction cost,
additional readiness, output mapping, graph metadata, memory pressure, and
backend behavior. Compute savings from grouping are **UNPROVEN**. Multiplying
removed submissions by 8 ms would incorrectly count mostly non-eliminable
matmul compute as setup savings.

## Candidate Ranking

### Candidate A: Same-expert gate/up plus activation multi-op graph

```text
CANDIDATE_A_SUBMISSIONS: 1248 structural graph executions
CANDIDATE_A_REDUCTION: 33.33% of current matmul-like submission count
CANDIDATE_A_TEMP_MEMORY: small; roughly one extra gate/up weight slice plus
  gate/up/SwiGLU float intermediates, estimated under 1 MiB incremental
CANDIDATE_A_SEMANTIC_RISK: LOW
CANDIDATE_A_IMPLEMENTATION_COMPLEXITY: LOW/MEDIUM
CANDIDATE_A_WEIGHT_COPY: NO for separate zero-copy views
CANDIDATE_A_REPACK: NO
```

This graph contains the two independent gate/up branches and their dependent
SwiGLU operation; down remains a separate submission. It is the least invasive
graph opportunity. It removes at most the directly
measured `86.5 ms` of graph/setup/result boundary cost under a linear upper-bound
model. It does not combine the two matmul kernels or prove a compute gain.

### Candidate B: Multi-expert per-layer graph

```text
CANDIDATE_B_SUBMISSIONS: 104 structural graph executions, or 208 conservative
  gate/up-plus-down graph executions
CANDIDATE_B_REDUCTION: 94.44% theoretical, or 88.89% conservative
CANDIDATE_B_TEMP_MEMORY: independent branches require about 18.7 MB of six
  selected-expert routed weights plus about 0.15 MB of float intermediates;
  a full-rank-3 mul_mat_id path may instead make about 199 MB of one-layer
  gate/up/down weights simultaneously resident
CANDIDATE_B_SEMANTIC_RISK: LOW only with explicit rank-slot mapping; otherwise HIGH
CANDIDATE_B_IMPLEMENTATION_COMPLEXITY: MEDIUM/HIGH
CANDIDATE_B_WEIGHT_COPY: NO for independent rank-2 branches; NO for full-rank
  mul_mat_id views, but full tensors must be available
CANDIDATE_B_REPACK: NO for existing GGML mul_mat_id; possible for custom packing
```

This has the largest structural submission reduction, but it changes the
working-set shape and needs a new multi-operation execution boundary. The
full-rank `mul_mat_id` alternative may fetch and hold all 64 experts instead of
the six selected slices, so zero-copy does not mean zero residency cost.

### Candidate C: True fused gate/up kernel

```text
CANDIDATE_C_SUBMISSIONS: 1248 per-expert fused-gate/up-plus-down lower bound
CANDIDATE_C_REDUCTION: 33.33% before any multi-expert extension
CANDIDATE_C_TEMP_MEMORY: implementation-dependent; stacked per-layer gate/up
  packing would duplicate about 90.8 MiB of full gate/up payload
CANDIDATE_C_SEMANTIC_RISK: MEDIUM; gate/up values and reduction slots must remain
  distinct and ordered
CANDIDATE_C_IMPLEMENTATION_COMPLEXITY: HIGH
CANDIDATE_C_WEIGHT_COPY: custom pointer-pair kernel could avoid it; stacked
  representation would require it
CANDIDATE_C_REPACK: YES for a conventional stacked packed-weight design
```

The current raw GGML API has no true fused gate/up primitive. Separate gate and
up full tensors happen to be adjacent in the source manifest, but their expert
planes are not interleaved, so a `[K, 2N, 64]` zero-copy stacked view is invalid.
No fused-kernel opportunity is justified without a separate backend experiment.

## Selected Next Experiment

The measured directly eliminable boundary cost is only `273 ms` across all
1,872 matmul-like submissions, versus `15,051 ms` of measured matmul compute.
Therefore graph grouping is not yet justified as a performance optimization by
submission count alone.

The smallest evidence-driven next experiment is:

```text
BEST_NEXT_EXPERIMENT:
  Host/native GGML microqualification of ggml_mul_mat_id for the actual
  F32 x IQ2_XXS gate/up and F32 x IQ4_NL down geometry, with six selected IDs
  and one token, compared against six existing ggml_mul_mat operations.
```

This should be separately authorized and must remain outside the production
vBuf runtime path until it measures both output parity and compute behavior.
It should explicitly record whether full rank-3 tensor materialization and
temporary memory erase any grouped-kernel benefit. It must preserve rank-slot
outputs and use a canonical rank-ordered CPU reduction in the oracle.

## Required Classification Fields

```text
CURRENT_ROUTED_EXPERT_COUNT: 624
CURRENT_GATE_SUBMISSIONS: 624
CURRENT_UP_SUBMISSIONS: 624
CURRENT_GATE_UP_SUBMISSIONS: 1248
CURRENT_DOWN_SUBMISSIONS: 624
CURRENT_TOTAL_EXPERT_SUBMISSIONS: 1872
MOE_INVOCATIONS: 104
SELECTED_EXPERTS_PER_MOE_INVOCATION: 6
GATE_UP_SIGNATURE_COUNT: 1
DOWN_SIGNATURE_COUNT: 1
GATE_UP_SHARE_INPUT: YES
GATE_UP_IDENTICAL_GEOMETRY: YES
GATE_UP_SEPARATE_WEIGHTS: YES
CURRENT_GRAPH_SUBMISSION_GRANULARITY: one backend graph compute per TensorWave operation
CURRENT_TOTAL_SUBMISSIONS: 1872 matmul-like; 2496 including SwiGLU
GROUPING_REQUIRES_WEIGHT_COPY: NO for A/B zero-copy graph paths; possible for C
GROUPING_REQUIRES_REPACK: NO for A/B; YES for stacked fused C
GROUPING_REQUIRES_MODEL_FORMAT_CHANGE: NO for the analyzed graph/API paths
STEP31K_GATE_UP_TOTAL_MS: 11289
STEP31K_GATE_UP_COMPUTE_MS: 10163
STEP31K_GATE_UP_SETUP_MS: 173 direct setup/build/allocation/result
MEASURED_SETUP_PER_SUBMISSION_MS: 0.146 all matmul-like direct categories
MEASURED_COMPUTE_PER_SUBMISSION_MS: 8.04 all matmul-like average
THEORETICAL_ELIMINABLE_SETUP_MS_CANDIDATE_A: 86.5 upper bound
THEORETICAL_ELIMINABLE_SETUP_MS_CANDIDATE_B: 257.8 upper bound
COMPUTE_SPEEDUP_FROM_GROUPING_PROVEN: NO
EXPECTED_COMPUTE_SPEEDUP: UNPROVEN
TOPK_SELECTION_CHANGED: NO
EXPERT_SET_CHANGED: NO
SEMANTIC_REDUCTION_ORDER_CHANGED: NO
HOST_GROUPING_PLANNER_ADDED: NO; Step 31K aggregate evidence plus source/API audit sufficed
PHYSICAL_RERUN_REQUIRED: NO
PHYSICAL_RERUN_PERFORMED: NO
GROUPING_OPPORTUNITY_CLASSIFICATION: CONDITIONAL; graph grouping is representable,
  current vBuf adapter is not, and compute benefit is unproven
BEST_NEXT_EXPERIMENT: host/native ggml_mul_mat_id microqualification
GROUPED_EXECUTION_IMPLEMENTED: NO
GRAPH_REUSE_IMPLEMENTED: NO
KERNEL_CHANGE_IMPLEMENTED: NO
RUNTIME_SEMANTICS_CHANGED: NO
MODEL_ARTIFACTS_COMMITTED: NO
TEMPORARY_DIAGNOSTICS_PRESENT: NO
```

## Verification Boundary

This Step 31L change is analysis and documentation only; no runtime or Android
native source was modified. Existing Step 31K physical qualification remains
the oracle. The final verification and commit record are captured after the
report and roadmap updates.
