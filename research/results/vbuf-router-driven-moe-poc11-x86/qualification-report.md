# POC11 Router-Driven MoE Selection

## Artifact And Router

- Artifact: `research-models/DeepSeek-V2-Lite.IQ1_S.vbuf`
- SHA256: `780a55b77d2730705a93622338d9747149624d72e558e26182176868210fafcc`
- Layer: `1`
- Router tensor: `blk.1.ffn_gate_inp.weight`
- Tensor ID: `19`
- Representation: `0` (`CanonicalPrimitive`, direct float32 payload)
- Dimensions: `[2048,64]`
- Payload offset: `99094944`
- Payload length: `524288`
- Routed expert count: `64`
- Model routing cardinality: `top_k=6`

The router graph is generic: a persistent router tensor and a real activation
feed a `MulMat` operation, producing a 64-score vector. Selection is a separate
generic `deterministic_top_k` operation.

## Activations And Routing

`ACTIVATION_SOURCE: DETERMINISTIC_FIXTURE`

Both activations are exact hidden-width `[2048,1]` float32 vectors:

- Activation A: one-hot index `0`; top-6 IDs `37,21,31,54,6,33`; selected top-1 `37`.
- Activation B: one-hot index `1`; top-6 IDs `3,53,47,5,20,35`; selected top-1 `3`.

Selection rule: descending score, then lower expert ID for ties. Scores must be
finite; `k` must be in `[1, expert_count]`; score count must equal expert count.

## Reference And Top-K Correctness

- Activation A router score parity: max abs `0`, max rel `0`, mean abs `0`.
- Activation B router score parity: max abs `0`, max rel `0`, mean abs `0`.
- Tolerance: `1e-5`.
- Top-k IDs match independent mapped-weight reference exactly for both activations.
- Tie ordering and invalid `k`, mismatch, and NaN cases are covered by
  `vbuf_topk_contract`.

## Working-Set Proof

Router selection completes before the selected expert graph is built. Both
graphs use graph-local refs `gate:0,up:1,down:2`, with no router-specific tensor
mapping below the selection boundary.

- Activation A selected top-1 expert: `37`; selected ranges are gate
  `124602848/563200`, up `160647664/563200`, down `235868672/1622016`.
- Activation B selected top-1 expert: `3`; selected ranges are gate
  `105454048/563200`, up `141498864/563200`, down `180720128/1622016`.
- Unselected expert tensors acquired: `0`.
- Unselected expert source reads: `0`.
- Routing changes active working set: PASS.

POC11 executes only the selected top-1 expert through the unchanged POC10
expert path. The remaining top-5 IDs are recorded but not aggregated.

## Residency, Materialization, And Leases

- Router cold source reads: `1`.
- Router warm source reads: `0`.
- Router residency hit: PASS.
- Router lease: one consumer, acquire step `1`, release step `1`.
- Selected expert gate is locally preloaded; selected up/down are HTTP misses.
- Selected up/down follow `IN_FLIGHT -> READY -> RELEASED`.
- Selected expert tensor leases retain the POC10 lifecycle: acquire/release
  steps gate `1/1`, up `2/2`, down `4/4`.
- Expert materializer teardown resources: `0` for both activations.

## Downstream Correctness

- Activation A selected expert reference parity: PASS, max abs `0`, max rel `0`.
- Activation B selected expert reference parity: PASS, max abs `0`, max rel `0`.
- Existing POC10 downstream semantics preserved: PASS.

## Failure Isolation

When the router endpoint is unavailable, router materialization fails before
router computation. No expert is selected, no expert graph is created, no
expert source is read, and resources after teardown are `0`.

Invalid top-k configuration and non-finite scores fail closed before expert
execution.

## Scope And Guards

- Source-call instrumentation: existing generic materialization trace only; no
  new policy instrumentation was added.
- Router persistent layout: `DIRECT`.
- vBuf layout change required: `NO`.
- vBuf format change required: `NO`.
- Architecture-specific runtime logic: `NO`.
- Full multi-expert aggregation: not implemented by design.
- Full transformer execution: not implemented by design.
- RV2 router compute: not executed.
