# POC13 Full Real DeepSeek MoE Layer Closure

## Boundary And Input

- Layer boundary: `blk.1` FFN/MoE sublayer.
- Input: deterministic float32 `[2048,1]` one-hot fixture.
- Activation A: index `0`.
- Activation B: index `1`.
- RMSNorm epsilon: `1e-6`, verified against the reference path.

## Composition

The complete graph is RMSNorm, real router matmul, deterministic top-6,
six selected routed expert FFNs, selected-probability weighted merge, shared
expert SiLU FFN, routed-plus-shared add, and residual add with the original
layer input.

Shared expert is required and executes through the same generic expert graph
machinery. No architecture-specific runtime branch was added.

## Results

- Activation A top-k: `37,21,31,54,6,33`.
- Activation A normalized weights: `0.182863,0.175766,0.175188,0.16509,0.153029,0.148064`.
- Activation B top-k: `3,53,47,5,20,35`.
- Activation B normalized weights: `0.241116,0.210624,0.159175,0.14877,0.126245,0.11407`.
- Normalized input parity: PASS for A and B.
- Router logits parity: PASS for A and B.
- Selected-weight parity: PASS for A and B.
- Six routed expert outputs: PASS for both activations.
- Routed merge parity: PASS.
- Shared expert parity: PASS.
- Pre-residual composition parity: PASS.
- Final layer parity: PASS, max abs/rel/mean abs `0`, tolerance `1e-5`.

## Persistent Metrics

- Selected routed expert bytes: `16490496`.
- Always-required bytes (norm, router, shared tensors): `4677632`.
- Total required bytes: `21168128`.
- Peak active persistent bytes: `1892352`.
- Peak active fraction: `0.0893963`.
- Peak resident bytes: `8245248`.
- Router/expert/shared layout: `DIRECT`.
- Bytes copied for execution preparation: `0` additional backend copies.
- Bytes repacked: `0`.
- Bytes transcoded: `0`.

## Residency And Source Behavior

The 8 MiB tensor residency budget is shared by norm, router, shared expert,
and selected expert slices. Full-layer per-run counters are in `execution.log`.

- A cold source reads: `20`, source bytes `19478528`.
- A warm source reads: `3`, source bytes `1689600`.
- B transition source reads: `18`, source bytes `16490496`.
- A replay source reads: `0`, source bytes `0`.
- A→B→A: PASS.
- Unselected expert graphs: `0`.
- Unselected expert acquisitions: `0`.
- Unselected expert source reads: `0`.
- Source striping on x86: none; all selected ranges used single-source HTTP.

Per-expert POC12 residency, source-policy, materialization, prefetch, and lease
traces remain embedded in the full execution log.

## Timing And Structural Open

- Artifact size: `4993331814` bytes.
- Structural records visited: `377`.
- Cold structural open: `2134805224 ns`.
- Warm structural open: `6588236 ns`.
- Payload bytes touched during structural open: not measurable through the
  current generic ABI.
- First-payload-to-shared-compute-start is recorded per run.
- Exact payload-to-first-consumer timing is not currently instrumented;
  execution-ready-to-consumer is explicitly reported as `NOT_INSTRUMENTED`.

The effective cold structural traversal rate is approximately `2.34 GB/s`;
this is a structural traversal metric, not storage bandwidth.

## Failure Isolation

- Selected expert failure: expert `37`, `blk.1.ffn_up_exps.weight` fails;
  consuming op, final composition, and final output are invalid; teardown is `0`.
- Always-required failure: `blk.1.ffn_norm.weight` fails before dependent
  execution; no final output is reported; teardown is `0`.

## Scope Guards

- Shared expert: required and qualified.
- Full transformer stack: not implemented.
- Router/expert/shared persistent layout change: none.
- vBuf layout change required: `NO`.
- vBuf format change required: `NO`.
- Architecture-specific runtime logic: `NO`.
