# POC14 Attention + KV Substrate Closure

## Boundary

The qualified boundary is real DeepSeek-V2-Lite `blk.1` attention from the
FFN/attention block input through attention residual output. It uses the
non-split MLA-compatible KV-B reference branch with 16 heads, 192-dimensional
Q/K heads, 64 rotary dimensions, 512 compressed KV dimensions, and 128 value
dimensions per head.

## Correctness

- Token 0 attention residual parity: PASS, max abs `0`.
- Token 1 attention residual parity: PASS, max abs `0`.
- Token 1 without token 0 state delta: `0.368056`, state influence PASS.
- State after token 0: K/V positions `1/1`.
- State after token 1: K/V positions `2/2`.
- Runtime state appends: `4`; state reads: `96`.
- Warm token 0/token 1 parity: PASS.

The independent reference uses mapped persistent payloads and a separate
state instance. The runtime path obtains the same quantized projections through
the TensorDependencyExecutor and materializer path.

## Residency And Timing

- Cold source reads: `6`.
- Cold source bytes: `2960384`.
- Warm source reads: `0`.
- Warm source bytes: `0`.
- Peak resident bytes: `2960384`.
- Persistent attention bytes: `2960384`.
- Additional execution-preparation copies: `0`.
- Repacked bytes: `0`.
- Transcoded bytes: `0`.
- Payload-to-first-consumer instrumentation: enabled.
- Representative cold q payload-to-first-consumer gap: `68869842 ns`.
- Representative cold output payload-to-first-consumer gap: `68756241 ns`.

All six persistent attention tensors are direct views/materialized ranges. KV
state remains transient and separate from residency.

## Failures And Guards

- Required `attn_q.weight` failure: dependent op not executed, final output
  invalid, resources after teardown `0`.
- Invalid state shape/capacity contract: PASS.
- Model artifact mutation: `NO`.
- Architecture-specific runtime logic: `NO`.
- vBuf layout change required: `NO`.
- vBuf format change required: `NO`.
- Structural-open baseline unchanged; POC13 baseline remains the reference
  structural benchmark.
