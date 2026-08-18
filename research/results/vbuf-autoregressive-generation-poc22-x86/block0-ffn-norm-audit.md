# DeepSeek Block 0 FFN Norm Audit

## Scope

Position 0 only, with the temporary llama F32 K/V diagnostic active. No FFN
gate/up/down projections or full-model generation were used for this audit.

## Result

The previously reported FFN-input parity compared the stored residual boundary,
not the input actually passed to the vBuf dense FFN norm graph.

The actual vBuf path is:

```text
attention output -> FFN RMSNorm
```

The stored/llama boundary is:

```text
attention output + residual input -> FFN RMSNorm
```

The actual vBuf TensorWave norm input has hash `c415092f31bd2d54`, equal to the
attention output hash. The stored residual boundary has hash
`fd017832a837be21`.

Representative values:

| boundary | first eight values |
| --- | --- |
| vBuf norm execution input | `-0.0656610206,-0.0337293595,0.0192056596,0.0199207626,-0.0434858724,0.0422239639,0.0315527618,0.00656052306` |
| stored residual FFN boundary | `0.0854009911,-0.0278089494,0.0251260698,0.0258411728,-0.0375654623,-0.0244264267,0.0374731719,0.0124809332` |

The first failed equality is therefore before RMS reduction. The earlier
`FFN_INPUT_PARITY: PASS` classification was instrumentation-invalid.

## Norm Tensor Trace

The dense block-0 plan selects source `blk.0.ffn_norm.weight`, aliases it as
runtime `blk.1.ffn_norm.weight`, and registers it as persistent tensor index 1.
The observed runtime descriptor was:

```text
name=blk.1.ffn_norm.weight
tensor_id=10008
source_offset=74755336
representation=0 (F32)
rank=1
elements=2048
bytes=8192
raw_hash=5ee229ba6b5d069c
bound_hash=5ee229ba6b5d069c
```

The materialized and unmaterialized vBuf norm runs produced the same wrong
output, so the current evidence does not implicate materializer replacement.

## Classification

```text
ROOT_CAUSE_CLASS=MODEL_PLAN_DATAFLOW
BUG_SIDE=VBUF_NATIVE
BUG_CAUSE=run_sequence passes actual_attention.output directly to run_dense_layer; the DeepSeek block boundary requires actual_attention.output + actual.values.
PERMANENT_FIX_APPLIED=YES (one native block-0 dataflow edge only)
```
