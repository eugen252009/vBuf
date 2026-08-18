# Position-0 FFN Input Boundary Audit

## Result

The pinned llama source proves that `ffn_inp-0` is not a naming-only
intermediate. In `src/models/deepseek2.cpp:648-649`, the graph creates

```cpp
ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
cb(ffn_inp, "ffn_inp", il);
```

For block 0, `cur` is the attention output and `inpSA` aliases the block input
`inpL` (`src/models/deepseek2.cpp:470-474`). The callback is attached
immediately after the ADD. The same `ffn_inp` is passed directly to
`build_norm` at `src/models/deepseek2.cpp:651`, using `LLM_NORM_RMS`, and is
later reused by the post-FFN residual ADD at line 692.

Therefore:

```text
ffn_inp-0 = actual post-attention residual
ffn_inp-0 = direct block-0 FFN RMSNorm operand
```

The F32-KV position-0 ADD audit captured `cur = c415092f31bd2d54`,
`inpSA = 69c5d6519b190f83`, and `ffn_inp-0 = fd017832a837be21`, matching the
corrected vBuf residual. The earlier `3d607311e32ecf27` value came from the
default F16 trace and is not a valid F32-KV comparison value.

## Exact Chain

```text
inpL / inpSA: pre-attention residual, 69c5d6519b190f83
  -> RMS_NORM: attn_norm-0, c92b4d34c0160a28
  -> Q/K/V and attention context, 9e37b0a2234c59cf
  -> attention output, c415092f31bd2d54
  -> ADD(cur, inpSA): ffn_inp-0, fd017832a837be21
  -> RMS_NORM + FFN norm weight: ffn_norm-0, 57a1e1bc0fb324b6
  -> dense FFN
  -> ADD(FFN output, ffn_inp), then build_cvec: l_out-0
```

`ffn_norm-0` is the RMSNorm output after `build_norm` applies the FFN norm
weight, not the RMSNorm input. `build_norm` dispatches `LLM_NORM_RMS` to
`ggml_rms_norm` in `src/llama-graph.cpp:1556-1565`, then multiplies by the
weight at lines 1577-1582.

`l_out-0` is the final block-0 output after the FFN residual composition and
`build_cvec`, from `src/models/deepseek2.cpp:692-695`. Its captured hash is
graph-identity evidence only; it is not a qualified parity comparison after
the `ffn_inp-0` divergence.

## Classification

```text
FFN_INP_CREATION_SITE_IDENTIFIED: YES
FFN_INP_PRODUCER_IDENTIFIED: YES
FFN_INP_PRODUCER_OP: ADD
FFN_INP_CONSUMERS_IDENTIFIED: YES
FFN_INP_IS_DIRECT_FFN_RMSNORM_OPERAND: YES
ACTUAL_FFN_RMSNORM_INPUT_IDENTIFIED: YES
ACTUAL_FFN_RMSNORM_INPUT_HASH: fd017832a837be21
FFN_RMSNORM_INPUT_MATCHES_FFN_INP_0: YES
FFN_RMSNORM_INPUT_MATCHES_VBUF_CORRECTED: YES
POST_ATTENTION_RESIDUAL_IDENTIFIED: YES
PRE_ATTENTION_RESIDUAL_HASH: 69c5d6519b190f83
POST_ATTENTION_RESIDUAL_HASH: fd017832a837be21
POST_ATTENTION_RESIDUAL_MATCHES_FFN_INP_0: YES
POST_ATTENTION_RESIDUAL_MATCHES_VBUF_CORRECTED: YES
FFN_INP_HOOK_LOCATION_IDENTIFIED: YES
FFN_INP_HOOK_SEMANTIC_LABEL_ACCURATE: YES
L_OUT_0_SEMANTIC_ROLE: FINAL_BLOCK_OUTPUT_AFTER_FFN_RESIDUAL_AND_CVEC
L_OUT_0_HASH: a4a9304b1af7ced8
FFN_NORM_0_SEMANTIC_ROLE: RMSNORM_OUTPUT_AFTER_WEIGHT_MULTIPLICATION
BLOCK0_GRAPH_CHAIN_CAPTURED: YES
SEMANTIC_BOUNDARY_MAPPING_COMPLETE: YES
FFN_INPUT_BOUNDARY_ROOT_CAUSE_CLASS: NONE_AT_POST_ATTENTION_ADD
PREVIOUS_EXTERNAL_DIVERGENCE_FFN_INP_0: SUPERSEDED_MIXED_MODE_EVIDENCE
NEW_DIVERGENCE_CONFIRMED: NO
FIRST_TRUE_EXTERNAL_DIVERGENCE_BOUNDARY: NONE_YET_AT_ADD
DOWNSTREAM_PARITY_CLAIMS_AFTER_DIVERGENCE: NO
```

GRAPH_SEMANTICS: CONFIRMED
OLD_HASH_COMPARISON: SUPERSEDED
F32_KV_SEMANTIC_PARITY: PASS

The original `ffn_inp-0` mismatch was a mixed F16/F32 diagnostic comparison,
not an F32-KV semantic divergence. The previous vBuf residual fix remains in
place and is not invalidated. No permanent fix is applied.

## Scope

Only block 0 graph identity was audited. No later model boundaries, later
positions, logits, recurrence, cache requalification, or benchmarks were run.
The next smallest experiment is to inspect the operand production and
materialization details of the two differing post-attention residual ADDs;
it should not proceed to downstream parity.
