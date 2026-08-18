# Position-0 First-Divergence Audit

## Scope

This audit compares the pinned llama.cpp DeepSeek GGUF path with the vBuf-native path at position 0, token 0. It does not change model semantics, update pins, or run autoregressive generation beyond this single position.

## Inputs

- llama.cpp commit: `4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c`
- GGUF SHA256: `9d3bc4a5bc25b7acb8bc31436745bd8cfaf94509fd1322bb36ab155b0daf1616`
- vBuf SHA256: `780a55b77d2730705a93622338d9747149624d72e558e26182176868210fafcc`
- Native run: one position, seed token `0`, full stack, cost-aware residency

## Boundary Hashes

Hashes are FNV-1a over the raw IEEE-754 F32 bit patterns in tensor order.

| Boundary | llama.cpp | vBuf-native | Result |
|---|---:|---:|---|
| Token-0 embedding (`inpL`) | `69c5d6519b190f83` | `69c5d6519b190f83` | PASS |
| Attention RMSNorm (`attn_norm-0`) | `c92b4d34c0160a28` | `c92b4d34c0160a28` | PASS |
| Attention Q/K/V inputs (`q`, `Kcur`, `Vcur_cont`) | `ddc768f806ab7888` / `fa9bffc62bd69b63` / `9e37b0a2234c59cf` | `ddc768f806ab7888` / `fa9bffc62bd69b63` / `9e37b0a2234c59cf` | PASS |
| Attention output projection (`attn_out-0` / `attn_out_0`) | `f0fe17cdbad2f808` | `c415092f31bd2d54` | MISMATCH |
| Transformer block 0 output (`l_out-0` / `block_0`) | `a4a9304b1af7ced8` | `9d1c4fbd3b3566c2` | MISMATCH |

Additional llama boundaries were captured for follow-up: `attn_norm-0=c92b4d34c0160a28`, `l_out-1=3d288617569bd8e1`, `l_out-26=eba8f36641fa35c7`, `result_norm=b0cdbe8dded7cd45`, and `result_output=53224aeaaba8bfd4`.

## Conclusion

The first confirmed divergent boundary is the block-0 attention output projection. The model artifact, metadata, token-0 embedding row, attention RMSNorm, Q/K/V construction, RoPE/attention inputs, and runtime state are not the cause. The mismatch occurs when projecting the matching `Vcur_cont` attention value through the output projection weight. The next diagnostic should compare the output-projection tensor interpretation, matrix orientation, and dequantized multiply inputs without changing semantics.

The single-position native run still passes its internal runtime/reference parity and produces token `86711`; this is not an oracle parity result.

## Revision: Immediate Projection Operand Audit

The earlier `Vcur_cont` hash was not the immediate llama projection operand. In pinned llama.cpp, the output projection is applied after `build_attn_mha`; the exact operand is `kqv_out-0`, not `Vcur_cont-0`.

| Boundary | llama.cpp | vBuf-native | Result |
|---|---:|---:|---|
| Exact output weight raw SHA-256 | `a434525333668d360c4843bb76ae4f84e437b9aff998c4035b2b3f49ddeae5bb` | `a434525333668d360c4843bb76ae4f84e437b9aff998c4035b2b3f49ddeae5bb` | PASS |
| V input / native attention context | `Vcur_cont-0=9e37b0a2234c59cf` | `attention_context=9e37b0a2234c59cf` | PASS |
| Immediate projection input | `kqv_out-0=50293333ac9ee383` | `attention_context=9e37b0a2234c59cf` | MISMATCH |

The immediate input comparison is numeric, not only hash-based:

- max absolute error: `9.88245e-05`
- max relative error: `0.0154622`
- mean absolute error: `4.11948e-06`
- first differing element: `0`
- llama first values: `0.0120239258,-0.00514984131,0.0311431885,0.00433731079,-0.00498199463,-0.0133285522,0.0114059448,0.00226020813`
- vBuf first values: `0.0120244212,-0.00515118148,0.0311453342,0.00433723722,-0.00498362351,-0.0133314226,0.0114051234,0.00226012385`

Pinned llama’s `build_attn_mha` casts F32 K/V tensors to F16 before `ggml_flash_attn_ext` (`llama-graph.cpp`, lines 2533-2542). At position 0, the independent scalar attention result is the value vector because the sole attention probability is 1. vBuf’s direct F32 attention context matches `Vcur_cont`; llama’s flash-attention result differs by the measured F16/backend rounding. The output projection is therefore not the first divergent operation.

Revised classification: `DIAGNOSTIC_BOUNDARY_MISIDENTIFIED` for the original projection claim, with the demonstrated semantic difference at llama `ggml_flash_attn_ext` versus vBuf’s F32 scalar attention composition. No weight or runtime fix was applied.
