# Attention Tensor Inventory

Artifact SHA256:
`780a55b77d2730705a93622338d9747149624d72e558e26176868210fafcc`

| Tensor | ID | Representation | Dimensions | Offset | Bytes | Layout |
|---|---:|---:|---|---:|---:|---|
| `blk.1.attn_norm.weight` | 13 | CanonicalPrimitive / F32 | [2048] | 96126272 | 8192 | DIRECT |
| `blk.1.attn_q.weight` | 15 | GGML_IQ1_S | [2048,3072] | 96134480 | 1228800 | DIRECT |
| `blk.1.attn_kv_a_mqa.weight` | 10 | GGML_IQ1_S | [2048,576] | 97363296 | 230400 | DIRECT |
| `blk.1.attn_kv_a_norm.weight` | 11 | CanonicalPrimitive / F32 | [512] | 97593704 | 2048 | DIRECT |
| `blk.1.attn_kv_b.weight` | 12 | GGML_IQ1_S | [512,4096] | 97595768 | 409600 | DIRECT |
| `blk.1.attn_output.weight` | 14 | GGML_IQ2_XXS | [2048,2048] | 98005384 | 1081344 | DIRECT |

Total persistent attention bytes: `2960384`.

Transient values include normalized input, q, kv-a, compressed latent, q/K/V
head values, scores, probabilities, context, and output. Runtime KV state is
not a persistent tensor and is not serialized into vBuf.
