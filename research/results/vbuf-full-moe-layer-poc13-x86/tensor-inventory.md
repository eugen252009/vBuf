# `blk.1` Persistent Tensor Inventory

Artifact SHA256:
`780a55b77d2730705a93622338d9747149624d72e558e26182176868210fafcc`

| Tensor | ID | Representation | Dimensions | Offset | Bytes | Layout |
|---|---:|---:|---|---:|---:|---|
| `blk.1.ffn_norm.weight` | 21 | CanonicalPrimitive / F32 | [2048] | 99086736 | 8192 | DIRECT |
| `blk.1.ffn_gate_inp.weight` | 19 | CanonicalPrimitive / F32 | [2048,64] | 99094944 | 524288 | DIRECT |
| `blk.1.ffn_gate_shexp.weight` | 20 | GGML_IQ1_S | [2048,2816] | 99619248 | 1126400 | DIRECT |
| `blk.1.ffn_up_shexp.weight` | 23 | GGML_IQ1_S | [2048,2816] | 100745664 | 1126400 | DIRECT |
| `blk.1.ffn_down_shexp.weight` | 17 | GGML_Q2_K | [2816,2048] | 101872080 | 1892352 | DIRECT |

Routed expert slices use the POC12 exact 3D views: gate/up slices are
`563200` bytes and down slices are `1622016` bytes per expert. No packed expert
tensor is materialized whole.

Persistent execution preparation copies: `0` additional bytes. Materializer
payload ownership copies are the requested source ranges; no reorder, repack,
or transcode occurs.
