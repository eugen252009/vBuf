# DeepSeek-V2-Lite `blk.1` Attention Inventory

Reference path: pinned `llama.cpp/src/models/deepseek2.cpp`, the non-split
`wkv_b` branch of `build_arch_graph`.

```text
block input [2048,position]
→ RMSNorm(attn_norm, epsilon=1e-6)
→ q = attn_q(input) [3072]
→ kv_a = attn_kv_a_mqa(input) [576]
→ split kv latent [512] and rotary key [64]
→ RMSNorm(kv latent, epsilon=1e-6)
→ kv_b(kv latent) [4096]
→ split per-head K-nope [128] and V [128]
→ repeat/rotate K-rope [64] across 16 heads
→ causal scaled dot-product attention
→ output projection attn_output [2048]
→ residual add with block input
```

This artifact uses the non-split KV-B representation, not the split MLA
`wk_b/wv_b` representation. The runtime state is two append-only position
slots: compressed K `[16,192]` and V `[16,128]`. It is owned independently
from persistent vBuf tensors and is never written to the model artifact.

Token positions are `0` and `1`; token 1 attends to both positions while token
0 attends only to itself.
