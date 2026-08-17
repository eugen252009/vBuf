# POC16 Semantic Inventory

Selected consecutive span: real `blk.1` through `blk.3`, three blocks.

All three blocks expose the same audited geometry:

- MLA non-split KV attention, 16 heads, Q/K head 192, RoPE 64, latent 512,
  value head 128.
- FFN RMSNorm, router over 64 experts, top-k 6, separate routed gate/up/down
  slices, shared expert, residual add.
- Width `2048`; two-position F32 fixtures.

The `ffn_gate_up_exps` tensor is present in the artifact inventory for these
blocks, but the selected POC13-compatible execution path uses the separate
`ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` tensors, as established by
the prior qualification.
