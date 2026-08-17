# Real DeepSeek-V2-Lite `blk.1` Layer Inventory

Reference path: pinned `llama.cpp/src/models/deepseek2.cpp`, `build_moe_ffn`
and the surrounding `ffn_inp` composition.

```text
ffn_inp = layer_input
normalized = RMSNorm(ffn_inp, blk.1.ffn_norm.weight, epsilon=1e-6)
router_logits = normalized @ blk.1.ffn_gate_inp.weight
router_probs = softmax(router_logits)
selected = deterministic top-6(router_probs)
selected_weights = selected_probs / sum(selected_probs)
routed = sum(selected_weights[i] * routed_expert_i(normalized))
shared = SiLU(normalized @ gate_shexp) * (normalized @ up_shexp)
shared = shared @ down_shexp
ffn_out = routed + shared
layer_output = ffn_out + ffn_inp
```

The shared expert is required for faithful `blk.1` closure. Attention and the
next transformer layer are outside this POC boundary.
