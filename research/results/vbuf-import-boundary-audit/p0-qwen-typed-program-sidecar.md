# P0 Qwen3.6 Typed Program Sidecar

This is an importer-owned semantic sidecar for the real
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` artifact. It is not a vBuf wire-region
specification, kernel implementation, runtime schedule, or residency plan.

## Source Evidence

- Architecture: `qwen35moe`
- Artifact: `research-models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- Layers: 41
- Experts: 256 total, top-8 active
- Attention metadata: 16 query heads, 2 KV heads, head dimension 256
- Hybrid metadata: SSM inner size 4096, state size 128, group count 16,
  convolution kernel 4, time-step rank 32

Representative real regions:

- Full attention: layer 3 (`blk.3.attn_q.weight` shape `2048x8192`,
  `attn_k.weight` `2048x512`, `attn_v.weight` `2048x512`,
  `attn_output.weight` `4096x2048`, all `Q8_0`)
- SSM: layer 0 (`ssm_alpha.weight` `2048x32`, `ssm_beta.weight` `2048x32`,
  `ssm_conv1d.weight` `4x8192`, `ssm_out.weight` `4096x2048`; F32 inputs and
  Q8_0 output)
- MoE: layer 0 (`ffn_gate_inp.weight` `2048x256` F32 router,
  `ffn_gate_exps.weight` `2048x512x256` Q4_K,
  `ffn_up_exps.weight` `2048x512x256` Q4_K,
  `ffn_down_exps.weight` `512x2048x256` Q5_K)

## Boundary

The sidecar stores semantic TensorRefs, shapes, representation contracts,
typed operations, state descriptors, and a dynamic expert alternative set.
Source names, source offsets, and GGML representation labels are provenance
for importer audit and validation only. The semantic program does not depend
on parsing names at runtime.

Excluded from the portable program are GGML numeric IDs, backend buffers,
device placement, source read plans, current-token expert selections, and
runtime scheduling/residency decisions.

## State And Unresolved Semantics

The descriptor includes sequence position, layer-3 KV cache, and layer-0
recurrent SSM state with explicit transitions. Qwen3.6-specific SSM equations
are represented by the importer-defined version marker
`importer_defined_qwen35moe_ssm_v1`; the descriptor does not claim that this
marker is a standardized vBuf primitive. Capacity and drop behavior for MoE
are explicitly `unspecified` because the source evidence inspected here does
not establish an architecturally observable policy.

Generated descriptor: `qwen35moe-typed-program.json`.
