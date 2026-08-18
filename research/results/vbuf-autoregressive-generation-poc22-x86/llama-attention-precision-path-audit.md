# Pinned llama Attention Precision-Path Audit

## Result

The pinned CPU execution has an F32-capable `FLASH_ATTN_EXT` implementation, but the llama context defaults its KV cache to F16. The graph then also explicitly casts F32 graph K/V inputs to F16 when flash attention is enabled. The active CPU path therefore receives F16 K/V.

The bounded diagnostic set `type_k=GGML_TYPE_F32`, `type_v=GGML_TYPE_F32` in the context parameters and removed only the graph-local casts in a separate temporary tree. The active CPU flash implementation then received F32 K/V and produced:

- `kqv_out-0`: `9e37b0a2234c59cf`
- `Vcur_cont-0`: `9e37b0a2234c59cf`
- context numerical parity: exact, max absolute error `0`
- attention output: `c415092f31bd2d54`, matching vBuf

The completed F32-KV position-0 ADD qualification matches both operands and
the post-attention residual bit-for-bit. The earlier `ffn_inp-0` mismatch was
from the default F16 trace and is superseded mixed-mode evidence. No later
block, output, or logit boundary is qualified by the bounded ADD audit.

## Source Findings

- `src/llama-context.cpp:3534-3535`: default `type_k` and `type_v` are `GGML_TYPE_F16`.
- `src/llama-graph.cpp:2523-2542`: flash graph path explicitly casts F32 K/V to F16, then calls `ggml_flash_attn_ext`.
- `src/llama-graph.cpp:2546-2547`: the graph sets attention precision to `GGML_PREC_F32`; this controls accumulation/precision mode, not KV-cache storage type.
- `ggml/src/ggml-cpu/ops.cpp:9111-9195`: CPU `FLASH_ATTN_EXT` accepts F32 K/V and uses the F32 accumulation branch when V is F32.
- `ggml/src/ggml-cpu/ops.cpp:9202-9210`: CPU dispatch accepts `GGML_PREC_DEFAULT` and `GGML_PREC_F32` and invokes that implementation.
- `ggml/src/ggml-et/ggml-et-ops.cpp:1522-1598`: ET has an explicit `flash_attn_ext_f32` kernel selection, but ET is not the active CPU backend in this oracle.

The canonical Step21 patch changes only model-source loading/metadata plumbing and does not touch attention precision. The same F16 default and graph cast are present in the pristine pinned source.

## Classification

`ATTENTION_PRECISION_POLICY`: the existing CPU F32 path is available, but the default F16 KV-cache policy and flash graph construction select F16. This is not a vBuf runtime issue, backend rejection, or Step21 patch regression.
