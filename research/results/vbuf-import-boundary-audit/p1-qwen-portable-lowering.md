# P1 Qwen Portable Lowering

Phase F was not started because Gate 2 failed before lowering:
`DEEPSEEK_PORTABLE_LOWERING_INCOMPLETE`.

The existing Qwen3.6 sidecar remains valid as an importer-owned descriptive
program for attention, SSM, and MoE regions. No Qwen execution path, Qwen
lowerer, PoC22 Qwen graph, kernel, runtime switch, or oracle comparison was
added. Qwen lowering is authorized only after DeepSeek demonstrates parity
through the same generic lowerer.

The existing `vbuf-runtime::parse_layer` name parser is also an unresolved
execution-boundary gap. Qwen cannot be called a same-lowerer proof while that
path remains responsible for layer grouping.
