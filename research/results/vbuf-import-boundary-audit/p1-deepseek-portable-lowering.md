# P1 DeepSeek Portable Import And Lowering Gate

## Import

The existing qualified `DeepSeek-V2-Lite.IQ1_S` manifest was imported into
`deepseek2-typed-program.json` using generic semantic bindings. The manifest
proves 377 tensors, `deepseek2`, 27 layers, 64 experts, top-6 routing, two
shared experts, 2048 embedding width, 163840 context, and the IQ1_S fixture
identity. GGML type IDs remain source provenance and are excluded from the
portable boundary.

The importer expresses compressed-KV attention, position state, dense and MoE
regions, shared experts, routed expert alternatives, and output projection
using the same descriptor vocabulary as Qwen, Phi, and Gemma.

## Legacy comparison

The legacy PoC22 path is known-correct on x86 for the same DeepSeek lineage:
four-token generation, router parity, zero logit error, generated-token
feedback, and zero-resource teardown. It is valid control evidence, not the
portable schema.

## Gate 2 result

`DEEPSEEK_PORTABLE_LOWERING_INCOMPLETE`.

The repository has no generic PortableProgram lowerer. `rust/vbuf-runtime`
contains only a storage/residency control plane and a graph seed. The active
PoC22 C++ path constructs `LayerPlan` from `blk.N.` names, directly looks up
DeepSeek tensor names such as `ffn_gate_inp.weight`, fixes 64 experts and fixed
state widths, and calls DeepSeek-shaped attention/expert helpers. It therefore
cannot consume the imported sidecar without introducing the prohibited
architecture-specific runtime path.

The existing Rust control plane also contains `parse_layer` in
`rust/vbuf-runtime/src/lib.rs`, which parses `blk.N.` names to group ranges.
That is a pre-existing runtime leakage and was not modified in this phase; a
generic lowerer must replace this convenience with imported dependency/region
references before neutrality can be claimed.

No lowering, execution, or parity claim was made. This is an intentional phase
stop, not a numerical failure. The exact missing seam is a generic lowerer and
backend-neutral execution-region representation between the sidecar and the
existing PoC22 compute helpers.

## Required next implementation

Implement a generic lowerer that consumes only `PORTABLE_SEMANTIC_V1` nodes,
TensorRefs, StateRefs, and dynamic alternatives. It may reuse GGML compute
helpers after backend binding, but it must not inspect architecture names,
source tensor names, GGUF IDs, or vendor identity. Only after the DeepSeek
control path passes through that lowerer may Qwen lowering begin.
