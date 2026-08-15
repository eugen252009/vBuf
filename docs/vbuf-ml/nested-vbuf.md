# Inline Nested vBuf

The current `vbuf-ml-0.1` profile supports an optional inline nested-directory
region. It is intended for hierarchical model data such as expert groups,
adapter groups, or other independently validated child payloads.

## Contract

The parent file contains:

```text
parent canonical vBuf
  -> optional NestedDirectory profile region
  -> one or more opaque parent blocks containing child bytes
```

Each directory entry contains a name, parent block Key-ID and occurrence, and
a child offset and length relative to that parent payload.

The child range must be non-empty, remain inside the parent payload, not overlap
another child range, have a unique strictly sorted UTF-8 name, and contain a
complete canonical v0.6 vBuf stream.

The child stream is parsed only after the parent canonical file and parent
bootstrap have been validated. Child bytes are never authorized by an
untrusted offset alone.

## Rust API

```rust
use vbuf_core::v06::parse_v06;
use vbuf_ml::{Bootstrap, NestedDirectory};

let parent = parse_v06(parent_bytes)?;
let bootstrap = Bootstrap::discover(&parent)?;
let nested = NestedDirectory::parse(&parent, &bootstrap)?;

for child in nested.children() {
    println!("{}: {} bytes", child.name, child.range.bytes().len());
    let _child_blocks = child.validated.blocks();
}
```

The nested directory payload is encoded with `encode_nested_payload`. The
nested directory itself remains a normal v0.6 opaque block referenced by the
profile bootstrap; the generic vBuf layer does not know that it is a model
directory.

## Generic MoE Catalog

An optional `MoeDirectory` profile region can accompany the nested directory.
It declares expert count, active-expert count, layer count, shared-expert
presence/count, and sorted `(layer, expert, role) -> nested child name` entries.
Every entry must resolve to an already validated nested child. The catalog is
architecture-neutral and does not encode router tensor names or graph rules.

The runtime dispatches architecture strings to `QwenMoE`, `DeepSeekMoE`, or
`Unsupported`; those loaders are responsible for architecture-specific graph
construction.

## Qwen MoE Loader

`QwenMoELoader` validates `qwen2moe`, `qwen3moe`, and `qwen3vlmoe` against the
llama.cpp tensor contract:

- `blk.{layer}.ffn_gate_inp.weight`: `[embedding, expert_count]`;
- `blk.{layer}.ffn_gate_exps.weight`: `[embedding, expert_ff, expert_count]`;
- `blk.{layer}.ffn_up_exps.weight`: `[embedding, expert_ff, expert_count]`;
- `blk.{layer}.ffn_down_exps.weight`: `[expert_ff, embedding, expert_count]`.

Qwen2 MoE additionally requires its four shared-expert tensors. The loader
validates descriptors and dimensions, then leaves graph construction to the
existing llama.cpp Qwen graph implementation.

## DeepSeek MoE Loader

`DeepSeekMoELoader` validates `deepseek2`, `deepseek32`, and `deepseek2-ocr`.
It supports dense leading layers and MoE layers. MoE layers require the
router, expert gate/up/down tensors, optional `ffn_exp_probs_b.bias`, and
shared expert gate/up/down tensors. Shared expert dimensions are checked as
`expert_ff * shared_expert_count`, matching llama.cpp's DeepSeek graph.

Conversion manifests for DeepSeek use `DEEPSEEK_HOT_ROLE_ORDER`: attention
weights come first within each layer, followed by the always-used router and
shared-expert weights; selectively-used expert gate/up/down matrices are last.
This keeps common per-token reads closer together while retaining layer-major
locality and canonical tensor names.

## MoE Status

Inline nested vBuf is the storage foundation for expert groups, but MoE runtime
routing is not inferred from child names. A complete MoE implementation still
needs an explicit architecture contract covering router tensor names/shapes,
expert tensor roles, shared experts, and llama.cpp graph construction. The
generic catalog now supplies the storage facts; nested children are still not
automatically routed by the llama adapter.
