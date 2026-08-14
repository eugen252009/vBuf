# ADR 0004: Tensor representation identity

- **Status:** Proposed; selection required before tensor-directory wire freeze
- **Scope:** vBuf-ML only
- **Related plan decision:** D

## Context

A scalar bit width does not identify a quantized tensor representation. Runtime kernels require exact block size, packed byte layout, scale/zero-point interpretation, and representation version. Directly freezing an external enum into vBuf-ML would couple profile evolution to that enum, while inventing a new representation would invalidate an equivalent-kernel GGUF comparison.

## Recommended decision

Use a namespaced representation registry:

```text
local compact ID
    -> namespace
    -> representation name/type
    -> representation version
    -> exact block and byte-layout contract
```

For initial qualification, preserve selected GGML packed tensor bytes unchanged. The registry can map to a pinned GGML representation without making external enum numbers the permanent profile namespace.

## Alternatives

1. **Namespaced registry (recommended):** compact entries with explicit external representation identity/version.
2. **Freeze `ggml_type` numeric values directly:** simpler initial loader, but permanent external-enum coupling.
3. **Invent vBuf-specific quantization now:** rejected for first qualification because repacking would confound format/runtime comparisons.

## Invariants

- Representation semantics remain vBuf-ML-specific and never enter `vbuf-core`.
- Mixed representations are allowed only when every entry resolves and validates independently.
- Stored byte length is checked against logical shape and representation block rules.
- Unsupported namespace/type/version fails before typed tensor access.
- Initial GGUF conversion preserves per-tensor names, shapes, representation identity, and packed-byte digests.

## Decision required

Explicitly select alternative 1 or 2 before Step 9 freezes tensor-directory fields. This ADR does not authorize tensor implementation.
