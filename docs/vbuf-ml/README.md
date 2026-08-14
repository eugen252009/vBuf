# vBuf-ML descendant boundary

`vbuf-ml` is a downstream profile crate. It consumes canonical, validated
vBuf state and generic checked ranges; it does not define or override generic
physical validity.

Dependency direction is one-way:

```text
vbuf-ml -> vbuf-core / vbuf-layout
vbuf-core -X-> vbuf-ml
vbuf-layout -X-> vbuf-ml
```

The crate currently defines only a small profile bootstrap and tensor-directory
semantic index. The tensor directory adds names, shapes, and canonical generic
value references; it does not duplicate physical descriptors. Model metadata,
tokenizers, quantization, backends, placement, and conversion remain deferred.
A request from this descendant is not by itself
a reason to promote a feature into generic vBuf; promotion requires independent
downstream-neutral utility and generic qualification.

The boundary follows:

```text
raw bytes -> canonical v0.6 validation -> checked ranges -> bootstrap roles -> tensor semantics
```

The bootstrap and tensor directory are located through canonical generic
blocks. Their semantic references resolve by generic Key-ID and physical
occurrence, never by trusted profile offsets.

The base remains a small compositional vocabulary: efficient composition is
preferred over maximal base functionality.
