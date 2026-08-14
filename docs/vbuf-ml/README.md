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

The crate intentionally does not define tensors, model metadata, tokenizers,
quantization, backends, placement, or a profile wire format. Those decisions
belong to later vBuf-ML steps. A request from this descendant is not by itself
a reason to promote a feature into generic vBuf; promotion requires independent
downstream-neutral utility and generic qualification.

The boundary follows:

```text
raw bytes -> canonical v0.6 validation -> checked ranges -> profile semantics
```

The base remains a small compositional vocabulary: efficient composition is
preferred over maximal base functionality.
