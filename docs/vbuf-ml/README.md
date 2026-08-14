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

The crate currently defines a small profile bootstrap, tensor-directory semantic
index, model-metadata index, and vocabulary-only tokenizer view. These add only
ML names, relationships, shapes, and domain validation over canonical generic
values; they do not duplicate physical descriptors. Quantization, tokenizer
algorithms, merges, backends, placement, and conversion remain deferred.
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

Step 11 keeps `VocabularyOnly` normative and qualifies lazy versus eager
access with file-backed mmap fixtures. Step 12 keeps `CanonicalPrimitive`
normative while quantized layouts await a pinned first target and upstream
revision. Step 13 adds only deterministic downstream writer ordering and
payload-alignment planning; BaseStep remains generic. Runtime-local indexes and future algorithm-specific structures remain
derived or deferred.

The base remains a small compositional vocabulary: efficient composition is
preferred over maximal base functionality.
