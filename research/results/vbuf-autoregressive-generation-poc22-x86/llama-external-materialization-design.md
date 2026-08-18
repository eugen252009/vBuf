# llama.cpp External Materialization Design

## Source-Agnostic Handoff

The C++ boundary receives only:

```text
tensor name
ggml type
dimensions
materialized pointer
materialized byte length
opaque lease held by the owner
```

Source descriptors, SourceId resolution, checked u64 offsets, file reads, and
HTTP Range reads remain Rust/runtime responsibilities. The new metadata-only
open path parses the semantic bootstrap but does not fetch payloads.

## Local Fast Path

`vbuf_ml_consumer_tensor_info` and the existing `VbufMlAdapter::tensor` path
remain unchanged for local artifacts. They continue to return pointers into
the mmap, and `VbufRuntime` keeps the owning consumer handle alive through
model destruction.

## External Path

`VbufMlAdapter::tensor_materialized` calls the descriptor-only FFI function,
checks that the supplied span length equals the validated TensorRef length,
then exposes the same `TensorDescriptor` shape/type contract with the supplied
pointer and lease. This is the same downstream descriptor shape as local
payloads; only the backing ownership differs.

## Width and Provenance

Logical source offsets remain u64 in Rust and in the diagnostic
`VbufMlTensorSourceInfo` FFI structure. C++ receives source ID and source
offset only for provenance; it does not resolve them or convert them to a
pointer. Materialized byte lengths are checked before crossing the boundary.

## Not Implemented Here

The full `llama_model_load_vbuf` path still opens a local payload-bearing
artifact and eagerly inventories direct pointers. It has not been converted
into a source-materializing model loader because that would require a larger
runtime factory or a C++ callback into source-aware Rust I/O. The bounded
descriptor/materialized-span seam is the narrow next boundary, while the real
ggml compute qualification remains unreached until the external llama.cpp
dependency is available.
