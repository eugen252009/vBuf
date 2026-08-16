# vBuf to ggml Tensor Adapter

This is the physical tensor compatibility boundary only. It has no model,
architecture, layer, expert, tokenizer, or scheduling knowledge.

## API and Ownership

`VbufTensorView` is the smallest native input:

```text
representation, rank, dimensions, payload pointer, payload length
```

It is a C-compatible subset of the existing Rust `VbufMlTensorView` payload
view. Tensor identity is not needed for descriptor construction; diagnostics
can retain it outside this boundary.

`BorrowedGgmlTensor` owns:

- the transient ggml context and tensor descriptor;
- the CPU backend handle;
- the non-owning ggml external buffer wrapper;
- a shared source lease supplied by the caller.

Destruction releases the ggml buffer, context, and backend before releasing the
source lease. The source lease therefore outlives every backend read. The
adapter never frees source bytes.

`VbufBorrowedStorage` is runtime-only containing-span metadata. It provides an
aligned base, byte length, payload offset, and lifetime lease. It is not a vBuf
format field.

## Representation Mapping

| vBuf profile ID | Representation | ggml type | Elements/block | Bytes/block |
|---:|---|---|---:|---:|
| 0 | F32 | `GGML_TYPE_F32` | 1 | 4 |
| 5 | IQ1_S | `GGML_TYPE_IQ1_S` | 256 | 50 |
| 7 | IQ4_NL | `GGML_TYPE_IQ4_NL` | 32 | 18 |
| 10 | IQ2_XXS | `GGML_TYPE_IQ2_XXS` | 256 | 66 |
| 4 | Q2_K | `GGML_TYPE_Q2_K` | 256 | 84 |
| 13 | Q5_K | `GGML_TYPE_Q5_K` | 256 | 176 |

Unknown profile IDs fail closed. Backend-local repacked types are not accepted
as canonical representations.

## Descriptor Rules

For `rows = product(shape[1..])` and
`blocks_per_row = shape[0] / elements_per_block`:

```text
nbytes = rows * blocks_per_row * bytes_per_block
nb[0]  = bytes_per_block
nb[1]  = nb[0] * blocks_per_row
nb[i]  = nb[i-1] * shape[i-1]  for i >= 2
```

The adapter rejects zero dimensions, unsupported ranks, signed-dimension
overflow, block-width mismatch, arithmetic overflow, and payload-length
mismatch. Q2_K is validated as 256 values and 84 bytes per block.

## CPU Binding

The direct path is:

```text
vBuf payload range
    -> aligned containing span
    -> ggml_backend_cpu_buffer_from_ptr()
    -> ggml_backend_tensor_alloc()
    -> ggml CPU/RVV graph
```

`ggml_backend_tensor_set()` is not used to bind weights. The adapter verifies
the containing span, alignment, offset, and bounds before binding. The
qualification tests verify that `tensor->data` equals the source payload
address and that source bytes remain unchanged.

The current implementation has no staging-copy fallback. An alignment or
range incompatibility fails explicitly as an adapter error rather than silently
becoming a canonical copy path. A future backend-local fallback must be
labelled `BACKEND_LOCAL_FALLBACK`.

CUDA is not implemented here. Its future path must materialize canonical bytes
into backend-owned device storage rather than imply that CPU borrowing works on
all backends.

## Qualification

`vbuf_tensor_adapter_qualification` covers all six supported representations,
expected ggml types, `ne[]`, `nb[]`, and `nbytes`, plus malformed and overflow
inputs. It also verifies:

- aligned containing-span binding;
- out-of-range span rejection;
- F32 borrowed execution and pointer identity;
- Q2_K borrowed CPU execution through `ggml_mul_mat`;
- source immutability and absence of a full source-to-destination weight copy.

The existing substrate smoke and borrowed-buffer probes remain in the CMake
test suite.

## Remaining Work

The next step is a real model-layer proof that supplies architecture-derived
graph lowering separately from this adapter. Async execution must retain the
source lease through backend synchronization. CUDA materialization and any
backend-local fallback require independent qualification.
