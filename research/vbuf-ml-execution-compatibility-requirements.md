# vBuf-ML Execution Compatibility Requirements

Status: **requirements profile; no runtime or format change proposed**.

## Scope

This document separates requirements that must be true of a persisted vBuf-ML
artifact from requirements imposed by a ggml adapter, a backend, or an
execution optimization. It is based on the repository contracts and the pinned
llama.cpp revision:

```text
4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
```

The conclusions below do not claim that every listed representation is already
qualified on every backend. Format compatibility and backend kernel coverage
are separate gates.

## Backend-Neutral Tensor Requirements

The smallest generic immutable model-tensor contract is:

| Information | Classification | Reason |
|---|---|---|
| Tensor identity | `REQUIRED_FOR_CORRECTNESS` | The runtime must select the tensor referenced by a graph or model role |
| Logical representation/dtype | `REQUIRED_FOR_CORRECTNESS` | Determines mathematical interpretation and byte geometry |
| Shape and rank | `REQUIRED_FOR_CORRECTNESS` | Required for indexing, operation shape inference, and payload sizing |
| Exact byte range and byte length | `REQUIRED_FOR_CORRECTNESS` | Bounds the immutable source and proves payload completeness |
| Block geometry | `DERIVABLE` or `REQUIRED_FOR_CORRECTNESS` | Derivable from a versioned representation ID; explicit only if the representation is otherwise ambiguous |
| Representation alignment | `DERIVABLE` | Comes from the representation contract; current qualified contracts require no extra payload alignment |
| Immutable/mutable status | `REQUIRED_FOR_CORRECTNESS` | Execution may borrow immutable weights, while activations and state need writable runtime storage |
| Semantic role | `OPTIONAL_HINT` unless graph semantics require it | Names/roles help model selection but do not change tensor bytes |
| Strides | `DERIVABLE` for canonical contiguous tensors | Shape, representation, and block geometry determine the runtime descriptor strides |
| Padding | `DERIVABLE` or absent | Canonical payload has no implicit padding; runtime padding is a view/materialization concern |
| Backend buffer type | `BACKEND_SPECIFIC` | Selected only after backend placement |
| Device placement | `BACKEND_SPECIFIC` | A deployment decision, not a model representation fact |
| Repacked representation | `BACKEND_SPECIFIC` | A backend-local execution cache or transform |
| Kernel tile shape | `BACKEND_SPECIFIC` | Depends on the selected kernel implementation |
| ggml type ID | `BACKEND_SPECIFIC` | External numeric IDs are an adapter mapping, not a generic wire contract |
| Allocation granularity | `BACKEND_SPECIFIC` | Allocator policy does not describe model meaning |
| Tensor ownership | `DERIVABLE` at runtime | Persist the lifetime/immutability semantic, not a pointer owner object |
| Graph lifetime | `DERIVABLE` at runtime | Graphs are reconstructed for an execution context or region |

Persisted alignment may be stronger than one for file-range placement, but that
is a vBuf range invariant, not a SIMD or ggml allocation invariant. No evidence
currently establishes a representation-intrinsic alignment greater than the
existing vBuf alignment rules.

## Executive Contract

A backend-neutral vBuf-ML tensor needs:

1. logical shape, with the first dimension as the contiguous row width;
2. a profile-local representation identifier;
3. the representation's canonical payload bytes, unchanged;
4. checked payload length matching the representation and shape;
5. canonical byte ordering where the representation defines scalar fields;
6. the existing vBuf range, alignment, and integrity rules.

The adapter derives the ggml type, `ne[]`, `nb[]`, ggml tensor object, backend
buffer, device placement, and any backend-specific packed or repacked view. None
of those are required persisted fields.

## Descriptor Compatibility

| ggml/runtime item | Persist? | Adapter/runtime action | Independent justification |
|---|---:|---|---|
| `ggml_tensor.ne[]` | No | Construct from rank and logical shape | Shape is already the generic tensor fact |
| `ggml_tensor.nb[]` | No | Derive contiguous strides from type geometry and shape | No information is lost for canonical contiguous payloads |
| `tensor->data` | No | Bind the validated range at execution time | It is a process address, not persistent model meaning |
| `ggml_backend_buffer_t` | No | Wrap host memory or allocate device memory | Backend ownership and placement are runtime state |
| Graph node descriptors | No | Build from model execution semantics | Graph structure is not a property of one immutable payload range |
| Allocator arena/reset state | No | Create and recycle per graph/region | Lifetime and reuse do not affect tensor interpretation |
| Scheduler split/event state | No | Select after backend and graph construction | Scheduling is implementation state |

For a canonical contiguous quantized tensor, the adapter sets logical row width
from `shape[0]`, checks block divisibility, and derives the physical row span
from the representation's bytes-per-block. A ggml `nb[]` convention can
therefore be synthesized without changing source bytes. A backend requiring a
different view is a `VIEW_ADAPTER` or backend-local materialization, not a vBuf
layout change.

## Alignment Provenance

| Alignment source | Classification | Finding |
|---|---|---|
| Quantized block field alignment | `FORMAT_INTRINSIC` only where the canonical byte encoding requires it | The qualified opaque contracts are byte-addressable and require no extra alignment |
| vBuf range/base alignment | `FORMAT_INTRINSIC` | Existing vBuf range validation remains required for safe mapped access |
| CPU scalar/SIMD load alignment | `BACKEND_INTRINSIC` | The backend or kernel must use legal loads, an aligned view, or staging |
| `ggml_backend_cpu_buffer_from_ptr()` wrapper alignment | `BACKEND_INTRINSIC` | The wrapper's accepted alignment is a host-backend API constraint, not a model invariant |
| CUDA allocation/upload alignment | `BACKEND_INTRINSIC` | Device allocation and transfer APIs own this requirement |
| Allocator alignment/granularity | `GGML_IMPLEMENTATION_OPINION` | Convenience for allocation and reuse |
| Larger alignment for throughput | `OPTIONAL_OPTIMIZATION` | May be obtained by aligned mapping/span selection or backend staging |

The CPU borrowed-storage path requires a live pointer, a valid byte length, and
stable lifetime for all tensor reads. The pointer must satisfy the pinned CPU
wrapper and selected kernel's host-access rules. If a particular range offset is
not suitable, the adapter can choose an aligned containing span or make a
backend-local copy; that does not establish a global vBuf payload alignment
requirement. No evidence shows that weights are written by the ordinary direct
CPU/RVV paths.

## Stride And Layout Provenance

The persisted layout is the canonical logical row order plus its representation
block encoding. The runtime view layout is a descriptor over that layout. The
kernel-preferred layout may be different.

- `shape + representation + block geometry` deterministically derive canonical
  contiguous strides and byte count.
- No persisted `nb[]` is needed for a canonical contiguous tensor.
- Padding is not silently inferred inside a payload; any runtime padding belongs
  to a view or materialized buffer.
- `IQ4_NL_16x1` and `Q2_K_16x1` are transformed execution layouts. They are not
  the original IQ4_NL and Q2_K representations.
- A transformation required by one kernel is `GGML_IMPLEMENTATION_OPINION` or
  `OPTIONAL_OPTIMIZATION`, never `REPACK_REQUIRED` for the vBuf source unless no
  direct or generic execution path exists at all.

## Quantization Representation Ownership

The IQ1_S, IQ4_NL, IQ2_XXS, Q2_K, and Q5_K encodings have explicit block sizes,
field layouts, and dequantization semantics. vBuf persists those source bytes
directly. ggml currently supplies type traits and kernels that consume them, but
the mathematical encoding is independently implementable by another runtime.
Thus the canonical encodings are model representations (`FORMAT_INTRINSIC`),
while ggml's type IDs and repacked layouts are not.

The Q2_K qualification is direct evidence of this ownership boundary: the
source payload is 84 bytes per 256-value block, and the adapter must expose all
84 bytes. The 82-byte interpretation was a contract error, not a required
conversion.

## CPU Borrowed-Storage Compatibility

The smallest CPU binding is:

```text
validated vBuf range
    -> live host pointer + length
    -> ggml CPU external buffer wrapper
    -> derived tensor descriptor
    -> direct CPU/RVV reads
```

The pinned CPU backend advertises host-pointer wrapping. The wrapper must retain
the source mapping for the complete execution lifetime, must not permit writes
to immutable weight ranges, and must bind tensor offsets within the supplied
buffer length. The current llama model-loading path copies into model-owned CPU
storage, but the region audit does not identify that copy as a ggml mathematical
requirement. A region runtime can instead own the wrapper and the vBuf view
together.

## CUDA Materialization Compatibility

CUDA is intentionally a separate path:

```text
HOST_PERSISTENT_LAYOUT
    -> upload, with optional device-local transform
DEVICE_EXECUTION_LAYOUT
    -> CUDA kernel
```

The pinned CUDA backend does not offer the CPU host-pointer capability, so a
device-owned allocation and upload/materialization are `BACKEND_INTRINSIC`.
Whether copy-only upload works, or whether a CUDA kernel needs a device-local
repack, must be qualified per representation and operation. A CUDA-preferred
tile or packed layout is not a reason to persist that layout in vBuf. Upload
and repack may be fused, but the source payload remains the canonical host
representation.

## Representation Compatibility Matrix

The payload geometry is the canonical per-tensor ggml encoding identified by the
pinned source. `rows = product(shape[1..])`; `blocks_per_row = shape[0] / block
elements` for packed types.

| Representation | GGML type | Elements/block | Bytes/block | Payload rule | Relationship | Persisted requirement | Required runtime strides | CPU/RVV status | CUDA status |
|---|---:|---:|---:|---|---|---|---|---|---|
| F32 | 0 | 1 | 4 | `elements * 4` little-endian bytes | `DIRECT` | primitive descriptor and exact bytes | contiguous row-major `nb[]`, derived | direct CPU; RVV ordinary float path | device-owned tensor required for execution |
| IQ1_S | 19 | 256 | 50 | `rows * blocks_per_row * 50` | `DIRECT` | opaque canonical bytes; row width divisible by 256 | type-derived row/block strides | direct ordinary RVV path qualified in DeepSeek corpus | upload/materialize device storage; direct CUDA coverage needs qualification |
| IQ4_NL | 20 | 32 | 18 | `rows * blocks_per_row * 18` | `DIRECT` | opaque canonical bytes; row width divisible by 32 | type-derived row/block strides | direct ordinary RVV path qualified in DeepSeek corpus | upload/materialize device storage; direct CUDA coverage needs qualification |
| IQ2_XXS | 16 | 256 | 66 | `rows * blocks_per_row * 66` | `DIRECT` | opaque canonical bytes; row width divisible by 256 | type-derived row/block strides | native CPU kernel evidence; RVV direct qualification remains a backend gate | upload/materialize device storage; direct CUDA coverage needs qualification |
| Q2_K | 10 | 256 | 84 | `rows * blocks_per_row * 84` | `DIRECT` | opaque canonical bytes; row width divisible by 256 | type-derived row/block strides | direct ordinary RVV path qualified; 84-byte geometry mandatory | upload/materialize device storage; direct CUDA coverage needs qualification |
| Q5_K | 13 | 256 | 176 | `rows * blocks_per_row * 176` | `DIRECT` | opaque canonical bytes; row width divisible by 256 | type-derived row/block strides | native CPU kernel evidence; RVV direct qualification remains a backend gate | upload/materialize device storage; direct CUDA coverage needs qualification |

The matrix describes storage compatibility, not a promise that a backend can
execute every operation for every type. If a backend lacks a native kernel, the
adapter may reject the execution, use a supported conversion path, or perform a
backend-local materialization. Such a choice must not alter canonical vBuf
bytes.

### Common representation rules

- Quantized rows must be divisible by the representation block width.
- The payload is an opaque byte array at the vBuf boundary; opening a model does
  not decode, requantize, or reorder blocks.
- Required payload alignment is 1 at the representation-contract level. The
  existing canonical vBuf `BaseStep` and range alignment remain authoritative.
- F32 and BF16 scalar fields use explicit little-endian representation. Packed
  quantized fields retain the byte layout defined by the canonical ggml type.
- The Q2_K block is 84 bytes. The previously investigated 82-byte interpretation
  is invalid and must not be reintroduced as an alternate format.

## Requirement Provenance

| Requirement or fact | Classification | Evidence | Consequence |
|---|---|---|---|
| Profile-local representation ID | `FORMAT_INTRINSIC` | `rust/vbuf-ml/src/representations.rs`; `docs/vbuf-ml/representations.md` | Persist an ID that is stable within the vBuf-ML profile; map to ggml at the adapter boundary |
| Logical shape and row width | `FORMAT_INTRINSIC` | `expected_payload_bytes()` and representation validation | Persist shape; reject non-divisible quantized rows |
| Exact block geometry | `FORMAT_INTRINSIC` | pinned ggml type traits; representation constants | Persist bytes exactly as the selected canonical representation |
| Exact payload length | `FORMAT_INTRINSIC` | `validate_tensor_representation()` | Validate without scanning or decoding payload contents |
| Canonical opaque storage for packed types | `FORMAT_INTRINSIC` | vBuf-ML representation contracts | No format-level dequantization or repacking |
| Little-endian scalar encoding | `FORMAT_INTRINSIC` | Step 17 qualification; pinned BF16 and quantized layouts | Do not serialize native Rust/C ABI structs |
| Tensor range and file alignment | `FORMAT_INTRINSIC` | vBuf core format and current consumer | Preserve existing range validation; no new SIMD alignment needed |
| ggml type enum value | `BACKEND_INTRINSIC` | pinned `enum ggml_type` | Adapter mapping only; do not make external enum numbers wire requirements |
| `ggml_tensor.ne[]` | `EXECUTION_SEMANTIC` | ggml tensor construction and shape semantics | Derive from persisted shape and representation |
| `ggml_tensor.nb[]` | `EXECUTION_SEMANTIC` | ggml contiguous tensor/type sizing | Derive from type block geometry and shape; not persisted |
| ggml tensor descriptor/object | `BACKEND_INTRINSIC` | ggml tensor API | Construct at import/execution time |
| Backend buffer association | `BACKEND_INTRINSIC` | `ggml_backend_tensor_alloc()` and backend buffer APIs | Bind borrowed CPU or owned device storage at runtime |
| CPU borrowed storage | `BACKEND_INTRINSIC` | CPU `buffer_from_host_ptr` capability and memory audit | CPU may wrap vBuf-backed storage without copying when lifetime is held |
| CUDA device ownership | `BACKEND_INTRINSIC` | CUDA backend placement/upload model | Materialize or upload into device-owned storage; host mmap is not a CUDA tensor buffer |
| CPU/RVV kernel availability | `BACKEND_INTRINSIC` | native kernel and RVV qualification evidence | Per-operation/per-type backend gate, not a file-format field |
| CUDA kernel availability | `BACKEND_INTRINSIC` | pinned CUDA implementation and future qualification | Per-type/per-operation execution gate |
| CPU_REPACK / `*_16x1` layout | `GGML_IMPLEMENTATION_OPINION` | CPU repack A/B and ggml repack paths | Optional backend cache; correctness must not depend on it |
| Graph allocator arena and reset | `EXECUTION_SEMANTIC` | ggml allocator APIs | Runtime lifetime/reuse policy; never persist as model data |
| Scheduler splits, events, and graph state | `EXECUTION_SEMANTIC` | ggml backend scheduler APIs | Reconstructed per execution context/region |
| Model-wide tensor inventory | `GGML_IMPLEMENTATION_OPINION` | llama.cpp model loader architecture audit | Not required by vBuf format; region runtime may select ranges lazily |
| Architecture graph recipe | `EXECUTION_SEMANTIC` | DeepSeek/Qwen graph builders and region audit | Consumer/runtime must provide semantics; not inferred from payload bytes |
| Repacking for performance | `OPTIONAL_OPTIMIZATION` | CPU_REPACK qualification | Must preserve canonical bytes and numerical behavior |

## ggml Opinions Below The vBuf Boundary

The following are useful implementation choices but must remain below the
vBuf boundary:

- Numeric ggml enum values as persisted identifiers.
- A particular `ggml_tensor` allocation or context lifetime.
- Contiguous `nb[]` construction details, provided the adapter presents the
  canonical logical shape and byte layout.
- CPU borrowed-buffer wrappers and their ownership callbacks.
- Device buffers, CUDA uploads, staging buffers, and synchronization events.
- CPU_REPACK, `*_16x1`, architecture-specific tile layouts, and cached packed
  weights.
- Graph allocator arenas, scheduler partitions, backend selection, and graph
  reset/reuse policy.
- llama.cpp's full-model loader, model object, and model-wide weight map.

Persisting any of these would couple vBuf artifacts to one ggml revision or
backend and would make a backend-neutral consumer harder, not more compatible.

## Adapter Responsibilities

At import/execution time, the adapter must:

1. map the profile-local representation to the consumer type;
2. construct ggml dimensions from the vBuf logical shape;
3. derive valid strides and row size from the canonical type geometry;
4. validate that the payload range is in bounds and has the exact expected size;
5. bind CPU borrowed storage only while the model view remains alive;
6. allocate/upload device-owned storage for CUDA or another device backend;
7. choose direct or repacked execution only when the backend supports it;
8. construct graph-local tensors, allocator state, and scheduler state;
9. preserve architecture and tensor-role semantics independently of storage.

The adapter must not silently truncate, reorder, reinterpret, or rewrite a
canonical payload. A backend conversion is an execution-local representation,
not a replacement for the vBuf source representation.

## Direction Of Adaptation

The current integration is closer to the default-risk direction: the llama.cpp
loader enumerates the full model and places tensors into model-scoped backend
storage. The source-neutral seam is useful, but it remains model-wide and has
no region lease or residency operation.

The preferred direction is:

```text
vBuf canonical representation
    -> narrow adapter
    -> derived ggml descriptors and host/device bindings
    -> backend-local materialization where necessary
    -> ggml execution
```

The smallest architectural move is therefore a vBuf-owned region/runtime layer
on top of ggml, not changes to the persisted representation. It can create
descriptors, aligned span views, CPU wrappers, device copies, and repack caches
as runtime objects.

## Necessary vBuf Changes

No independent format change is required by the execution boundary identified
here. The current profile already has the necessary representation identifier,
shape, opaque payload, and checked size contract for the six matrix entries.

The one proven correction remains the existing Q2_K geometry correction from 82
to 84 bytes. It is a correction to the canonical representation contract, not a
new backend-specific field or repack format.

Potential future metadata additions, such as explicit KV head counts or tied
weight alias relations, are consumer/model-semantics questions and are not
required to make the canonical packed payload executable.

| Proposed change | Classification | Decision |
|---|---|---|
| Correct Q2_K from 82 to 84 bytes | `REQUIRED` | Generic payload parity requires the complete canonical block; this is already corrected |
| Persist representation ID, shape, and exact range/length | `REQUIRED` | Generic runtime correctness requires identifying and bounding the bytes |
| Persist ggml enum IDs | `GGML_SPECIFIC_DO_NOT_ADD` | Use profile-local IDs and adapter mapping |
| Persist `nb[]` strides | `GGML_SPECIFIC_DO_NOT_ADD` | Derive for canonical contiguous tensors |
| Persist backend buffer/device/ownership fields | `GGML_SPECIFIC_DO_NOT_ADD` | Runtime handles and addresses are not persistent facts |
| Persist CPU_REPACK or `*_16x1` payloads | `GGML_SPECIFIC_DO_NOT_ADD` | Direct canonical paths exist; repack is backend-local |
| Add stronger global SIMD alignment | `NOT_JUSTIFIED` | No independent representation requirement was found |
| Add optional backend hints | `BENEFICIAL_BUT_OPTIONAL` | Only if versioned and ignored for correctness |

## Qualification Tests

Each representation/backend pair should be qualified independently:

- descriptor test: type mapping, shape, row divisibility, block count, and
  exact payload length;
- byte test: adapter-visible source bytes equal the vBuf payload bytes;
- CPU borrowed-storage test: no payload copy is required and lifetime remains
  valid through execution;
- CPU/RVV numerical test: direct canonical execution matches a trusted
  dequantized/reference path;
- CUDA placement test: device tensor owns valid storage and produces matching
  output;
- fallback test: unavailable native kernels fail explicitly or use a tested
  backend-local conversion;
- lifecycle test: graph allocation, reset, scheduler state, and region release
  do not mutate persisted payload bytes;
- Q2_K regression: reject 82-byte blocks and accept the qualified 84-byte
  layout with full payload parity.

The qualification relationship labels have a precise meaning here:

- `DIRECT`: canonical bytes and a derived descriptor can feed the selected
  direct kernel;
- `VIEW_ADAPTER`: bytes remain unchanged but descriptor/view translation is
  required;
- `COPY_ONLY`: a backend-owned copy is needed without a byte transform;
- `REPACK_REQUIRED`: only a particular execution path requires a transform;
- `UNSUPPORTED`: no tested implementation can consume the representation.

The six matrix entries are `DIRECT` at the canonical CPU representation seam.
CUDA execution is at least `COPY_ONLY` because device ownership is required;
the final CUDA classification must be measured per type and operation rather
than inferred from the host format.

## Final Classification

The vBuf-ML boundary is backend-neutral when it persists canonical logical
shape, profile representation, exact payload bytes, and existing vBuf range
metadata. ggml type construction, strides, buffers, device placement, kernels,
repacking, graph allocation, and scheduling remain adapter/runtime concerns.

The recommended architecture therefore remains:

```text
VBUF_RUNTIME_ON_GGML
```

The runtime may lower validated vBuf tensors into ggml descriptors and graphs at
import or execution time without making ggml's internal allocation or backend
state part of the artifact format.

## Final Decision

```text
VBUF_FORMAT_COMPATIBLE_WITH_GGML:
YES_WITH_OPTIONAL_BACKEND_TRANSFORMS

VBUF_FORMAT_CHANGES_REQUIRED:
NONE
```

`MINOR_GENERIC` refers only to the already-proven Q2_K geometry correction and
the existing generic representation metadata contract. No new ggml-specific
format field is justified. The Q2_K correction is already part of the current
baseline, so the remaining execution audit requires no format changes.
