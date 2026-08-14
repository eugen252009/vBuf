# Step 24 — borrowed runtime views

Status: **implemented and qualified**.

Step 24 replaces the Rust `ConsumerModel` snapshot on the direct vBuf source
path with a validated mmap-backed `BorrowedModel`. The v0.6 wire format,
vBuf-ML semantic representation, llama source seam, GGML runtime, tensor
layout, and kernels are unchanged.

## Architecture

Old direct path:

```text
mmap
→ ConsumerModel::open
→ owned Snapshot
→ owned token strings / merge pairs
→ bulk FFI tables
→ llama_model_source
```

New direct path:

```text
mmap
→ BorrowedModelView::parse
→ checked canonical/semantic views
→ direct token and merge array ABI
→ llama_model_source
```

`ConsumerModel` remains available as an owned convenience/compatibility API.
The compatibility adapter remains available and correctness-qualified.

## Borrowed model layer

Added in `rust/vbuf-ml/src/consumer.rs`:

```text
BorrowedModelView<'a>
├── ValidatedV06<'a>
├── Bootstrap<'a>
├── ModelMetadata<'a>
├── TensorDirectory<'a>
└── TokenizerMetadata<'a>

BorrowedModel
├── mmap owner
└── validated BorrowedModelView
```

`BorrowedModelView::parse` performs canonical validation, semantic role
resolution, metadata validation, tensor-directory validation, tokenizer
array validation, offset/UTF-8 validation, and merge-ID validation once.

`BorrowedModel` retains the mmap for the complete lifetime of the view. The
single lifetime extension is confined to the owner constructor and is safe
because the mapping is immutable, private, and dropped after the view.

## ModelMetadataView

Model metadata remains a small scalar boundary.

The runtime-facing view reads the validated metadata fields and copies only:

```text
architecture string
10–11 scalar numeric values
```

This is intentionally not optimized further. It is negligible compared with
vocabulary and merge state.

Classification:

```text
SMALL_SCALAR_COPY_ACCEPTABLE
```

## TensorDirectoryView

The existing validated `TensorDirectory` remains the semantic tensor view.
Its descriptors already retain checked payload ranges and direct mmap payload
identity. The current parser still owns tensor names and dimension vectors;
this is bounded to approximately 310/311 tensors and names/dimensions must
ultimately cross into llama-owned values.

The direct source still uses a small final tensor descriptor table. It does
not copy tensor payloads, repack payloads, or reorder tensor bytes.

Classification:

```text
payloads: BORROWABLE_DIRECTLY
ranges: BORROWABLE_DIRECTLY
names/dimensions: small final-consumer materialization
```

A fully raw borrowed directory-record wrapper is a possible follow-up, but is
not materialized into the hot tokenizer path in Step 24.

## TokenizerView

`TokenizerMetadata` now exposes validated canonical spans directly:

```text
text_bytes()
offset_bytes()
type_bytes()
score_bytes()
merge_left_bytes()
merge_right_bytes()
```

The direct ABI now exposes:

```text
text pointer + length
offset pointer + count
type pointer + byte length
optional score pointer + byte length
token count

merge-left pointer
merge-right pointer
merge count
```

No per-token Rust record table is constructed by the direct path.
No per-merge Rust record table is constructed by the direct path.
No merge strings or GGUF tokenizer representation are created.

## Token representation

Token IDs remain ordinals:

```text
start = offsets[id]
end   = offsets[id + 1]
text  = text_bytes[start..end]
```

Rust validates the complete offset table and UTF-8 semantics once during view
establishment. C++ performs only bounded little-endian decoding and final
`std::string` construction at the llama consumer boundary.

## Token types and scores

Types are borrowed with their canonical width.
Scores are optional and remain absent for the qualified Qwen3 vBuf artifacts.
No synthesized full score array is created.

The direct source supplies llama's default score only at final token access
when the canonical score span is absent.

## MergeView

Merge state remains canonical numeric SoA:

```text
MergeLeftIds[]
MergeRightIds[]
```

The direct ABI returns both spans and the ordinal count. The C++ source decodes
little-endian u32 values directly.

No Rust merge pair vector and no merge strings are created.

## Validation provenance

The provenance chain is:

```text
mmap
→ parse_v06 / ValidatedV06
→ Bootstrap role/reference validation
→ CheckedRange-backed ModelMetadata/TensorDirectory/TokenizerMetadata
→ typed borrowed spans
```

Raw mmap bytes are never exposed as typed arrays without prior semantic and
geometry checks.

## Validation passes

Old `ConsumerModel::open` path:

```text
canonical parse
semantic parse
snapshot tensor materialization
151,936 token String clones
151,936 type/score projections
151,387 merge pair materializations
special/control copies
```

New direct path:

```text
canonical parse once
Bootstrap once
metadata/directory/tokenizer qualification once
full offset and UTF-8 validation once
full token-type validation once
full merge-ID bounds validation once
no token/merge snapshot pass
```

The direct view benchmark separately reports canonical validation and borrowed
view establishment. The latter still includes required semantic validation;
view establishment is not incorrectly reported as O(1) for the whole model.

## Runtime indexes

The view layer does not persist or canonicalize runtime indexes.

The final llama tokenizer builder still constructs its required runtime BPE
state. A Rust measurement of equivalent eager indexes gives:

```text
Token byte-key index: approximately 13.5 ms
Merge pair-key index: approximately 7.4 ms
```

These are runtime-derived costs, not vBuf serialization overhead.

The deferred/lazy alternative is represented by `BorrowedModelView` readiness:
views can be established without either index. The current llama integration
uses eager construction because llama tokenizer construction requires the
runtime vocabulary state before `MODEL_READY`.

## Physical index comparison

An additional Step-24 experiment compares a derived physical index built from:

| Source | BF16 median | Q8_0 median | Classification |
|---|---:|---:|---|
| canonical block descriptors | ~1.0 µs | ~1.0 µs | direct physical descriptors |
| Nano + minimal headers | ~7.1 ms | ~3.0 ms | optional physical bootstrap |
| semantic TensorDirectory | ~16.6 µs | ~15.7 µs | semantic runtime index |

Nano remains an optional physical bootstrap input only. It is not used during
normal Step-24 dense tokenizer/model loading and is not itself treated as a
semantic index.

## Materialization and memory

Old semantic state included owned token strings, optional arrays, merge pairs,
tensor names, and dimensions.

New direct tokenizer canonical copied bytes:

```text
0
```

Tensor payload copying remains:

```text
0
```

Runtime-derived key storage remains intentionally owned:

```text
token key bytes: approximately 1.37 MB minimum
merge index estimate: approximately 3.63 MB key/value storage
```

The direct borrowed benchmark establishes views without allocating token or
merge object tables. The bounded tensor descriptor table remains a final
consumer boundary.

## Phase measurements

Release-mode, ten samples per artifact. Warm model-ready samples were run with
CPU-only llama.cpp and the pinned Step-21 llama build.

### Rust-side phases

| Phase | BF16 | Q8_0 |
|---|---:|---:|
| canonical validation | ~39.9 µs | ~39.3 µs |
| borrowed view establishment | ~4.53 ms | ~4.68 ms |
| old owned `ConsumerModel::open` | ~13.8 ms | ~13.6 ms |
| eager token index | ~13.5 ms | ~13.3 ms |
| eager merge index | ~7.4 ms | ~7.3 ms |

The first sample includes page/cache effects; reported phase values are medians.

### llama model-ready

| Artifact | GGUF | old compatibility | Step-24 borrowed direct |
|---|---:|---:|---:|
| BF16 | 249.9 ms | 430.5 ms | 152.0 ms |
| Q8_0 | 196.0 ms | 430.7 ms | 149.7 ms |

Historical Step-23 values remain unchanged in
`benchmark-results/vbuf-ml-step23/`:

```text
BF16 direct: 247.1 ms
Q8_0 direct: 247.0 ms
```

The Step-24 direct path therefore removes approximately:

```text
BF16: 95 ms versus Step-23 direct
Q8_0: 97 ms versus Step-23 direct
```

under this warm CPU qualification environment.

The old compatibility path remains intentionally GGUF-shaped and is not the
Step-24 optimization target.

## Readiness milestones

Step-24 distinguishes:

```text
SOURCE_VIEW_READY:
  BorrowedModelView established after canonical and semantic validation.

TOKENIZER_READY:
  final llama tokenizer runtime indexes available.

MODEL_READY:
  common llama model construction complete.

INFERENCE_READY:
  unchanged llama/GGML context construction and decode path.
```

The current llama source uses eager tokenizer construction, so
`TOKENIZER_READY` is inside model construction rather than exposed as a
separate public API milestone.

## Correctness

Both Qwen3 artifacts passed:

```text
structural parity: PASS
tokenizer corpus parity: PASS
logit max_abs_diff versus GGUF: 0
deterministic generation parity: PASS
```

The tested tokenizer corpus included:

```text
empty input
ASCII text
whitespace/newline text
CJK text
Qwen control-token text
```

Tensor payload behavior:

```text
copy: 0
repack: 0
reorder: 0
```

The source-neutral `llama_model_source` interface is unchanged.

## ABI

The coarse direct path remains coarse:

```text
no per-token FFI calls
no per-merge FFI calls
3 bulk semantic calls for token arrays, merge arrays, and tensors
```

The old bulk table APIs remain available for compatibility tests, but the new
direct source uses the pointer/count array APIs.

## Architecture guards

`verify_architecture.py` now rejects direct-source regressions that:

```text
construct GGUF metadata/context
use owned token-view tables
use owned merge-view tables
introduce reverse generic-vBuf dependencies
```

No Nano construction is present in the normal direct source.

## Format and ownership invariants

```text
v0.6 wire changes: 0
vBuf-ML wire changes: 0
BaseStep changes: 0
Nano wire changes: 0
payload copies: 0
payload repacks: 0
payload reorders: 0
```

The mmap owner remains alive through source/model destruction.

## Limitations

- Tensor directory names/dimensions still use the existing bounded Rust
  descriptor representation before final llama conversion.
- Runtime RSS/allocator counters were not available in this environment.
- UBSan passed for the instrumented direct Q8_0 integration probe; full ASan
  qualification against instrumented llama/Rust dependencies remains deferred.
- GPU, cold-cache, first-touch, and prefetch behavior remain out of scope.
- The eager/lazy comparison measures view readiness versus explicit Rust index
  construction; llama's current tokenizer contract remains eager.

## Decision gate

After removing the large semantic snapshot/materialization step, the remaining
startup costs are primarily:

```text
required vBuf validation: small
borrowed view establishment: approximately 4.5–4.7 ms
runtime token index: approximately 13 ms
runtime merge index: approximately 7 ms
common llama model construction: dominant remainder
```

The remaining costs do not require physical-layout awareness. Nano therefore
remains deferred for layer readiness, residency, streaming, and prefetch work.

## Recommended next step

Keep the borrowed direct semantic source as the vBuf-ML baseline. A future
optimization should separately address runtime tokenizer index construction or
llama common model construction. Do not begin Nano or physical-runtime work
until first-touch, residency, or layer-streaming evidence justifies it.
