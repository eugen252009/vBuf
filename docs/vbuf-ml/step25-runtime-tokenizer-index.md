# Step 25 — runtime-local tokenizer indexes

Status: **implemented and qualified**.

## Frozen Step-24 baseline

The immutable annotated tag is:

```text
vbuf-ml-0.1-direct-view-baseline
```

Tag target:

```text
b569f26effaba1e358985d37e9464093fc59657c
```

Annotated tag object:

```text
38d416445bfffae68bfb01dd4edd561ed08b64ae
```

The tag points to the final Step-24 qualification/cleanup commit and was not
moved or rewritten.

The qualified artifacts remain:

```text
Qwen3-0.6B-BF16.vbuf
SHA-256 6ec3db0bb8a26914be7312cc26c3ec0fb6945202659c46b26b4506f4de75a806

Qwen3-0.6B-Q8_0.vbuf
SHA-256 2982cedd0ddc12d762d12ff3426bcc105b2cee1c675ca13bbb5ca4c3cce9a998
```

Step-24 evidence was preserved unchanged. Its historical environment record
contains the earlier qualification commit `0575599`; the immutable baseline
tag targets the final cleanup commit `b569f26`. This provenance distinction is
recorded in `benchmark-results/vbuf-ml-step25-runtime-index/step24-provenance.json`.

## Current index audit

The pinned llama implementation currently owns:

```text
std::unordered_map<std::string, llama_token>
std::unordered_map<std::pair<std::string,std::string>, int, pair_hash>
```

Both structures own string keys. No explicit reserve was observed in the pinned
implementation. Duplicate token text follows last-ordinal assignment.

The Step-24 Rust benchmark used equivalent owned-key structures as a portable
construction proxy; the actual llama maps remain llama-owned final-consumer
state.

## Reusable runtime index layer

Added `rust/vbuf-ml/src/runtime_tokenizer.rs`:

```text
TokenIndex<'a>
  HashMap<&'a [u8], u32>

MergeRankIndex
  HashMap<u64, u32>
  packed key = (left << 32) | right

RuntimeTokenizerIndexes<'a>
  TokenIndex + MergeRankIndex
```

The indexes:

```text
consume validated TokenizerMetadata
borrow canonical token bytes
copy no canonical token key bytes
remain runtime-local
are not persisted in vBuf
contain no llama.cpp types
```

The C ABI exposes lazy construction and lookup for both the current consumer
handle and future native runtimes:

```text
vbuf_ml_consumer_runtime_indexes
vbuf_ml_consumer_token_id
vbuf_ml_consumer_merge_rank
```

The normal llama source path does not eagerly build a second Rust index because
llama still requires its own final vocabulary maps. This avoids adding duplicate
startup work. The current `VbufMlAdapter` exposes the same lazy build/token
lookup/merge-rank API for integration-side consumers, while future native
runtimes can consume the Rust API directly.

## Candidate strategies

Token candidates:

```text
owned HashMap<Vec<u8>, u32>
borrowed HashMap<&[u8], u32>
sorted borrowed spans + binary search
```

Merge candidates:

```text
owned HashMap<(u64,u64), u32>
packed HashMap<u64, u32>
sorted packed pairs + binary search
```

## Selected strategies

### Token index

Selected:

```text
borrowed HashMap<&[u8], u32>
```

Reason:

```text
zero canonical key-byte copies
about 2× faster construction than the owned proxy
much faster lookup than sorted spans
portable Rust-only implementation
```

### Merge index

Selected:

```text
packed HashMap<u64, u32>
```

The qualified profile bounds token IDs to u32-compatible values. The builder
checks this bound before packing, so the representation is not silently applied
to wider future profiles.

The packed map reduces reported key/value storage from approximately 3.63 MB
to 2.42 MB for the Qwen3 merge count.

## Construction measurements

Thirty release-mode samples per artifact are in
`benchmark-results/vbuf-ml-step25-runtime-index/`.

### Token construction

| Artifact | Owned proxy | Borrowed map | Sorted spans |
|---|---:|---:|---:|
| BF16 | 14.71 ms | 7.57 ms | 15.56 ms |
| Q8_0 | 13.51 ms | 7.50 ms | 15.40 ms |

### Merge construction

| Artifact | Tuple hash map | Packed map | Sorted pairs |
|---|---:|---:|---:|
| BF16 | 7.34 ms | 4.40 ms | 3.40 ms |
| Q8_0 | 7.40 ms | 4.35 ms | 3.37 ms |

Sorted merge construction is slightly faster, but packed lookup is faster and
has simpler predictable runtime behavior, so the packed map is selected.

## Lookup measurements

The benchmark performs one lookup for every qualified token or merge.

### Token lookup

| Artifact | Borrowed map | Sorted spans |
|---|---:|---:|
| BF16 | 10.67 ms | 45.92 ms |
| Q8_0 | 10.45 ms | 45.29 ms |

### Merge lookup

| Artifact | Packed map | Sorted pairs |
|---|---:|---:|
| BF16 | 8.01 ms | 8.59 ms |
| Q8_0 | 7.67 ms | 8.54 ms |

The lookup benchmark is a full-vocabulary/full-merge sweep, not a single
lookup latency. It establishes the relative strategy tradeoff.

## Eager/lazy behavior

The reusable C ABI index path is lazy:

```text
SOURCE_VIEW_READY
→ first token_id/merge_rank request
→ build and retain RuntimeTokenizerIndexes
```

The current llama default remains eager internally because llama constructs its
final vocabulary maps during model construction. No llama tokenizer semantics
were changed.

Measured reusable view milestones:

```text
BorrowedModelView establishment:
BF16 ≈4.61 ms
Q8_0 ≈4.60 ms

BorrowedModel open total:
BF16 ≈5.49 ms
Q8_0 ≈5.38 ms
```

Split eager/lazy variants were not selected because the current source-neutral
llama seam does not expose a separate tokenizer-readiness phase.

## Parallel construction

Not selected.

The two indexes are independent, but the qualified work is only approximately
12 ms with the selected builders. Thread creation/synchronization overhead was
not justified without a dedicated runtime executor and was not introduced.

## Incremental construction

Feasible in the reusable API, but not implemented as the default.

Both builders already consume ordinal canonical arrays and can be adapted to
chunked insertion in a future progressive runtime. The current API intentionally
retains a simple deterministic complete-build operation.

## Model-ready qualification

The normal llama direct source behavior was not changed to construct duplicate
Rust indexes. New ten-sample CPU controls were recorded:

| Artifact | GGUF | Step-24 recorded direct | Step-25 direct control |
|---|---:|---:|---:|
| BF16 | 250.4 ms | 152.0 ms | 163.5 ms |
| Q8_0 | 194.7 ms | 149.7 ms | 158.0 ms |

A separate 30-sample direct control gave approximately:

```text
BF16: 163.6 ms median
Q8_0: 157.2 ms median
```

These are environment controls, not a claimed regression caused by the index
module: the default llama direct path does not invoke the new lazy index APIs.
The Step-24 values remain the immutable historical baseline. The runtime-index
microbenchmark isolates the reusable index work without conflating it with
common llama construction.

## Remaining-cost decomposition

For the selected portable index builders:

```text
BF16:
  borrowed token index ≈7.6 ms
  packed merge index   ≈4.4 ms
  total                ≈12.0 ms

Q8_0:
  borrowed token index ≈7.5 ms
  packed merge index   ≈4.4 ms
  total                ≈11.9 ms
```

The remaining model-ready time is predominantly common llama model/vocabulary
construction and tensor/backend setup. No common llama rewrite was attempted.

## Memory

Canonical copies:

```text
Token key bytes copied: 0
Merge key bytes copied: 0
```

Reported selected index storage estimates:

```text
Token key bytes retained through mmap: 1.37 MB logical span
Merge packed key/value storage:        2.42 MB
```

Hash table allocator overhead and peak RSS were not instrumented.

## Correctness

Both Qwen3 artifacts passed the existing three-way qualification after the
index layer changes:

```text
structural parity: PASS
tokenizer parity: PASS
BF16 logit max_abs_diff: 0
Q8_0 logit max_abs_diff: 0
deterministic generation: PASS
merge ordinal lookup: PASS
packed-width rejection: PASS
```

No token IDs, special-token behavior, BOS behavior, merge ranks, logits, or
generation output changed.

## Format and dependency boundaries

```text
v0.6 wire changes: 0
vBuf-ML wire changes: 0
persisted index changes: 0
Nano/LayerView/RuntimeChunk work: 0
llama types in reusable index module: 0
vbuf-core/vbuf-layout ML dependency: 0
```

## Large-model benchmark seam

The Step-25 evidence harness accepts an arbitrary vBuf path and run count; it
does not hardcode model file size or tensor count. The same schema can later
qualify a larger Qwen artifact with:

```text
artifact identity
file size
parameter/model metadata
vocabulary/merge counts
token/merge index construction
model-ready
```

Future size-normalized metrics should include:

```text
model-ready ms
model-ready ms / GB
bytes touched before ready
view/index milliseconds
common runtime milliseconds
payload I/O/page-touch milliseconds
first useful compute
```

No crossover point between structure/index work and raw weight I/O is inferred
from the current 0.6B artifacts.

## Limitations

- The selected Rust indexes are exposed for current/future runtime use but are
  not substituted for llama's internal final maps in this step.
- Actual pinned llama map allocation internals were audited from source but not
  separately timed inside `llama_vocab`.
- Full allocator/RSS and ASan measurements remain unavailable.
- Parallel and incremental construction were not implemented.
- Physical runtime work remains explicitly deferred.

## Architecture impact

The reusable architecture is now:

```text
vBuf mmap
↓
validated borrowed TokenizerView
↓
optional runtime-local indexes
├── borrowed token byte index
└── packed merge rank index
↓
consumer adapter
├── current llama integration
└── future native vBuf runtime
```

Canonical vBuf remains portable and unchanged.

## Next recommended step

Keep the selected indexes as reusable runtime-local infrastructure. If future
measurements show the pinned llama-owned maps dominate startup, consider a
separate llama-specific seam study. Do not begin Nano, LayerView,
RuntimeChunk, prefetch, residency, GPU streaming, or physical-runtime work in
that study.
