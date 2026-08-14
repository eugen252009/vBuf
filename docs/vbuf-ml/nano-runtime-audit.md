# Nano-backed vBuf-ML runtime audit

Status: **audit only; no native runtime implemented and no wire changes made**.

Evidence is under `benchmark-results/vbuf-ml-nano-runtime-audit/`.
The audit uses the existing Qwen3 artifacts and a release-mode read-only
prototype. It does not add Nano to v0.6 or persist a new artifact.

## Current Nano implementation

The current v0.6 implementation does **not** contain a Nano wire artifact.
Normative v0.6 explicitly leaves Nano unselected. The only current Nano code is
the benchmark-only reconstruction in
`rust/src/bin/v06_navigation_bench.rs`:

```text
validated canonical blocks
→ one bit per BaseStep slot
→ set-bit enumeration/checkpoint experiments
```

That benchmark implementation:

- derives Nano from already validated canonical blocks;
- uses `BaseStep` as the slot size;
- marks every canonical physical block start, including continuation members;
- does not encode Key-ID, semantic role, tensor ID, token ID, or length;
- has no production persisted representation or bootstrap API.

## Nano semantics and the intended property

The precise property is narrower than “jump directly to objects.” Given a
known physical region and geometry, Nano identifies **candidate canonical block
starts**. To recover a block extent, a consumer still needs either:

```text
next set bit / region end
→ canonical header parsing
→ checked payload/range validation
```

Nano does not provide semantic identity, payload length, continuation validity,
or typed record boundaries. It is physical discovery/navigation only.

Therefore the intended statement is **conditionally true**:

> Nano can avoid sequential discovery of candidate physical starts when the
> consumer does not otherwise know the physical topology, but it does not
> replace canonical validation or semantic-directory lookup.

For known dense tokenizer arrays and the existing TensorDirectory, direct
addressing is stronger and Nano is redundant.

## Qualified artifact geometry

Both artifacts use `BaseStep = 8`:

| Artifact | File | Canonical blocks | Set bits | Nano bytes | Bytes per set bit |
|---|---:|---:|---:|---:|---:|
| BF16 | 1,507,825,912 | 338 | 338 | 23,559,780 | 69,703 |
| Q8_0 | 637,925,504 | 337 | 337 | 9,967,586 | 29,577 |

There are no continuation blocks in either qualified artifact.

The block composition is:

```text
BF16: 311 tensor payload blocks, 11 metadata value blocks,
      1 token text block, 1 offset block, 1 token-type block,
      2 merge-ID blocks, plus small control/role blocks.
Q8_0: 310 tensor payload blocks; tokenizer/model structure is otherwise the same.
```

The enormous hypothetical Nano sizes are a direct consequence of one bit per
8-byte physical slot over a 0.6–1.5 GB data region. This is not a production
recommendation; it is evidence against using Nano as a dense-array index.

## Discovery / validation / materialization

| Structure | Discovery | Validation | Materialization |
|---|---|---|---|
| Bootstrap | scan canonical blocks for bootstrap Key-ID | bootstrap framing and role/reference checks | small role vector |
| ModelMetadata | Bootstrap role reference | 11 field/reference/type checks | small field vector; scalar values |
| TensorDirectory | Bootstrap role reference | variable-record/name/shape/reference checks | 310/311 owned names and dimension vectors today |
| TokenizerMetadata | Bootstrap role reference | role table, offsets, UTF-8, types, merge bounds | range handles and special-token vector |
| Token text/offset/type/score | tokenizer role references | dense-array bounds and element validation | direct borrowed view is possible |
| Merge IDs | tokenizer role references | two-array length and ID bounds | direct borrowed pair view is possible |
| Tensor payloads | TensorDirectory Key-ID/occurrence | checked canonical payload range | direct mmap payload pointer |

The current Rust parser already preserves checked ranges for tokenizer arrays,
but `ConsumerModel` later copies token strings, scores/types, and merge pairs
into an owned snapshot. A future native runtime can stop at validated range
views and avoid that materialization.

## Measured traversal prototype

Thirty release-mode samples per artifact were collected. These are in-memory
warm-process measurements, not cold filesystem measurements.

| Operation | BF16 | Q8_0 |
|---|---:|---:|
| canonical parse | 43.2 µs | 34.0 µs |
| semantic parse/materialization path | 4.80 ms | 4.78 ms |
| sequential traversal of validated block records | 0.76 µs | 0.50 µs |
| reconstruct Nano from validated blocks | 0.86 ms | 0.25 ms |
| iterate all bytes/set bits of existing Nano | 8.54 ms | 3.62 ms |
| binary TensorDirectory lookup for all tensors | 30.0 µs | 29.4 µs |
| dense ordinal tensor view for all tensors | 0.12 µs | 0.11 µs |

The canonical parse timing touches only structural bytes and is not equivalent
to a disk read. The Nano reconstruction and scan touch approximately the full
hypothetical Nano byte count. The direct dense ordinal benchmark is intentionally
not a full parser benchmark; it demonstrates the addressing cost after a view
has been established.

The result is clear for these artifacts: Nano is much more work than iterating
the already validated block vector, and it is unnecessary for dense ordinal
arrays. A persisted Nano could avoid reconstruction, but scanning it still
costs milliseconds and its size is tens of megabytes.

## Tokenizer audit

Qualified tokenizer data contains:

```text
151,936 tokens
151,387 merges
```

Current wire representation:

```text
TokenTextBytes   : 1,372,760 bytes
TokenOffsets     : 1,215,496 bytes (151,937 u64 offsets)
TokenScores      : absent in qualified artifacts
TokenTypes       : 607,744 bytes
MergeLeft/Right  : 1,211,096 bytes total
```

The current representation already supports direct mmap-backed views:

```text
text[i] = TextBytes[offsets[i] .. offsets[i+1]]
type[i] = typed view at i
merge[i] = (left[i], right[i])
rank[i] = i
```

No TokenRecord wire table is structurally required.

### TokenRecord AoS versus current SoA

A hypothetical record:

```text
{text_offset, text_length, score, type}
```

would duplicate information already represented by offsets and parallel arrays.
Current SoA is preferable for this profile because it provides:

- zero-copy text spans;
- direct ordinal access;
- optional scores/types without padding every record;
- field-local validation and cache access;
- explicit cross-language little-endian decoding.

An AoS table could be useful for a runtime-local wrapper if a tokenizer performs
many mixed-field sequential scans, but it is not justified as a new canonical
wire structure.

### Merge representation

The existing two u32 arrays are already direct-view ready. A `MergeRecord{u32,
u32}` would be 8 bytes per pair, equal to the packed SoA storage, but would not
remove the need for the same runtime `(left,right) → rank` index.

The current SoA representation is therefore retained for audit purposes:

```text
wire-size difference: none of consequence
validation: simple parallel-array bounds
ordinal rank: already canonical
Nano value: redundant
```

A runtime hash/map remains separate work. Direct views do not eliminate it.

## Per-element startup work

A native direct-view bootstrap could eliminate the following transitions:

```text
151,936 token projections
151,387 merge projections
per-element C ABI crossings
Rust snapshot token String creation
Rust snapshot merge pair copying
GGUF-shaped tokenizer reconstruction
```

It cannot eliminate:

```text
UTF-8/offset validation if safety is required
merge-ID bounds validation
runtime token-text → ID index construction
runtime merge-pair → rank index construction
```

The precise claim is therefore **no per-element deserialization/materialization
for establishing canonical views**, not “zero tokenizer startup work.”

## Runtime index work

Likely unavoidable runtime-local indexes:

```text
token bytes/string → token ID
(left ID, right ID) → merge rank
```

These should remain runtime-local. They may be evaluated lazily or incrementally,
but this audit did not change or select a lazy strategy.

Tradeoff:

```text
lazy index: lower open latency, higher first-use latency, more complexity
full index: deterministic startup cost, faster steady tokenization
```

Persisting hash indexes would bind the file to hash/runtime behavior and is not
recommended as canonical state.

## Nano classifications

| Use case | Classification | Reason |
|---|---|---|
| Bootstrap role discovery | NANO_REDUNDANT | Bootstrap already identifies semantic regions |
| Model metadata values | NANO_REDUNDANT | 11 small referenced values |
| TensorDirectory lookup | NANO_REDUNDANT | directory references exact blocks |
| Token text/offset/type/score | NANO_REDUNDANT | dense ordinal/offset addressing |
| Merge arrays | NANO_REDUNDANT | dense ordinal arrays |
| Tensor payload lookup | NANO_REDUNDANT | checked TensorDirectory ranges |
| Continuation-chain physical enumeration | NANO_ASSISTED | can enumerate physical members, but headers/Key-IDs remain authoritative |
| Unknown/mixed canonical physical bootstrap | NANO_ASSISTED | candidate starts without semantic assumptions |
| Layer-major physical scan | NANO_ASSISTED | future readiness/prefetch discovery |
| Partial loading | NANO_ASSISTED | possible physical candidate enumeration, not semantic selection |
| Prefetch/residency planning | NANO_ASSISTED | physical topology may be useful |
| Persisted accelerator for token lookup | NANO_HARMFUL/EXTRA | duplicates runtime-derived semantic indexes |

## Layered bootstrap architecture

The evidence supports a compositional design:

```text
mmap
↓
bounded canonical validation
↓
Bootstrap / semantic role resolution
↓
Nano only when physical topology is otherwise unknown or streamed
↓
borrowed dense/variable record views
↓
runtime-local lookup indexes
↓
execution
```

For normal qualified vBuf-ML model open, the better path is:

```text
mmap
↓
canonical validation once
↓
Bootstrap
↓
semantic region wrappers
↓
dense TokenizerView / TensorDirectoryView
↓
runtime-local indexes
```

## Continuation semantics

Nano marks every physical start independently. It does not collapse a
continuation chain into one bit or one logical object. A native wrapper must
still validate:

```text
same Key-ID
continuation sequence
next canonical member
final continuation=false
```

The qualified Qwen3 artifacts contain no continuation chains, so no performance
claim is made for this case.

## Tensor and layer use

For individual tensors, Nano provides no benefit beyond TensorDirectory:

```text
name/ordinal
→ checked payload range
→ mmap pointer
```

The current tensor path is already direct and negligible in Step 22A.

Nano may become useful later for physical layer operations:

```text
next layer-region discovery
layer-ready detection
rolling residency
prefetch planning
backend transfer planning
```

Those operations require runtime-local layer/region wrappers and are not
implemented here.

## Runtime wrapper proposal

Minimal future wrappers should contain:

```text
VbufModelSource
  mapping/lifetime handle
  validated canonical source
  BootstrapView
  ModelMetadataView
  TensorDirectoryView
  TokenizerView
  optional NanoView

TokenizerView
  text-pool span
  offset span
  score/type spans
  merge-left/right spans
  special/control views
  runtime-local indexes only when needed

TensorView
  borrowed name/dimensions metadata
  representation
  checked payload range
```

Construction should be O(roles + directory records) for variable record scans,
O(1) for dense array view establishment after validation, and O(runtime index
size) only for required lookup maps.

## Memory audit

The Rust `ConsumerModel` snapshot has a conservative lower-bound estimate of
approximately 12.3 MB for the qualified tokenizer/tensor semantic fields,
excluding allocator overhead and capacity slack. This includes:

```text
token String headers and text
Option<u64> token type/score vectors
merge pair vector
TensorSnapshot headers, names, and dimensions
```

A direct canonical view needs zero copied element bytes. It needs only small
range/wrapper state plus any runtime-local indexes. Allocator counters were not
instrumented, so this is an estimate rather than an allocator trace.

## Validation-once opportunity

Current parsing already creates checked ranges, but downstream consumers should
be able to retain those validated ranges instead of reparsing or copying:

```text
canonical validation once
→ CheckedRange provenance
→ Bootstrap region handles
→ borrowed metadata/token/tensor wrappers
```

This is the strongest native-runtime opportunity identified by the audit. It is
independent of Nano and does not require a wire change.

## Complexity accounting

Current semantic startup:

```text
O(canonical blocks)
+ O(metadata fields)
+ O(tensors plus variable directory records)
+ O(tokens for offset/text/type validation)
+ O(merges for bounds validation)
+ O(snapshot materialization)
```

Native direct-view startup:

```text
O(canonical blocks) for safety validation
+ O(region count) for wrappers
+ O(variable TensorDirectory scan) unless a runtime index is built
+ O(tokens/merges) only for required validation
+ O(runtime indexes actually required)
```

Nano does not change the safety-validation term. It can reduce physical
candidate discovery in unknown/mixed/streamed layouts, but adds a Nano scan or
rank/select structure cost.

## Conclusion

For the qualified Qwen3 vBuf-ML artifacts:

```text
Nano is not useful for dense tokenizer arrays.
Nano is not useful for individual tensor lookup.
Nano is redundant for Bootstrap, ModelMetadata, and TokenizerMetadata roles.
Nano is potentially useful for future physical layer/stream/residency discovery.
```

The stronger immediate architecture is:

```text
mmap
→ validate once
→ semantic role resolution
→ borrowed dense/variable views
→ runtime-local indexes
```

Nano should remain an optional physical bootstrap mechanism, not a semantic
index and not a mandatory feature of a native vBuf runtime.
