# Step 31B Persistent Source Representation Decision

Status: **DECIDED**

Step 31B is a design gate only. It adds no persistent source implementation,
coverage writes, hybrid source, or runtime behavior change.

## Decision

Select **A: a sparse canonical same-offset mirror**, with:

```text
existing semantic bootstrap
        +
immutable external artifact identity
        +
same-size sparse payload mirror
        +
4 KiB authoritative chunk bitmap
        -> existing RangeSource/materialization/residency path
```

The mirror is payload-only relative to the semantic bootstrap, but its logical
offset space is the complete external vBuf payload artifact. It is not a second
semantic model representation and it does not contain tensor names, shapes,
types, graph data, or residency state.

## Evidence Classification

### CODE-AUDITED

- `SourceDescriptor` requires a declared size when a source profile is encoded.
- The qualified semantic-bootstrap generator assigns the external source
  `SourceId(1)`, records the original payload length, and stores a full-source
  SHA-256 `SourceHash` with algorithm `1`.
- `SourceId` deliberately excludes the locator and is not a global content
  identity by itself.
- `TensorRef` range validation checks `offset + length` against the declared
  source size.
- Rust `MmapSource` and `PositionedFileSource` already provide checked local
  range reads.
- C++ `LocalVbufRangeSource` provides checked mapped-artifact reads, while
  `HttpRangeSource` validates HTTP 206 framing, content length, and
  `Content-Range`.
- `LocalVbufRangeMaterializer` consumes a `RangeSource`; it does not own source
  selection or persistence policy.
- `TensorResidencyStore` is a separate bounded RAM tensor store keyed by
  runtime tensor references and leases.
- v0.6 does not define a cache record, index, coverage, or external-artifact
  identity format.

### MEASURED

The Step 31A Pixel run measured 4,287 ranges over the 5,639,819,878-byte
external payload, with 1,377,067,264 bytes in the bytewise union and
3,368,434,656 repeated/reloaded bytes. The semantic bootstrap remained a
separate local artifact.

### DERIVED

Using 4 KiB coverage chunks for the captured ranges would produce:

```text
logical payload size:       5,639,819,878 bytes
chunk size:                         4,096 bytes
chunk count:                    1,376,910
one-bit bitmap size:              172,114 bytes
rounded acquisition bytes:  4,763,426,816 bytes
rounded overfetch:              17,924,896 bytes
```

The rounded-overfetch figure is an offline calculation from the 31A trace, not
a physical 31C result. It is approximately 0.38% of measured source traffic
and is much smaller than the measured repeated traffic.

### DECIDED

- Same-offset sparse payload mirror.
- Exact logical size equal to the external source's declared size.
- Fixed 4 KiB source-coverage chunks for the first implementation.
- One authoritative source-identity/coverage sidecar containing the expected
  artifact identity, logical size, chunk geometry, and bitmap.
- A bit means the complete corresponding chunk, including the final short
  chunk, is locally valid under the Step 31C source-trust contract.
- Coverage is availability of canonical source bytes, never tensor identity.
- A rebuildable in-memory interval/chunk lookup index may accelerate reads, but
  the bitmap is authoritative.
- A missing or invalid bit causes a remote acquisition of all touched chunks;
  the first implementation does not split local and remote subranges.

### DEFERRED

- Full-artifact hash verification as a completion/finalization operation.
- Per-range cryptographic digests or a signed source manifest.
- Power-loss durability.
- Retention, eviction, offline completion, and cache policy.
- Coalescing, prefetch, and partial-range optimization.
- Multi-source mirror orchestration.

## Artifact Identity

The persistent payload identity is exactly:

```text
ArtifactIdentity {
    declared_size: 5,639,819,878,
    hash_algorithm: SourceHash algorithm 1 (SHA-256),
    full_source_hash: the 32-byte SourceHash value
}
```

`SourceId(1)` and the source-profile binding are required context for selecting
the descriptor, but are not sufficient identity values. The locator, endpoint,
file path, cache filename, model tensor names, and RAM tensor references are
not identity. A different semantic bootstrap may reuse the same mirror only
when it resolves the same external artifact identity and compatible bindings.

The existing source-profile `SourceHash` and `declared_size` contracts are
reused. No second hashing or identity subsystem is selected. A thin future
source-boundary seam is still required because the current Android C++ path
passes offsets to `RangeSource` but does not expose the source hash to a
persistent source constructor.

If the expected full SHA-256 or declared size is absent, malformed, or differs
from the sidecar, the mirror is not reusable. The locator alone must never
authorize reuse.

## Integrity and Trust Boundary

Step 31C can honestly guarantee:

1. checked source containment against the declared size;
2. exact transport framing and returned length for each remote acquisition;
3. complete 4 KiB writes before a coverage bit is published;
4. no reuse across a mismatched source identity header;
5. unchanged bytes/offsets through the existing materializer boundary.

This is **structurally validated and source-identity-bound**, not
per-range-cryptographically authenticated. The source profile supplies an
expected whole-artifact SHA-256, but the current range path cannot verify a
whole-artifact digest from one range and has no external per-range digest
manifest. The C++ materializer's FNV-1a telemetry hash is not an authority.

Therefore:

```text
WHOLE_ARTIFACT_HASH_METADATA: available for the qualified source
WHOLE_ARTIFACT_HASH_VERIFIED: deferred until complete-source verification
PER_RANGE_DIGEST:              unavailable
PUBLISHING_TRUST MODEL:        trusted immutable source for 31C qualification
```

The first implementation must not claim protection against a malicious or
mutable endpoint substituting payload B while presenting payload A's expected
metadata. A future strict-integrity mode needs a verified whole-artifact hash
before declaring complete coverage, or a separately defined per-range manifest.
This limitation applies to all three representation alternatives; changing
from bytes to records would not create missing integrity evidence.

## Coverage Authority

The authoritative state is a source-cache sidecar with:

```text
sidecar identity header:
    format/version
    declared source size
    SourceHash algorithm and full digest
    fixed chunk size
    bitmap length

bitmap:
    one bit per 4 KiB logical source chunk
```

The sidecar is not model metadata. It contains no semantic directory, tensor
name, shape, representation, graph, or runtime residency information. Sparse
file holes are never consulted as coverage authority. A zero byte in the
payload is valid data and cannot distinguish a hole from a valid range.

The bitmap is authoritative. An optional in-memory merged-range lookup is a
derived accelerator and can be deleted and rebuilt by replaying the bitmap and
sidecar header. No append-only payload record store is required.

## Publication Contract

Externally observable states are only:

```text
MISSING  (bitmap bit = 0)
VALID   (bitmap bit = 1)
```

`ACQUIRING` is an in-memory synchronization state and is never published as
coverage. The required transition is:

```text
MISSING
  -> acquire every touched 4 KiB chunk from the canonical remote source
  -> validate exact response framing and source bounds
  -> pwrite complete chunk bytes at canonical offsets
  -> confirm each full write
  -> publish the corresponding bitmap bits
  -> VALID
```

The first implementation never overwrites a `VALID` chunk. For every failure,
the bit remains clear: partial HTTP body, truncation, failed framing, invalid
source bounds, short write, disk full, cancellation, or process death before
bitmap publication. Orphan bytes in an unmarked sparse-file region are ignored.

Concurrent requests touching the same chunk serialize through the source-cache
publication lock. A duplicate request rechecks the bit after acquiring the
lock and then reads the now-local chunk. Overlapping requests publish whole
chunks independently; no reader can observe a chunk as valid before its full
write and publication step completes.

## Crash and Durability Contract

For the first Step 31C qualification:

- process-crash safety is required;
- power-loss durability is not required;
- `fsync` per source range is not required;
- ordinary process restart must accept only a complete, parseable sidecar header
  and published bits;
- a process death during a chunk write or before bit publication leaves that
  chunk missing on restart;
- a complete bit published after a successful write remains valid under the
  process-crash contract.

The sidecar update must use a publication ordering that cannot expose a bit
before the corresponding complete write. Sudden power loss may leave the
mirror and sidecar at different durability points without this contract; the
future durability qualification must add and measure the required flush,
journal, or replacement protocol before claiming persistence across power
loss or device restart.

## Local File and Existing Sources

```text
logical file size:              5,639,819,878 bytes
offset space:                  [0, 5,639,819,878)
prepopulated bytes:            none for the cold 31C run
semantic bootstrap included:   no
payload-only mirror:           yes, relative to the bootstrap
payload header handling:       preserve external bytes at canonical offsets;
                               do not reconstruct or merge the bootstrap
full coverage equivalence:     yes, if every logical chunk is valid and the
                               complete source bytes pass the expected hash
```

“Payload-only” means the mirror is separate from semantic metadata. The mirror
still spans the complete external vBuf artifact address space, including its
own v0.6 header, alignment, control, and non-tensor bytes when those bytes are
acquired. Observed tensor coverage is not full artifact coverage.

`PositionedFileSource` or `MmapSource` can serve as the Rust local leaf source;
the C++ `LocalVbufRangeSource` can serve a mapped local artifact. A thin future
progressive source composition is required to check the bitmap and choose the
local leaf or existing HTTP source. The materializer remains below neither
policy nor persistence and does not change.

## Alternative Matrix

| Criterion | A. Sparse canonical mirror | B. Canonical block persistence | C. Runtime-local envelope/records |
|---|---|---|---|
| Canonical offsets | Direct, proven by 31A | Requires block/context reconstruction | Requires a translation layer |
| New payload format | No | Yes, block records/context | Yes, envelope/record format |
| Model-semantics duplication | None | Risks stream/block semantics | High; likely tensor/object metadata |
| Identity complexity | Existing full hash + size | Identity plus block provenance | Identity plus record provenance |
| Coverage complexity | Small fixed bitmap | Block map plus context rules | Record index and completeness rules |
| Crash publication | Full chunk write then one bit | Record and block publication | Record/index transaction protocol |
| Existing RangeSource reuse | Direct leaf/composition reuse | Low | Low |
| 100% local equivalence | Byte-for-byte same artifact | Must reconstruct/validate source | Not naturally canonical |
| Offline completion | Enumerate all source chunks | Must enumerate all canonical blocks | Must define record completeness |
| Qualified one-source suitability | **High** | Conditional | Low |
| Multi-source extension | One mirror/identity per source | Possible but source-heavy | Flexible but complex |
| Implementation complexity | **Lowest** | Medium/high | Highest |

Option B is rejected for this gate because the target persistence object is the
external byte artifact, not an isolated v0.6 block. v0.6 block validation needs
stream context, and block persistence would no longer be consumed through the
same canonical offset source boundary. Option C is rejected because it creates
a second model/payload representation and loses direct byte-for-byte source
equivalence without solving identity or integrity limitations.

## Step 31C Acceptance Checklist

Step 31C may implement only this bounded slice:

1. Require one qualified external `SourceDescriptor` with declared size and
   algorithm-1 full SHA-256 identity.
2. Create/open a same-size sparse payload mirror and identity/bitmap sidecar.
3. Reject an existing mirror when identity, logical size, or chunk geometry
   differs.
4. On a local request, require every touched 4 KiB bit to be `VALID` before
   reading from the local leaf source.
5. On a miss, acquire complete touched chunks from the canonical remote source,
   validate transport and bounds, write canonical offsets, then publish bits.
6. Reuse the existing materializer, leases, tensor references, 256 MiB
   residency, backend, and inference semantics unchanged.
7. Prove cold remote acquisition followed by warm local reuse for the same
   qualified source range sequence.
8. Test process death at write/publication boundaries and verify unmarked bytes
   never become local hits.
9. Measure chunk-rounding overfetch and local/remote bytes separately.
10. Do not add eviction, offline completion, prefetch, or multi-source policy.

## Deferred Product Work

Retention/eviction, storage budgets, offline completion, mobile-data UX, idle
warmup, prefetch, compute/network overlap, residency tuning, backend tuning,
and broad multi-source qualification remain outside Step 31C.
