# Progressive Local Source Design Checklist

Status: **STEP 31H CACHED GEOMETRY OWNERSHIP REPAIRED; IQ2_XXS INPUTS RESTORED; PHYSICAL CONFIRMATION BLOCKED BY ADB**

Step 31B decision: **SPARSE CANONICAL MIRROR + 4 KiB AUTHORITATIVE BITMAP**
for the qualified single external payload source. Step 31C and the Step 31D
bounded acquisition follow-up are host- and Android-qualified with a completed
31D cold/warm and generation result. Step 31E measures the active-RAM curve
without changing persistent source state.

This document records the repository-grounded design and bounded Step 31C/31D
implementation for progressively persisting remote vBuf source data. It does
not change TensorRef semantics, materialization semantics, source acquisition,
or inference behavior. Retention, offline completion, and power-loss durability
remain deferred. Step 31D adds demand-driven missing-chunk coalescing with a
measured 1 MiB maximum acquisition window; 4 KiB coverage remains authoritative.
Step 31E varies only the numeric active residency budget and records aggregate
materialization/reacquisition counters. Step 31F preserves the existing
executor and runtime error detail through the Android boundary and physically
identifies the fixed 512 MiB first-decode failure as malformed tensor geometry
for `blk.1.ffn_down_shexp.weight` in `block 7` `expert_down_matmul`; it does not
fix or reinterpret that failure.

The physical application baseline is commit `80409a4 Add Android app baseline
harness` on a Pixel 7 Pro, Android 17, `arm64-v8a`, using the DeepSeek-V2-Lite
IQ2_XXS payload and a `268,435,456` byte (`256 MiB`) runtime residency cap.

## 1. Current State

### 1.1 Generic vBuf wire contract

The normative contract is `spec/spec_0.6.md`. It defines a canonical stream of
validated physical blocks. The global header supplies `BaseShift`,
`BaseStep`, data-region boundaries, and optional generic header extensions. A
block anchor supplies semantic and physical representation, continuation,
count, payload alignment, `KeyID`, and bit width.

The Rust parser in `rust/src/v06.rs` validates:

- global magic, version, flags, header size, reserved fields, and data-region size;
- checked `u64` arithmetic for block, header, payload, and alignment ranges;
- block starts, payload starts, canonical count encoding, and zero padding;
- semantic/physical/bit-width combinations and continuation chains;
- truncation, invalid tail padding, and ranges outside the data region.

`ValidatedV06::block_range()` returns the physical block plus canonical
inter-block padding. `ValidatedV06::payload_range()` returns only the payload.
These ranges are valid because the complete source stream was parsed with its
global header and stream context.

The v0.6 contract does **not** select an index, checksum table, finalization
footer, or cache record format. Canonical parsing does not require one. The
block anchor is self-describing for canonical geometry when interpreted with
the stream's `BaseStep`/`BaseShift` and position; it does not contain source
artifact identity, a global model/version identity, or an integrity digest.

Therefore the current format does not yet prove that an isolated byte slice
`[block header][payload]` can be treated as a complete authoritative cache
record. An isolated record would need stream context or a separately defined
runtime-local envelope before it could be safely scanned and validated.

### 1.2 vBuf-ML semantic source identity

`rust/vbuf-ml/src/source.rs` already provides the main source-neutral seam:

```text
SourceId
SourceDescriptor { id, declared_size, locator, hashes }
TensorRef { source_id, offset, length }
SourceRegistry
```

Confirmed behavior:

- `SourceId` is separate from `SourceLocator`; changing a file path or HTTP
  endpoint does not change the declared source ID.
- `SourceDescriptor::checked_range()` performs checked `offset + length` and
  declared-size containment validation.
- `SourceRegistry` rejects duplicate IDs, requires `SourceId::SELF`, and
  rejects malformed locators and empty hash values.
- Persistent source-profile encoding requires a declared source size and can
  carry zero or more binary hash descriptors.
- Source-profile parsing is bounded, validates reserved fields and bindings,
  performs no source I/O, and preserves old self-source artifacts when the
  optional source role is absent.
- Tensor-directory external bindings are keyed by canonical `(KeyID,
  occurrence)` and resolve to the same normalized `TensorRef` used by runtime
  range loading.

The identity is stable only within the profile's identity contract. A numeric
`SourceId` is not a content hash and is not, by itself, proof that two source
artifacts are the same model. Source hashes are optional and currently are
descriptive metadata; no source-wide verification is automatically performed.
For the qualified semantic-bootstrap generator, the external descriptor does
carry a full-source SHA-256 (`SourceHash.algorithm == 1`) and the exact source
length. Step 31B selects that existing pair as the persistent mirror identity;
it does not make `SourceId` or the locator authoritative identity.

### 1.3 Current source and range APIs

Rust `rust/vbuf-ml/src/range_loading.rs` provides:

- `RangeSource::read_exact_at(offset, destination)` and `len()`;
- `SourceResolver` and `SourceSet` mapping `SourceId` to a source;
- `MmapSource` for checked borrowed local mappings;
- `PositionedFileSource` for selected local file reads;
- `ReadPlan` with semantic targets, physical reads, relative target slices,
  semantic/physical byte accounting, and configurable coalescing;
- `execute_plan_with_sources()` which resolves source identity at read time.

`ReadPlan` never coalesces ranges from different source IDs. It is a read
executor, not a persistent store. `LoadedPlan` owns temporary byte vectors and
is released with the plan result.

The C++ GGML seam in `integrations/ggml/include/vbuf_range_source.h` provides:

- `RangeSource::read_range(offset, length, destination, result)`;
- `LocalVbufRangeSource` over an already mapped local artifact;
- `HttpRangeSource` with HTTP 206, `Content-Length`, `Content-Range`, and
  checked-range validation;
- `RangeReadResult` with offsets, returned bytes, source label, HTTP status,
  content range, endpoint information, and error text;
- `RangeSourceMetrics` with completed request count, transferred bytes,
  exact-range unique bytes, connections, and request duration summary.

The C++ source API has no persistent write operation and no standard local
partial-coverage query. `LocalVbufRangeSource` assumes a mapped artifact, not a
block store.

### 1.4 Materialization, leases, and runtime residency

The existing C++ materializer seam is in
`integrations/ggml/include/vbuf_materializer.h` and
`integrations/ggml/src/vbuf_materializer.cpp`:

```text
request -> InFlight -> source read -> payload hash/telemetry
         -> Ready -> obtain_ready_tensor -> release
         -> Failed
```

`LocalVbufRangeMaterializer` owns an aligned temporary buffer and one worker per
request. It accepts a primary source and an optional fallback source, but the
fallback is only tried after the primary full-range read fails; it is not a
local-coverage/remote-coverage composer.

`ResidentTensorMaterializer` and `TensorResidencyStore` provide the current
runtime RAM layer. The store owns retained immutable materialized tensors,
tracks resident bytes, records hit/miss/materialize/insert/lease/eviction
events, and refuses eviction while leases are active. LRU and cost-aware
replacement policies already exist. The store identity is currently a C++
`uint32_t tensor_ref`, not a persistent source-range key.

The runtime residency budget is independent of source location. The Android
baseline proves the `256 MiB` bound under actual eviction and reacquisition; it
does not establish an optimal RAM budget.

### 1.5 Android direct runtime seam

The qualified application path is in
`integrations/android-vbuf-chat/app/src/main/cpp/vbuf_android_direct.cpp` and
includes the PoC22-derived execution implementation. `DirectSession` currently:

1. opens semantic metadata through the vBuf-ML consumer;
2. creates one `HttpRangeSource` from the configured endpoint;
3. creates `LocalVbufRangeMaterializer(source)`;
4. creates `TensorResidencyStore(256 MiB, CostAware)`;
5. executes the existing batched prefill and single-position decode path.

The Android C++ `PersistentTensorRef` in
`integrations/ggml/include/vbuf_tensor_wave.h` carries tensor ID, name, view,
and `source_offset`, but not a `SourceId`. The Android PoC22 path therefore
assumes one source and routes all physical offsets through that one HTTP source.
The Rust C ABI can expose `source_id`, offset, and length through
`vbuf_ml_consumer_tensor_source()`, but the Android direct adapter does not
currently carry that identity into its C++ persistent tensor representation.

This is an important implementation seam: the canonical Rust source model is
source-aware, while the qualified Android PoC22 adapter is currently
single-source and source-ID-blind above `RangeSource`.

### 1.6 Integrity and validation

`rust/vbuf-ml/src/integrity.rs` and `docs/vbuf-ml/integrity.md` define optional
SHA-256 metadata for selected canonical payloads. Verification is explicit and
target-scoped. It does not authenticate a publisher, protect against a changed
remote locator, or verify an external source-wide artifact automatically.

Separately, the target source profile carries a full-source SHA-256 through
`SourceHash`. That value supplies an expected artifact identity, but the current
range-loading path does not verify it while serving individual external ranges.

The current C++ materializer records an FNV-1a payload hash in telemetry. That
hash is not an integrity contract and must not be used as the cache's authority.
HTTP validation proves response/range framing and length, not model identity or
payload authenticity.

### 1.7 Existing telemetry and evidence

The Android baseline records aggregate per-token source request/byte and
residency counters, plus inclusive compute intervals. The current physical
baseline is approximately:

```text
MEAN_DECODE_MS: 55069.25
MEAN_SOURCE_BYTES_PER_DECODE_TOKEN: 816272544
MEAN_SOURCE_REQUESTS_PER_DECODE_TOKEN: 697
MEAN_SOURCE_FRACTION_OF_MODEL: 0.1447338
PEAK_RESIDENT_BYTES: 267452416 <= 268435456
```

The C++ HTTP source can count exact repeated `(offset, length)` requests, but
that is not the same as a bytewise interval union. The materialization trace
contains requested offsets, returned lengths, source labels, and state events.
The residency trace contains tensor identities and source labels, but not a
complete canonical range identity.

Existing POC22 evidence records per-position source and reload totals, but not
the complete per-token interval set needed to calculate union, overlap, and
reuse distance. The Android app baseline does not retain a range trace artifact.
Thus unique-vs-repeated range telemetry is **partially available**, not
available as the requested first-class evidence.

The Step 31A diagnostic run subsequently captured a complete four-token
range ledger under the unchanged runtime semantics. It recorded phase/token
bounds and exact `(offset, length)` requests, allowing bytewise interval union
and reuse analysis without inferring ranges from residency counters. The
qualified result is recorded in
`docs/vbuf-ml/step31a-canonical-offset-source-authority-audit.md`.

The current-state summary also retains an older header baseline label
(`27c8bd9`) while the application baseline is recorded at `80409a4`; this is a
documentation bookkeeping issue, not evidence that the app baseline is absent.

### 1.8 Canonical-offset audit

The repository's semantic-bootstrap generator in
`rust/vbuf-ml/src/bin/vbuf-ml-semantic-bootstrap.rs` establishes the relevant
relationship directly:

1. It opens and validates the original full v0.6 source artifact.
2. It creates an external `SourceDescriptor` with `SourceId::new(1)`, the
   original source byte length, and the source locator/hash.
3. It binds each external tensor with `TensorRef::new()` using the original
   canonical tensor payload offset and length.
4. It writes a separate semantic bootstrap whose tensor blocks are zero-payload
   placeholders while retaining the external bindings.

The Android baseline records the resulting deployment explicitly:

```text
APP_SEMANTIC_BOOTSTRAP: local app-private semantic .vbuf
PAYLOAD_SOURCE: HTTP RangeSource serving the original DeepSeek payload .vbuf
```

The Android `DirectSession` passes each `PersistentTensorRef.source_offset`
directly to `RangeSource::read_range()`. The Rust range path likewise passes
`TensorRef.offset()` and `TensorRef.length()` to `RangeSource::read_exact_at()`.
No source-relative rebasing, tensor-object serialization, or backend-specific
offset translation occurs at this boundary.

For the current external TensorRef path, the confirmed model is therefore:

```text
semantic bootstrap artifact: local discovery/metadata artifact
external payload artifact:   one immutable random-access canonical vBuf source
TensorRef offset/length:     payload artifact offset/length
local mirror offset/length:  same payload artifact offset/length
```

This proves canonical offset preservation for the qualified single external
payload source, subject to validating that every binding used by the target
model resolves to that same immutable source identity. It does not prove the
property for arbitrary multi-source `SourceSet` deployments, where each source
has its own offset space.

The physical Step 31A run used 1,499 batched-prefill requests and 2,788 decode
requests across four generated tokens. It requested `4,745,501,920` bytes and
returned the exact requested lengths. The global bytewise union was
`1,377,067,264` bytes, with `907,623,424` bytes shared between prefill and
decode. The runtime peak remained `267,452,416` bytes under the
`268,435,456`-byte residency cap, while reload bytes reached `2,791,685,088`.
These are source/range and RAM-residency observations, not a persistent-cache
qualification.

The separate semantic bootstrap does not need to be copied into the payload
mirror. It remains the existing local semantic artifact. The payload mirror is
not itself a replacement semantic bootstrap. The original payload artifact may
contain its own v0.6 global header, block anchors, payloads, alignment padding,
and non-tensor control regions, but ordinary external TensorRef materialization
requests only the bound payload ranges.

Consequences:

- `pwrite(local_mirror, bytes, TensorRef.offset())` is a viable first-principles
  hypothesis for the current one-source payload path.
- A sparse mirror sized to the declared payload source can serve covered
  TensorRef payload ranges without reconstructing tensor metadata.
- A sparse mirror is not byte-for-byte complete merely because every observed
  TensorRef is covered. Header, padding, control-region, and never-touched
  expert bytes may remain holes.
- Full byte-for-byte equivalence to the remote payload artifact requires
  coverage of every byte, including structural and non-tensor ranges. That is a
  separate completion state from “all observed inference ranges are local.”
- The existing semantic bootstrap can remain unchanged while a local payload
  mirror is used as the external `SourceId(1)` source, provided identity,
  declared size, coverage, and read validation are enforced.

## 2. First-Principles Design

The intended architecture keeps five resources distinct:

```text
semantic TensorRef
    -> immutable source identity and physical range
    -> persistent/local source availability
    -> materialization
    -> bounded runtime residency and lease
    -> backend execution representation
```

The source layer should eventually support one semantic path with three source
states:

```text
REMOTE:
TensorRef -> remote RangeSource -> materialization -> residency -> compute

HYBRID:
TensorRef -> local persisted coverage, otherwise remote + publish ->
             materialization -> residency -> compute

LOCAL:
TensorRef -> local persisted coverage -> materialization -> residency -> compute
```

The persistent source store is not the RAM residency store. It is a separate
remote/network-to-local-flash resource. A model may have multiple gigabytes of
persisted source coverage while only `256 MiB` is resident in active runtime
memory.

The minimal future source composition should be below semantic execution and
above transport details. It may eventually resemble a hybrid range source, but
the exact type name and ownership should follow the existing Rust `SourceSet`
and C++ `RangeSource` boundaries rather than introducing an Android-specific
loader or a backend-owned cache.

Step 31B selects a sparse mirror of the same immutable canonical payload
artifact. The selected first coverage unit is a fixed 4 KiB logical source
chunk, represented by a minimal authoritative bitmap:

```text
remote payload artifact: [----------------------------------]
local sparse mirror:     [██████......████████....██........]
                          same logical offsets and declared size
```

For a missing range, the conceptual operation is:

```text
identify all touched 4 KiB chunks
    -> read complete canonical chunks remotely
    -> validate response and source identity
    -> use the requested bytes for the current materialization
    -> pwrite complete chunks at the same local offsets
    -> publish bitmap bits only after the writes succeed
```

For a covered range, the source reads the same offset and length from local
storage. The semantic runtime, materializer, leases, active residency, and
backend path remain unchanged.

This avoids assuming a new append-only cache-record format. It also avoids
turning persisted bytes into a tensor database or deserialized-object cache.
The sparse mirror is a partially materialized copy of the existing source
artifact. A separate minimal coverage state is still required because sparse
holes read as zeroes and zeroes may be valid source bytes.

The representation decision is no longer conditional for the qualified
single-source path. Compare:

1. sparse canonical mirror plus minimal coverage state;
2. canonical block persistence with enough source/stream context;
3. a runtime-local envelope/record store only if the first two cannot preserve
   correctness or practical lookup.

Step 31B selects option 1 for the qualified single external payload artifact.
It does not authorize implementation yet, and it does not establish that a
sparse mirror is sufficient for arbitrary multi-source models or complete
canonical-artifact reconstruction. The exact identity, bitmap publication,
crash boundary, integrity limitation, and alternative comparison are recorded
in `docs/vbuf-ml/step31b-persistent-source-representation-decision.md`.

## 3. Required Invariants

### Format and scope

- The v0.6 wire contract remains unchanged unless a separately scoped format
  proposal is approved and qualified.
- Generic vBuf parsing/navigation remains model-neutral.
- Persistent source retention belongs to vBuf-ML/runtime source layers, not
  `llama_model_loader` or a GGML loader callback.
- The backend continues to consume validated materialized or borrowed tensors.

### Source identity and correctness

- A cache hit requires immutable source identity, exact range identity, length,
  and validation. Matching only `offset + length` is insufficient.
- A locator, endpoint, file path, or cache filename is not source identity.
- A numeric `SourceId` must be tied to a model/source identity contract before
  it can authorize reuse across process or model sessions.
- Old and new source artifacts must never be combined into one logical model.
- Local and remote representations of the same source range must produce the
  same validated bytes before materialization exposes them to compute.
- Source identity and source storage policy remain independent of tensor
  semantic identity and active RAM residency.

### Persistence and crash safety

- The local mirror has the same declared logical size and offset space as the
  selected immutable payload source.
- A local range is readable as a cache hit only when coverage says the complete
  range was successfully acquired, validated, and published.
- A remote range becomes covered only after its complete bytes have been written
  at the canonical offset and the required process-crash publication step has
  succeeded.
- Partial writes, failed downloads, truncated ranges, malformed source data,
  integrity mismatches, and disk-full results never publish coverage.
- Coverage publication must be atomic with respect to readers and concurrent
  writers for the same canonical interval.
- Coverage state must not be inferred from sparse-file hole reads.
- Coverage state must not duplicate tensor/model semantics; it answers only
  whether canonical source bytes are safely present.
- A stale or deleted coverage accelerator must not make valid mirror bytes
  permanently unreachable; the minimal authoritative coverage representation
  must be recoverable or rebuildable under its declared contract.
- Deleting a model/source must prevent future reuse while not invalidating an
  active materialization lease.
- Persistent source retention/eviction must never be confused with active RAM
  residency or evict an active RAM lease; the two lifetimes are separate.

Process-crash safety and power-loss durability are separate requirements. The
first implementation decision must state whether a successful process-local
publication is sufficient for the first qualification or whether durable
power-loss recovery is required before reporting coverage as retained across a
device restart. It must not silently impose `fsync` per range without measuring
the cost and defining the failure boundary.

### Policy and measurement

- Stream-only, cache-as-used, and keep-model policies are explicit source
  retention policies, not implicit changes to inference semantics.
- Persistent storage budgets are independent from runtime residency budgets.
- A cache policy must not be selected from one fixed byte value for all models.
- Every performance result records cold/warm source state, source requests,
  remote/local bytes, unique/repeated bytes, residency events, materialization
  timing, and compute timing with non-overlapping boundaries.
- No source-only performance improvement is claimed until remote and local
  bytes are separately measured.

## 4. Existing Reusable Seams

The following should be reused before adding new abstractions:

| Existing seam | Reuse in this design | Constraint |
|---|---|---|
| `SourceId`, `SourceDescriptor`, `TensorRef` | Persistent source identity/range key input | Numeric IDs and optional hashes do not yet provide global artifact identity |
| `SourceRegistry` / `SourceSet` | Source resolution and local/remote composition boundary | Current `SourceSet` is static and non-persistent |
| Rust `RangeSource` | File, HTTP, hybrid source-neutral range reads | Trait has no source telemetry or write-through operation |
| `MmapSource` / `PositionedFileSource` | Existing local artifact/file reads | Neither provides partial-coverage lookup or persistence |
| C++ `RangeSource` | Android/GGML source boundary | Current Android path supplies one HTTP instance |
| `LocalVbufRangeSource` | Local mapped artifact control | Assumes an entire mapped artifact, not a cache store |
| `HttpRangeSource` | Remote range fallback and transport control | Exact-range unique metrics are not interval-union telemetry |
| `ReadPlan` | Preserve source IDs and physical ranges through planning | It loads temporary vectors and does not publish blocks |
| `LocalVbufRangeMaterializer` | Keep materialization/readiness/lease contract unchanged | Its fallback is failure fallback, not per-range hybrid resolution |
| `ResidentTensorMaterializer` | Keep active RAM residency independent | Identity is C++ tensor-ref based, not source-range based |
| `TensorResidencyStore` | Preserve 256 MiB bound and existing replacement seam | Must not become the persistent source cache |
| `MaterializationTrace` / `RangeReadResult` | Extend evidence at the source boundary later | Current Android report lacks retained range-level traces |
| `SourceSelectionPolicy` | Future local-vs-remote selection evidence | It selects whole source candidates, not partial local coverage |
| v0.6 `ValidatedV06` | Validate canonical stream/block geometry | Requires full stream context; block slices are not standalone records |
| `IntegrityMetadata` / `SourceHash` | Candidate correctness checks | Optional, target/source-scoped, not authentication or publication state |

## 5. Missing Seams

The genuinely missing pieces are:

1. **Source-artifact identity contract.** Define how the sparse mirror is bound
   to immutable model/source identity, declared size, profile/version, and any
   available source hash. Decide whether existing hashes are sufficient or
   whether a future source manifest/profile must carry the missing binding.
2. **Canonical-offset contract.** Qualify that every external TensorRef used by
   the target model addresses one immutable payload artifact directly and that
   local reads may use the same offset/length without rebasing.
3. **Minimal coverage state.** Choose the smallest authoritative representation
   of safely present canonical byte intervals. It must not duplicate tensor or
   model semantics and must not infer state from sparse-hole reads.
4. **Sparse mirror read/write seam.** Add no implementation yet; the future
   source layer needs exact-offset local reads, remote fallback, canonical-offset
   writes, and coverage publication. A new record/envelope format is not the
   default.
5. **Crash and durability contract.** Define temporary-write, validation,
   process-crash publication, power-loss durability, atomic replacement, and
   recovery semantics without assuming `fsync` per range is acceptable.
6. **Rebuildable coverage accelerator.** A compact interval/chunk index may be
   needed for performance, but it must be derivable from authoritative coverage
   state and safe to delete. It must not become model metadata.
7. **Hybrid partial-coverage source.** Compose local coverage and remote fallback
   for fully local, fully remote, and partially local requests. The current C++
   primary/fallback materializer path is not sufficient because it retries only
   after a complete primary read failure.
8. **Persistent retention policy.** Add stream-only, bounded cache-as-used, and
   keep-model behavior independently from `TensorResidencyStore` policy.
9. **Source-aware native runtime plumbing.** The Android PoC22 C++ tensor ref
   currently carries `source_offset` but not `SourceId`; a future source-cache
   implementation must either qualify the single-source restriction or carry
   source identity through the native runtime boundary.
10. **Range-level telemetry.** Record requested intervals and source outcome per
    token/phase so unique, repeated, local, remote, and overfetch bytes can be
    computed without guessing from residency events.
11. **Qualification fixtures.** Add sparse-hole, partial-write, coverage
    publication, corruption, source mismatch, concurrent request, disk-full,
    deletion, and coverage-rebuild tests before device integration.

## 6. Open Questions

### Canonical mirror authority

- Does every target-model external TensorRef resolve to one complete immutable
  payload artifact, or are there hidden multi-source bindings?
- Are all source offsets relative to the beginning of that payload artifact,
  with no source-specific rebasing in the native path?
- Is the declared source size exactly the remote payload artifact size used to
  size a sparse mirror?
- Is the first useful local state “all observed TensorRef payload ranges
  covered,” or must it also cover source headers, padding, and control regions?
- For a complete offline source, how are non-tensor and never-requested bytes
  acquired and validated?
- Is a mirror of payload bytes sufficient for the existing semantic bootstrap,
  or does any local consumer need to parse the original payload artifact's v0.6
  headers too?

### Coverage representation

**Step 31B resolution:** use fixed 4 KiB source chunks and a one-bit
authoritative bitmap. The measured irregular request boundaries make exact
TensorRef coverage a poor authority, while a 4 KiB bitmap remains compact and
adds only derived low single-digit-percent rounded acquisition overhead on the
31A trace. Arbitrary interval journals and canonical block maps are deferred or
rejected for the first implementation.

- Should coverage track arbitrary byte intervals, fixed source chunks, canonical
  v0.6 physical blocks, or TensorRef ranges?
- What request geometry and overlap distribution make one representation cheaper
  and less error-prone than the others?
- Can coverage be rebuilt from a compact bitmap/interval file, or must it retain
  a journal/checkpoint to distinguish published writes after a crash?
- How are overlapping writes and partial-range reads published atomically?

### Identity and integrity

**Step 31B resolution:** the mirror key is `(declared_size, full-source
SHA-256 SourceHash)`. `SourceId` remains profile binding context, not identity.
The target profile has the expected full hash, but range-only acquisition does
not verify it per range; the first implementation uses a trusted immutable
source contract and defers whole-artifact final verification.

- Are `SourceId` values generated deterministically and immutably for each
  artifact, or are they merely profile-local labels?
- Which existing `SourceHash` algorithms and values are present in the real
  DeepSeek semantic bootstrap? Are they full-source hashes, and can they be
  checked without defeating range loading?
- Is per-block/per-payload integrity available for the external source? The
  existing `IntegrityMetadata` records canonical bootstrap blocks and does not
  automatically authenticate an external payload range.
- Does the source-profile SHA-256 hash cover the exact remote payload artifact,
  and can it be used as a persistent mirror binding without requiring a full
  hash before the first range is served?
- What trust model is required for accidental corruption versus malicious
  source replacement? Existing SHA-256 metadata is not a signature.

### Store and platform behavior

**Step 31B resolution:** the semantic bootstrap remains separate; the payload
mirror is model/source scoped by its identity header. Process-crash publication
is required for 31C, power-loss durability is deferred, and per-range `fsync`
is not required for the first qualification.

- What Android flash directory and quota API should own the sparse payload mirror
  and its minimal coverage state?
- What durability level is required after a successful publish: close, `fsync`,
  directory durability, or a platform-specific equivalent?
- How should concurrent reads/writes for the same canonical interval coordinate
  without exposing a partially written file or prematurely publishing coverage?
- What should happen when the cache is full, the source is removed, or a model
  is replaced while a lease is active?
- Is a model/source-scoped sparse mirror safer and easier to delete than a
  content-addressed layout, given that offsets are meaningful only within one
  source artifact?

### Policy and performance

- Which persistent storage curve is useful for this model after real range
  overlap is known?
- How much local flash read latency and filesystem overhead is added per range?
- Does serving the same requested range locally reduce wall time, or does
  materialization/compute dominate the current `~55 s/token` result?
- Which user intent vocabulary should map to retention policy? This document
  does not finalize UI terminology.

## 7. Telemetry / Evidence Needed

### First evidence pass

Before implementing persistence, capture or recover a range trace for the same
model, prompt, runtime mode, residency cap, GGML configuration, and token
sequence as the Android baseline. Do not change batching, residency, thread
count, backend, or source scheduling in that experiment.

The trace should record at minimum:

```text
generation_id
token_index
phase
tensor_ref / semantic tensor identity
source_id
requested_offset
requested_length
returned_length
source_kind (remote/local)
request_start / first-byte / completion timestamps
materialization-ready timestamp
residency hit/miss/eviction/lease events
```

The analysis must distinguish:

```text
REQUESTED_SOURCE_BYTES
UNIQUE_BYTE_UNION
PREVIOUSLY_SEEN_BYTE_UNION
REPEATED_BYTE_OVERLAP
LOCAL_BYTES
REMOTE_BYTES
OVERFETCH_BYTES
```

The existing `HttpRangeSource::unique_bytes` is exact-range uniqueness, not a
general interval union. It should not be reported as `UNIQUE_BYTE_UNION`
without an interval analysis.

### Offline reuse and capacity simulation

For each token, compute the interval union and overlaps against all previous
tokens. Then replay the same range sequence against hypothetical persistent
coverage/retention capacities, at least:

```text
stream-only
256 MiB
512 MiB
1 GiB
2 GiB
keep-all
```

This is a source-cache simulation, not a runtime residency simulation. The
existing `TensorResidencyStore` trace and its cost-aware policy can inform the
comparison, but persistent coverage must be modeled independently.

### Evidence classification

Every result must be labeled measured, derived, comparable, observer-distorted,
or unresolved. In particular:

- the `816 MB/token` value is requested source traffic, not unique bytes;
- the `14.47%` payload fraction rules out full-model retransmission but does
  not identify the network/materialization/compute split;
- the `~55 s/token` value is not evidence that persistence will help by itself;
- local/warm results must keep source state, residency state, process state,
  model, prompt, and backend configuration explicit.

## 8. Implementation Checklist

This is an ordered design checklist, not an implementation script.

### Phase 1: Canonical-offset and source-authority audit

- [x] Inventory the target model's source-profile `SourceId`, declared size,
      locator, source hash, and binding count.
- [x] Prove whether all target external TensorRefs address one immutable random-
      access payload artifact directly, without offset rebasing.
- [x] Record the exact relationship between the separate semantic bootstrap,
      payload artifact, v0.6 headers, and TensorRef payload ranges.
- [x] Determine whether an existing local semantic bootstrap can consume a
      same-offset payload mirror unchanged.
- [x] Capture a no-cache-change source range trace and derive bytewise union,
      new unique bytes, repeated overlap, reuse distance, request count,
      requested bytes, and overfetch where measurable.
- [ ] Simulate source-coverage policies and candidate storage budgets before
      selecting retention or eviction behavior.

### Phase 2: Persistent source representation decision

- [x] Compare, using the Phase 1 facts, (A) sparse canonical mirror plus minimal
      coverage, (B) canonical block persistence, and (C) a runtime-local
      envelope/record store.
- [x] Prefer the sparse mirror if remote and local offsets are provably
      identical, the declared size is stable, and coverage can be published
      safely without duplicating model semantics.
- [x] Define whether coverage tracks arbitrary intervals, fixed source chunks,
      canonical blocks, or TensorRef ranges from actual request geometry.
- [x] Define source identity, coverage, validation, and incomplete/full-mirror
      states independently from the semantic bootstrap.
- [x] Reject a more complex record/envelope format unless the mirror alternatives
      fail a correctness, identity, recovery, or practical-lookup requirement.

### Phase 3: Minimal local-hit / remote-miss source mechanism

- [x] Add the smallest source-layer local sparse-mirror read and coverage lookup
      seam, without changing `TensorRef`, materializer, lease, residency, or
      GGML contracts.
- [x] On a miss, read the complete touched 4 KiB chunks from the remote
      canonical source, validate them, use the requested bytes for the current
      request, write chunks at the same canonical offsets, and publish coverage
      only after the defined publication step succeeds.
- [ ] Keep stream-only as a policy that bypasses persistence entirely.
- [x] Define the simplest correct handling for partial requested ranges: fetch
      complete touched 4 KiB chunks first; defer interval splitting/coalescing.
- [x] Keep persistent source coverage separate from active RAM residency.

### Phase 4: Cold-remote versus warm-local qualification

- [x] Qualify the deterministic host remote-cold fixture first.
- [x] Qualify warm local reuse with the same Pixel, model, prompt, generated
      sequence, threads, batching, GGML configuration, and 256 MiB residency.
- [x] Measure remote bytes, local bytes, source requests, and coverage
      hits/misses separately in the host fixture.
- [ ] Measure physical materialization timing, decode time, and compute timing
      separately.
- [x] Verify output/source-range correctness and unchanged lease/residency
      behavior in the completed cold/warm runs.
- [x] Confirm that the local mirror served the measured warm workload without
      remote traffic or semantic-path branching.

### Phase 5: Retention policy

- [ ] Keep stream-only, keep-as-used, bounded retention, and keep-model policies
      above the source mechanism and independent from RAM residency.
- [ ] Add persistent eviction only after local-hit correctness and warm/cold
      attribution pass.
- [ ] Use measured reuse distance and coverage simulation to choose any initial
      persistent storage budget; do not reuse the 256 MiB RAM value by default.
- [ ] Distinguish progressively local coverage from guaranteed complete offline
      coverage, especially for MoE expert ranges.

### Phase 6: Offline completion using the same mirror

- [ ] Determine declared source coverage required for a complete offline state.
- [ ] Discover missing canonical ranges without creating a second downloader
      representation.
- [ ] Fill only missing holes through the same validated remote-to-mirror path.
- [ ] Report exact incomplete coverage when inference use has not touched all
      model bytes.

### Phase 7: Independent later experiments

- [x] Attempt the runtime residency curve independently at 256 MiB, 512 MiB,
      1 GiB, and 2 GiB with persistent source state held constant. Only 256 MiB
      completed decode; larger points were physically constrained at first
      decode and are not timing comparisons.
- [ ] Later evaluate idle warmup, next-use planning, prefetch/compute overlap,
      advanced replacement, mobile-data UX, and model-aware policy tuning.
- [ ] Tune backend compute and threads only in separate experiments.

### Phase 8: Step 31E attribution boundary

- [x] Expose only the numeric active residency budget through the existing
      qualification build seam; retain the 256 MiB default.
- [x] Add aggregate materialization and repeated-materialization counters at the
      existing generic residency boundary without changing eviction behavior.
- [x] Reuse the persistent mirror and authoritative coverage across fresh
      process lifecycles.
- [x] Record zero remote traffic for every measured curve point.
- [x] Record that 512 MiB, 1 GiB, and 2 GiB did not produce comparable decode
      timing because the first decode failed at the existing runtime boundary.
- [ ] Choose a production residency default from this result.
- [ ] Tune residency or backend behavior from this result.

## 9. Qualification Matrix

| Stage | Required proof | Must remain unchanged |
|---|---|---|
| Authority audit | Canonical offset equality, source artifact size, bootstrap relationship, identity, and hash scope recorded | v0.6 wire contract and current remote baseline |
| Trace analysis | Exact interval union, overlap, reuse distance, and source/residency distinction | Runtime mode, prompt, batching, residency, backend |
| Representation decision | Sparse mirror, block persistence, and envelope alternatives compared against measured geometry and invariants | No implementation or runtime behavior |
| Mirror/coverage fixture | Sparse holes are not hits; complete canonical writes publish coverage only after validation/publication; stale identity fails closed | Semantic TensorRef and backend boundary |
| Mirror recovery | Crash at each write/publication point leaves missing or previously published coverage; coverage rebuild follows its contract | Current source behavior when persistence is disabled |
| Remote cold control | Existing remote requests, bytes, output, and residency remain qualified | All runtime/source policy except explicit instrumentation |
| Local warm control | Same output and semantic ranges; local-hit and remote-miss accounting is correct | Same prompt/model/GGML/residency/thread configuration |
| Hybrid coverage | Full-local, full-remote, and qualified partial-local reads produce identical validated target bytes | Tensor materialization and leases |
| Cache-as-used | Retention budget and eviction do not alter source correctness; network bytes decrease as measured | 256 MiB active residency semantics |
| Keep-model/offline | Missing-only acquisition reaches complete declared source coverage, or reports exact incompleteness | No separate model-download representation |
| Android physical | APK uses the same runtime path; cold/warm source states and local/remote bytes are visible | No Android-specific inference semantics; no residency change |
| Residency curve | Step 31E curve report; 256 MiB completed and larger caps classified as device-constrained | Source identity, source policy, batching, backend, and thread count |

## 10. UX / Policy Mapping

The low-level source mechanism should expose intent, not arbitrary cache numbers:

| Future intent | Source behavior | Not implied |
|---|---|---|
| Stream only | Read remote ranges and discard persistent copies | No change to RAM residency |
| Cache as used | Publish encountered valid units under a model-specific storage budget | Not guaranteed offline completeness |
| Keep model / make available offline | Retain encountered units and acquire declared missing units on demand | Not loading the full model into RAM |
| Runtime memory setting | Change only active `TensorResidencyStore` budget | Does not change persistent source coverage |
| Data-use safeguard | Gate or warn on remote acquisition using measured accounting | Does not redefine source identity or inference semantics |

“Data saver” or “storage saver” terminology is intentionally not finalized
here. A policy must not mean “exactly 1 GiB” for every architecture. Dense and
MoE models can have different useful persistent working sets because routing,
shared paths, tensor sizes, and reuse distances differ.

## 11. Failure / Recovery Cases

| Failure | Required result |
|---|---|
| Remote read interrupted | No publication; current request fails or retries under an explicit future source policy |
| Flash write interrupted | Unpublished mirror bytes are ignored; coverage is not published for the incomplete interval |
| Process death during tee | Only previously published coverage remains visible; the remote result may still complete the current request |
| Partial requested range | The complete touched 4 KiB coverage chunks are fetched remotely; a partial local read is never treated as complete |
| Sparse hole read | Coverage lookup rejects it; hole contents never authorize a cache hit |
| Corrupt mirror bytes | Coverage is invalidated or quarantined for the affected interval; remote fallback may repair it |
| Source identity mismatch | The local mirror is not reused, even when offset/length match |
| Source version changes at same locator | Old and new records remain separated or old records are invalidated |
| Stale coverage accelerator | Rebuild from authoritative coverage state; stale entries cannot authorize a hit |
| Coverage accelerator deleted | Published valid coverage remains recoverable under the declared coverage contract |
| Disk full | Current computation may continue from remote/temporary bytes; no partial publication |
| Concurrent same-range requests | One valid publication wins; readers see uncovered, old valid, or new valid bytes, never partial coverage |
| Concurrent read during publication | Reader uses the previous covered bytes or remote fallback; it never maps a growing range as valid |
| Model/source removed | New requests fail or use an explicitly selected replacement; active leases remain valid until release |
| Integrity mismatch | Local record is invalid; remote replacement is validated before publication |
| Missing canonical source context | Do not claim a canonical mirror or self-describing block; retain only a representation with an explicit validation contract |

## 12. Deferred Ideas

The following are explicitly deferred and are not required for the minimal
source-persistence mechanism:

- idle-time warmup;
- next-layer prefetch;
- compute/prefetch overlap;
- advanced cache replacement beyond the existing source-policy experiment;
- model-aware automatic policy tuning;
- mobile-data UX and warning policy;
- full offline completion, beyond designing the same-store path;
- runtime residency tuning;
- backend compute tuning;
- speculative expert acquisition before routing;
- Android-specific inference or a separate model loader;
- changing batching, thread count, GGML backend, or quantization as part of the
  first source-persistence experiment.
- successful larger-cap physical residency attribution beyond the Step 31E
  device-constrained curve;

## Decision Snapshot

```text
VBUF_ML_PROGRESSIVE_LOCAL_SOURCE_PLAN: STEP_31E_RESIDENCY_CURVE_INCOMPLETE_DEVICE_CONSTRAINED

CURRENT_ARCHITECTURE:
  TensorRef -> SourceSet/RangeSource -> materialization -> lease/residency -> GGML
  Android qualification currently uses one HttpRangeSource over the PoC22 path.

SELF_DESCRIBING_BLOCK_SUPPORT:
  PARTIAL. v0.6 blocks describe canonical geometry in stream context, but an
  isolated block lacks global BaseShift/stream context, source identity, and digest.

SOURCE_IDENTITY_SUPPORT:
  DECIDED. The persistent key is declared source size plus SourceHash algorithm
  1 (full SHA-256) and digest. SourceId(1) selects the profile binding but is
  not identity; the locator is not identity.

LOCAL_FILE_SOURCE_SUPPORT:
  YES. Rust MmapSource/PositionedFileSource and C++ mapped LocalVbufRangeSource
  exist; no persistent partial-coverage store exists.

HYBRID_SOURCE_SUPPORT:
  YES for the bounded C++ source path. `ProgressiveRangeSource` composes an
  existing remote RangeSource with a same-offset sparse mirror and bitmap;
  materialization remains unchanged.

REBUILDABLE_COVERAGE_FEASIBLE:
  DECIDED. A one-bit-per-4-KiB-chunk bitmap is authoritative; an in-memory
  lookup accelerator is optional and rebuildable. Sparse-file holes are not
  coverage.

CRASH_SAFE_PUBLICATION_SUPPORT:
  HOST-QUALIFIED for process ordering: complete chunk write precedes one-bit
  publication, and reopen trusts only a valid sidecar. Power-loss durability
  and fsync policy remain deferred.

UNIQUE_RANGE_TELEMETRY_AVAILABLE:
  PARTIAL. Exact-range unique counters and materialization offsets exist; a
  bytewise per-token union/overlap trace is not currently retained.

REUSABLE_COMPONENTS:
  SourceId, SourceDescriptor, TensorRef, SourceSet, RangeSource, MmapSource,
  PositionedFileSource, ReadPlan, RangeReadResult, LocalVbufRangeMaterializer,
  ResidentTensorMaterializer, TensorResidencyStore, SourceHash, IntegrityMetadata.

MISSING_COMPONENTS:
  First-decode backend error attribution for larger residency caps,
  process-death fault injection, retention, power-loss durability, offline
  completion, and multi-source policy remain later work.

OPEN_ARCHITECTURE_QUESTIONS:
  No representation choice remains open for the qualified single-source path.
  Deferred questions are whole-artifact hash finalization, power-loss durability,
  retention policy, complete/offline coverage, and multi-source extension.

PROPOSED_PHASE_COUNT: 8
PHASE_1: canonical-offset and source-authority audit
PHASE_2: persistent source representation decision (DECIDED: A)
PHASE_3: minimal local-hit/remote-miss source mechanism (31C)
PHASE_4: bounded demand-driven acquisition (31D; HOST QUALIFIED)
PHASE_5: physical residency curve (31E; device-constrained above 256 MiB)
PHASE_6: crash/corruption/recovery qualification
PHASE_7: extended cold/warm attribution
PHASE_8: retention and offline completion

FIRST_MEASUREMENT_TO_RUN:
  Completed in Step 31A: captured and offline-analyzed requested source
  intervals for the unchanged 80409a4 Android configuration.

FIRST_IMPLEMENTATION_COMPLETED:
  Step 31C: one qualified source, same-size sparse payload mirror, existing
  SourceHash/declared-size identity, 4 KiB bitmap, local-hit/remote-miss
  resolution, canonical writes followed by bit publication, and no residency
  or execution changes.

FIRST_PHYSICAL_QUALIFICATION:
  Step 31D completed the same Pixel/model/semantic bootstrap cold and warm
  inference workload with persistent source reuse and output parity. Step 31E
  completed the 256 MiB local curve point and attempted 512 MiB, 1 GiB, and
  2 GiB; the larger points failed at first decode before comparable timing.

RESIDENCY_SEPARATE_FROM_SOURCE_CACHE: YES, REQUIRED
ANDROID_SPECIFIC_RUNTIME_REQUIRED:
  NO for the architecture; only a thin source plumbing adapter is needed to
  qualify the existing Android path, not Android-specific inference semantics.
SEPARATE_MODEL_DOWNLOADER_REQUIRED:
  NO by intent; offline completion should use the same source mirror.
AUTHORITATIVE_COVERAGE_STATE_REQUIRED:
  YES. A minimal coverage representation is required; an accelerator index may
  be rebuildable, but sparse-hole state alone cannot authorize a cache hit.

DOCUMENT_CREATED_OR_UPDATED:
  docs/vbuf-ml/progressive-local-source-design-checklist.md
  docs/vbuf-ml/step31b-persistent-source-representation-decision.md
  research/results/vbuf-android-demo-poc/step31c-progressive-local-source.md
  research/results/vbuf-android-demo-poc/step31d-bounded-acquisition.md
STEP_31B_REPRESENTATION: SPARSE_CANONICAL_MIRROR_WITH_4_KIB_BITMAP
STEP_31C_IMPLEMENTATION: HOST_QUALIFIED
STEP_31D_IMPLEMENTATION: HOST_QUALIFIED_1_MIB_DEMAND_WINDOWS
STEP_31E_IMPLEMENTATION: MINIMAL_BUDGET_AND_REACQUISITION_COUNTER_SEAM
PHYSICAL_ANDROID_QUALIFICATION: STEP_31D_COLD_WARM_AND_STEP31E_CURVE_ATTEMPTED
PHYSICAL_ACQUISITION_LIMITATION: RESOLVED_BY_1_MIB_DEMAND_WINDOWS
NEXT_SOURCE_EXPERIMENT: FIRST_DECODE_BACKEND_ERROR_ATTRIBUTION
RUNTIME_BEHAVIOR_CHANGED: NO
COMMIT_PERFORMED: PENDING_STEP31E_COMMIT
```

## Review Questions

1. **What is the smallest mechanism required to persist normally used remote
   bytes?**

   If the qualified payload is one immutable same-offset source, it is a sparse
   canonical mirror, explicit source identity, minimal coverage state, validated
   remote-to-local writes, and local-hit/remote-miss lookup. A record/envelope
   store is not the default.

2. **When is the sparse mirror authoritative?**

   Only for a qualified source identity and declared size, where local offset `o`
   denotes the same source byte as remote offset `o`, and coverage proves the
   touched 4 KiB chunks were completely validated and published. The mirror is
   not automatically valid for arbitrary multi-source models.

3. **What auxiliary coverage structure, if any, is required?**

    An explicit one-bit-per-4 KiB-chunk bitmap is required; sparse-hole reads are
    not coverage. A compact lookup accelerator may be added for performance, but
    it must be rebuildable and never authorize a hit by itself.

4. **How is source identity guaranteed across remote and local copies?**

    By binding both to the existing full-source SHA-256 `SourceHash` and declared
    size from the source profile. `SourceId` is useful profile plumbing but
    insufficient alone because it is numeric/profile-scoped; the locator is not
    identity and range-only reads do not independently verify the whole hash.

5. **What does complete/offline coverage mean?**

   It must be defined against the declared source artifact, not only the ranges
   encountered during inference. TensorRef coverage may be sufficient for warm
   reuse but cannot be reported as full offline availability until headers,
   padding/control regions, and never-requested ranges are accounted for.

6. **What must be measured before choosing a cache budget?**

   Per-token requested interval unions, repeated overlap, reuse distance, local
   versus remote bytes, overfetch, materialization wait, decode wall time, and
   compute timing under unchanged residency. Then simulate storage budgets from
   the real trace rather than selecting a universal byte value.

7. **How should cold remote vs warm local qualification be performed?**

   Use the same model, semantic bootstrap, prompt, generated sequence, runtime
   mode, batching, GGML configuration, thread configuration, and `256 MiB`
   residency. Run an empty persistent store as the remote cold control, then
   repeat with the same encountered source coverage served locally. Record
   remote/local bytes, requests, materialization timings, residency events,
   decode time, output, and correctness separately.

8. **Which parts should explicitly remain deferred?**

    Idle-time warmup, next-layer prefetch, compute/prefetch overlap, advanced
    replacement, model-aware automatic tuning, mobile-data UX, the implementation
    of full offline completion, residency tuning, backend compute tuning, and any
    Android-specific inference path remain deferred.
