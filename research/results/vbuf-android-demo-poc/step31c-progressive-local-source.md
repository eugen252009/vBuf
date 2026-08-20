# Step 31C Progressive Local Source

Date: 2026-08-20
Branch: `vbuf-ml`
Starting commit: `f832333 Select Step 31B persistent source representation`
Status: **IMPLEMENTED / HOST QUALIFIED; PHYSICAL ANDROID QUALIFICATION PARTIAL**

## Objective

Implement only the Step 31C source mechanism:

```text
remote miss
  -> complete canonical 4 KiB acquisition
  -> same-offset sparse payload write
  -> coverage-bit publication
  -> local read through the existing RangeSource contract
```

No TensorRef, materializer, lease, residency, backend, scheduler, or inference
semantics were changed.

## Selected Representation

```text
payload mirror: same logical size and canonical offsets as the external source
coverage:       `.coverage` sidecar, one authoritative bit per 4096-byte chunk
identity:       declared source size + SourceHash algorithm 1 + 32-byte SHA-256
bootstrap:      separate; never copied into the payload mirror
```

The payload file is opened with `O_CREAT|O_RDWR` and sized with `ftruncate`; no
payload bytes are eagerly allocated. Sparse holes are never used as coverage.
The sidecar contains only a versioned identity/geometry header and bitmap. It
contains no tensor or model metadata.

## Implementation

### CODE-AUDITED

- `ProgressiveRangeSource` was added below the existing C++ `RangeSource`
  boundary in `integrations/ggml/include/vbuf_range_source.h` and
  `integrations/ggml/src/vbuf_range_source.cpp`.
- Existing `HttpRangeSource` remains the remote leaf source.
- `LocalVbufRangeMaterializer` receives the progressive source through its
  existing `std::shared_ptr<RangeSource>` constructor.
- The Android direct session obtains the existing source-profile identity via
  `vbuf_ml_consumer_source_identity`; no TensorRef or metadata representation
  was duplicated.
- Android derives the mirror path from the semantic bootstrap path as
  `<bootstrap>.payload`, with `<bootstrap>.payload.coverage` as the sidecar.

### DECIDED / ENFORCED

- Requests are rejected for zero length, overflow, or declared-size overflow.
- A request maps to every intersecting 4 KiB chunk, including a short final EOF
  chunk.
- Missing chunks are acquired individually from the existing remote source.
- Returned offset, requested length, and returned byte count must match the
  canonical chunk request before persistence.
- A complete chunk is written at its canonical offset before its bitmap bit is
  published.
- The first implementation serializes the whole source operation with one
  mutex, preventing duplicate same-chunk acquisition and partial visibility.
- Identity, logical size, chunk geometry, sidecar format, or trailing bitmap
  bits that do not match fail closed on reopen.
- `fsync` is not performed per chunk. Power-loss durability remains deferred;
  the qualified guarantee is process-ordering and restart acceptance of a
  complete sidecar.

### MEASURED: HOST FIXTURE

The deterministic fixture used a 10,000-byte immutable source and a 4 KiB
bitmap:

```text
cold request:             offset=5000, length=5000
remote chunk requests:    [4096, 4096] and [8192, 1808]
remote bytes acquired:    5904
published chunks:         2 of 3
coverage sidecar bytes:   81 (80-byte header + 1-byte bitmap)
mirror logical size:      10000 bytes
```

The same request after publication returned identical bytes with zero additional
remote calls and two local chunk hits. A subsequent request populated chunk 0;
the reopen test then served the previously published range with zero remote
calls from a new source instance.

The concurrent same-chunk test used two overlapping requests and observed one
remote acquisition and one published chunk. Remote failure and a short returned
length left coverage clear. Hash and declared-size identity mismatches rejected
reuse before any local hit.

## Tests And Builds

### UNIT-TESTED

- Empty/cold mirror and canonical chunk writes.
- Warm hit with unchanged remote call count.
- Unaligned, cross-chunk, exact-chunk, EOF, zero-length, and out-of-range
  requests.
- Partial coverage without refetching covered chunks.
- Identity mismatch for SHA-256 and declared size.
- Reopen of published coverage.
- Remote failure and truncated result without publication.
- Concurrent same-chunk acquisition and overlapping reads.
- Logical mirror size and sidecar geometry.
- Invalid persistent store fails closed.

### HOST-QUALIFIED / VERIFIED

```text
cargo fmt --all -- --check: PASS
cargo test --workspace: PASS
neutrality guard: PASS, FORBIDDEN_LEAKAGE_COUNT=0
direct host C++ source contract: PASS
native CTest range contracts: PASS
```

The final native CTest run used the fetched pinned ggml checkout at revision
`2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`. The focused source contract also
compiled independently of ggml model behavior.

### ANDROID-BUILD-QUALIFIED

- ARM64 native build: PASS.
- Android `assembleDebug`: PASS.
- APK installed and the real Pixel runtime opened the semantic model: PASS.

### PHYSICALLY OBSERVED / PARTIALLY QUALIFIED

- Normal Android runtime reached real batched prefill with
  `ProgressiveRangeSource` active: PASS.
- Remote payload access, sparse mirror population, and coverage publication:
  OBSERVED.
- A stopped-process restart reopened the same payload/coverage state with the
  same identity and reported `Chunks covered: 7000` and `Coverage bytes:
  172114`: PASS for persistent-state resumption and local-reuse eligibility.
  No generation read was issued after restart, so a post-restart local-hit
  counter was not collected.

### NOT MEASURED

- Full physical cold/warm traffic, timing, output parity, and RAM residency
  qualification remain incomplete.
- Physical prefill speedup, decode speedup, FFN speedup, and remote reduction
  remain unmeasured.

### PHYSICAL OBSERVATION: ANDROID SOURCE SMOKE

The Android environment was repaired and the ARM64 APK was rebuilt with the
documented endpoint override. The connected device was a Pixel 7 Pro
(`192.168.188.33:46013`, `arm64-v8a`). The semantic bootstrap was already
present on-device and the initial payload mirror was absent.

The host payload was verified against the bootstrap before the run: the
semantic parity digest was
`72d5398296f01fbe6d7e90bef8f855c554c938fb3df061f1b04555d043d0aa00`, and all
four sampled external ranges matched byte-for-byte.

The first run used nginx and returned valid HTTP 206 responses, but stopped at
the exact default nginx `keepalive_requests 1000` boundary: 1,001 progressive
misses were attempted, 1,000 chunks were published, and generation failed with
`batched dense FFN failed`. This is classified as test-server connection
boundary evidence, not a payload or mirror-integrity failure.

A replacement range server was started on host port 18125 and exposed through
`adb reverse tcp:18124 tcp:18125`. A fresh cold run then entered normal batched
prefill and published approximately 22 MiB of sparse payload blocks in about
3.5 minutes at the captured checkpoint, without the 1,000-request failure.
The logical payload mirror was `5,639,819,878` bytes. At final cleanup the
coverage sidecar was `172,194` bytes and the restarted app accepted 7,000
published chunks. These are physical observations, not a controlled throughput
benchmark. The run was force-stopped after the device-side cancel could not
interrupt the current serialized tensor acquisition. The partial mirror was
shown to survive the stopped process and remain resumable.

The request shape is the separate limitation exposed by this run: each missing
4 KiB coverage chunk currently becomes a similarly fine upstream HTTP
acquisition. Coverage granularity is therefore not invalidated, but acquisition
granularity needs a later source-layer experiment. The replacement server
functioned correctly; this is request amplification, not a source-correctness
failure.

No generation output, warm-run result, source-complete hash verification, or
residency qualification is claimed from this smoke run.

## Architecture Boundary

```text
TensorRef
  -> existing source/materializer boundary
  -> ProgressiveRangeSource
       -> sparse mirror + coverage bitmap
       -> existing remote RangeSource
  -> existing materializer
  -> existing residency/leases
  -> existing GGML execution
```

```text
TENSORREF_CHANGED:            NO
MATERIALIZER_CHANGED:         NO
RESIDENCY_CHANGED:            NO
BACKEND_CHANGED:              NO
ANDROID_INFERENCE_CHANGED:    NO
```

## Integrity Boundary

Artifact identity is bound to the source profile's declared size and full
SHA-256 metadata. Individual range acquisition has no per-range cryptographic
digest and does not verify the complete artifact hash. Step 31C therefore
claims structural framing, bounds, complete-write ordering, and identity-bound
coverage under the trusted immutable-source qualification contract only.

## DEFERRED

Step 31C does not implement retention, eviction, budgets, offline completion,
prefetch, background download, mobile-data policy, power-loss durability,
whole-artifact final verification, multi-source orchestration, residency tuning,
backend tuning, full physical Android cold/warm qualification, or acquisition
coalescing.
