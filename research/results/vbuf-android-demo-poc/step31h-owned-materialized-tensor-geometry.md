# Step 31H Owned Materialized Tensor Geometry

Date: 2026-08-20
Starting commit: `096752e`
Status: **HOST-QUALIFIED; PHYSICAL CONFIRMATION BLOCKED**

## Proven 31G Root Cause

Step 31G established that `executor_persistent` was the last valid boundary
and `resident_ready` was the first invalid boundary. A residency hit returned
a cached `MaterializedTensor` with valid leased payload storage but a dangling
borrowed `dimensions` pointer. The proven class was a shape-owner lifetime
mismatch/dangling borrowed descriptor view. Residency exposed the latent bug;
the replacement policy and cache algorithm were not shown to be defective.

## Ownership Contract

Before Step 31H, `MaterializedTensor` contained a shallow `VbufTensorView`:
rank and representation were values, but `dimensions`, `payload`, and
`payload_len` were borrowed fields. Its payload storage had a lease owner, but
its shape storage had no owner. Local and striped materializers copied the
source view, and residency copied that object again.

The audited contract was:

```text
MATERIALIZED_TENSOR_TYPE: internal C++ value type
RANK_STORAGE: uint8_t value
DIMENSION_STORAGE: borrowed const uint64_t pointer (before 31H)
PAYLOAD_STORAGE: borrowed payload pointer plus VbufBorrowedStorage
PAYLOAD_OWNER: VbufBorrowedStorage::lease
SHAPE_OWNER: caller/source descriptor (before 31H)
COPY_BEHAVIOR: shallow VbufTensorView copy (before 31H)
MOVE_BEHAVIOR: shallow VbufTensorView move (before 31H)
CACHE_LIFETIME: TensorResidencyStore entry outlives request descriptor
BORROWED_FIELDS: dimensions, payload, and storage base
```

## Repair

`MaterializedTensor` now owns `representation`, `rank`, and an inline
`std::array<uint64_t, GGML_MAX_DIMS>` geometry buffer. `view()` creates a
transient backend view pointing at that object's owned array. The type has no
self-referential view pointer, so ordinary copy, move, return-by-value, and
container relocation preserve geometry.

Payload behavior was not changed. Materializers still use the existing owned
payload allocation and `VbufBorrowedStorage::lease`; residency still stores
and returns the payload without an additional tensor-payload copy.

## Regression Coverage

`vbuf_residency_contract` now exercises the actual residency path with
stack-backed source geometry, then verifies rank, dimensions, representation,
payload pointer, payload length, and bytes after:

- source geometry scope destruction;
- cold materialization retrieval;
- warm residency-hit retrieval;
- value copy;
- value move;
- vector relocation.

The existing materializer, striped materializer, tensor adapter, source
fallback, and tensor-wave contracts were updated to consume the value-generated
view without changing their payload assertions.

## Qualification

```text
CACHED_GEOMETRY_SURVIVES_SOURCE_SCOPE: PASS
RESIDENCY_HIT_GEOMETRY: PASS
PAYLOAD_POINTER_LEASE: PASS; unchanged
COPY_MOVE_GEOMETRY: PASS
HOST_ROOT_CAUSE_REPRODUCED_BEFORE_FIX: NOT EXECUTED; Step 31G physical trace is the pre-fix evidence
HOST_ROOT_CAUSE_FIXED_AFTER_FIX: PASS; deterministic cache-lifetime regression
PHYSICAL_ENVIRONMENT_AVAILABLE: NO
PHYSICAL_512_RUN_EXECUTED: NO
PHYSICAL_FIX_CONFIRMATION: BLOCKED_BY_MISSING_QUALIFICATION_ARTIFACT
```

The canonical local model artifact and range server are unavailable in this
workspace. No substitute model, prompt, or source was used. The former
512 MiB Android failure is therefore not claimed physically fixed, and no
new downstream physical failure is claimed.

## Scope Classification

```text
RESIDENCY_POLICY_CHANGED: NO
RESIDENCY_ALGORITHM_CHANGED: NO
PERSISTENCE_CHANGED: NO
ACQUISITION_CHANGED: NO
TENSORREF_CHANGED: NO
MATERIALIZER_PAYLOAD_SEMANTICS_CHANGED: NO
BACKEND_CHANGED: NO
INFERENCE_SEMANTICS_CHANGED: NO
ABI_CHANGED: NO; MaterializedTensor is internal C++ and C ABI views remain POD
RESIDENCY_CURVE_RESUMED: NO
```
