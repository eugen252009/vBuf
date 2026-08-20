# Step 31H Owned Materialized Tensor Geometry

Date: 2026-08-20
Starting commit: `096752e`
Status: **QUALIFICATION INPUTS RESTORED; PHYSICAL CONFIRMATION BLOCKED BY ADB**

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
PHYSICAL_ENVIRONMENT_AVAILABLE: YES; ADB_SERVICE_UNAVAILABLE
PHYSICAL_512_RUN_EXECUTED: NO
PHYSICAL_FIX_CONFIRMATION: BLOCKED_BY_ADB_CONNECTION
```

The exact qualification source was restored from the recorded Hugging Face
provenance:

```text
HF_REPOSITORY: legraphista/DeepSeek-V2-Lite-IMat-GGUF
HF_REVISION: 3048fc1df365e992c92a055324e8fd872e5763b9
HF_FILENAME: DeepSeek-V2-Lite.IQ2_XXS.gguf
SOURCE_SIZE: 5640619552
SOURCE_SHA256: 3b7da33584bebf89afcdbdd2e7a8e3e47e11092971559371f13b510f475e3c0c

IMPORTER: scripts/build_step18_manifest.py + scripts/convert_gguf_to_vbuf_ml.py
IMPORTER_COMMIT: 41945b3
IMPORT_COMMAND: |
  python3 scripts/build_step18_manifest.py --root /home/eugen/projekte/vBuf
    --source /home/eugen/projekte/vBuf/.qualification-iq2xxs/DeepSeek-V2-Lite.IQ2_XXS.gguf
    --output-dir /tmp/opencode/deepseek-v2-lite-imat
    --manifest /tmp/opencode/deepseek-v2-lite-imat/DeepSeek-V2-Lite.IQ2_XXS-manifest.json
  python3 scripts/convert_gguf_to_vbuf_ml.py
    /home/eugen/projekte/vBuf/.qualification-iq2xxs/DeepSeek-V2-Lite.IQ2_XXS.gguf
    /tmp/opencode/deepseek-v2-lite-imat/DeepSeek-V2-Lite.IQ2_XXS-manifest.json
    /tmp/opencode/deepseek-v2-lite-imat/DeepSeek-V2-Lite.IQ2_XXS.vbuf
    --integrity none --evidence-dir /tmp/opencode/deepseek-v2-lite-imat/evidence
  cargo run --quiet --manifest-path rust/Cargo.toml -p vbuf-ml
    --bin vbuf-ml-semantic-bootstrap --
    /tmp/opencode/deepseek-v2-lite-imat/DeepSeek-V2-Lite.IQ2_XXS.vbuf
    /tmp/opencode/deepseek-v2-lite-imat/DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf
    http://127.0.0.1:18124/DeepSeek-V2-Lite.IQ2_XXS.vbuf
PAYLOAD_OUTPUT: /tmp/opencode/deepseek-v2-lite-imat/DeepSeek-V2-Lite.IQ2_XXS.vbuf
PAYLOAD_SIZE: 5639819878
PAYLOAD_SHA256: 2ef0cdde67154ecc68bd82558007a4ea9da36262c4405007cb83306f68456e47
HISTORICAL_PAYLOAD_SHA256: 2ef0cdde67154ecc68bd82558007a4ea9da36262c4405007cb83306f68456e47
PAYLOAD_IDENTITY: EXACT_IDENTITY_MATCH

SEMANTIC_OUTPUT: /tmp/opencode/deepseek-v2-lite-imat/DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf
SEMANTIC_SIZE: 3206424
SEMANTIC_SHA256: 668bf438b7f170d885eff987d7796312d078a026ffbd65998ea17d5a28e3ab9c
HISTORICAL_SEMANTIC_SHA256: 639ac345136de7f3d36c8fea15a8bf7fca70d3d518915a3371a9bd9cc02df910
SEMANTIC_HASH_RELATIONSHIP: STRUCTURALLY_REGENERATED; BYTE_HASH_DIFFERS
```

The payload `SourceHash` is the authoritative artifact identity and matches
exactly. The semantic bootstrap has the recorded historical size and the
same separate external-source structure, but its historical byte hash is not
reproduced by the current generator; that distinction is retained rather than
silently claiming semantic byte identity.

The unchanged range server was started with:

```text
python3 scripts/range_server.py --file /tmp/opencode/deepseek-v2-lite-imat/DeepSeek-V2-Lite.IQ2_XXS.vbuf --port 18124 --log
```

An HTTP range probe returned `206 Partial Content`,
`Content-Range: bytes 0-15/5639819878`, 16 bytes, and byte-exact payload
content. Android deployment was then blocked when the previously connected
wireless ADB endpoint `192.168.188.33:46013` began refusing connections. The
semantic bootstrap was not copied to the device, `adb reverse` was not
restored, and no physical 512 MiB run was started. No substitute model,
prompt, or source was used; the former Android failure remains physically
unconfirmed and no new downstream physical failure is claimed.

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
