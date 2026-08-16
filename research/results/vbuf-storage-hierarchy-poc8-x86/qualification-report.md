# Storage Hierarchy / RAM Residency POC8

## Environment

- Architecture: x86_64
- ggml: `2d191b5dee1a591c41ee8a653ce42bfcd9c8716`
- `CPU_REPACK=OFF`
- CUDA: OFF
- Fixture: `DeepSeek-V2-Lite.IQ1_S.vbuf`
- Tensor: `blk.0.ffn_down.weight`, TensorRef `3`
- Range: offset `83518776`, length `12607488`
- Payload hash: `0c7bdf85b162206b`

## Architecture

`TensorResidencyStore` owns immutable persistent tensor buffers at TensorRef
granularity. `ResidentTensorMaterializer` wraps the existing materializer and
retains its owned buffer through a shared lease; it does not copy payload bytes.
The store's `LEASE_ACQUIRE` and `LEASE_RELEASE` events are execution leases.
`resident_bytes` remains unchanged after lease release.

The budgeted policy is deterministic oldest-use-first eviction, skipping entries
with active leases. The qualification budget was `12607488` bytes, so only
`ffn_down.weight` remained resident. This is partial, irregular FFN residency,
not a layer or region owner.

## Cold/Warm Local

- Source-selection calls: 1 for the two-run qualification.
- Run 1: `MISS -> MATERIALIZE(local-vbuf) -> INSERT(12607488) -> LEASE_ACQUIRE -> LEASE_RELEASE`.
- Run 2: `HIT -> LEASE_ACQUIRE -> LEASE_RELEASE`.
- Local materialization events: one request; no second `RangeSource` read.
- Execution: cold `46.905 ms`; warm `10.292 ms`.
- Both outputs: exact output hash `4a6526893cc2f652`.

## Cold/Warm HTTP

- Source-selection calls: 1 for the two-run qualification.
- Run 1: one HTTP `206` exact range `83518776-96126263/4993331814`, then insert.
- Run 2: RAM `HIT -> LEASE_ACQUIRE -> LEASE_RELEASE`; no HTTP request.
- Source provenance is diagnostic only; execution consumed the same resident RAM entry.
- Execution: cold `87.965 ms`; warm `26.392 ms`.
- Both outputs: exact output hash `4a6526893cc2f652`.

The warm resident lookup/acquire path was sub-microsecond in the trace
(`110 ns` local, `490 ns` HTTP); warm consumer wait was `0`. Cold consumer wait
uses the already-qualified POC6 measurements: `15.572810 ms` local and
`38.879683 ms` HTTP.

## Eviction and Failure Contracts

The residency contract test proves duplicate insertion rejection, insertion
rejection when the only entry is actively leased, deterministic eviction after
lease release, and teardown. Existing materializer contracts retain source
failure behavior. No actively leased entry is evicted.

## Accounting

- Peak resident weight bytes: `12607488`.
- Stable resident bytes after each cold/warm run: `12607488`.
- Active leased bytes after each graph: `0`.
- Resident resources after teardown: `0`.
- Materialization buffer ownership is handed to the residency entry through the
  existing shared storage lease; no tensor-sized payload copy was introduced.

## Results

- Cold local parity: PASS
- Warm local parity: PASS
- Cold HTTP parity: PASS
- Warm HTTP parity: PASS
- Tensor-granular partial FFN residency: PASS
- Source-independent execution: PASS
- Full CTest: 9/9 PASS
- RV2 compute: NOT_EXECUTED; existing pinned ggml RVV FP16 toolchain blocker remains unchanged.
