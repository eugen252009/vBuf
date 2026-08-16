# Multi-Source Parallel Range Materialization POC9

## Environment

- Architecture: x86_64
- ggml: `2d191b5dee1a591c41ee8a653ce42bfcd9c8716`
- `CPU_REPACK=OFF`
- CUDA: OFF
- Artifact: `DeepSeek-V2-Lite.IQ1_S.vbuf`
- Tensor: `blk.0.ffn_down.weight`, TensorRef `3`
- Tensor range: `[83518776, 96126264)`
- Tensor length: `12607488`
- Expected payload hash: `0c7bdf85b162206b`

## API and Partition

`RangeStripingPlan` is separate from `SourceSelectionPolicy`. It contains two
`RangeStripe` entries, each with a source, source offset, destination offset, and
length. `two_way()` creates a deterministic 50/50 split aligned to 32 bytes.

The qualified plan was:

```text
A: source [83518776, 89822520), destination [0, 6303744)
B: source [89822520, 96126264), destination [6303744, 12607488)
```

The plan validator proves no gap, no overlap, exact source offsets, and union
equal to the complete tensor range. The split is transport-only; the final
payload is one contiguous existing tensor view and no format change was made.

## Ownership and Concurrency

`ParallelRangeMaterializer` allocates exactly one aligned destination buffer.
Each worker writes only its assigned destination interval. Small HTTP/library
buffers are permitted, but there is no full-stripe or full-tensor concatenation
buffer. The completed `MaterializedTensor` is handed to the existing
`ResidentTensorMaterializer` and `TensorResidencyStore` unchanged.

The two request intervals overlapped for `18.485091 ms`. Both workers started
before either completed. Both returned HTTP 206 and exact assigned lengths.

## Transfer Results

- Single source A end-to-end graph time: `94.115 ms`.
- Single source B end-to-end graph time: `65.236 ms`.
- Striped cold end-to-end graph time: `46.870 ms`.
- Warm striped graph time: `42.764 ms`.
- Speedup versus fastest single-source graph run: `1.392x`.
- A effective first-byte-to-complete rate: approximately `221.79 MB/s`.
- B effective first-byte-to-complete rate: approximately `313.90 MB/s`.
- Striped A rate: approximately `348.70 MB/s`.
- Striped B rate: approximately `336.74 MB/s`.
- Aggregate striped rate over the striped wall interval: approximately `501.33 MB/s`.
- Total received bytes: `6303744 + 6303744 = 12607488`.

These are loopback qualification measurements, not a claim of model or network
throughput. The aggregate is below an ideal 2x result.

## Residency and Parity

Cold lifecycle: `MISS -> STRIPED_MATERIALIZE -> INSERT -> LEASE_ACQUIRE -> LEASE_RELEASE`.
Warm lifecycle: `HIT -> LEASE_ACQUIRE -> LEASE_RELEASE`.

The warm run produced no stripe events and no network reads. Source selection is
not invoked for the warm hit. The striped payload hash was
`0c7bdf85b162206b`, and the cold and warm output hashes were both
`4a6526893cc2f652`.

## Failure Behavior

If either stripe fails, the materializer joins both workers, discards the
incomplete destination, and remains non-READY. The contract suite covers
source-A failure, source-B failure, and both-source failure. Existing execution
fallback behavior remains available after striped materialization failure.

## Teardown

- Stripe workers joined: PASS
- In-flight materialization after teardown: `0`
- Execution leases after graph: `0`
- Resident buffers after teardown: `0`
- Full CTest: `10/10 PASS`
- RV2 transport striping: not executed
