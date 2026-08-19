# Android ARM64 Phase C

Phase C characterizes remote model-open performance from the Phase B baseline at
commit `2905b88`. All measurements use the Pixel 7 Pro, Android 17, arm64-v8a,
the local semantic bootstrap, the full artifact absent from the device, and
direct Wi-Fi HTTP Range transport. The experiment changes one transport/planning
variable at a time and does not alter vBuf formats, vbuf-ML semantics, model
semantics, or the pinned llama.cpp construction behavior.

| Variant | Concurrency | Requests | Connections | Bytes | Overfetch | Median open | Effective MB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase B baseline | 1 | 310 | 310 | 633495552 | 0 | 53203 ms | 11.907 |
| C1 keep-alive | 1 | 310 | 1 | 633495552 | 0 | 44451 ms | 14.252 |
| C2 zero-gap | 1 | 310 | 1 | 633495552 | 0 | 44451 ms | 14.252 |
| C3 4 KiB gap | 1 | 1 | 1 | 633499592 | 4040 | 81905 ms | 7.735 |
| C3 64 KiB gap | 1 | 1 | 1 | 633499592 | 4040 | 85314 ms | 7.425 |
| C3 256 KiB gap | 1 | 1 | 1 | 633499592 | 4040 | 84321 ms | 7.513 |
| C3 1 MiB gap | 1 | 1 | 1 | 633499592 | 4040 | 84721 ms | 7.478 |
| C4 concurrency 2 | 2 | 310 | 2 | 633495552 | 0 | 45384 ms | 13.959 |
| C4 concurrency 4 | 4 | 310 | 4 | 633495552 | 0 | 46083 ms | 13.764 |
| C4 concurrency 8 | 8 | 310 | 8 | 633495552 | 0 | 46709 ms | 13.580 |

## C0 Transport Characterization

The raw Pixel-to-server large-range median was `29.763 MB/s` for a 256 MiB
range. The complete Phase B model-open path achieved `11.907 MB/s` for
`633,495,552` bytes, only `0.400` of that synthetic transport ceiling. The
large-range result is a transport characterization, not the achievable
complete model-open throughput.

The baseline performed 310 logical range requests over 310 separate HTTP
connections. This established connection/request overhead as a material part
of the remaining load time before any planning change was attempted.

## Selection Evidence

- Connection reuse saved `8752 ms` at the median, a `16.450%` reduction versus Phase B.
- Zero-gap coalescing changed nothing because the TensorRef gaps are `16` bytes, not zero.
- Positive-gap coalescing reduced the request count to one, but introduced `4040` bytes of overfetch and regressed open time to approximately `82-85 s`; it was not retained.
- Bounded concurrency was measured only after batching was fixed. Concurrency 2, 4, and 8 all regressed versus C1, so it was not retained and is not a viable production alternative under this architecture.

The only retained production optimization is C1 persistent HTTP connection reuse
in the generic `HttpRangeSource`. It keeps all 310 logical ranges, changes the
connection count from 310 to 1, preserves zero overfetch, and reduces median
model open from `53.203 s` to `44.451 s` (`1.197x`, `16.450%`).

## Final Qualification

Correctness gates for the selected C1 configuration: all 310 tensor payloads
matched byte-for-byte, remote 16-token generation passed, and local SELF
regression passed. Rust tests, local and remote Android builds, JSON validation,
the code index, and diff checks passed. A separate final-state observation
opened the model in `43848 ms`; the reported Phase C benchmark remains the
five-run median `44451 ms`. The final Pixel state contains only the semantic
bootstrap, whose SHA-256 matched the Phase B artifact.

## Stopping Boundary

The backend residency policy remains `EAGER_ALL`. The full model artifact is not
copied to the Pixel, but the selected path still transfers and retains almost
all tensor payload bytes during pinned llama.cpp model construction. Phase C
therefore stops at transport connection reuse. The dominant remaining cost is
serialized per-tensor HTTP/materialization overhead together with the pinned
`EAGER_ALL` construction path. Residency redesign, lazy backing, eviction,
prefetch, batching policy, and further concurrency work require a later phase
with an explicitly changed architecture; none is part of this result.
