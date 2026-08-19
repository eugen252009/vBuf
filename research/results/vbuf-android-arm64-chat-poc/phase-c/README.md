# Android ARM64 Phase C

Phase C measures remote-load transport/planning changes from the Phase B
baseline at commit `2905b88`. All measurements use the Pixel 7 Pro, Android 17,
arm64-v8a, the local semantic bootstrap, the full artifact absent from the
device, and direct Wi-Fi HTTP Range transport.

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

## Attribution

- Connection reuse saved `8752 ms` at the median, a `16.450%` reduction versus Phase B.
- Zero-gap coalescing changed nothing because the TensorRef gaps are `16` bytes, not zero.
- Positive-gap coalescing reduced request count but regressed open time and increased peak RSS, so it was not retained.
- Bounded concurrency was measured only after batching was fixed. Concurrency 2, 4, and 8 all regressed versus C1, so no concurrency code was retained.

The selected production change is C1 persistent HTTP connection reuse in the
generic `HttpRangeSource`. EAGER_ALL remains unchanged. No vBuf format,
vbuf-ML semantic, llama.cpp model-construction, tokenizer, or generation
behavior was changed.

Correctness gates for the selected C1 configuration: all 310 tensor payloads
matched byte-for-byte, remote 16-token generation passed, and local SELF
regression passed. The final Pixel state contains only the semantic bootstrap.
