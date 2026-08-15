# vBuf Cold-Load Throughput Qualification

Payload: `34.812 GB` (`4443880`..`34816197376`)
Storage classification: **STORAGE_BASELINE_NOT_REPRODUCED**

## Main Answer
The claimed 3.2 GB/s cold baseline was not reproduced in this run; full physical reads measured substantially lower for both dd and exact-range readers, so a 2.2x loader gap cannot yet be attributed to vBuf software.

## Best Controls
- Best dd: `1.359 GB/s`
- Direct-I/O dd: `1.802 GB/s` at 32 MiB
- Best exact-range reader: `1.410 GB/s`, `4` workers, `16777216` bytes
- Expected full-payload time at best reader: `24.70 s`
- Single-reader 32 MiB control: approximately `1.34 GB/s`, `1038` reads
- RAM-destination 32 MiB control: `1.085 GB/s`
- S4 corrected cap wait: `0 ms` / `0` worker sleep events; gate wait remained compute-side

## Root-Cause Evidence
- All exact-range controls read the complete `34,811,753,496` payload bytes.
- Physical `read_bytes` was approximately the full payload, so these were not warm-cache results.
- Chunk sizes from 256 KiB through 64 MiB stayed near 1.34 GB/s for one worker.
- Worker counts 1/2/4/8 produced approximately 1.34/1.19/1.41/1.36 GB/s.
- The claimed 3.2 GB/s device baseline was not reproduced; the limiting cause therefore remains unisolated rather than being assigned to vBuf bookkeeping or caps.

## Classifications
- Storage ceiling: `STORAGE_BASELINE_NOT_REPRODUCED`
- Existing loader: `EXISTING_LOADER_NEAR_DEVICE_LIMIT`
- Root cause: `ROOT_CAUSE_NOT_ISOLATED`
- Best isolated strategy: `MULTIWORKER_READER`
- End-to-end: `HORIZON_PREFETCH_BEST` in the existing one-sample S4 comparison, but the corrected S4 run is reported separately in `end-to-end-strategies.csv`
- Final: `LOADER_HAS_MINOR_HEADROOM`
- Recommendation: `KEEP_CURRENT_LOADER`

The 3.2 GB/s claim requires a repeat with the requested `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` method or equivalent validated device state.
