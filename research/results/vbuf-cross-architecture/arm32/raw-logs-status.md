# ARM32 Raw Evidence Status

## Backed Up

- `device-baseline.txt`: direct target capture including CPU, memory, model, kernel, mounts, and network state.
- `gt4gib-range.headers`: direct HTTP HEAD and `206 Content-Range` response headers.
- `scaffolding/arm32_bootstrap_qualification.rs`: exact temporary semantic/FFI qualification source.
- `scaffolding/vbuf-arm32-range-server.py`: exact controlled Range server implementation.
- `/home/eugen/arm32-qualification-backup/arm32_bootstrap_qualification`: ARM32 release binary, with size and SHA-256 recorded below.
- `/home/eugen/arm32-qualification-backup/qwen32.semantic.vbuf`: exact bootstrap artifact, with size and SHA-256 recorded below.

## Not Originally Persisted

The original Cargo test stdout and the first semantic qualification stdout were
printed in the SSH session but not redirected to target files. They are not
reconstructed here. The durable JSON preserves their measured pass counts and
outputs, and the target binary/source, range headers, and hashes are preserved
for honest rerun.

Therefore:

```text
RAW_QUALIFICATION_LOGS_BACKED_UP: PARTIAL

## Follow-up ABI Fix

The initial ARM32 run correctly recorded the committed `consumer_ffi` harness
as blocked by hard-coded `i8` buffers. The generic follow-up changed four test
buffers to `core::ffi::c_char`; no production FFI declaration changed. The
fixed committed test then passed on both x86_64 and ARM32:

```text
x86_64 consumer_ffi: 2 passed
ARM32 consumer_ffi: 2 passed
ARM32 external_source: 5 passed
ARM32 range_loading: 4 passed
ARM32 source_profile: 2 passed
```
```

## Durable Hashes

```text
arm32_bootstrap_qualification
  bytes: 467212
  sha256: 7f38878a86e01e98c2f22ac002bddd3d889ee9f44fef7f61f97b71693db01ab0

qwen32.semantic.vbuf
  bytes: 4472327
  sha256: dc3b0c755b5bbb4949ca813131c3a74bd33ea21368c5ac42c59feb3adfdeb278

arm32_bootstrap_qualification.rs
  bytes: 5017
  sha256: 2b2b3db29718883186480dd5b9d6ffe01c3b2716fa4a832561a3584b66965613

vbuf-arm32-range-server.py
  sha256: c69d9b42b976811a50937b0c489c8fb2449b3c50eace85a144df3f1b237fc254
```
