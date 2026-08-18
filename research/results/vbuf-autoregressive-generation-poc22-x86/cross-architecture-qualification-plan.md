# Cross-Architecture Remote Bootstrap Qualification Plan

## Topology

```text
NAS / x86 host
  - authoritative model.vbuf
  - semantic-bootstrap.vbuf
  - dumb HTTP Range server
  - minimal result receiver

clients
  - x86_64
  - ARM32
  - riscv64
  - ARM64
```

The NAS/server remains model-agnostic. The client performs semantic discovery,
source resolution, exact range materialization, and local qualification.

The result receiver should remain dumb. Preferred persistence is append-only
NDJSON or one immutable JSON file per run. A database is not required.

## Self-Reporting Result

Conceptual endpoint:

```text
POST /result
```

Each immutable result should include:

```text
run_id
architecture
OS
model identity/hash
bootstrap identity/hash
source kind
SourceId
logical offset and length
payload/result hashes
numeric metrics
timestamps
```

The result schema must preserve source provenance and distinguish discovery,
range, materialization, FFI, and backend gates.

## Test Ladder

Each device stops at the first unsupported layer:

1. generic semantic-bootstrap parse;
2. discovery digest;
3. exact TensorRef;
4. real >4 GiB logical offset;
5. local file range;
6. HTTP Range;
7. payload hash;
8. source-independent materialization;
9. compute kernel where the backend exists;
10. optional token-0/position-0 path.

The first eight gates are storage/source plumbing. Backend or compiler failure
must be recorded separately and must not invalidate earlier portability gates.

## Architecture Checks

The next implementation must retain:

```text
FFI_POINTER_WIDTH_DEPENDENCY_FOR_LOGICAL_OFFSET: NO
LOGICAL_OFFSETS_CROSS_FFI_AS_U64: YES
MATERIALIZED_LOCAL_LENGTH_FITS_USIZE_CHECKED: YES
```

No ARM32, RISC-V, ARM64, Android, SSH, NAS deployment, result receiver, or
new ggml backend work is performed in this milestone.
