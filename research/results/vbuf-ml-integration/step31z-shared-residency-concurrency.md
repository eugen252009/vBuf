# Step 31Z Shared Residency Concurrency Qualification

Status: **X86_64 SHARED IMMUTABLE RESIDENT-WEIGHT READERS QUALIFIED FOR THE
TESTED RESIDENCY SEAM; 512-MIB CAP QUALIFIED; PRODUCTION SERIAL POLICY
PRESERVED**.

Target: Linux x86_64 workstation. Android, ADB, APK, external hardware, and
RISC-V execution were not required or used.

The available GGML checkout was
`a97123e497968f3440264c0464a7adc7c999c027`. The canonical pinned revision
remains unavailable and was not changed. The production server remains
serial:

```text
MAX_ACTIVE_GENERATIONS = 1
```

No server slots, batching, interleaving, KV sharing, backend changes, format
changes, or production parallel scheduling were introduced.

## Qualification Boundary

The research probe creates two independent `VbufGenerationSession` objects.
Each session retains private source, HTTP connection, materializer backing,
request maps, KV state, and backend execution contexts. The sessions share
only one `TensorResidencyStore`, which owns immutable materialized payload
views and lease counts. This qualifies the shared immutable residency seam; it
does not qualify shared source or shared materializer ownership, nor concurrent
calls on one session.

The low-level contract covers cold same-key readers, cold different-key
readers, release ordering, eviction pressure while leased, rewarm after
eviction, and payload byte/pointer identity. Cold same-key readers may perform
one or two backing materializations under the intentionally separate
materializer backings, but only one resident payload is retained.

## Implementation Boundary

The qualification changes are limited to the vBuf-ML materialization and
residency seam:

- residency maps, counters, traces, leases, eviction, and clear operations are
  mutex-protected;
- resident acquisition returns a value-owned materialized view while holding
  a store lease;
- an already-resident request acquires its lease before `wait()` so another
  reader cannot evict it between request and obtain;
- request records distinguish backing requests from shared-residency leases;
- tensor-wave execution tracks materializer leases across the whole wave and
  releases all remaining leases on exceptional exits;
- generation boundaries release stale request-local backing state while
  preserving resident payloads;
- a research-only shared-residency contract is registered with CTest.

The legacy raw-pointer `lookup()` and `peek()` APIs remain compatibility
interfaces. The canonical runtime path uses value-returning acquisition and
does not retain raw store pointers across lock boundaries.

## Measurements

Workload: two blocks, `NormalInference`, prompts `Say hi` and `Count to one`,
eight generated tokens, one warmup per session, and three measured A/B pairs in
each fresh process. Prompt hashes were `232a1eb1cb807d1b` and
`b0b030c4617b9bde`; all qualified modes returned the same token hashes:

```text
Say hi       -> 9b2aafabc894a70e
Count to one -> 1c9544a3bd9109be
```

Residency capacity was `536870912` bytes. Serial, private isolated
concurrent, shared-serial, and shared-concurrent modes all completed with
token parity and zero post-run active leases, inflight bytes, and active
generations.

| Metric | Serial A/B | Private concurrent A/B | Shared concurrent A/B |
|---|---:|---:|---:|
| mean pair makespan | `8.737 s` | `4.924 s` | `4.507 s` |
| aggregate speedup vs serial | `1.000x` | `1.774x` | `1.938x` |
| shared individual latency factor vs serial | `1.000x` | not primary metric | `0.976x` |
| peak RSS | `1269452 KiB` | `1505264 KiB` | `936496 KiB` |
| peak threads | `5` | `10` | `10` |

The shared run observed real overlap (`shared_concurrent_pair_p50` was
`4.446 s`) and retained resident bytes under the 512-MiB cap while recording
evictions and reacquisitions. The private-concurrent versus shared-concurrent
peak RSS difference was `568768 KiB` in this fresh-process measurement.

Evidence is measured host evidence, not a general thread-safety proof. No
complete-stack ThreadSanitizer run was available. The raw result is:

```text
/tmp/opencode/vbuf-step31z-shared-residency.json
```

## Bounded-Cap Limitation

A `268435456`-byte exploratory run was not qualified. Under repeated shared
concurrent eviction pressure it intermittently reached the existing
single-backing materializer request-budget rejection path while another
per-session request was in flight. The observed result was a failed
qualification, not a claim of safe operation at that cap. The low-level
eviction/lease contract still passes at its bounded test capacity.

This distinction matters: the shared store is synchronized, but the current
per-session source/materializer path intentionally does not become a parallel
request scheduler. The next experiment is a separately measured 256-MiB
bounded-cap qualification after an explicit request-queue/backing-budget
contract is defined; that experiment must not change production admission.

## Reproduction

Build and test:

```text
cmake -S integrations/ggml -B /tmp/opencode/vbuf-step31u-host-build
cmake --build /tmp/opencode/vbuf-step31u-host-build --target vbuf_step31y_backend_concurrency vbuf_shared_residency_concurrency_contract -j2
ctest --test-dir /tmp/opencode/vbuf-step31u-host-build --output-on-failure
```

The research coordinator is
`integrations/ggml/tests/vbuf_step31z_shared_residency_test.py`. It runs the
serial, private-concurrent, shared-serial, and shared-concurrent modes and
writes JSON evidence. The low-level contract is
`integrations/ggml/tests/shared_residency_concurrency_contract.cpp`.
