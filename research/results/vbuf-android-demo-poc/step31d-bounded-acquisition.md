# Step 31D Bounded Demand-Driven Acquisition

Date: 2026-08-20
Starting commit: `e67c352687b3c96beec26d25f02f6d45d885c7e3`
Status: **IMPLEMENTED / HOST ANDROID QUALIFIED**

## Objective

Reduce upstream request amplification below `ProgressiveRangeSource` without
changing its consumer contract, 4 KiB coverage authority, source identity, or
any materialization, residency, backend, or inference behavior.

## Selected Policy

For each consumer `read_range`, the source finds contiguous missing 4 KiB
chunks within the requested interval. Each run is split into windows no larger
than `1 MiB` (`256` chunks). A window is fetched once, written at its canonical
offset, and only then publishes its individual complete-chunk coverage bits.
Covered chunks are never fetched as coalescing gaps. Future ranges are not
prefetched.

The source reports consumer requests, requested bytes, upstream remote requests
and bytes, acquisition-window requests and bytes, local hits, misses, and
published coverage separately.

## Host Evidence

The deterministic progressive-source contract passes with assertions enabled in
the focused Debug build. It covers:

- bounded `1 MiB` windows over a 700-chunk request;
- partial local coverage and disjoint missing runs;
- exact warm reuse and final-EOF behavior;
- remote failure and truncated response without publication;
- deterministic local mirror write failure without publication;
- identity mismatch, reopen, concurrent overlap, and invalid-store fail-closed behavior.

The Release native build completed all 20 registered CTest tests. The full
Debug suite is not a valid repository baseline because five unrelated existing
contracts assert behaviors that are only configured to pass with `NDEBUG`; the
focused progressive-source contract passes in Debug with its assertions active.

## Trace Replay

Input: `/tmp/opencode/step31a-source-range-audit.tsv`, containing the measured
4,287 semantic requests from Step 31A. This is an offline simulation, not a
network measurement.

| Acquisition policy | Upstream requests | Rounded bytes | Request reduction |
|---|---:|---:|---:|
| Existing one-chunk acquisition | 336,990 | 1,380,311,040 | baseline |
| 64 KiB bounded windows | 21,654 | 1,380,311,040 | 93.574% |
| 256 KiB bounded windows | 5,813 | 1,380,311,040 | 98.275% |
| **1 MiB bounded windows** | **1,850** | **1,380,311,040** | **99.451%** |
| 4 MiB bounded windows | 1,282 | 1,380,311,040 | 99.620% |

The bytewise range union was `1,377,067,264` bytes. The 1 MiB policy therefore
adds `3,243,776` bytes of chunk-boundary rounding in this trace, approximately
`0.236%` over the bytewise union. The replay does not establish throughput,
latency, server behavior, or mobile energy use.

## Architecture Boundary

```text
TensorRef
  -> existing RangeSource/materializer boundary
  -> ProgressiveRangeSource
       -> bounded missing-chunk acquisition
       -> sparse canonical mirror + authoritative 4 KiB bitmap
  -> existing materializer/residency/GGML path
```

`TensorRef`, source identity, materialization, leases, residency, batching,
GGML execution, and inference semantics are unchanged. The implementation does
not add retention, eviction, offline completion, prefetch, or background work.

## Android Build Qualification

The documented shell environment resolved the previously false discovery
blocker. The known SDK, NDK, SDK CMake 3.22.1, JDK, GGML checkout, Gradle
wrapper, and Pixel 7 Pro were present. The ARM64 Rust library rebuilt
successfully with the installed NDK linker
`aarch64-linux-android29-clang`; the bare Cargo invocation had selected host
`cc` and failed with the concrete wrong-format `EM: 183` linker error before
the linker was supplied. No source or toolchain version change was made.

The APK build passed with the documented GGML source and remote payload
properties:

```text
ARM64 vbuf-ml build: PASS
Android assembleDebug: PASS
GGML: /home/eugen/projekte/llama.cpp/ggml
Remote URL: http://127.0.0.1:18124/models/DeepSeek-V2-Lite.IQ2_XXS.vbuf
```

## Physical Qualification

Device: Pixel 7 Pro, Android 17, `arm64-v8a`. The semantic bootstrap remained
on device. The Step 31D cold control began with no payload mirror or coverage
sidecar; the range server returned valid HTTP 206 responses for the intended
`5,639,819,878`-byte IQ2_XXS payload. The runtime remained
`RuntimeMode::NormalInference`, batched prefill, four bounded decode tokens,
and a `268,435,456`-byte residency cap.

Coverage advanced from 3,581 to 336,990 chunks during the cold run. The run
completed normally. The process was restarted for warm qualification without
deleting the mirror or sidecar, and the identical prompt/workload completed
normally again.

### Measured Cold/Warm Metrics

| Metric | Cold | Warm |
|---|---:|---:|
| Consumer requests | 4,287 | 4,287 |
| Consumer requested bytes | 4,745,501,920 | 4,745,501,920 |
| Upstream remote requests | 1,850 | 0 |
| Upstream remote bytes | 1,380,311,040 | 0 |
| Acquisition windows | 1,850 | 0 |
| Acquisition window bytes | 1,380,311,040 | 0 |
| Local source bytes | 4,745,501,920 | 4,745,501,920 |
| Local chunk hits | 825,956 | 1,162,946 |
| Remote chunk misses | 336,990 | 0 |
| Covered chunks after run | 336,990 | 336,990 |
| Covered rounded payload bytes | 1,380,311,040 | 1,380,311,040 |
| Coverage bitmap bytes | 172,114 | 172,114 |
| Prefill | 179,800 ms | 67,329 ms |
| Decode token 1 | 26,087 ms | 24,094 ms |
| Decode token 2 | 25,958 ms | 25,482 ms |
| Decode token 3 | 30,794 ms | 27,711 ms |
| Decode token 4 | 26,788 ms | 25,174 ms |
| Mean decode | 27,408 ms | 25,616 ms |
| Decode total | 109,632 ms | 102,465 ms |
| Total generation | 289,443 ms | 169,806 ms |
| Attention | 41,735 ms | 30,381 ms |
| FFN | 234,585 ms | 133,824 ms |
| Output head | 12,491 ms | 5,354 ms |
| Peak resident bytes | 267,452,416 | 267,452,416 |

Both runs produced the exact output `----`. The cold-to-warm upstream request
and remote-byte reductions were physically measured at 100%. Prefill improved
by 62.553% and decode by 6.537%. These timers include their existing compute
and materialization scopes; they do not isolate source-only latency. FFN/MoE
compute remained the dominant measured warm generation cost.

The range-server log contained one additional 4 KiB HTTP probe before the cold
run; application metrics above exclude that probe. The server observed the
same 1,850 cold acquisition requests and no warm requests from the app.

## Deferred

Crash/power-loss durability, retention, offline completion, complete-artifact
verification, multi-source orchestration, residency tuning, and backend tuning
remain outside this step.
