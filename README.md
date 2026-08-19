# vBuf

vBuf is a self-navigating binary substrate for validated persistent structure
whose payload source, residency, and execution representation can remain
independent runtime concerns. It is designed so a reader can validate and
navigate canonical structure without a mandatory global index, while a runtime
can materialize only the ranges currently required by a consumer.

vBuf itself is a generic binary substrate. [`vbuf-ML`](rust/vbuf-ml/) is an
optional semantic/profile layer for model metadata, tokenizers, tensor
directories, and external tensor sources. vBuf is not defined by machine
learning or by any particular backend such as llama.cpp or ggml.

## Core Model

The architecture keeps these concepts separate:

```text
semantic tensor identity
        !=
metadata artifact
        !=
payload source
        !=
payload residency
        !=
backend execution representation
```

Likewise, these sizes are different measurements:

```text
persistent model size
        != bootstrap transfer size
        != local storage requirement
        != process virtual address space
        != resident working set
        != active compute memory
```

The common external-source path is:

```text
semantic-bootstrap.vbuf
        |
        v
normal vbuf-ML discovery
        |
        v
TensorRef(SourceId, u64 offset, u64 length)
        |
        v
SourceSet / RangeSource
        |
        v
file or HTTP Range
        |
        v
bounded materialization
        |
        v
pointer + length + provenance + lease
        |
        v
source-agnostic backend or compute
```

The local path remains supported and keeps its borrowed fast path:

```text
full model.vbuf
        |
        v
SELF source -> mmap -> CheckedRange -> borrowed payload
```

Remote/bootstrap loading is therefore not a replacement for local mmap loading.
They are two source and residency strategies under the same semantic model.

## Design Principles

- Canonical structure is self-navigating: the next block follows checked geometry, not a required index.
- Wire offsets and arithmetic stay in the checked `u64` domain until a host pointer or allocation is explicitly created.
- Alignment is explicit and orthogonal to payload representation.
- Persistent identity is independent of whether bytes are mapped, borrowed, cached, or materialized.
- Source identity is separate from source location. A `SourceId` identifies the authoritative byte artifact; a locator describes where bytes may be fetched.
- Materialization is bounded and lazy. A remote logical offset may exceed 4 GiB while the local materialized span remains checked against the host `usize`.
- Backend representations may differ from persistent representations without changing the vBuf artifact.
- Existing payload-bearing artifacts remain usable through the `SELF` source default.

## vBuf v0.6

The stable generic wire contract is [`spec/spec_0.6.md`](spec/spec_0.6.md).
Its high-level rules are:

- little-endian fields and payload encodings;
- version code `0x00060000`;
- minimum global header size of 24 bytes;
- `BaseShift` in `3..=8`, giving `BaseStep` values from 8 through 256 bytes;
- `PayloadAlignment = 1 << (BaseShift + PayloadShift)`;
- `next_block_start = align_up_checked(payload_end, BaseStep)`;
- checked `u64` arithmetic for offsets, sizes, alignment, and conversions;
- no required index, directory, checkpoint, checksum, or finalization artifact.

Optional physical structures may accelerate a particular workload, but they are
not normative v0.6 requirements. In particular, Nano is an optional physical
or topological acceleration structure, not a semantic index and not a
guaranteed speedup.

## vbuf-ML Profile

vbuf-ML adds semantic interpretation without creating a second container
format. Its additive source profile uses:

```text
RegionRole::SourceMetadata = 7
```

The profile can persist:

- `SourceDescriptor` and `SourceId`;
- file, HTTP, or other `SourceLocator` values;
- optional binary source hashes, with zero or multiple hashes allowed;
- external tensor bindings represented as checked `TensorRef` values.

Older payload-bearing artifacts without this optional role remain compatible and
resolve their tensor payloads through `SELF`.

The vbuf-ML implementation and focused profile documentation are in
[`docs/vbuf-ml/`](docs/vbuf-ml/) and [`rust/vbuf-ml/`](rust/vbuf-ml/).

## Semantic Bootstrap

A structural header-only derivative and a semantic bootstrap have different
purposes. The structural derivative measures a lower bound for generic
navigation. A semantic bootstrap contains the metadata needed for complete
vbuf-ML discovery while copying zero tensor payload bytes.

| Artifact | Full artifact | Structural derivative | Semantic bootstrap |
|---|---:|---:|---:|
| Qwen3-0.6B-Q8_0 | 637,925,504 bytes | 24,288 bytes | 4,438,480 bytes |
| Qwen3-32B-Q8_0 | 34,816,197,376 bytes | 52,872 bytes | 4,472,327 bytes |

The approximately 53 KiB 32B result is therefore a structural lower bound, not
a complete semantic model bootstrap. The 32B bootstrap contains complete
bootstrap, model metadata, tensor directory, tokenizer, and source metadata;
its tensor payload bytes copied during discovery are zero.

## Cross-Architecture Qualification

The same 32B semantic bootstrap was parsed on four real architectures. The
shared normalized discovery digest is
`f731868eafcd1830244c4dbb28c198b13a204e468adb1d89859c127eccce1f3f`.

| Capability | x86_64 | ARM32 / ARMv7 | ARM64 / AArch64 | riscv64 |
|---|---:|---:|---:|---:|
| Generic vBuf parse | PASS | PASS | PASS | PASS |
| Semantic bootstrap parse | PASS | PASS | PASS | PASS |
| Discovery digest parity | PASS | PASS | PASS | PASS |
| u64 TensorRef | PASS | PASS | PASS | PASS |
| Real >4 GiB tensor | PASS | PASS | PASS | PASS |
| HTTP Range source | PASS | PASS | PASS | PASS |
| File RangeSource | PASS | NOT_QUALIFIED | NOT_QUALIFIED | NOT_QUALIFIED |
| Exact payload hash parity | PASS | PASS | PASS | PASS |
| Bounded materialization | PASS | PASS | PASS | PASS |
| Committed FFI materialization | PASS | PASS | PASS | PASS |
| Real ggml descriptor | PASS | NOT_QUALIFIED | NOT_QUALIFIED | NOT_QUALIFIED |
| Bounded ggml compute | PASS | NOT_QUALIFIED | PASS (Android local) | NOT_QUALIFIED |
| Full external token path | NOT_RUN | NOT_QUALIFIED | NOT_QUALIFIED | NOT_QUALIFIED |

Hardware identities and detailed raw results:

- x86_64: development workstation. See [`research/results/vbuf-autoregressive-generation-poc22-x86/`](research/results/vbuf-autoregressive-generation-poc22-x86/).
- ARM32: Cubietech Cubietruck Plus, Allwinner A83T, `armv7l`, 32-bit userspace. See [`ARM32 evidence`](research/results/vbuf-cross-architecture/arm32/).
- ARM64: Google Pixel 7 Pro, GS201, Android 17, Termux, `aarch64`. See [`ARM64 evidence`](research/results/vbuf-cross-architecture/arm64/).
- riscv64: Orange Pi RV2, Ky X1 / `ky,x60`, Ubuntu 24.04.4, `riscv64`. See [`RISC-V evidence`](research/results/vbuf-cross-architecture/riscv64/).

The table proves storage, source, addressing, materialization, and committed
FFI portability. It does not imply that every architecture has a qualified
llama/ggml backend or token-generation path.

The ARM64 Android local-compute result is a separate Phase A proof: the Pixel 7
Pro opened the local Qwen3-0.6B vBuf artifact and completed bounded real
autoregressive generation. It does not qualify ARM64 remote/external token
generation; see [`Android Phase A evidence`](research/results/vbuf-android-arm64-chat-poc/).

The Android Phase B proof adds a direct Wi-Fi HTTP Range source: the Pixel 7 Pro
opened the Qwen3-0.6B semantic bootstrap without the full model artifact on the
device, materialized the external tensor payloads, and completed bounded real
autoregressive generation. Tokenizer text/type, merge, and all 310 tensor
payloads matched the direct source. The 53.203 s model open transferred
633,495,552 bytes in 310 HTTP Range requests under the pinned EAGER_ALL backend
construction behavior; remote transfer and residency are not optimized yet.
See [`Android Phase B evidence`](research/results/vbuf-android-arm64-chat-poc/phase-b-remote-chat.json).

### Android Remote Loading

Phase C measured remote-load transport without changing `EAGER_ALL` residency.
The large-range C0 transport characterization reached `29.763 MB/s`, while the
complete Phase B model-open path reached `11.907 MB/s`, or `40.0%` of that
synthetic ceiling.

| Measurement | Phase B baseline | Selected C1 keep-alive |
|---|---:|---:|
| Requests | 310 | 310 |
| Connections | 310 | 1 |
| Transferred bytes | 633,495,552 | 633,495,552 |
| Overfetch | 0 | 0 |
| Median model open | 53.203 s | 44.451 s |
| Effective throughput | 11.907 MB/s | 14.252 MB/s |

C1 is the only retained production optimization: it reuses one persistent
generic HTTP connection, giving a `1.197x` speedup and `16.450%` lower median
model-open time without changing payload semantics or persistent formats.
Payload parity, remote generation, and local/direct-source regression all
passed. Batching/coalescing and bounded concurrency were tested as selection
evidence but regressed and were not retained.

The full artifact is not copied onto the Pixel, but the backend remains
`EAGER_ALL`, so the selected path still transfers and retains almost all tensor
payload bytes during model construction. The remaining bottleneck is therefore
serialized per-tensor HTTP/materialization overhead combined with the pinned
llama.cpp construction path, not raw HTTP bandwidth. Detailed results are in
[`Android Phase C evidence`](research/results/vbuf-android-arm64-chat-poc/phase-c/).

## Shared 32B Proof

The authoritative source is `Qwen3-32B-Q8_0.vbuf`,
`34,816,197,376` bytes, SHA-256
`84597064d5b3530572959345286368b17e891e640b5958bba7f0b67980cd119d`.

The shared semantic bootstrap is `4,472,327` bytes, SHA-256
`dc3b0c755b5bbb4949ca813131c3a74bd33ea21368c5ac42c59feb3adfdeb278`.

All four architectures resolved the same real tensor:

```text
ordinal:          22
SourceId:         1
logical offset:   6,056,603,320
logical length:   5,570,560
end exclusive:    6,062,173,880
representation:   GGML_Q8_0
dimensions:       [5120, 1024]
payload SHA-256:  bfb4d69b99058c659ce075ec794b5d923b8a513ab6b7745a042383aa8dac13bf
```

The same discovery digest, exact range, and payload hash matched across all
four qualified architectures. ARM32, ARM64, and riscv64 completed this proof
without possessing, mapping, or downloading the full model.

## Performance Snapshot

These are directly measured ARM64 and riscv64 results. TensorRef lookup is a
hot-loop microbenchmark, not end-to-end latency. ARM64 source transfer used
ADB reverse HTTP and is not directly comparable to riscv64 direct LAN HTTP.

| Target | Semantic discovery median | TensorRef lookup | Local materialization | Source transfer |
|---|---:|---:|---:|---:|
| ARM64 | 9.344 ms | 0.852213 ns/op | 2.919 ms, 1908.641 MB/s | 0.506011 s, 11.009 MB/s, ADB reverse HTTP |
| riscv64 | 38.978 ms | 3.18285 ns/op | 1.994 ms, 2793.924 MB/s | 49.094 ms, 113.467 MB/s, direct LAN HTTP |

The local-layout benchmark remains a negative result. On the measured 337-record
artifact:

| Traversal | Time |
|---|---:|
| Canonical | 0.016849 ms |
| Generic header-only | 0.019990 ms |
| Scalar/direct | 0.016500 ms |
| SIMD | 0.019355 ms |
| Nano | 0.020385 ms |

The measured conclusion is `LOCAL_LAYOUT_LATENCY_BENEFIT: NO`,
`SIMD_BENEFIT: NO`, and `NANO_BENEFIT: NO`. The value of semantic bootstrap is
remote deployment, source indirection, and bounded residency, not local parser
acceleration.

## Backend Separation

The llama.cpp integration is one source-agnostic consumer, not a definition of
vBuf. On x86_64, materialized external bytes crossed the committed FFI boundary
into the pinned embedded ggml CPU backend. The bounded operation was:

```text
ggml_sum
local:         220.409927
external file: 220.409927
external HTTP: 220.409927
```

The outputs matched exactly, with zero observed absolute or relative error and
without full-source mapping or download. ARM32, ARM64, and riscv64 compute
backends remain separately unqualified; this is not a storage or source failure.

The ARM32 qualification also found a hard-coded `i8` assumption in the FFI test
harness. It was fixed generically with `core::ffi::c_char`; no production FFI,
storage, or persistent-format semantics changed.

## Build and Test

From the repository root:

```sh
cargo test --manifest-path rust/Cargo.toml -p vbuf-ml
```

The current focused evidence includes the committed vbuf-ML tests for bootstrap,
consumer, consumer FFI, external sources, range loading, source profiles, and
tensor directories. The Rust package is at [`rust/vbuf-ml`](rust/vbuf-ml/), the
llama adapter is at [`integrations/llama.cpp`](integrations/llama.cpp/), and
qualification records are under [`research/results`](research/results/).

## Evidence and Scope

The detailed architecture and qualification evidence is preserved in:

- [`vbuf-ML documentation`](docs/vbuf-ml/);
- [`cross-architecture matrix`](research/results/vbuf-autoregressive-generation-poc22-x86/cross-architecture-qualification-matrix.md);
- [`ARM32 records`](research/results/vbuf-cross-architecture/arm32/);
- [`ARM64 records`](research/results/vbuf-cross-architecture/arm64/);
- [`riscv64 records`](research/results/vbuf-cross-architecture/riscv64/);
- [`v0.6 specification`](spec/spec_0.6.md).

Historical experiments remain under `research/` as evidence. They are not
alternate definitions of the current architecture. This README does not claim
full 32B inference, universal cache behavior, universal parser speedups, or
compute support on every qualified architecture.

## License

MIT
