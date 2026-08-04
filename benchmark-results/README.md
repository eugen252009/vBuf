# Isolated Protobuf decode investigation

**“Protobuf showed a 6.6% median variation between two benchmark executions; the cause has not been isolated.”**

This replaces the removed cross-format benchmark artifacts with a
single-format Prost benchmark.  It makes no comparison against simulated or
otherwise unfair format implementations.

## Reproduction

The runner is `rust/src/bin/protobuf_decode_bench.rs`.  It measures only:

1. `ProtoRecordList::decode(&bytes[..])` into a new owned
   `Vec<ProtoRecord>`, followed by a value checksum; and
2. the same decode followed by an id-plus-value checksum.

The encoded input is built once and prefaulted before measurement.  Every
sample allocates and drops the decoded `Vec`; the borrowed input buffer is
reused.  `black_box(checksum)` remains after the end timestamp.  There are no
vBuf, FlatBuffers, Cap'n Proto, or native-format targets.

Each report was run with:

```text
taskset -c 0 cargo run --release --bin protobuf_decode_bench
```

All runs used benchmark commit `5b26ec3`, source SHA-256
`08938af1dc465b362e4d14fbab3a54fba9119e2afcb48dc30f7fb454c010532b`,
`rustc 1.95.0 (59807616e 2026-04-14)` / LLVM 22.1.2, and the release
`target-cpu=native` Cargo configuration.  They use 1,000,000 records, five
warm-ups per workload, and 30 accepted samples per workload.  The two
workloads are shuffled every round by the documented fixed-seed LCG; the raw
files record each round and execution position.

## Raw results

The CSVs are the authoritative evidence.  They preserve every duration,
TSC-tick count, derived TSC frequency, CPU before/after, migration status, and
position, as well as per-run summary statistics.

| Run | Raw result |
|---|---|
| 01 | `protobuf-isolated-pinned-01.csv` |
| 02 | `protobuf-isolated-pinned-02.csv` |
| 03 | `protobuf-isolated-pinned-03.csv` |
| 04 | `protobuf-isolated-pinned-04.csv` |
| 05 | `protobuf-isolated-pinned-05.csv` |

All 300 samples were accepted on CPU 0; no migration samples were rejected.

### Per-run decode summaries (ms)

| Run | Workload | Median | Mean | SD | p5 | p95 | Min | Max | Median TSC GHz |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | Value-only | 15.885294 | 15.982081 | 0.216003 | 15.761587 | 16.378730 | 15.737537 | 16.652332 | 4.199992861 |
| 01 | Full-record | 15.962278 | 16.056372 | 0.236536 | 15.807457 | 16.572071 | 15.727887 | 16.580501 | 4.199992809 |
| 02 | Value-only | 15.778387 | 16.067666 | 0.625990 | 15.609606 | 17.821127 | 15.595087 | 18.006838 | 4.199993628 |
| 02 | Full-record | 15.834952 | 15.975125 | 0.347814 | 15.642877 | 16.676072 | 15.602337 | 17.040963 | 4.199995140 |
| 03 | Value-only | 16.057849 | 16.349529 | 1.338644 | 15.803118 | 18.043808 | 15.787638 | 23.209404 | 4.199994053 |
| 03 | Full-record | 16.042844 | 16.703781 | 2.270966 | 15.853638 | 20.995063 | 15.853257 | 27.821127 | 4.199992847 |
| 04 | Value-only | 15.875867 | 16.008399 | 0.231300 | 15.765027 | 16.454661 | 15.745457 | 16.504461 | 4.199992165 |
| 04 | Full-record | 15.993378 | 16.083095 | 0.204840 | 15.891378 | 16.472010 | 15.859907 | 16.755922 | 4.199992465 |
| 05 | Value-only | 15.902818 | 16.003135 | 0.207514 | 15.803227 | 16.370991 | 15.783048 | 16.630602 | 4.199992586 |
| 05 | Full-record | 15.968753 | 16.122940 | 0.251710 | 15.893937 | 16.560381 | 15.893108 | 16.889723 | 4.199992614 |

## Findings

* **Entire distribution vs. outliers:** the five run medians are tight:
  15.778–16.058 ms (value-only) and 15.835–16.043 ms (full-record).  The wide
  pooled upper tails are primarily individual long samples in run 03 (23.209
  and 27.821 ms).  This recreated isolated benchmark therefore shows changed
  outliers, not a reproduced 6.6% whole-distribution shift.
* **CPU/TSC frequency:** no matching shift is observed.  Median TSC frequency
  spans only 4.199992165–4.199994053 GHz (value-only) and
  4.199992465–4.199995140 GHz (full-record).
* **Execution position:** no material correlation.  Pearson
  `r(position,duration)` is -0.042 for value-only and +0.056 for full-record.
  Position-one minus position-zero mean is -0.059 ms and +0.120 ms,
  respectively; per-run signs are mixed.
* **Allocator/memory reuse:** allocation behavior is fixed by the source, but
  allocator bins, addresses, and reuse history are not instrumented.  Runtime
  allocator state remains unisolated.
* **Semantic/assembly comparison:** the unfair legacy benchmark sources and
  reports were intentionally removed.  There is no trustworthy prior,
  byte-identical benchmark commit or assembly artifact to compare.  This
  runner is a new committed, single-format baseline rather than evidence that
  an earlier implementation was semantically unchanged.

The legacy `bench.md` values are deliberately not retained as comparison data:
they combined different workloads and did not provide source-commit or raw-run
provenance.
