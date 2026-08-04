# Real vBuf SoA/AoS baseline

This is a reproducible native-versus-real-vBuf baseline. It contains no Protobuf, FlatBuffers, Cap'n Proto, simulated target, or cross-format ranking.

## Scope

The dataset has exactly one million records: `id: u32 = index` and `value: f64 = index * 1.5`. All targets validate count, `sum(value) = 749999250000.0`, and `sum(id as f64 + value) = 1249998750000.0` before measurement.

* **Native Rust SoA:** `Vec<u32>` plus `Vec<f64>`.
* **Native Rust AoS:** `Vec<NativeRecord>` with `#[repr(C)]`, `id` and `value`.
* **real vBuf SoA:** current `VBufWriter` writes actual id and value columns; `VBufInstance::get_as` reads them.
* **real vBuf AoS:** current `VBufWriter` writes one `NativeRecord` column; `VBufInstance::get_as<NativeRecord>` reads it.

Prepared scan category A resolves every vBuf typed slice before timing. Category B puts the actual `VBufInstance::get_as` call(s) inside the timer. Category C starts with the same native AoS source and includes real current `VBufWriter` construction and writes; SoA C extracts two vectors before writing two columns, while AoS C writes one `NativeRecord` column. Categories are reported separately and are not combined into one ranking.

The timing interval begins after `LFENCE; RDTSC`, starts `Instant`, runs the stated target, makes its result opaque with `black_box`, takes `Instant::elapsed`, and ends with `RDTSCP; LFENCE`. The numeric aggregation and any category-specific allocation/drop are inside the interval. The reported diagnostic TSC rate is not active core frequency.

## Provenance and reproduction

Runner commit `a8ed76df2ee46bc8e8a8396942b48b4a96d691bc`; runner SHA-256 `e732340d9e3aec15f6dcecd82681b84fbaca81647e5842e5615e993b087daa52`; current vBuf core SHA-256 `613e9d2c6531ae4a571ecc28d0cab6031d7c0928389f5a018e62e209633e1376`. The raw reports record the dirty worktree state, compiler, effective flags, CPU affinity, and environment. They were not generated from a clean worktree.

Run one non-overwriting report with `BENCH_CPU=0 sh scripts/run_vbuf_baseline.sh LABEL`. The script verifies affinity, records metadata, uses `cargo run --locked --release`, and writes a temporary real vBuf SoA file and a temporary real vBuf AoS file using current public APIs.

Five independent runs use five warm-ups and 30 accepted samples per target. Each round uses deterministic cyclic rotation of all 16 targets; positions are recorded. Every accepted sample observed CPU 0 before and after timing; no migration was accepted.

## Primary result: category A prepared scans

Medians below pool 150 raw samples per layout/workload (five runs × 30), not category B or C samples. “SoA minus AoS” is the real-vBuf SoA median minus the real-vBuf AoS median; negative favors SoA.

| Workload | Native SoA median ms | Native AoS median ms | vBuf SoA median ms | vBuf AoS median ms | vBuf SoA vs Native SoA | vBuf AoS vs Native AoS | vBuf SoA minus AoS ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| value-only | 0.755812 | 0.783757 | 0.753837 | 0.794577 | -0.261% | +1.380% | -0.040740 ms |
| full-record | 0.812882 | 0.848361 | 0.815022 | 0.841066 | +0.263% | -0.860% | -0.026045 ms |

## Categories B and C (separate)

| Category | Workload | Layout | Pooled median ms |
|---|---|---|---:|
| B reader/view setup plus scan | value-only | real vBuf SoA | 0.752836 |
| B reader/view setup plus scan | value-only | real vBuf AoS | 0.791427 |
| B reader/view setup plus scan | full-record | real vBuf SoA | 0.817007 |
| B reader/view setup plus scan | full-record | real vBuf AoS | 0.858342 |
| C pack/encode | value-only | real vBuf SoA | 7.733057 |
| C pack/encode | value-only | real vBuf AoS | 5.560107 |
| C pack/encode | full-record | real vBuf SoA | 7.682397 |
| C pack/encode | full-record | real vBuf AoS | 5.621937 |

## Raw evidence

| Run | Report |
|---|---|
| final-01 | `vbuf-baseline-final-01.csv` |
| final-02 | `vbuf-baseline-final-02.csv` |
| final-03 | `vbuf-baseline-final-03.csv` |
| final-04 | `vbuf-baseline-final-04.csv` |
| final-05 | `vbuf-baseline-final-05.csv` |

`scripts/validate_vbuf_baseline.py` parses raw CSVs and verifies all target counts, rounds, positions, CPU/migration fields, and every summary statistic. Percentiles are floor-index quantiles (`floor(n*p)`, clamped); standard deviation divides by N. No causal explanation is inferred from timing variation.
