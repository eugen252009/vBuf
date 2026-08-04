# Investigation of a preliminary 6.6% run-to-run variation

> “Protobuf showed a 6.6% median variation between two benchmark executions; the cause has not been isolated.”

This is historical investigation context, not the result of this committed baseline and not a regression claim. The two preliminary executions were not an official committed baseline. The five final isolated runs do not corroborate that exact 6.6% observation: final run 05 instead shows a larger broad variation. No causal explanation has been established.

## Scope and timed operation

This is a reproducible **single-format Prost benchmark**. Because it contains no cross-format ranking, it avoids the invalid comparison methodology of the removed legacy runner. Single-format isolation does not establish that this workload represents all Protobuf implementations, schemas, wire patterns, or production uses.

The baseline applies only to Prost, this generated `ProtoRecordList` / `ProtoRecord` schema, one million records with the values constructed by the runner, this compiler and machine, decode into owned Rust objects, and these two numeric sum reductions:

1. Decode the prefaulted encoded byte buffer with `ProtoRecordList::decode`, creating a new owned `Vec<ProtoRecord>`, then calculate `sum(value)`.
2. Decode the same reused input buffer into a new owned `Vec<ProtoRecord>`, then calculate `sum(id as f64 + value)`.

No encoding occurs inside the timer. `decoded` is closure-local; Rust drops `ProtoRecordList` and deallocates its vector at closure scope exit, before the closure returns and therefore before `Instant::elapsed` and `RDTSCP`. Allocation, decode, numeric reduction, destruction, and deallocation are timed. The input byte buffer is reused.

The aggregation result is passed through `black_box` immediately after the closure returns and before the end timestamps. This makes the returned numeric reduction observable so the compiler cannot eliminate the decode/reduction as unused. The tiny barrier operation is therefore included in the timed interval; it is not placed in the per-record loop. Assembly inspection is recorded in `benchmark-results/assembly/` by the audit procedure.

The archived pre-audit runner put its `black_box` after the end timestamps. The audited runner intentionally uses `black_box((run)())` before those timestamps, so the reported timing boundary now includes barrier overhead. This does not alter the decode or numeric aggregation, but it means pre-audit and final medians are not a controlled before/after timing comparison. The final assembly artifact shows calls to `prost::message::Message::decode` in both workload closures; no decoded object is carried between calls.

The wall-clock interval begins at `Instant::now()` after `LFENCE; RDTSC`, and ends at `Instant::elapsed()` before `RDTSCP; LFENCE`. Thus the TSC numerator includes the two `Instant` calls and is not identical to the wall-clock denominator. `tsc_ticks / Instant_duration` is reported only as a **diagnostic measured TSC rate**, not as active core frequency. On this x86 system the TSC may be invariant while core clock changes.

## Method and provenance

Runner source commit: `6391cee1cacced866037966a7c2510f47dc10337`. The final reports were run before the later results/documentation commit; the final documentation commit is recorded in `benchmark-results/manifest.json`. Runner SHA-256: `1e6c968aa59d2edb01f505dce7940113300ce2a3d6a38181d5a71ce1f9f2892b`. The raw reports record ` M rust/src/lib.rs; M rust/src/main.rs; M scripts/run_protobuf_decode_bench.sh;?? benchmark-results/pre-audit/protobuf-isolated-final-01-before-script-preflight-fix.csv;?? c/active_tests/test_aos_soa.so;?? c/active_tests/test_pack_block.so` worktree status, effective rustc flags, `RUSTFLAGS`, compiler identity, CPU affinity, and all required run metadata.

`benchmark-results/pre-audit/` preserves earlier valid raw reports without merging them into the final baseline. In particular, `protobuf-isolated-final-01-before-script-preflight-fix.csv` was collected before the reproducibility script reliably forced a fresh effective-rustc capture; it is retained but excluded from final statistics.

The captured effective Cargo rustc invocation includes `-C target-cpu=native`. This audit found no repository `.cargo/config.toml`, no `~/.cargo/config.toml`, and no set `RUSTFLAGS` or `CARGO_ENCODED_RUSTFLAGS`; therefore the origin of that effective flag is not attributable from available configuration files or environment. The script records the effective invocation for every official run rather than assuming an origin.

Use `scripts/run_protobuf_decode_bench.sh LABEL` with `BENCH_CPU=0` (default). The script verifies affinity, warns on a dirty worktree, captures the runner hash, commit, rustc/LLVM, effective Cargo rustc invocation, and Rust flags, runs `cargo run --locked --release`, and refuses to overwrite an existing report.

The execution order is **deterministic balanced two-target rotation**: value-only is position 0 in each even round, full-record is position 0 in each odd round. Across 30 rounds each workload appears 15 times at position 0 and 15 at position 1. No LCG or Fisher–Yates shuffle is used.

Quantiles use named **floor-index** selection on sorted samples: `floor(n*p)`, clamped to `n-1`. For n=30, p5 is index 1 and p95 is index 28; they are coarse tail indicators. Standard deviation is population SD (divide by N).

## Raw reports

| Run | Report |
|---|---|
| final-01 | `protobuf-isolated-final-01.csv` |
| final-02 | `protobuf-isolated-final-02.csv` |
| final-03 | `protobuf-isolated-final-03.csv` |
| final-04 | `protobuf-isolated-final-04.csv` |
| final-05 | `protobuf-isolated-final-05.csv` |

The validation script parses the CSVs rather than trusting prose. It verifies 30 accepted samples for each workload per report (60/report), 150/workload and 300 total, CPU 0 before/after every accepted sample, no accepted migrations, summary recomputation, valid positions, every round exactly once per workload, and one sample at each position per round.

## Generated summaries (ms)

<!-- BEGIN GENERATED SUMMARY -->
| Run | Workload | Median | Mean | Population SD | p5 | p95 | Min | Max | Median diagnostic TSC rate (GHz) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| final-01 | Value-Only | 16.492086 | 16.634206 | 0.366416 | 16.190235 | 17.246480 | 16.118055 | 17.504061 | 4.199998242 |
| final-01 | Full-Record | 16.431401 | 16.579243 | 0.486031 | 16.244736 | 17.181119 | 16.233696 | 18.775338 | 4.199998594 |
| final-02 | Value-Only | 16.323290 | 16.371808 | 0.239635 | 16.117694 | 16.827057 | 16.088484 | 17.083429 | 4.199998232 |
| final-02 | Full-Record | 16.401796 | 16.512762 | 0.337619 | 16.150985 | 16.998409 | 16.139915 | 17.445311 | 4.200003689 |
| final-03 | Value-Only | 16.571476 | 16.624192 | 0.330750 | 16.121235 | 17.287550 | 16.015674 | 17.394830 | 4.199998235 |
| final-03 | Full-Record | 16.755677 | 16.750582 | 0.366974 | 16.203105 | 17.466031 | 16.190805 | 17.481801 | 4.199997670 |
| final-04 | Value-Only | 16.585017 | 16.573255 | 0.341891 | 16.153614 | 17.271530 | 16.117664 | 17.285730 | 4.199993521 |
| final-04 | Full-Record | 16.344471 | 16.458565 | 0.272853 | 16.158855 | 16.968519 | 16.134675 | 17.162209 | 4.199993283 |
| final-05 | Value-Only | 19.405184 | 19.342985 | 0.990915 | 18.219534 | 21.121127 | 16.342186 | 22.012252 | 4.199995796 |
| final-05 | Full-Record | 19.224199 | 19.135766 | 0.743401 | 17.856733 | 20.288424 | 17.525011 | 20.704005 | 4.199996665 |
<!-- END GENERATED SUMMARY -->

The validator compares every displayed table value to raw-derived values; its permitted display rounding difference is at most 0.0000005 in the displayed unit.

## Evidence-limited findings

Pooled value-only: median 16.605556 ms and mean 17.109289 ms. Pooled full-record: median 16.644307 ms and mean 17.087383 ms. Run-median ranges are 16.323290–19.405184 ms (value-only) and 16.344471–19.224199 ms (full-record).

Position Pearson correlation pools all five final runs (150 samples/workload): value-only r=-0.047, full-record r=-0.001. Position-0/position-1 counts are 75/75 for each workload. Position-one minus position-zero mean is -0.117708 ms (value-only) and -0.001987 ms (full-record). Per-run position-one-minus-position-zero deltas, final-01 through final-05, are -0.184488, +0.178897, -0.060732, -0.092562, -0.429657 ms (value-only) and +0.137114, -0.203062, -0.084736, -0.025773, +0.166520 ms (full-record). No material position association was observed in these five runs; these small descriptive correlations do not establish causality.

Final run 03 diagnostic analysis: Value-Only: median 16.571476 ms; diagnostic 10%-per-tail trimmed mean 16.603863 ms; three highest samples 17.260490, 17.287550, 17.394830 ms; Full-Record: median 16.755677 ms; diagnostic 10%-per-tail trimmed mean 16.731574 ms; three highest samples 17.409971, 17.466031, 17.481801 ms. The authoritative results are untrimmed. The diagnostic trimmed mean removes exactly the three smallest and three largest observations from each 30-sample workload. Final run 03 has no exceptional maximum comparable to the archived pre-audit run below.

Final run 05 is different: Value-Only: median 19.405184 ms; diagnostic 10%-per-tail trimmed mean 19.314121 ms; three highest samples 20.762856, 21.121127, 22.012252 ms; Full-Record: median 19.224199 ms; diagnostic 10%-per-tail trimmed mean 19.161433 ms; three highest samples 19.963542, 20.288424, 20.704005 ms. Its median and its diagnostic trimmed mean remain elevated relative to final runs 01–04, so this final five-run set contains a broad run-level shift, not only one or two maximum samples. The raw samples do not establish a cause.

The archived pre-audit `pinned-03` report is retained as evidence from the earlier runner implementation and is not merged with final statistics: Value-Only: median 16.057849 ms; diagnostic 10%-per-tail trimmed mean 16.057668 ms; three highest samples 16.450931, 18.043808, 23.209404 ms; Full-Record: median 16.042844 ms; diagnostic 10%-per-tail trimmed mean 16.129077 ms; three highest samples 17.637926, 20.995063, 27.821127 ms. It contains several upper-tail samples that increased its mean and population SD while its median stayed closer to its companion pre-audit medians. It is not evidence for a causal label.

The raw diagnostic TSC rates are not evidence of active-core-frequency changes. Allocator state, allocator-bin selection, and memory addresses/reuse history are not collected, so no allocator cause is established. There is intentionally no byte-for-byte or assembly comparison with the removed legacy runner.
