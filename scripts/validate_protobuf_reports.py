#!/usr/bin/env python3
"""Validate isolated Prost raw reports and regenerate their README table."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmark-results" / "final"
README = ROOT / "benchmark-results" / "README.md"
WORKLOADS = ("Value-Only", "Full-Record")
EXPECTED_SAMPLES = 30


def quantile_floor(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * probability), len(ordered) - 1)]


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (ordered[middle - 1] + ordered[middle]) / 2 if len(ordered) % 2 == 0 else ordered[middle]


def summary(samples: list[dict[str, str]]) -> dict[str, float]:
    values = [float(sample["duration_ns"]) / 1e6 for sample in samples]
    mean = sum(values) / len(values)
    rates = [float(sample["tsc_ticks"]) / float(sample["duration_ns"]) for sample in samples]
    return {
        "median_ms": median(values),
        "mean_ms": mean,
        "population_stddev_ms": math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
        "p5_ms": quantile_floor(values, 0.05),
        "p95_ms": quantile_floor(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "median_diagnostic_tsc_rate_ghz": median(rates),
    }


def parse_report(path: Path) -> dict:
    metadata: dict[str, str] = {}
    summaries: dict[str, dict[str, str]] = {}
    samples: dict[str, list[dict[str, str]]] = defaultdict(list)
    header: list[str] | None = None
    for raw_line in path.read_text().splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        if raw_line.startswith("kind,"):
            header = next(csv.reader([raw_line]))
            continue
        if header is None:
            key, value = raw_line.split("=", 1)
            metadata[key] = value
            continue
        row = dict(zip(header, next(csv.reader([raw_line])), strict=True))
        if row["kind"] == "summary":
            summaries[row["workload"]] = row
        elif row["kind"] == "sample":
            samples[row["workload"]].append(row)
        else:
            raise ValueError(f"{path}: unknown row kind {row['kind']}")
    return {"path": path, "metadata": metadata, "summaries": summaries, "samples": samples}


def close(left: float, right: float, name: str, path: Path) -> None:
    if abs(left - right) > 5e-7:
        raise ValueError(f"{path}: {name}: expected {left:.12f}, report has {right:.12f}")


def validate_report(report: dict) -> None:
    path = report["path"]
    metadata = report["metadata"]
    required = {
        "run_label", "commit", "source_sha256", "rustc", "affinity", "worktree_status",
        "rustflags", "effective_rustc_flags", "cpu_identification", "records", "warm_up_runs",
        "measured_runs", "execution_order", "timed_operation", "timing_windows", "black_box",
        "quantiles", "standard_deviation",
    }
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"{path}: missing metadata {sorted(missing)}")
    if metadata["records"] != "1000000" or metadata["warm_up_runs"] != "5" or metadata["measured_runs"] != "30":
        raise ValueError(f"{path}: configuration metadata mismatch")
    for workload in WORKLOADS:
        rows = report["samples"].get(workload, [])
        if len(rows) != EXPECTED_SAMPLES:
            raise ValueError(f"{path}: {workload} has {len(rows)}, expected {EXPECTED_SAMPLES} samples")
        if report["summaries"].get(workload, {}).get("accepted_samples") != str(EXPECTED_SAMPLES):
            raise ValueError(f"{path}: {workload} accepted count mismatch")
        if report["summaries"][workload]["rejected_migration_samples"] != "0":
            raise ValueError(f"{path}: {workload} rejected migration count is nonzero")
        rounds = sorted(int(row["round"]) for row in rows)
        if rounds != list(range(EXPECTED_SAMPLES)):
            raise ValueError(f"{path}: {workload} does not contain every round exactly once")
        if any(row["position"] not in {"0", "1"} for row in rows):
            raise ValueError(f"{path}: {workload} has invalid position")
        if any(row["cpu_before"] != "0" or row["cpu_after"] != "0" for row in rows):
            raise ValueError(f"{path}: {workload} is not pinned to CPU 0")
        if any(row["migrated"] != "false" for row in rows):
            raise ValueError(f"{path}: {workload} accepted a migration")
        computed = summary(rows)
        for key, value in computed.items():
            close(value, float(report["summaries"][workload][key]), key, path)
    for round_number in range(EXPECTED_SAMPLES):
        positions = [
            row["position"]
            for workload in WORKLOADS
            for row in report["samples"][workload]
            if int(row["round"]) == round_number
        ]
        if sorted(positions) != ["0", "1"]:
            raise ValueError(f"{path}: round {round_number} does not contain one target at each position")


def pearson(xs: list[float], ys: list[float]) -> float:
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
    return numerator / denominator


def trim_details(report: dict, workload: str) -> tuple[float, float, list[float]]:
    values = sorted(float(row["duration_ns"]) / 1e6 for row in report["samples"][workload])
    trim_count = len(values) // 10
    trimmed = values[trim_count:-trim_count]
    return median(values), sum(trimmed) / len(trimmed), values[-3:]


def render_readme(reports: list[dict]) -> str:
    summaries = []
    for report in reports:
        for workload in WORKLOADS:
            summaries.append((report, workload, summary(report["samples"][workload])))
    rows = "\n".join(
        f"| {report['metadata']['run_label']} | {workload} | {values['median_ms']:.6f} | {values['mean_ms']:.6f} | {values['population_stddev_ms']:.6f} | {values['p5_ms']:.6f} | {values['p95_ms']:.6f} | {values['min_ms']:.6f} | {values['max_ms']:.6f} | {values['median_diagnostic_tsc_rate_ghz']:.9f} |"
        for report, workload, values in summaries
    )
    all_by_workload = {workload: [row for report in reports for row in report["samples"][workload]] for workload in WORKLOADS}
    analyses = {}
    for workload, rows_for_workload in all_by_workload.items():
        durations = [float(row["duration_ns"]) / 1e6 for row in rows_for_workload]
        positions = [float(row["position"]) for row in rows_for_workload]
        position_means = {
            position: sum(float(row["duration_ns"]) / 1e6 for row in rows_for_workload if row["position"] == str(position))
            / sum(row["position"] == str(position) for row in rows_for_workload)
            for position in (0, 1)
        }
        analyses[workload] = (median(durations), sum(durations) / len(durations), pearson(positions, durations), position_means)
    run3 = next(report for report in reports if report["metadata"]["run_label"] == "final-03")
    run5 = next(report for report in reports if report["metadata"]["run_label"] == "final-05")
    run3_text = []
    run5_text = []
    for workload in WORKLOADS:
        run3_median, run3_trimmed_mean, run3_highest = trim_details(run3, workload)
        run5_median, run5_trimmed_mean, run5_highest = trim_details(run5, workload)
        run3_text.append(f"{workload}: median {run3_median:.6f} ms; diagnostic 10%-per-tail trimmed mean {run3_trimmed_mean:.6f} ms; three highest samples {', '.join(f'{value:.6f}' for value in run3_highest)} ms")
        run5_text.append(f"{workload}: median {run5_median:.6f} ms; diagnostic 10%-per-tail trimmed mean {run5_trimmed_mean:.6f} ms; three highest samples {', '.join(f'{value:.6f}' for value in run5_highest)} ms")
    archived_run3_path = ROOT / "benchmark-results" / "pre-audit" / "protobuf-isolated-pinned-03.csv"
    archived_run3 = parse_report(archived_run3_path) if archived_run3_path.exists() else None
    archived_run3_text = []
    if archived_run3:
        for workload in WORKLOADS:
            old_median, old_trimmed_mean, old_highest = trim_details(archived_run3, workload)
            archived_run3_text.append(f"{workload}: median {old_median:.6f} ms; diagnostic 10%-per-tail trimmed mean {old_trimmed_mean:.6f} ms; three highest samples {', '.join(f'{value:.6f}' for value in old_highest)} ms")
    per_run_deltas = {
        workload: [
            sum(float(row["duration_ns"]) / 1e6 for row in report["samples"][workload] if row["position"] == "1") / 15
            - sum(float(row["duration_ns"]) / 1e6 for row in report["samples"][workload] if row["position"] == "0") / 15
            for report in reports
        ]
        for workload in WORKLOADS
    }
    metadata = reports[0]["metadata"]
    report_files = "\n".join(f"| {report['metadata']['run_label']} | `{report['path'].name}` |" for report in reports)
    return f"""# Investigation of a preliminary 6.6% run-to-run variation

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

Runner source commit: `{metadata['commit']}`. The final reports were run before the later results/documentation commit; the final documentation commit is recorded in `benchmark-results/manifest.json`. Runner SHA-256: `{metadata['source_sha256']}`. The raw reports record `{metadata['worktree_status']}` worktree status, effective rustc flags, `RUSTFLAGS`, compiler identity, CPU affinity, and all required run metadata.

`benchmark-results/pre-audit/` preserves earlier valid raw reports without merging them into the final baseline. In particular, `protobuf-isolated-final-01-before-script-preflight-fix.csv` was collected before the reproducibility script reliably forced a fresh effective-rustc capture; it is retained but excluded from final statistics.

The captured effective Cargo rustc invocation includes `-C target-cpu=native`. This audit found no repository `.cargo/config.toml`, no `~/.cargo/config.toml`, and no set `RUSTFLAGS` or `CARGO_ENCODED_RUSTFLAGS`; therefore the origin of that effective flag is not attributable from available configuration files or environment. The script records the effective invocation for every official run rather than assuming an origin.

Use `scripts/run_protobuf_decode_bench.sh LABEL` with `BENCH_CPU=0` (default). The script verifies affinity, warns on a dirty worktree, captures the runner hash, commit, rustc/LLVM, effective Cargo rustc invocation, and Rust flags, runs `cargo run --locked --release`, and refuses to overwrite an existing report.

The execution order is **deterministic balanced two-target rotation**: value-only is position 0 in each even round, full-record is position 0 in each odd round. Across 30 rounds each workload appears 15 times at position 0 and 15 at position 1. No LCG or Fisher–Yates shuffle is used.

Quantiles use named **floor-index** selection on sorted samples: `floor(n*p)`, clamped to `n-1`. For n=30, p5 is index 1 and p95 is index 28; they are coarse tail indicators. Standard deviation is population SD (divide by N).

## Raw reports

| Run | Report |
|---|---|
{report_files}

The validation script parses the CSVs rather than trusting prose. It verifies 30 accepted samples for each workload per report (60/report), 150/workload and 300 total, CPU 0 before/after every accepted sample, no accepted migrations, summary recomputation, valid positions, every round exactly once per workload, and one sample at each position per round.

## Generated summaries (ms)

<!-- BEGIN GENERATED SUMMARY -->
| Run | Workload | Median | Mean | Population SD | p5 | p95 | Min | Max | Median diagnostic TSC rate (GHz) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}
<!-- END GENERATED SUMMARY -->

The validator compares every displayed table value to raw-derived values; its permitted display rounding difference is at most 0.0000005 in the displayed unit.

## Evidence-limited findings

Pooled value-only: median {analyses['Value-Only'][0]:.6f} ms and mean {analyses['Value-Only'][1]:.6f} ms. Pooled full-record: median {analyses['Full-Record'][0]:.6f} ms and mean {analyses['Full-Record'][1]:.6f} ms. Run-median ranges are {min(value['median_ms'] for report, workload, value in summaries if workload == 'Value-Only'):.6f}–{max(value['median_ms'] for report, workload, value in summaries if workload == 'Value-Only'):.6f} ms (value-only) and {min(value['median_ms'] for report, workload, value in summaries if workload == 'Full-Record'):.6f}–{max(value['median_ms'] for report, workload, value in summaries if workload == 'Full-Record'):.6f} ms (full-record).

Position Pearson correlation pools all five final runs (150 samples/workload): value-only r={analyses['Value-Only'][2]:+.3f}, full-record r={analyses['Full-Record'][2]:+.3f}. Position-0/position-1 counts are 75/75 for each workload. Position-one minus position-zero mean is {analyses['Value-Only'][3][1] - analyses['Value-Only'][3][0]:+.6f} ms (value-only) and {analyses['Full-Record'][3][1] - analyses['Full-Record'][3][0]:+.6f} ms (full-record). Per-run position-one-minus-position-zero deltas, final-01 through final-05, are {', '.join(f'{value:+.6f}' for value in per_run_deltas['Value-Only'])} ms (value-only) and {', '.join(f'{value:+.6f}' for value in per_run_deltas['Full-Record'])} ms (full-record). No material position association was observed in these five runs; these small descriptive correlations do not establish causality.

Final run 03 diagnostic analysis: {'; '.join(run3_text)}. The authoritative results are untrimmed. The diagnostic trimmed mean removes exactly the three smallest and three largest observations from each 30-sample workload. Final run 03 has no exceptional maximum comparable to the archived pre-audit run below.

Final run 05 is different: {'; '.join(run5_text)}. Its median and its diagnostic trimmed mean remain elevated relative to final runs 01–04, so this final five-run set contains a broad run-level shift, not only one or two maximum samples. The raw samples do not establish a cause.

The archived pre-audit `pinned-03` report is retained as evidence from the earlier runner implementation and is not merged with final statistics: {'; '.join(archived_run3_text) if archived_run3_text else 'not present'}. It contains several upper-tail samples that increased its mean and population SD while its median stayed closer to its companion pre-audit medians. It is not evidence for a causal label.

The raw diagnostic TSC rates are not evidence of active-core-frequency changes. Allocator state, allocator-bin selection, and memory addresses/reuse history are not collected, so no allocator cause is established. There is intentionally no byte-for-byte or assembly comparison with the removed legacy runner.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-readme", action="store_true")
    arguments = parser.parse_args()
    paths = sorted(RESULT_DIR.glob("protobuf-isolated-final-*.csv"))
    if len(paths) != 5:
        raise SystemExit(f"expected exactly five final reports in {RESULT_DIR}, found {len(paths)}")
    reports = [parse_report(path) for path in paths]
    for report in reports:
        validate_report(report)
    commits = {report["metadata"]["commit"] for report in reports}
    sources = {report["metadata"]["source_sha256"] for report in reports}
    if len(commits) != 1 or len(sources) != 1:
        raise SystemExit("reports do not share one runner commit and source hash")
    if arguments.write_readme:
        README.write_text(render_readme(reports))
    print(f"validated {len(reports)} reports: 150 Value-Only, 150 Full-Record, 300 accepted samples")


if __name__ == "__main__":
    main()
