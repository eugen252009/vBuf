#!/usr/bin/env python3
"""Validate real-vBuf SoA/AoS reports and generate their README."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark-results" / "vbuf" / "final"
README = ROOT / "benchmark-results" / "vbuf" / "README.md"
RUNS = 5
SAMPLES = 30
PREPARED = "A prepared scan"
LAYOUTS = ("Native Rust SoA", "Native Rust AoS", "real vBuf SoA", "real vBuf AoS")
WORKLOADS = ("value-only", "full-record")


def median(values: list[float]) -> float:
    values = sorted(values)
    mid = len(values) // 2
    return (values[mid - 1] + values[mid]) / 2 if len(values) % 2 == 0 else values[mid]


def stats(rows: list[dict[str, str]]) -> dict[str, float]:
    values = [float(row["duration_ns"]) / 1e6 for row in rows]
    mean = sum(values) / len(values)
    rates = [float(row["tsc_ticks"]) / float(row["duration_ns"]) for row in rows]
    ordered = sorted(values)
    return {
        "median_ms": median(values), "mean_ms": mean,
        "population_stddev_ms": math.sqrt(sum((x - mean) ** 2 for x in values) / len(values)),
        "p5_ms": ordered[min(int(len(values) * .05), len(values) - 1)],
        "p95_ms": ordered[min(int(len(values) * .95), len(values) - 1)],
        "min_ms": min(values), "max_ms": max(values),
        "median_diagnostic_tsc_rate_ghz": median(rates),
    }


def parse(path: Path) -> dict:
    metadata, summaries, samples = {}, {}, defaultdict(list)
    header = None
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("kind,"):
            header = next(csv.reader([line]))
            continue
        if header is None:
            key, value = line.split("=", 1)
            metadata[key] = value
            continue
        row = dict(zip(header, next(csv.reader([line])), strict=True))
        key = (row["target"], row["category"], row["workload"])
        if row["kind"] == "summary": summaries[key] = row
        elif row["kind"] == "sample": samples[key].append(row)
        else: raise ValueError(f"{path}: unknown row kind")
    return {"path": path, "metadata": metadata, "summaries": summaries, "samples": samples}


def validate(report: dict) -> None:
    required = {"commit", "source_sha256", "vbuf_core_sha256", "rustc", "affinity", "worktree_status", "records", "warm_up_runs", "measured_runs", "execution_order"}
    missing = required - report["metadata"].keys()
    if missing: raise ValueError(f"{report['path']}: missing metadata {sorted(missing)}")
    if (report["metadata"]["records"], report["metadata"]["warm_up_runs"], report["metadata"]["measured_runs"]) != ("1000000", "5", "30"):
        raise ValueError(f"{report['path']}: configuration mismatch")
    if len(report["samples"]) != 16: raise ValueError(f"{report['path']}: expected 16 targets, found {len(report['samples'])}")
    for key, rows in report["samples"].items():
        if len(rows) != SAMPLES: raise ValueError(f"{report['path']}: {key} has {len(rows)} samples")
        if sorted(int(row["round"]) for row in rows) != list(range(SAMPLES)): raise ValueError(f"{report['path']}: {key} rounds invalid")
        if any(row["cpu_before"] != "0" or row["cpu_after"] != "0" or row["migrated"] != "false" for row in rows): raise ValueError(f"{report['path']}: {key} CPU/migration invalid")
        if report["summaries"][key]["accepted_samples"] != "30" or report["summaries"][key]["rejected_migration_samples"] != "0": raise ValueError(f"{report['path']}: {key} count mismatch")
        calculated = stats(rows)
        for name, value in calculated.items():
            if abs(value - float(report["summaries"][key][name])) > 5e-7: raise ValueError(f"{report['path']}: {key} {name} mismatch")
    for round_number in range(SAMPLES):
        positions = sorted(int(row["position"]) for rows in report["samples"].values() for row in rows if int(row["round"]) == round_number)
        if positions != list(range(16)): raise ValueError(f"{report['path']}: round {round_number} position rotation invalid")


def rows_for(reports: list[dict], layout: str, category: str, workload: str) -> list[dict[str, str]]:
    return [row for report in reports for (target, found_category, found_workload), items in report["samples"].items() for row in items if report["summaries"][(target, found_category, found_workload)]["layout"] == layout and found_category == category and found_workload == workload]


def render(reports: list[dict]) -> str:
    first = reports[0]["metadata"]
    prepared = {workload: {layout: stats(rows_for(reports, layout, PREPARED, workload)) for layout in LAYOUTS} for workload in WORKLOADS}
    primary_rows = []
    for workload in WORKLOADS:
        native_soa, native_aos = prepared[workload]["Native Rust SoA"]["median_ms"], prepared[workload]["Native Rust AoS"]["median_ms"]
        vbuf_soa, vbuf_aos = prepared[workload]["real vBuf SoA"]["median_ms"], prepared[workload]["real vBuf AoS"]["median_ms"]
        primary_rows.append(f"| {workload} | {native_soa:.6f} | {native_aos:.6f} | {vbuf_soa:.6f} | {vbuf_aos:.6f} | {(vbuf_soa/native_soa-1)*100:+.3f}% | {(vbuf_aos/native_aos-1)*100:+.3f}% | {(vbuf_soa-vbuf_aos):+.6f} ms |")
    category_rows = []
    for category in ("B reader/view setup plus scan", "C pack/encode"):
        for workload in WORKLOADS:
            for layout in ("real vBuf SoA", "real vBuf AoS"):
                value = stats(rows_for(reports, layout, category, workload))["median_ms"]
                category_rows.append(f"| {category} | {workload} | {layout} | {value:.6f} |")
    primary_table = "\n".join(primary_rows)
    category_table = "\n".join(category_rows)
    report_files = "\n".join(f"| {report['metadata']['run_label']} | `{report['path'].name}` |" for report in reports)
    return f"""# Real vBuf SoA/AoS baseline

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

Runner commit `{first['commit']}`; runner SHA-256 `{first['source_sha256']}`; current vBuf core SHA-256 `{first['vbuf_core_sha256']}`. The raw reports record the dirty worktree state, compiler, effective flags, CPU affinity, and environment. They were not generated from a clean worktree.

Run one non-overwriting report with `BENCH_CPU=0 sh scripts/run_vbuf_baseline.sh LABEL`. The script verifies affinity, records metadata, uses `cargo run --locked --release`, and writes a temporary real vBuf SoA file and a temporary real vBuf AoS file using current public APIs.

Five independent runs use five warm-ups and 30 accepted samples per target. Each round uses deterministic cyclic rotation of all 16 targets; positions are recorded. Every accepted sample observed CPU 0 before and after timing; no migration was accepted.

## Primary result: category A prepared scans

Medians below pool 150 raw samples per layout/workload (five runs × 30), not category B or C samples. “SoA minus AoS” is the real-vBuf SoA median minus the real-vBuf AoS median; negative favors SoA.

| Workload | Native SoA median ms | Native AoS median ms | vBuf SoA median ms | vBuf AoS median ms | vBuf SoA vs Native SoA | vBuf AoS vs Native AoS | vBuf SoA minus AoS ms |
|---|---:|---:|---:|---:|---:|---:|---:|
{primary_table}

## Categories B and C (separate)

| Category | Workload | Layout | Pooled median ms |
|---|---|---|---:|
{category_table}

## Raw evidence

| Run | Report |
|---|---|
{report_files}

`scripts/validate_vbuf_baseline.py` parses raw CSVs and verifies all target counts, rounds, positions, CPU/migration fields, and every summary statistic. Percentiles are floor-index quantiles (`floor(n*p)`, clamped); standard deviation divides by N. No causal explanation is inferred from timing variation.
"""


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--write-readme", action="store_true"); args = parser.parse_args()
    paths = sorted(RESULTS.glob("vbuf-baseline-final-*.csv"))
    if len(paths) != RUNS: raise SystemExit(f"expected {RUNS} final reports, found {len(paths)}")
    reports = [parse(path) for path in paths]
    for report in reports: validate(report)
    for key in ("commit", "source_sha256", "vbuf_core_sha256"):
        if len({report["metadata"][key] for report in reports}) != 1: raise SystemExit(f"reports disagree on {key}")
    if args.write_readme: README.parent.mkdir(parents=True, exist_ok=True); README.write_text(render(reports))
    print("validated 5 reports: 16 targets × 30 accepted samples × 5 runs = 2400 accepted samples")


if __name__ == "__main__": main()
