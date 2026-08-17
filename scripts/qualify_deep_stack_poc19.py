#!/usr/bin/env python3
"""Build POC19 capacity, reuse, and layer-survival evidence from harness logs."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


IDENTITY = re.compile(r"poc17_identity ref=(\d+) offset=(\d+) bytes=(\d+) class=(\S+) expert=(-?\d+)")
EVENT = re.compile(
    r"poc17_event ordinal=(\d+) ref=(\d+) name=(\S+) kind=(\S+) before=(\d+) after=(\d+) "
    r"leases=(\d+) timestamp_ns=(\d+) offset=(\d+) bytes=(\d+) class=(\S+) expert=(-?\d+) source=(\S*)"
)
PHASE = re.compile(r"poc17_phase name=(\S+) event_begin=(\d+)")
SUMMARY = re.compile(
    r"configured_residency_capacity_bytes=(\d+)[^\n]*\n"
    r"total_logical_persistent_bytes=(\d+) peak_active_persistent_bytes=(\d+) peak_active_fraction=([\deE+.-]+) logical_to_active_ratio=([\deE+.-]+)[^\n]*\n"
    r"[^\n]*\n"
    r".*residency_unique_loaded=(\d+) residency_load_events=(\d+) residency_hits=(\d+) residency_misses=(\d+) "
    r"residency_evictions=(\d+) residency_reload_events=(\d+) residency_unique_reloaded=(\d+) residency_reload_bytes=(\d+) "
    r"reload_fraction=([\deE+.-]+) total_source_reads=(\d+) total_source_bytes=(\d+) peak_resident_bytes=(\d+) "
    r"average_resident_bytes=(\d+) current_resident_bytes=(\d+)"
)


def parse_log(path: Path):
    identities = {}
    events = []
    phases = {}
    lines = path.read_text().splitlines()
    for line in lines:
        match = IDENTITY.search(line)
        if match:
            ref, offset, size, kind, expert = match.groups()
            identities[int(ref)] = {"ref": int(ref), "offset": int(offset), "bytes": int(size),
                "class": kind, "expert": int(expert)}
        match = EVENT.search(line)
        if match:
            values = match.groups()
            events.append({"ordinal": int(values[0]), "ref": int(values[1]), "name": values[2],
                "kind": values[3], "before": int(values[4]), "after": int(values[5]),
                "leases": int(values[6]), "timestamp": int(values[7]), "offset": int(values[8]),
                "bytes": int(values[9]), "class": values[10], "expert": int(values[11]),
                "source": values[12]})
        match = PHASE.search(line)
        if match:
            phases[match.group(1)] = int(match.group(2))
    summary = SUMMARY.search("\n".join(lines))
    if not summary or not events:
        raise RuntimeError(f"incomplete POC19 log: {path}")
    values = summary.groups()
    fields = ("capacity", "logical_bytes", "peak_active_bytes", "peak_active_fraction", "logical_active_ratio",
        "unique_loaded", "load_events", "hits", "misses", "evictions", "reload_events", "unique_reloaded",
        "reload_bytes", "reload_fraction", "source_reads", "source_bytes", "peak_resident", "average_resident",
        "current_resident")
    parsed = {field: (float(value) if field in ("peak_active_fraction", "logical_active_ratio", "reload_fraction")
        else int(value)) for field, value in zip(fields, values)}
    return identities, events, phases, parsed


def phase_events(events, phases, name):
    start = phases[name]
    later = [value for value in phases.values() if value > start]
    end = min(later) if later else len(events)
    return [event for event in events if start <= event["ordinal"] < end]


def requests(events):
    return [event for event in events if event["kind"] == "REQUEST"]


def block(ref):
    return ref // 100000 if ref >= 100000 else ref // 10000


def reuse_rows(sequence):
    positions = defaultdict(list)
    for index, event in enumerate(sequence):
        positions[event["ref"]].append(index)
    rows = []
    for ref, uses in positions.items():
        for previous, current in zip(uses, uses[1:]):
            intervening = {event["ref"] for event in sequence[previous + 1:current]}
            intervening_bytes = sum(next((item["bytes"] for item in sequence if item["ref"] == value), 0)
                for value in intervening)
            bucket = next((label for limit, label in ((8, "<=8MiB"), (32, "<=32MiB"),
                (64, "<=64MiB"), (128, "<=128MiB"), (256, "<=256MiB"))
                if intervening_bytes <= limit * 1024 * 1024), ">256MiB")
            rows.append({"ref": ref, "block": block(ref), "class": sequence[current]["class"],
                "first_use_block": block(ref), "request_distance": current - previous,
                "intervening_unique_identities": len(intervening), "intervening_bytes": intervening_bytes,
                "tensor_bytes": sequence[current]["bytes"], "bucket": bucket})
    return rows


def retention(identities, events, phases):
    token0 = phase_events(events, phases, "warm1_token0")
    token1 = phase_events(events, phases, "warm1_token1")
    used0 = {event["ref"] for event in token0 if event["kind"] == "REQUEST"}
    used1 = {event["ref"] for event in token1 if event["kind"] == "REQUEST"}
    reusable = used0 & used1
    boundary = phases["warm1_token1"]
    resident = set()
    for event in events:
        if event["ordinal"] >= boundary:
            break
        if event["kind"] in ("INSERT", "LEASE_ACQUIRE"):
            resident.add(event["ref"])
        elif event["kind"] == "EVICT":
            resident.discard(event["ref"])
    first_use = {}
    reloaded = set()
    for event in token1:
        if event["kind"] == "MATERIALIZE":
            reloaded.add(event["ref"])
        if event["kind"] == "REQUEST" and event["ref"] not in first_use:
            first_use[event["ref"]] = event["ordinal"]
    rows = []
    for ref in sorted(reusable):
        identity = identities.get(ref, {"bytes": 0, "class": "unknown"})
        rows.append({"ref": ref, "block": block(ref), "class": identity["class"],
            "bytes": identity["bytes"], "retained": ref in resident,
            "reloaded_before_token1_use": ref in reloaded})
    by_block = defaultdict(lambda: {"token0_bytes": 0, "reused_bytes": 0, "retained_bytes": 0, "reloaded_bytes": 0})
    for ref in used0:
        identity = identities.get(ref, {"bytes": 0})
        row = by_block[block(ref)]
        row["token0_bytes"] += identity["bytes"]
        if ref in reusable:
            row["reused_bytes"] += identity["bytes"]
            row["retained_bytes"] += identity["bytes"] if ref in resident else 0
            row["reloaded_bytes"] += identity["bytes"] if ref in reloaded else 0
    for row in by_block.values():
        row["retention_rate"] = row["retained_bytes"] / row["reused_bytes"] if row["reused_bytes"] else 0.0  # type: ignore[assignment]
    def byte_total(refs):
        return sum(identities.get(ref, {"bytes": 0})["bytes"] for ref in refs)
    return rows, [{"block": key, **value} for key, value in sorted(by_block.items())], {
        "reusable_bytes": byte_total(reusable),
        "retained_bytes": byte_total(ref for ref in reusable if ref in resident),
        "token0_only_bytes": byte_total(used0 - used1),
        "token1_only_bytes": byte_total(used1 - used0),
        "selected_in_both_routed_expert_bytes": byte_total(ref for ref in reusable
            if identities.get(ref, {"class": ""})["class"].startswith("routed_expert")),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / "research/results/vbuf-deep-stack-poc19-x86"
    logs = [(8, evidence / "run-8MiB.log"), (32, evidence / "run-32MiB.log"),
        (64, evidence / "run-64MiB.log"), (128, evidence / "run-128MiB.log"),
        (256, evidence / "run-256MiB.log")]
    reports = []
    parsed_runs = []
    first = None
    for mib, path in logs:
        identities, events, phases, summary = parse_log(path)
        if first is None:
            first = (identities, events, phases)
        parsed_runs.append((mib, identities, events, phases))
        capacity = summary["capacity"]
        summary["configured_mib"] = mib
        summary["unused_capacity"] = capacity - summary["peak_resident"]
        summary["utilization_percent"] = 100.0 * summary["peak_resident"] / capacity
        summary["source_bytes_avoided_vs_8m"] = 0
        summary["reload_bytes_avoided_vs_8m"] = 0
        reports.append(summary)
    baseline = reports[0]
    for report in reports:
        report["source_bytes_avoided_vs_8m"] = baseline["source_bytes"] - report["source_bytes"]
        report["reload_bytes_avoided_vs_8m"] = baseline["reload_bytes"] - report["reload_bytes"]
        report["source_reduction_percent"] = 100.0 * report["source_bytes_avoided_vs_8m"] / baseline["source_bytes"]
        report["reload_reduction_percent"] = (100.0 * report["reload_bytes_avoided_vs_8m"] /
            baseline["reload_bytes"] if baseline["reload_bytes"] else 0.0)
    assert first is not None
    identities, events, phases = first
    sequence = requests(phase_events(events, phases, "warm1_token0") + phase_events(events, phases, "warm1_token1"))
    reuse, layer, retention_summary = retention(identities, events, phases)
    retention_by_capacity = []
    layers_by_capacity = []
    for mib, run_identities, run_events, run_phases in parsed_runs:
        run_reuse, run_layer, run_summary = retention(run_identities, run_events, run_phases)
        retention_by_capacity.append({"capacity_mib": mib, **run_summary,
            "retention_rate": run_summary["retained_bytes"] / run_summary["reusable_bytes"]
            if run_summary["reusable_bytes"] else 0.0})
        layers_by_capacity.append({"capacity_mib": mib, "blocks": run_layer})
    distances = reuse_rows(sequence)
    (evidence / "capacity-comparison.json").write_text(json.dumps(reports, indent=2) + "\n")
    (evidence / "token-to-token-reuse.json").write_text(json.dumps(reuse, indent=2) + "\n")
    (evidence / "layer-cache-survival.json").write_text(json.dumps(layer, indent=2) + "\n")
    (evidence / "token-retention-by-capacity.json").write_text(json.dumps(retention_by_capacity, indent=2) + "\n")
    (evidence / "layer-cache-survival-by-capacity.json").write_text(json.dumps(layers_by_capacity, indent=2) + "\n")
    (evidence / "reuse-distance.json").write_text(json.dumps(distances, indent=2) + "\n")
    with (evidence / "capacity-comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=reports[0].keys())
        writer.writeheader()
        writer.writerows(reports)
    bucket_counts = Counter(row["bucket"] for row in distances)
    (evidence / "reuse-distance-summary.json").write_text(json.dumps({"counts": bucket_counts,
        "total_pairs": len(distances)}, indent=2, default=dict) + "\n")
    knee = next((report["capacity"] for report in reports[1:]
        if report["source_reduction_percent"] >= 25.0 or report["reload_reduction_percent"] >= 25.0), None)
    summary = {"block_range": "blk.1..blk.8", "block_count": 8, "token_positions": [0, 1],
        "capacity_knee_bytes": knee or "NOT_OBSERVED", **retention_summary,
        "retention_by_capacity": retention_by_capacity,
        "always_required_reuse_bytes": sum(row["bytes"] for row in reuse if row["class"] != "routed_expert_gate" and
            row["class"] != "routed_expert_up" and row["class"] != "routed_expert_down"),
        "routed_expert_reuse_bytes": sum(row["bytes"] for row in reuse if row["class"].startswith("routed_expert")),
        "reuse_distance_buckets": dict(bucket_counts)}
    (evidence / "poc19-analysis-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"capacity": reports, "analysis": summary}, indent=2))


if __name__ == "__main__":
    main()
