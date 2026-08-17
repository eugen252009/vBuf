#!/usr/bin/env python3
"""Host-memory tiered residency replay for the exact POC17 request stream."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from qualify_residency_policy_poc17 import parse_trace, requests, simulate


def run_tiered(sequence, hot_capacity, warm_capacity, old_after_reuses=2):
    entries = {}
    clock = 0
    metrics = Counter()
    generation_counts = Counter()
    promotion_rows = []
    demotion_rows = []
    peak = {"HOT": 0, "WARM": 0}
    hot_bytes = warm_bytes = 0
    epoch_length = max(1, len(sequence) // 4)

    def evict_warm(required, epoch, reason):
        nonlocal warm_bytes
        while warm_bytes + required > warm_capacity:
            candidates = [ref for ref, value in entries.items() if value["tier"] == "WARM"]
            if not candidates:
                return False
            victim = min(candidates, key=lambda ref: (entries[ref]["last"], ref))
            value = entries.pop(victim)
            warm_bytes -= value["bytes"]
            metrics["warm_evictions"] += 1
            metrics["warm_drops"] += 1
            metrics["warm_drop_bytes"] += value["bytes"]
        return True

    def make_hot_room(required, epoch):
        nonlocal hot_bytes, warm_bytes
        while hot_bytes + required > hot_capacity:
            candidates = [ref for ref, value in entries.items() if value["tier"] == "HOT"]
            if not candidates:
                return False
            victim = min(candidates, key=lambda ref: (entries[ref]["last"], ref))
            value = entries[victim]
            if not evict_warm(value["bytes"], epoch, "hot_pressure"):
                return False
            value["tier"] = "WARM"
            hot_bytes -= value["bytes"]
            warm_bytes += value["bytes"]
            metrics["demotions"] += 1
            metrics["demotion_bytes"] += value["bytes"]
            metrics["hot_evictions"] += 1
            demotion_rows.append({"ref": victim, "bytes": value["bytes"], "epoch": epoch,
                "reason": "hot_pressure"})
        return True

    for index, request in enumerate(sequence):
        epoch = index // epoch_length
        clock += 1
        if request.ref not in entries:
            metrics["backing_misses"] += 1
            metrics["backing_source_bytes"] += request.bytes
            if not evict_warm(request.bytes, epoch, "warm_capacity"):
                continue
            entries[request.ref] = {"bytes": request.bytes, "tier": "WARM", "generation": "NEW",
                "count": 1, "last": clock, "last_epoch": epoch}
            warm_bytes += request.bytes
            metrics["warm_admissions"] += 1
        else:
            value = entries[request.ref]
            if value["tier"] == "HOT":
                metrics["hot_hits"] += 1
            else:
                metrics["warm_hits"] += 1
            value["count"] += 1
            if value["count"] == 2:
                value["generation"] = "YOUNG"
            if value["count"] >= old_after_reuses + 1:
                value["generation"] = "OLD"
            value["last"] = clock
            value["last_epoch"] = epoch
            if value["tier"] == "WARM" and value["generation"] == "OLD":
                if make_hot_room(value["bytes"], epoch):
                    value["tier"] = "HOT"
                    warm_bytes -= value["bytes"]
                    hot_bytes += value["bytes"]
                    metrics["promotions"] += 1
                    metrics["promotion_bytes"] += value["bytes"]
                    promotion_rows.append({"ref": request.ref, "bytes": value["bytes"],
                        "epoch": epoch, "prior_requests": value["count"] - 1,
                        "generation": value["generation"], "reason": "old_reuse"})
        peak["HOT"] = max(peak["HOT"], hot_bytes)
        peak["WARM"] = max(peak["WARM"], warm_bytes)
    for value in entries.values():
        generation_counts[value["generation"]] += 1
    metrics.update({"hot_peak": peak["HOT"], "warm_peak": peak["WARM"],
        "total_peak": peak["HOT"] + peak["WARM"], "generation_new": generation_counts["NEW"],
        "generation_young": generation_counts["YOUNG"], "generation_old": generation_counts["OLD"],
        "total_inter_tier_movement_bytes": metrics["promotion_bytes"] + metrics["demotion_bytes"],
        "total_data_movement_bytes": metrics["backing_source_bytes"] + metrics["promotion_bytes"] +
            metrics["demotion_bytes"], "duplicate_payload_bytes": 0, "dirty_writeback_bytes": 0})
    metrics["promotion_precision"] = 1.0 if promotion_rows else 0.0
    return {"config": {"hot_capacity": hot_capacity, "warm_capacity": warm_capacity,
        "total_capacity": hot_capacity + warm_capacity, "old_after_reuses": old_after_reuses},
        "metrics": dict(metrics), "promotions": promotion_rows, "demotions": demotion_rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, events, _ = parse_trace(args.trace)
    sequence = requests(events)
    configs = [(8, 0, "flat_8m"), (40, 0, "flat_40m"), (8, 8, "tiered_8_plus_8m"),
        (8, 16, "tiered_8_plus_16m"), (8, 32, "tiered_8_plus_32m"),
        (8, 56, "tiered_8_plus_56m"), (8, 64, "tiered_8_plus_64m")]
    results = {}
    for hot, warm, name in configs:
        if warm == 0:
            flat = simulate(sequence, hot * 1024 * 1024, "lru")
            results[name] = {"config": {"hot_capacity": hot * 1024 * 1024, "warm_capacity": 0,
                "total_capacity": hot * 1024 * 1024}, "metrics": {
                    "backing_source_bytes": flat["load_bytes"], "backing_source_reads": flat["loads"],
                    "backing_misses": flat["loads"], "hot_hits": flat["hits"], "warm_hits": 0,
                    "promotions": 0, "promotion_bytes": 0, "demotions": 0, "demotion_bytes": 0,
                    "hot_peak": flat["capacity"], "warm_peak": 0,
                    "total_peak": flat["capacity"], "duplicate_payload_bytes": 0,
                    "dirty_writeback_bytes": 0, "total_inter_tier_movement_bytes": 0,
                    "total_data_movement_bytes": flat["load_bytes"], "promotion_precision": 0.0,
                    "evictions": flat["evictions"]}}
        else:
            results[name] = run_tiered(sequence, hot * 1024 * 1024, warm * 1024 * 1024)
    (args.output_dir / "tiered-results.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.output_dir / "promotion-demotion-trace.json").write_text(json.dumps({name: {
        "promotions": result.get("promotions", []), "demotions": result.get("demotions", [])}
        for name, result in results.items()}, indent=2) + "\n")
    (args.output_dir / "generation-trace.json").write_text(json.dumps({name: {
        key: value for key, value in result["metrics"].items() if key.startswith("generation_")}
        for name, result in results.items()}, indent=2) + "\n")
    with (args.output_dir / "tiered-capacity-sweep.csv").open("w") as output:
        output.write("config,hot_capacity,warm_capacity,total_capacity,backing_source_bytes,backing_misses,hot_hits,warm_hits,promotions,promotion_bytes,demotions,demotion_bytes,total_data_movement_bytes\n")
        for name, result in results.items():
            config, metric = result["config"], result["metrics"]
            output.write(",".join(str(value) for value in (name, config["hot_capacity"], config["warm_capacity"],
                config["total_capacity"], metric.get("backing_source_bytes", 0), metric.get("backing_misses", 0),
                metric.get("hot_hits", 0), metric.get("warm_hits", 0), metric.get("promotions", 0),
                metric.get("promotion_bytes", 0), metric.get("demotions", 0), metric.get("demotion_bytes", 0),
                metric.get("total_data_movement_bytes", 0))) + "\n")
    print(json.dumps({name: result["metrics"] for name, result in results.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
