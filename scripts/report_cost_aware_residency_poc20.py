#!/usr/bin/env python3
"""Generate the self-contained POC20 runtime evidence from matched harness logs."""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qualify_deep_stack_poc19 as poc19
import qualify_residency_policy_poc17 as poc17


SUMMARY = re.compile(
    r"configured_residency_capacity_bytes=(\d+).*\n"
    r"total_logical_persistent_bytes=(\d+) peak_active_persistent_bytes=(\d+) peak_active_fraction=([\deE+.-]+) logical_to_active_ratio=([\deE+.-]+).*\n"
    r"[^\n]*\n"
    r".*residency_unique_loaded=(\d+) residency_load_events=(\d+) residency_hits=(\d+) residency_misses=(\d+) "
    r"residency_evictions=(\d+) residency_reload_events=(\d+) residency_unique_reloaded=(\d+) residency_reload_bytes=(\d+) "
    r"reload_fraction=([\deE+.-]+) total_source_reads=(\d+) total_source_bytes=(\d+) peak_resident_bytes=(\d+) "
    r"average_resident_bytes=(\d+) current_resident_bytes=(\d+)"
)
POLICY = re.compile(r"policy_decisions=(\d+) policy_candidates_evaluated=(\d+) policy_cpu_time_ns=(\d+) policy_max_decision_ns=(\d+)")


def parse(path: Path):
    identities, events, phases, summary = poc19.parse_log(path)
    text = path.read_text()
    policy_match = re.search(r"replacement_policy=(\S+)", text)
    policy = policy_match.group(1) if policy_match else "LRU"
    policy_match = POLICY.search(text)
    policy_values = tuple(int(value) for value in policy_match.groups()) if policy_match else (0, 0, 0, 0)
    return identities, events, phases, summary, policy, policy_values


def retention(identities, events, phases):
    token0 = poc19.phase_events(events, phases, "warm1_token0")
    token1 = poc19.phase_events(events, phases, "warm1_token1")
    used0 = {event["ref"] for event in token0 if event["kind"] == "REQUEST"}
    used1 = {event["ref"] for event in token1 if event["kind"] == "REQUEST"}
    boundary = phases["warm1_token1"]
    resident = set()
    for event in events:
        if event["ordinal"] >= boundary:
            break
        if event["kind"] in ("INSERT", "LEASE_ACQUIRE"):
            resident.add(event["ref"])
        elif event["kind"] == "EVICT":
            resident.discard(event["ref"])
    def bytes_for(refs):
        return sum(identities.get(ref, {"bytes": 0})["bytes"] for ref in refs)
    reusable = used0 & used1
    retained = reusable & resident
    dead = resident - used1
    return {"reusable_bytes": bytes_for(reusable), "retained_bytes": bytes_for(retained),
        "useful_retained_bytes": bytes_for(retained), "dead_retained_bytes": bytes_for(dead),
        "retention_rate": bytes_for(retained) / bytes_for(reusable) if reusable else 0.0}


def eviction_quality(events):
    requests = [(event["ordinal"], event["ref"], event["bytes"]) for event in events if event["kind"] == "REQUEST"]
    future = {}
    for index, (ordinal, ref, _) in enumerate(requests):
        future.setdefault(ref, []).append((ordinal, index))
    never = reused = reused_bytes = 0
    distances = []
    for event in events:
        if event["kind"] != "EVICT":
            continue
        next_use = next((item for item in future.get(event["ref"], []) if item[0] > event["ordinal"]), None)
        if next_use is None:
            never += 1
        else:
            reused += 1
            reused_bytes += event["bytes"]
            distances.append(next_use[0] - event["ordinal"])
    return {"evictions_never_reused": never, "evictions_reused": reused,
        "bytes_evicted_then_reused": reused_bytes,
        "median_reuse_distance_after_eviction": statistics.median(distances) if distances else 0}


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / "research/results/vbuf-cost-aware-residency-poc20-x86"
    p19_evidence = root / "research/results/vbuf-deep-stack-poc19-x86"
    names = {
        "8_lru": p19_evidence / "run-8MiB.log",
        "8_cost": "runtime-8MiB-cost-aware.log",
        "64_lru": "runtime-64MiB-lru.log",
        "64_cost": "runtime-64MiB-cost-aware.log",
        "128_lru": "runtime-128MiB-lru.log",
        "128_cost": "runtime-128MiB-cost-aware.log",
        "256_lru": p19_evidence / "run-256MiB.log",
        "256_cost": "runtime-256MiB-cost-aware.log",
    }
    parsed = {key: parse(name if isinstance(name, Path) else evidence / name) for key, name in names.items()}
    comparison = {}
    retention_rows = {}
    eviction_rows = {}
    for key, (identities, events, phases, summary, policy, overhead) in parsed.items():
        comparison[key] = {**summary, "policy": policy, "policy_decisions": overhead[0],
            "policy_candidates_evaluated": overhead[1], "policy_cpu_time_ns": overhead[2],
            "policy_max_decision_ns": overhead[3]}
        retention_rows[key] = retention(identities, events, phases)
        eviction_rows[key] = eviction_quality(events)
    (evidence / "runtime-policy-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (evidence / "retention-analysis.json").write_text(json.dumps(retention_rows, indent=2) + "\n")
    (evidence / "eviction-quality.json").write_text(json.dumps(eviction_rows, indent=2) + "\n")
    lines = ["# Runtime POC20 Policy Comparison", "", "| Capacity | Policy | Source bytes | Reload bytes | Reload events | Evictions | Useful retained bytes | Peak resident |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for key in ("8_lru", "8_cost", "64_lru", "64_cost", "128_lru", "128_cost", "256_lru", "256_cost"):
        row = comparison[key]
        lines.append(f"| {row['capacity'] // (1024 * 1024)} MiB | {row['policy']} | {row['source_bytes']} | {row['reload_bytes']} | {row['reload_events']} | {row['evictions']} | {retention_rows[key]['useful_retained_bytes']} | {row['peak_resident']} |")
    (evidence / "runtime-policy-comparison.md").write_text("\n".join(lines) + "\n")
    gate = {}
    for mib in (8, 64, 128, 256):
        lru = comparison[f"{mib}_lru"]
        cost = comparison[f"{mib}_cost"]
        min_reload = {8: 495354880, 64: 319285248, 128: 118007808, 256: 0}[mib]
        available = lru["reload_bytes"] - min_reload
        saved = lru["reload_bytes"] - cost["reload_bytes"]
        gate[str(mib)] = {"lru_reload_bytes": lru["reload_bytes"], "min_reload_bytes": min_reload,
            "cost_aware_reload_bytes": cost["reload_bytes"], "reload_bytes_saved": saved,
            "headroom_capture_percent": 100.0 * saved / available if available else 0.0,
            "source_bytes_saved": lru["source_bytes"] - cost["source_bytes"]}
    (evidence / "headroom-analysis.json").write_text(json.dumps(gate, indent=2) + "\n")
    failures = []
    for name in ("early", "middle", "late"):
        text = (evidence / f"failure-{name}-128MiB.log").read_text()
        match = re.search(r"target_block_failure .*", text)
        failures.append(match.group(0) if match else f"{name}: MISSING")
    (evidence / "failure-results.md").write_text("# POC20 Failure Results\n\n" + "\n".join(f"- `{line}`" for line in failures) + "\n")
    return comparison, retention_rows, eviction_rows, gate


if __name__ == "__main__":
    main()
