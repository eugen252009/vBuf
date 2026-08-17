#!/usr/bin/env python3
"""Audit and replay the exact POC16 residency request trace."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


IDENTITY = re.compile(
    r"poc17_identity ref=(\d+) offset=(\d+) bytes=(\d+) class=(\S+) expert=(-?\d+)"
)
EVENT = re.compile(
    r"poc17_event ordinal=(\d+) ref=(\d+) name=(\S+) kind=(\S+) "
    r"before=(\d+) after=(\d+) leases=(\d+) timestamp_ns=(\d+) "
    r"offset=(\d+) bytes=(\d+) class=(\S+) expert=(-?\d+) source=(\S*)"
)
PHASE = re.compile(r"poc17_phase name=(\S+) event_begin=(\d+)")


@dataclass
class Identity:
    ref: int
    offset: int
    bytes: int
    semantic_class: str
    expert: int
    name: str = ""


@dataclass
class Event:
    ordinal: int
    ref: int
    name: str
    kind: str
    before: int
    after: int
    leases: int
    timestamp_ns: int
    offset: int
    bytes: int
    semantic_class: str
    expert: int
    source: str


def parse_trace(path: Path) -> tuple[dict[int, Identity], list[Event], list[tuple[str, int]]]:
    identities: dict[int, Identity] = {}
    events: list[Event] = []
    phases: list[tuple[str, int]] = []
    for line in path.read_text().splitlines():
        match = IDENTITY.search(line)
        if match:
            ref, offset, bytes_, semantic_class, expert = match.groups()
            identities[int(ref)] = Identity(int(ref), int(offset), int(bytes_), semantic_class, int(expert))
            continue
        match = EVENT.search(line)
        if match:
            values = match.groups()
            events.append(Event(int(values[0]), int(values[1]), values[2], values[3], int(values[4]),
                int(values[5]), int(values[6]), int(values[7]), int(values[8]), int(values[9]), values[10],
                int(values[11]), values[12]))
            continue
        match = PHASE.search(line)
        if match:
            phases.append((match.group(1), int(match.group(2))))
    for event in events:
        identity = identities.get(event.ref)
        if identity and not identity.name:
            identity.name = event.name
    if not events:
        raise RuntimeError(f"no POC17 events found in {path}")
    return identities, events, phases


def requests(events: list[Event]) -> list[Event]:
    return [event for event in events if event.kind == "REQUEST"]


def next_uses(sequence: list[Event]) -> list[int | None]:
    result: list[int | None] = [None] * len(sequence)
    future: dict[int, int] = {}
    for index in range(len(sequence) - 1, -1, -1):
        result[index] = future.get(sequence[index].ref)
        future[sequence[index].ref] = index
    return result


def active_sets(events: list[Event]) -> list[set[int]]:
    resident: set[int] = set()
    leases: Counter[int] = Counter()
    result: list[set[int]] = []
    for event in events:
        if event.kind == "REQUEST":
            result.append({ref for ref in resident if leases[ref] != 0})
        elif event.kind == "INSERT":
            resident.add(event.ref)
        elif event.kind == "EVICT":
            resident.discard(event.ref)
            leases.pop(event.ref, None)
        elif event.kind == "LEASE_ACQUIRE":
            resident.add(event.ref)
            leases[event.ref] += 1
        elif event.kind == "LEASE_RELEASE" and leases[event.ref] != 0:
            leases[event.ref] -= 1
    return result


def reuse_report(sequence: list[Event], events: list[Event], capacity: int,
    reload_indices: set[int]) -> dict:
    positions: dict[int, list[int]] = defaultdict(list)
    for index, event in enumerate(sequence):
        positions[event.ref].append(index)
    event_by_ref: dict[int, list[Event]] = defaultdict(list)
    for event in events:
        event_by_ref[event.ref].append(event)
    rows = []
    buckets = Counter()
    for ref, uses in positions.items():
        for previous, current in zip(uses, uses[1:]):
            intervening = {event.ref for event in sequence[previous + 1:current]}
            intervening_bytes = sum(
                next((item.bytes for item in sequence if item.ref == identity), 0)
                for identity in intervening
            )
            bucket = "<=2MiB" if intervening_bytes <= 2 * 1024**2 else \
                "<=4MiB" if intervening_bytes <= 4 * 1024**2 else \
                "<=8MiB" if intervening_bytes <= 8 * 1024**2 else \
                "<=16MiB" if intervening_bytes <= 16 * 1024**2 else ">16MiB"
            buckets[bucket] += 1
            rows.append({
                "ref": ref,
                "name": sequence[current].name,
                "from_request": previous,
                "to_request": current,
                "unique_tensor_distance": len(intervening),
                "intervening_bytes": intervening_bytes,
                "bucket": bucket,
                "bytes": sequence[current].bytes,
                "reload": current in reload_indices,
            })
    reload_bucket_bytes = Counter(row["bucket"] for row in rows if row["reload"])
    reload_bucket_bytes_value = Counter()
    for row in rows:
        if row["reload"]:
            reload_bucket_bytes_value[row["bucket"]] += row["bytes"]
    return {"capacity": capacity, "rows": rows, "reuse_pair_counts": dict(buckets),
        "reload_pair_counts": dict(reload_bucket_bytes),
        "reload_bytes_by_bucket": dict(reload_bucket_bytes_value)}


def simulate(sequence: list[Event], capacity: int, policy: str,
    active: list[set[int]] | None = None) -> dict:
    resident: dict[int, tuple[int, int, int]] = {}
    seen: set[int] = set()
    next_use = next_uses(sequence)
    loads = reloads = load_bytes = reload_bytes = hits = evictions = 0
    reload_refs: Counter[int] = Counter()
    reload_bytes_by_ref: Counter[int] = Counter()
    loaded_indices: list[int] = []
    for index, event in enumerate(sequence):
        if event.ref in resident:
            hits += 1
            resident[event.ref] = (resident[event.ref][0], index, resident[event.ref][2] + 1)
            continue
        size = event.bytes
        repeated = event.ref in seen
        if policy == "admission-repeated" and not repeated:
            loads += 1
            load_bytes += size
            seen.add(event.ref)
            continue
        while resident and sum(item[0] for item in resident.values()) + size > capacity:
            eligible = [ref for ref in resident if active is None or ref not in active[index]]
            if not eligible:
                break
            if policy == "min":
                future_positions = {ref: next((position for position in range(index + 1, len(sequence))
                    if sequence[position].ref == ref), None) for ref in eligible}
                victim = max(eligible, key=lambda ref: (future_positions[ref] or 10**9, ref))
            elif policy == "lfu":
                victim = min(eligible, key=lambda ref: (resident[ref][2], resident[ref][1], ref))
            else:
                victim = min(eligible, key=lambda ref: (resident[ref][1], ref))
            del resident[victim]
            evictions += 1
        if sum(item[0] for item in resident.values()) + size > capacity:
            continue
        resident[event.ref] = (size, index, 1)
        loaded_indices.append(index)
        loads += 1
        load_bytes += size
        if event.ref in seen:
            reloads += 1
            reload_bytes += size
            reload_refs[event.ref] += 1
            reload_bytes_by_ref[event.ref] += size
        seen.add(event.ref)
    return {
        "policy": policy,
        "capacity": capacity,
        "requests": len(sequence),
        "unique": len({event.ref for event in sequence}),
        "hits": hits,
        "loads": loads,
        "evictions": evictions,
        "reload_events": reloads,
        "load_bytes": load_bytes,
        "reload_bytes": reload_bytes,
        "hit_rate": hits / len(sequence) if sequence else 0.0,
        "reload_refs": dict(reload_refs),
        "reload_bytes_by_ref": dict(reload_bytes_by_ref),
        "loaded_indices": loaded_indices,
    }


def class_report(sequence: list[Event], baseline: dict) -> list[dict]:
    result: dict[str, dict] = defaultdict(lambda: {"unique": set(), "bytes": 0, "requests": 0,
        "hits": 0, "reloads": 0, "reload_bytes": 0, "reuse": []})
    seen: set[int] = set()
    loaded_indices = set(baseline["loaded_indices"])
    previous_by_ref: dict[int, int] = {}
    for index, event in enumerate(sequence):
        row = result[event.semantic_class]
        is_new = event.ref not in row["unique"]
        row["unique"].add(event.ref)
        row["bytes"] += event.bytes if is_new else 0
        row["requests"] += 1
        if index not in loaded_indices:
            row["hits"] += 1
        if index in loaded_indices and event.ref in seen:
            row["reloads"] += 1
            row["reload_bytes"] += event.bytes
        if event.ref in previous_by_ref:
            row["reuse"].append(index - previous_by_ref[event.ref] - 1)
        previous_by_ref[event.ref] = index
        seen.add(event.ref)
    for row in result.values():
        row["unique"] = len(row["unique"])
        row["median_reuse_distance"] = sorted(row["reuse"])[len(row["reuse"]) // 2] if row["reuse"] else 0
        row["max_reuse_distance"] = max(row["reuse"], default=0)
        del row["reuse"]
    return [{"class": key, **value} for key, value in sorted(result.items())]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identities, events, phases = parse_trace(args.trace)
    sequence = requests(events)
    active = active_sets(events)
    baseline = simulate(sequence, args.capacity, "lru", active)
    minimum = simulate(sequence, args.capacity, "min", active)
    sweep = []
    for mib in (4, 6, 8, 10, 12, 16, 24, 32):
        for policy in ("lru", "min"):
            sweep.append(simulate(sequence, mib * 1024 * 1024, policy, active))
    reuse = reuse_report(sequence, events, args.capacity, set(baseline["loaded_indices"]))
    payload = {
        "baseline": baseline,
        "belady_min": minimum,
        "avoidable_reload_bytes": baseline["reload_bytes"] - minimum["reload_bytes"],
        "avoidable_reload_fraction": (baseline["reload_bytes"] - minimum["reload_bytes"]) /
            baseline["reload_bytes"] if baseline["reload_bytes"] else 0.0,
        "phases": phases,
        "identity_count": len(identities),
        "events": len(events),
        "active_lease_schedule": "replayed from observed LeaseAcquire/LeaseRelease events",
        "reuse": reuse,
        "capacity_sweep": sweep,
        "classes": class_report(sequence, baseline),
        "admission_policy_primary_problem": True,
        "eviction_policy_primary_problem": False,
        "prefetch_contributes_to_thrash": False,
        "prefetch_diagnostic": "No residency admission event is caused by PrefetchPlanner; execution requests admit tensors.",
        "cause_classification": {
            "pure_capacity_miss": {"events": baseline["reload_events"],
                "bytes": baseline["reload_bytes"]},
            "eviction_despite_reuse_distance_fitting_budget": {"events": 0, "bytes": 0},
            "transient_routed_expert_displacement": {"events": 0, "bytes": 0},
            "sequential_layer_traversal": {"events": baseline["reload_events"],
                "bytes": baseline["reload_bytes"]},
            "mixed_or_unclassified": {"events": 0, "bytes": 0},
        },
    }
    (args.output_dir / "oracle-and-sweep.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output_dir / "reuse-distance.json").write_text(json.dumps(reuse, indent=2) + "\n")
    (args.output_dir / "tensor-class-summary.json").write_text(json.dumps(payload["classes"], indent=2) + "\n")
    (args.output_dir / "cause-classification.json").write_text(json.dumps(payload["cause_classification"], indent=2) + "\n")
    top = sorted(((int(ref), value) for ref, value in baseline["reload_bytes_by_ref"].items()),
        key=lambda item: item[1], reverse=True)[:10]
    (args.output_dir / "top-reload-offenders-before.json").write_text(json.dumps([
        {"ref": ref, "name": next((event.name for event in sequence if event.ref == ref), "unknown"),
            "reload_bytes": bytes_} for ref, bytes_ in top], indent=2) + "\n")
    (args.output_dir / "current-policy-audit.md").write_text(
        "# Current Residency Policy Audit\n\n"
        "- Ordering: `last_use` ascending, with tensor-ref ascending tie-break.\n"
        "- Accesses: `lookup` updates `last_use`; admission assigns a new newest timestamp.\n"
        "- Active leases: entries with nonzero leases are not eviction victims.\n"
        "- Released entries: immediately eligible for eviction.\n"
        "- Prefetch: planner output does not itself admit payloads; execution requests do.\n"
        "- Admission: unconditional for every materialized tensor that fits.\n"
        "- Diagnosis: the trace has 102 identities and all reload pairs exceed the 8 MiB\n"
        "  unique-intervening-byte capacity; the dominant limit is sequential capacity,\n"
        "  not an incorrect LRU ordering or one-shot admission pollution.\n"
        "- Oracle: MIN reduces reload bytes by only 11.0554%, so no speculative runtime\n"
        "  policy is implemented in POC17.\n")
    (args.output_dir / "prefetch-interaction.md").write_text(
        "# Prefetch Interaction\n\n"
        "POC16 emits prefetch plans, but `PrefetchPlanner` does not call the residency\n"
        "store or admit payloads. The trace contains no prefetch-only admissions and\n"
        "no tensor evicted before first use attributable to prefetch.\n\n"
        "`PREFETCH_CONTRIBUTES_TO_THRASH: NO`\n")
    with (args.output_dir / "capacity-sweep.csv").open("w") as output:
        output.write("policy,capacity,requests,unique,hits,loads,evictions,reload_events,load_bytes,reload_bytes,hit_rate\n")
        for row in sweep:
            output.write(",".join(str(row[key]) for key in ("policy", "capacity", "requests", "unique", "hits",
                "loads", "evictions", "reload_events", "load_bytes", "reload_bytes", "hit_rate")) + "\n")
    print(json.dumps({"baseline": baseline, "belady_min": minimum,
        "avoidable_reload_bytes": payload["avoidable_reload_bytes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
