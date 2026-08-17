#!/usr/bin/env python3
"""Offline POC20 comparison of LRU, generic cost-aware, and MIN replacement."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qualify_residency_policy_poc17 as poc17


def simulate(sequence, active, capacity, policy):
    resident = {}
    seen = set()
    requests_seen = {}
    last_request = {}
    loads = reloads = load_bytes = reload_bytes = hits = evictions = 0
    decisions = 0
    candidates_evaluated = 0
    decision_rows = []
    for index, event in enumerate(sequence):
        ref = event.ref
        requests_seen[ref] = requests_seen.get(ref, 0) + 1
        if ref in resident:
            hits += 1
            last_request[ref] = index
            continue
        while resident and sum(item[0] for item in resident.values()) + event.bytes > capacity:
            eligible = [candidate for candidate in resident if candidate not in active[index]]
            if not eligible:
                break
            if policy == "lru":
                victim = min(eligible, key=lambda candidate: (last_request[candidate], candidate))
                scores = {}
            elif policy == "min":
                future = {candidate: next((position for position in range(index + 1, len(sequence))
                    if sequence[position].ref == candidate), None) for candidate in eligible}
                victim = max(eligible, key=lambda candidate: (future[candidate] or 10**9, candidate))
                scores = {}
            else:
                density = {candidate: requests_seen[candidate] / resident[candidate][0]
                    for candidate in eligible}
                recency = {candidate: 1.0 / (index - last_request[candidate] + 1)
                    for candidate in eligible}
                max_density = max(density.values(), default=0.0)
                max_recency = max(recency.values(), default=0.0)
                scores = {candidate: {
                    "size": resident[candidate][0],
                    "reacquire_cost_bytes": resident[candidate][0],
                    "observed_request_count": requests_seen[candidate],
                    "next_use_distance_estimate": index - last_request[candidate],
                    "reuse_density": density[candidate],
                    "normalized_reuse_density": density[candidate] / max_density if max_density else 0.0,
                    "normalized_recency": recency[candidate] / max_recency if max_recency else 0.0,
                    "retention_score": 0.9 * (density[candidate] / max_density if max_density else 0.0) +
                        0.1 * (recency[candidate] / max_recency if max_recency else 0.0),
                } for candidate in eligible}
                victim = min(eligible, key=lambda candidate: (scores[candidate]["retention_score"], candidate))
            decisions += 1
            candidates_evaluated += len(eligible)
            decision_rows.append({"request_index": index, "incoming_ref": ref, "incoming_bytes": event.bytes,
                "eligible": eligible, "victim": victim, "policy": policy, "scores": scores})
            del resident[victim]
            evictions += 1
        if sum(item[0] for item in resident.values()) + event.bytes > capacity:
            continue
        resident[ref] = (event.bytes, index)
        last_request[ref] = index
        load_bytes += event.bytes
        if ref in seen:
            reloads += 1
            reload_bytes += event.bytes
        else:
            loads += 1
        seen.add(ref)
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
        "source_bytes": load_bytes,
        "reload_bytes": reload_bytes,
        "policy_decisions": decisions,
        "policy_candidates_evaluated": candidates_evaluated,
        "decision_rows": decision_rows,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    evidence = root / "research/results/vbuf-cost-aware-residency-poc20-x86"
    evidence.mkdir(parents=True, exist_ok=True)
    trace_path = root / "research/results/vbuf-deep-stack-poc19-x86/run-64MiB.log"
    _, events, _ = poc17.parse_trace(trace_path)
    sequence = poc17.requests(events)
    active = poc17.active_sets(events)
    rows = []
    decisions = []
    for mib in (8, 64, 128, 256):
        for policy in ("lru", "cost-aware", "min"):
            result = simulate(sequence, active, mib * 1024 * 1024, policy)
            decisions.extend(result.pop("decision_rows"))
            rows.append(result)
    comparison = {f"{row['capacity']}_{row['policy']}": row for row in rows}
    (evidence / "offline-policy-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    representative = [row for row in decisions if row["policy"] == "cost-aware"]
    representative = representative[:20]
    (evidence / "representative-decisions.json").write_text(json.dumps(representative, indent=2) + "\n")
    gate = {}
    for mib in (64, 128):
        lru = comparison[f"{mib * 1024 * 1024}_lru"]
        cost = comparison[f"{mib * 1024 * 1024}_cost-aware"]
        oracle = comparison[f"{mib * 1024 * 1024}_min"]
        available = lru["reload_bytes"] - oracle["reload_bytes"]
        saved = lru["reload_bytes"] - cost["reload_bytes"]
        gate[str(mib)] = {"lru_reload_bytes": lru["reload_bytes"], "min_reload_bytes": oracle["reload_bytes"],
            "cost_aware_reload_bytes": cost["reload_bytes"], "saved_vs_lru": saved,
            "headroom_capture_percent": 100.0 * saved / available if available else 0.0}
    (evidence / "offline-acceptance-gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    lines = ["# Offline POC20 Policy Comparison", "", "The trace is the real POC19 8-block request trace.",
        "COST_AWARE uses only observed request count, recency, size, and byte-based reacquisition cost.", "",
        "| Capacity | Policy | Source bytes | Reload bytes | Reload events | Evictions | Decisions | Candidates |",
        "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['capacity'] // (1024 * 1024)} MiB | {row['policy']} | {row['source_bytes']} | "
            f"{row['reload_bytes']} | {row['reload_events']} | {row['evictions']} | {row['policy_decisions']} | "
            f"{row['policy_candidates_evaluated']} |")
    lines.extend(["", "## Acceptance", ""])
    for mib, result in gate.items():
        lines.append(f"- {mib} MiB: COST_AWARE reload bytes {result['cost_aware_reload_bytes']} "
            f"versus LRU {result['lru_reload_bytes']}; headroom capture {result['headroom_capture_percent']:.2f}%.")
    (evidence / "offline-policy-comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"gate": gate, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
