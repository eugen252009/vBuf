#!/usr/bin/env python3
import csv
import json
import pathlib
import re
import statistics

ROOT = pathlib.Path.home() / "projekte/vBuf/benchmark-results/vbuf-ml-rvv-gguf-comparison"
RUNS = {
    "vbuf": ["vbuf-warm-00-final4", "vbuf-warm-01-final4", "vbuf-warm-02-final4", "vbuf-cold-00-final4", "vbuf-cold-01-final4"],
    "gguf": ["gguf-warm-00-final4", "gguf-warm-01-final4", "gguf-warm-02-final4", "gguf-cold-00-final4", "gguf-cold-01-final4"],
}


def trace(path):
    events = []
    repacks = []
    for line in path.read_text().splitlines():
        fields = line.split(",")
        if fields[0] == "EVENT":
            events.append({"name": fields[1], "us": int(fields[2]), "detail": fields[5] if len(fields) > 5 else ""})
        elif fields[0] == "REPACK":
            repacks.append({"tensor": fields[1], "source_type": fields[2], "destination": fields[3], "source_bytes": int(fields[4]), "destination_bytes": int(fields[5]), "start_us": int(fields[6]), "end_us": int(fields[7]), "duration_us": int(fields[8])})
    return events, repacks


def pair(events, begin, end, detail=None):
    starts = [x for x in events if x["name"] == begin and (detail is None or x["detail"] == detail)]
    for start in starts:
        ends = [x for x in events if x["name"] == end and x["us"] >= start["us"] and (detail is None or x["detail"] == detail)]
        if ends:
            return start["us"], ends[0]["us"]
    return None


def last_pair(events, begin, end):
    starts = [x for x in events if x["name"] == begin]
    for start in reversed(starts):
        ends = [x for x in events if x["name"] == end and x["us"] >= start["us"]]
        if ends:
            return start["us"], ends[0]["us"]
    return None


def log_stats(path):
    text = path.read_text()
    buffers = {}
    for name, mib in re.findall(r"load_tensors: +([A-Z_]+) model buffer size = +([0-9.]+) MiB", text):
        buffers[name] = float(mib)
    reserve = re.search(r"reserve took ([0-9.]+) ms", text)
    nodes = re.search(r"graph nodes += ([0-9]+)", text)
    splits = re.search(r"graph splits += ([0-9]+)", text)
    return {"buffers_mib": buffers, "scheduler_reserve_ms": float(reserve.group(1)) if reserve else None, "graph_nodes": int(nodes.group(1)) if nodes else None, "graph_splits": int(splits.group(1)) if splits else None}


def interval_union(repacks):
    intervals = sorted((x["start_us"], x["end_us"]) for x in repacks)
    total = 0
    current_start = current_end = None
    for start, end in intervals:
        if current_start is None:
            current_start, current_end = start, end
        elif start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None:
        total += current_end - current_start
    return total


def sample_stats(path):
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return {}
    numeric = lambda key: [int(float(row[key])) for row in rows if row.get(key, "") not in ("", None)]
    return {"sample_count": len(rows), "max_rss_bytes": max(numeric("rss_bytes"), default=0), "max_available_bytes": min(numeric("available_bytes"), default=0), "max_minor_faults": max(numeric("minor_faults"), default=0), "max_major_faults": max(numeric("major_faults"), default=0), "max_read_bytes": max(numeric("read_bytes"), default=0), "thread_count_max": max(numeric("thread_count"), default=0)}


def main():
    phase_rows = []
    repack_rows = []
    memory_rows = []
    io_rows = []
    summary = {}
    for kind, names in RUNS.items():
        summary[kind] = {"warm": [], "cold": []}
        for name in names:
            run = ROOT / name
            result = json.loads((run / "run.json").read_text())
            events, repacks = trace(run / "trace.csv")
            start = next(x["us"] for x in events if x["name"] == "PROCESS_START")
            runtime_enum = pair(events, "TENSOR_ENUMERATION_BEGIN", "TENSOR_ENUMERATION_END", "runtime")
            metadata = pair(events, "MODEL_METADATA_BEGIN", "MODEL_METADATA_END", "vbuf") or pair(events, "MODEL_METADATA_BEGIN", "MODEL_METADATA_END")
            tokenizer = last_pair(events, "TOKENIZER_BEGIN", "TOKENIZER_END")
            phases = {
                "format_open": pair(events, "FORMAT_OPEN_BEGIN", "FORMAT_OPEN_END"),
                "metadata": metadata,
                "tokenizer": tokenizer,
                "tensor_enumeration": runtime_enum,
                "model_structure_construction": pair(events, "MODEL_OPEN_BEGIN", "MODEL_STRUCTURE_READY"),
                "backend_allocation": pair(events, "BACKEND_BUFFER_ALLOCATION_BEGIN", "BACKEND_BUFFER_ALLOCATION_END"),
                "payload_materialization": pair(events, "PAYLOAD_MATERIALIZATION_BEGIN", "PAYLOAD_MATERIALIZATION_END"),
                "graph_reserve": pair(events, "GRAPH_RESERVE_BEGIN", "GRAPH_RESERVE_END"),
                "context_create": pair(events, "CONTEXT_CREATE_BEGIN", "CONTEXT_CREATE_END"),
                "execution_ready_total": pair(events, "PROCESS_START", "EXECUTION_READY"),
                "prompt_eval": pair(events, "PROMPT_EVAL_BEGIN", "PROMPT_EVAL_END"),
                "first_decode": pair(events, "FIRST_DECODE_BEGIN", "FIRST_DECODE_END"),
                "first_token": pair(events, "FIRST_TOKEN", "FIRST_TOKEN"),
                "total_run": pair(events, "PROCESS_START", "RUN_END"),
            }
            phase_values = {}
            for phase, bounds in phases.items():
                duration = (bounds[1] - bounds[0]) if bounds else 0
                phase_values[phase] = duration
                phase_rows.append({"run": name, "kind": kind, "regime": "warm" if "warm" in name else "cold", "phase": phase, "start_us": bounds[0] if bounds else "", "end_us": bounds[1] if bounds else "", "duration_us": duration, "classification": "MEASURED" if bounds else "NOT_REQUIRED"})
            repack_wall = interval_union(repacks)
            repack_cpu = sum(x["duration_us"] for x in repacks)
            repack_bytes = sum(x["source_bytes"] for x in repacks)
            repack_dest = sum(x["destination_bytes"] for x in repacks)
            repack_summary = {"repack_tensors": len(repacks), "repack_wall_us": repack_wall, "repack_cpu_sum_us": repack_cpu, "repack_source_bytes": repack_bytes, "repack_destination_bytes": repack_dest, "repack_throughput_gib_s": repack_bytes / repack_cpu * 1e6 / 1024**3 if repack_cpu else 0}
            for item in repacks:
                item.update({"run": name, "kind": kind})
                repack_rows.append(item)
            s = sample_stats(run / "cpu-samples.csv")
            l = log_stats(run / f"{kind}.log")
            summary_row = {"run": name, "kind": kind, "regime": "warm" if "warm" in name else "cold", "elapsed_us": result["elapsed_us"], "peak_rss_bytes": result["peak_rss_bytes"], **repack_summary, **s, **l, "phases": phase_values}
            summary[kind][summary_row["regime"]].append(summary_row)
            for row in csv.DictReader((run / "cpu-samples.csv").open()):
                row.update({"run": name, "kind": kind})
                memory_rows.append({"run": name, "kind": kind, "elapsed_us": row["elapsed_us"], "rss_bytes": row["rss_bytes"], "vsize_bytes": row["vsize_bytes"], "available_bytes": row["available_bytes"], "thread_count": row["thread_count"]})
                io_rows.append({"run": name, "kind": kind, "elapsed_us": row["elapsed_us"], "read_bytes": row["read_bytes"], "minor_faults": row["minor_faults"], "major_faults": row["major_faults"], "system_iowait_ticks": row["system_iowait_ticks"], "thread_delta_ticks": row["thread_delta_ticks"], "process_cpu_pct": row.get("process_cpu_pct", ""), "disk_reads_completed": row.get("disk_reads_completed", ""), "disk_sectors_read": row.get("disk_sectors_read", ""), "disk_io_ticks": row.get("disk_io_ticks", "")})

    def write_csv(path, rows):
        if not rows: return
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
    write_csv(ROOT / "phase-timings.csv", phase_rows)
    write_csv(ROOT / "repack-gguf.csv", [x for x in repack_rows if x["kind"] == "gguf"])
    write_csv(ROOT / "repack-vbuf.csv", [x for x in repack_rows if x["kind"] == "vbuf"])
    write_csv(ROOT / "memory.csv", memory_rows)
    write_csv(ROOT / "io.csv", io_rows)
    (ROOT / "phase-timings.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__": main()
