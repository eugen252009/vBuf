#!/usr/bin/env python3
import csv
import json
import pathlib
import re
import statistics


ROOT = pathlib.Path(__file__).resolve().parents[1] / "research/results/cpu-repack-rv2-ab/runs"
RUNS = ["on-00", "off-00", "off-01", "on-01", "on-02", "off-02"]


def events(path):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split(",")
        if fields and fields[0] == "EVENT":
            rows.append({"name": fields[1], "us": int(fields[2]), "detail": fields[5] if len(fields) > 5 else ""})
    start = next(row["us"] for row in rows if row["name"] == "PROCESS_START")
    return {row["name"]: row["us"] - start for row in rows}


def repack(path):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split(",")
        if fields and fields[0] == "REPACK":
            rows.append({"tensor": fields[1], "source_bytes": int(fields[4]), "destination_bytes": int(fields[5]), "duration_us": int(fields[8])})
    return rows


def samples(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def at_or_before(rows, elapsed_us):
    values = [row for row in rows if row.get("rss_bytes", "") not in ("", None) and int(row["elapsed_us"]) <= elapsed_us]
    return int(values[-1]["rss_bytes"]) if values else 0


def one(name):
    root = ROOT / name
    ev = events(root / "trace.csv")
    reps = repack(root / "trace.csv")
    rows = samples(root / "cpu-samples.csv")
    log = (root / "run.log").read_text()
    metrics_match = re.search(r"AB_METRICS prompt_us=(\d+) prompt_tokens=(\d+) prompt_tok_s=([0-9.]+) first_decode_us=(\d+) decode_us=(\d+) decode_tokens=(\d+) decode_tok_s=([0-9.]+) total_us=(\d+)", log)
    if not metrics_match:
        raise RuntimeError(f"missing AB_METRICS in {name}")
    values = metrics_match.groups()
    buffers = {key: float(value) for key, value in re.findall(r"load_tensors:\s+([A-Z_]+) model buffer size =\s+([0-9.]+) MiB", log)}
    ready = ev["EXECUTION_READY"]
    decode_start = ev["FIRST_DECODE_BEGIN"]
    decode_end = ev["RUN_END"]
    steady_rows = [row for row in rows if row.get("rss_bytes", "") not in ("", None) and decode_start <= int(row["elapsed_us"]) <= decode_end]
    rss = [int(row["rss_bytes"]) for row in rows if row.get("rss_bytes", "") not in ("", None)]
    available = [int(row["available_bytes"]) for row in rows if row.get("available_bytes", "") not in ("", None)]
    temperatures = [value for row in rows for value in json.loads(row["thermal_json"]).values()]
    frequencies = [value for row in rows for value in json.loads(row["frequency_json"]).values()]
    result = {
        "run": name,
        "variant": "on" if name.startswith("on") else "off",
        "return_code": json.loads((root / "run.json").read_text())["return_code"],
        "structure_ready_us": ev["MODEL_STRUCTURE_READY"],
        "payload_end_us": ev["PAYLOAD_MATERIALIZATION_END"],
        "execution_ready_us": ready,
        "first_token_us": ev["FIRST_TOKEN"],
        "total_us": ev["RUN_END"],
        "prompt_us": int(values[0]),
        "prompt_tokens": int(values[1]),
        "prompt_tok_s": float(values[2]),
        "first_decode_us": int(values[3]),
        "decode_us": int(values[4]),
        "decode_tokens": int(values[5]),
        "decode_tok_s": float(values[6]),
        "repack_tensors": len(reps),
        "repack_source_bytes": sum(row["source_bytes"] for row in reps),
        "repack_destination_bytes": sum(row["destination_bytes"] for row in reps),
        "repack_us": sum(row["duration_us"] for row in reps),
        "peak_rss_bytes": max(rss, default=0),
        "rss_execution_ready_bytes": at_or_before(rows, ready),
        "steady_rss_bytes": statistics.median(int(row["rss_bytes"]) for row in steady_rows) if steady_rows else 0,
        "min_available_bytes": min(available, default=0),
        "max_minor_faults": max((int(row["minor_faults"]) for row in rows if row.get("minor_faults", "") not in ("", None)), default=0),
        "max_major_faults": max((int(row["major_faults"]) for row in rows if row.get("major_faults", "") not in ("", None)), default=0),
        "temperature_max_milli_c": max(temperatures, default=0),
        "frequency_min_hz": min(frequencies, default=0),
        "frequency_max_hz": max(frequencies, default=0),
        "sample_count": len(rows),
        "buffers_mib": buffers,
    }
    return result


def stats(rows, field):
    values = sorted(row[field] for row in rows)
    return {"min": values[0], "median": statistics.median(values), "max": values[-1]}


def main():
    rows = [one(name) for name in RUNS]
    summary = {"runs": rows, "by_variant": {}}
    for variant in ["on", "off"]:
        selected = [row for row in rows if row["variant"] == variant]
        summary["by_variant"][variant] = {field: stats(selected, field) for field in [
            "structure_ready_us", "payload_end_us", "execution_ready_us", "first_token_us", "total_us",
            "prompt_us", "prompt_tok_s", "first_decode_us", "decode_tok_s", "repack_tensors",
            "repack_source_bytes", "repack_destination_bytes", "repack_us", "peak_rss_bytes",
            "rss_execution_ready_bytes", "steady_rss_bytes", "min_available_bytes", "max_minor_faults", "max_major_faults"]}
    (ROOT.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (ROOT.parent / "run-metrics.csv").open("w", newline="") as file:
        fieldnames = [key for key, value in rows[0].items() if not isinstance(value, dict)]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fieldnames} for row in rows])


if __name__ == "__main__":
    main()
