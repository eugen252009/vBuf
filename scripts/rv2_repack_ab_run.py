#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pathlib
import subprocess
import time


def read_stat(pid):
    try:
        fields = pathlib.Path(f"/proc/{pid}/stat").read_text().split()
        status = pathlib.Path(f"/proc/{pid}/status").read_text()
        rss = next(int(x.split()[1]) for x in status.splitlines() if x.startswith("VmRSS:")) * 1024
        return int(fields[13]), int(fields[14]), int(fields[22]), rss, int(fields[9]), int(fields[11])
    except (FileNotFoundError, ProcessLookupError, StopIteration, ValueError):
        return None


def read_threads(pid):
    result = {}
    try:
        for entry in pathlib.Path(f"/proc/{pid}/task").iterdir():
            value = read_stat(entry.name)
            if value:
                result[entry.name] = value[0] + value[1]
    except (FileNotFoundError, ProcessLookupError):
        pass
    return result


def read_system():
    try:
        values = list(map(int, pathlib.Path("/proc/stat").read_text().splitlines()[0].split()[1:]))
        return {"user": values[0], "system": values[2], "idle": values[3], "iowait": values[4]}
    except (FileNotFoundError, ValueError, IndexError):
        return {}


def read_mem():
    result = {}
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            result[key] = int(value.strip().split()[0]) * 1024
    except (FileNotFoundError, ValueError):
        pass
    return result


def read_io(pid):
    result = {}
    try:
        for line in pathlib.Path(f"/proc/{pid}/io").read_text().splitlines():
            key, value = line.split(":", 1)
            result[key] = int(value.strip())
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return result


def read_thermal():
    values = {}
    for path in pathlib.Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            values[path.parent.name] = int(path.read_text().strip())
        except (FileNotFoundError, ValueError):
            pass
    return values


def read_frequency():
    values = {}
    for path in pathlib.Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_cur_freq"):
        try:
            values[path.parent.parent.name] = int(path.read_text().strip())
        except (FileNotFoundError, ValueError):
            pass
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["on", "off"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--off-lib", default="")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trace = out / "trace.csv"
    log_path = out / "run.log"
    command = [args.binary, "vbuf", args.model]
    env = os.environ.copy()
    env["VBUF_BENCH_TRACE"] = str(trace)
    env.pop("VBUF_AB_TOKENS", None)
    if args.variant == "off":
        env["LD_LIBRARY_PATH"] = args.off_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    config = {
        "variant": args.variant,
        "run_id": args.run_id,
        "command": command,
        "model": args.model,
        "environment": {key: env[key] for key in ["VBUF_BENCH_TRACE", "LD_LIBRARY_PATH"] if key in env},
        "parameters": {"context": 256, "batch": 64, "ubatch": 64, "threads": 8, "prompt": "Hello world", "generated_tokens": 32, "sampling": "greedy", "seed": "default"},
        "cache_regime": "WARM_CACHE",
    }
    (out / "commands.txt").write_text(" ".join(command) + "\n")
    (out / "environment.json").write_text(json.dumps(config, indent=2) + "\n")

    start = time.monotonic_ns()
    with log_path.open("w") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        previous = read_stat(process.pid)
        previous_threads = read_threads(process.pid)
        previous_system = read_system()
        previous_time = start
        rows = []
        peak_rss = 0
        while process.poll() is None:
            now = time.monotonic_ns()
            current = read_stat(process.pid)
            current_threads = read_threads(process.pid)
            current_system = read_system()
            mem = read_mem()
            io = read_io(process.pid)
            if current:
                peak_rss = max(peak_rss, current[3])
            process_cpu_pct = ""
            if current and previous:
                process_cpu_pct = ((current[0] + current[1] - previous[0] - previous[1]) / ticks) / max((now - previous_time) / 1e9, 1e-9) * 100.0
            rows.append({
                "elapsed_us": (now - start) // 1000,
                "rss_bytes": current[3] if current else "",
                "vsize_bytes": current[2] if current else "",
                "minor_faults": current[4] if current else "",
                "major_faults": current[5] if current else "",
                "read_bytes": io.get("read_bytes", ""),
                "available_bytes": mem.get("MemAvailable", ""),
                "thread_count": len(current_threads),
                "process_cpu_pct": process_cpu_pct,
                "system_user_ticks": current_system.get("user", ""),
                "system_system_ticks": current_system.get("system", ""),
                "system_idle_ticks": current_system.get("idle", ""),
                "system_iowait_ticks": current_system.get("iowait", ""),
                "thermal_json": json.dumps(read_thermal(), sort_keys=True),
                "frequency_json": json.dumps(read_frequency(), sort_keys=True),
            })
            previous, previous_threads, previous_system, previous_time = current, current_threads, current_system, now
            time.sleep(0.1)
        return_code = process.wait()

    with (out / "cpu-samples.csv").open("w", newline="") as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    result = {
        "variant": args.variant,
        "run_id": args.run_id,
        "return_code": return_code,
        "elapsed_us": (time.monotonic_ns() - start) // 1000,
        "peak_rss_bytes": peak_rss,
        "sample_count": len(rows),
    }
    (out / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
