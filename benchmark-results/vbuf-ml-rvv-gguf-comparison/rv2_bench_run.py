#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pathlib
import shutil
import subprocess
import sys
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
    task_dir = pathlib.Path(f"/proc/{pid}/task")
    try:
        for entry in task_dir.iterdir():
            value = read_stat(entry.name)
            if value:
                result[entry.name] = value[0] + value[1]
    except (FileNotFoundError, ProcessLookupError):
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


def read_system():
    try:
        first = pathlib.Path("/proc/stat").read_text().splitlines()[0].split()
        values = list(map(int, first[1:]))
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


def read_disk():
    try:
        fields = pathlib.Path("/sys/block/mmcblk0/stat").read_text().split()
        return {"disk_reads_completed": int(fields[0]), "disk_sectors_read": int(fields[2]), "disk_io_ticks": int(fields[9])}
    except (FileNotFoundError, ValueError, IndexError):
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["gguf", "vbuf"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cold", action="store_true")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trace = out / "trace.csv"
    log = out / f"{args.kind}.log"
    command = ["/home/eugen/rv2_gguf_vbuf_bench", args.kind, args.model]
    env = os.environ.copy()
    env["VBUF_BENCH_TRACE"] = str(trace)

    cache_mode = "WARM_CACHE"
    if args.cold:
        subprocess.run(["sync"], check=False)
        drop = pathlib.Path("/proc/sys/vm/drop_caches")
        if os.access(drop, os.W_OK):
            drop.write_text("3\n")
            cache_mode = "COLD_CACHE_DROPPED"
        else:
            cache_mode = "COLD_CACHE_NOT_GUARANTEED"

    (out / "commands.txt").write_text(" ".join(command) + "\n")
    (out / "regime.txt").write_text(cache_mode + "\n")
    start = time.monotonic_ns()
    with log.open("w") as log_file:
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
            io = read_io(process.pid)
            mem = read_mem()
            disk = read_disk()
            if current:
                peak_rss = max(peak_rss, current[3])
            elapsed_us = (now - start) // 1000
            thread_deltas = ""
            process_cpu_pct = ""
            if current and previous:
                thread_deltas = ";".join(f"{tid}:{ticks_now - previous_threads.get(tid, ticks_now)}" for tid, ticks_now in current_threads.items())
                process_cpu_pct = ((current[0] + current[1] - previous[0] - previous[1]) / ticks) / max((now - previous_time) / 1e9, 1e-9) * 100.0
            rows.append({
                "elapsed_us": elapsed_us,
                "pid": process.pid,
                "rss_bytes": current[3] if current else "",
                "vsize_bytes": current[2] if current else "",
                "process_utime_ticks": current[0] if current else "",
                "process_stime_ticks": current[1] if current else "",
                "minor_faults": current[4] if current else "",
                "major_faults": current[5] if current else "",
                "read_bytes": io.get("read_bytes", ""),
                "syscr": io.get("syscr", ""),
                "system_user_ticks": current_system.get("user", ""),
                "system_ticks": current_system.get("system", ""),
                "system_idle_ticks": current_system.get("idle", ""),
                "system_iowait_ticks": current_system.get("iowait", ""),
                "available_bytes": mem.get("MemAvailable", ""),
                "thread_count": len(current_threads),
                "thread_delta_ticks": thread_deltas,
                "process_cpu_pct": process_cpu_pct,
                "disk_reads_completed": disk.get("disk_reads_completed", ""),
                "disk_sectors_read": disk.get("disk_sectors_read", ""),
                "disk_io_ticks": disk.get("disk_io_ticks", ""),
            })
            previous, previous_threads, previous_system, previous_time = current, current_threads, current_system, now
            time.sleep(0.1)
        return_code = process.wait()
    (out / "cpu-samples.csv").open("w").write("")
    if rows:
        with (out / "cpu-samples.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    result = {
        "kind": args.kind,
        "model": args.model,
        "cache_regime": cache_mode,
        "return_code": return_code,
        "elapsed_us": (time.monotonic_ns() - start) // 1000,
        "peak_rss_bytes": peak_rss,
        "sample_count": len(rows),
        "command": command,
    }
    (out / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return return_code


if __name__ == "__main__":
    sys.exit(main())
